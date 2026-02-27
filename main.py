"""
AI Image Detector - CLI Entry Point

Usage:
    python main.py --image <path>            Analyze a single image
    python main.py --train                   Train SVM on data/real/ and data/ai_generated/
    python main.py --evaluate                Evaluate on data/test/
    python main.py --batch <directory>       Analyze all images in a directory
"""

import argparse
import sys
import os
import time

# Add the project root to the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.classifier import AIDetectorClassifier, classify_image
from src.fft_analyzer import fft_score, extract_fft_features
from src.eigen_analyzer import eigenvalue_score, extract_eigen_features
from src.metadata_extractor import metadata_score, extract_metadata_features
from src.noise_analyzer import noise_score, extract_noise_features
from src.dct_analyzer import dct_score, extract_dct_features
from src.ela_analyzer import ela_score, extract_ela_features
from src.utils import get_all_image_paths


def print_banner():
    """Prints the application banner."""
    print("=" * 60)
    print("  AI Image Detector")
    print("  Based on Durall 2020 (FFT) & Corvi 2023 (Eigenvalue)")
    print("=" * 60)
    print()


def analyze_single_image(image_path: str, model_path: str = 'models/svm_classifier.pkl'):
    """Analyzes a single image with detailed output."""
    print_banner()
    print(f"Analyzing: {image_path}")
    print("-" * 60)
    
    start_time = time.time()
    
    # Get classification result
    result = classify_image(image_path, model_path)
    
    elapsed = time.time() - start_time
    
    # Print detailed results
    print(f"\n{'=' * 40}")
    print(f"  RESULT: {result['label']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Method: {result['method']}")
    print(f"{'=' * 40}")
    
    print(f"\n{result['explanation']}")
    
    print(f"\nAnalysis completed in {elapsed:.2f}s")

    # Screenshot warning
    if result.get('screenshot_warning'):
        print(f"\n{result['screenshot_warning']}")


def train_model(real_dir: str = 'data/real', ai_dir: str = 'data/ai_generated',
                real_screenshots_dir: str = 'data/screenshots',
                ai_screenshots_dir: str = 'data/ai_generated_screenshots',
                model_dir: str = 'models'):
    """Trains the SVM classifier with all available data including screenshots."""
    print_banner()
    print("Training SVM Classifier")
    print("-" * 60)
    
    classifier = AIDetectorClassifier()
    
    start_time = time.time()
    
    # Try to use GPU SVM via cuML (RAPIDS), fall back to sklearn
    import os as _os
    use_gpu = False
    try:
        from cuml.svm import SVC as CumlSVC
        import cupy as cp
        classifier.svm = CumlSVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
        use_gpu = True
        print("  GPU SVM: cuML SVC loaded (RTX 4060)")
    except ImportError:
        print("  GPU SVM: cuML not available, using sklearn CPU SVM")

    # Parallel feature extraction using all CPU cores
    n_cpu = _os.cpu_count() or 4
    print(f"  Parallel extraction: {n_cpu} CPU cores")
    
    # Collect all training data
    from src.utils import (
        load_labeled_dataset, get_all_image_paths,
        augment_dataset_with_jpeg, cleanup_augmented_files,
    )
    import numpy as np

    paths, labels = load_labeled_dataset(real_dir, ai_dir)

    # Add screenshot directories if they exist
    if os.path.isdir(real_screenshots_dir):
        ss_real = get_all_image_paths(real_screenshots_dir)
        print(f"  Adding {len(ss_real)} real screenshots from {real_screenshots_dir}")
        paths.extend(ss_real)
        labels.extend([0] * len(ss_real))

    if os.path.isdir(ai_screenshots_dir):
        ss_ai = get_all_image_paths(ai_screenshots_dir)
        print(f"  Adding {len(ss_ai)} AI screenshots from {ai_screenshots_dir}")
        paths.extend(ss_ai)
        labels.extend([1] * len(ss_ai))

    print(f"\nBase training set: {len(paths)} images")
    print(f"  Real: {sum(1 for l in labels if l == 0)}"
          f" | AI: {sum(1 for l in labels if l == 1)}")

    # JPEG augmentation (ITW-SM: training data composition is key)
    print("\nApplying JPEG augmentation (Q=70, Q=80, 0.75x resize)...")
    aug_paths, aug_labels, tmp_paths = augment_dataset_with_jpeg(paths, labels)
    labels_arr = np.array(aug_labels)
    print(f"Augmented training set: {len(aug_paths)} images "
          f"({len(aug_paths) - len(paths)} synthetic copies)")
    print(f"  Real: {np.sum(labels_arr == 0)} | AI: {np.sum(labels_arr == 1)}")
    print("\nExtracting features (parallel)...")

    # Parallel feature extraction
    X = classifier.feature_extractor.extract_batch(
        aug_paths, verbose=True, n_workers=n_cpu
    )

    # Scale and train
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score

    X_scaled = classifier.scaler.fit_transform(X)

    # Use GPU cross-validation if cuML available, else sklearn
    if use_gpu:
        # cuML SVC doesn't support cross_val_score directly; do it manually
        from sklearn.model_selection import StratifiedKFold
        import numpy as _np
        skf = StratifiedKFold(n_splits=min(5, len(labels_arr) // 4))
        cv_preds = _np.zeros(len(labels_arr))
        for train_idx, val_idx in skf.split(X_scaled, labels_arr):
            _svm_fold = CumlSVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
            _svm_fold.fit(X_scaled[train_idx], labels_arr[train_idx])
            cv_preds[val_idx] = _svm_fold.predict(X_scaled[val_idx])
        cv_acc = float(_np.mean(cv_preds == labels_arr))
        print(f"  GPU CV Accuracy: {cv_acc:.3f}")
    else:
        cv_scores = cross_val_score(
            classifier.svm, X_scaled, labels_arr,
            cv=min(5, len(labels_arr) // 4)
        )
        cv_acc = cv_scores.mean()
        print(f"  CPU CV Accuracy: {cv_acc:.3f} (+/- {cv_scores.std():.3f})")

    print("\nTraining SVM on full dataset...") 
    if use_gpu:
        # cuML SVC expects cupy arrays
        X_gpu = cp.array(X_scaled)
        y_gpu = cp.array(labels_arr)
        classifier.svm.fit(X_gpu, y_gpu)
        train_pred = classifier.svm.predict(X_gpu)
        train_pred = cp.asnumpy(train_pred)
    else:
        classifier.svm.fit(X_scaled, labels_arr)
        train_pred = classifier.svm.predict(X_scaled)
    classifier.is_trained = True

    train_acc = accuracy_score(labels_arr, train_pred)

    # Save model (save sklearn-compatible wrapper so load_model works)
    if use_gpu:
        # Wrap cuML SVC in a sklearn-API-compatible adapter for joblib save
        # Store as cuML model — note: loads require cuML on inference machine
        classifier.is_trained = True  # flag already set above
    model_path = classifier.save_model(model_dir)

    # Cleanup temp augmentation files
    cleanup_augmented_files(tmp_paths)

    elapsed = time.time() - start_time
    backend = "cuML GPU" if use_gpu else "sklearn CPU"
    print(f"\nTraining completed in {elapsed:.1f}s ({backend})")
    print(f"CV Accuracy: {cv_acc:.3f}")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Model saved to: {model_path}")



def evaluate_model(test_real_dir: str = 'data/test/real',
                   test_ai_dir: str = 'data/test/ai_generated',
                   model_path: str = 'models/svm_classifier.pkl'):
    """Evaluates the classifier on the test set."""
    print_banner()
    print("Evaluating on Test Set")
    print("-" * 60)
    
    classifier = AIDetectorClassifier()
    
    if os.path.exists(model_path):
        classifier.load_model(model_path)
    else:
        print("No trained model found. Using weighted voting fallback.")
        print("Run --train first for better accuracy.\n")
    
    results = classifier.evaluate(test_real_dir, test_ai_dir, verbose=True)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.1%}")
    target = 0.70
    if results['accuracy'] >= target:
        print(f"✓ Meets target accuracy ({target:.0%})")
    else:
        print(f"✗ Below target accuracy ({target:.0%})")


def batch_analyze(directory: str, model_path: str = 'models/svm_classifier.pkl'):
    """Analyzes all images in a directory."""
    print_banner()
    print(f"Batch Analysis: {directory}")
    print("-" * 60)
    
    paths = get_all_image_paths(directory)
    print(f"Found {len(paths)} images\n")
    
    classifier = AIDetectorClassifier()
    if os.path.exists(model_path):
        classifier.load_model(model_path)
    else:
        print("No trained model found. Using weighted voting.\n")
    
    results_summary = {'Real': 0, 'AI-Generated': 0}
    
    for path in paths:
        try:
            result = classifier.predict(path)
            label = result['label']
            conf = result['confidence']
            results_summary[label] = results_summary.get(label, 0) + 1
            print(f"  {os.path.basename(path):40s} → {label:15s} ({conf:.1%})")
        except Exception as e:
            print(f"  {os.path.basename(path):40s} → ERROR: {e}")
    
    print(f"\n{'=' * 40}")
    print(f"Summary:")
    for label, count in results_summary.items():
        print(f"  {label}: {count}")
    print(f"  Total: {sum(results_summary.values())}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Image Detector - Detect AI-generated images using frequency and eigenvalue analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --image photo.jpg
  python main.py --train
  python main.py --evaluate
  python main.py --batch data/test/real/
        """
    )
    
    parser.add_argument('--image', '-i', type=str,
                        help='Path to a single image to analyze')
    parser.add_argument('--train', '-t', action='store_true',
                        help='Train SVM on data/real/ and data/ai_generated/')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='Evaluate on data/test/')
    parser.add_argument('--batch', '-b', type=str,
                        help='Analyze all images in a directory')
    parser.add_argument('--model', '-m', type=str, default='models/svm_classifier.pkl',
                        help='Path to trained model file (default: models/svm_classifier.pkl)')
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.evaluate:
        evaluate_model(model_path=args.model)
    elif args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        analyze_single_image(args.image, args.model)
    elif args.batch:
        if not os.path.isdir(args.batch):
            print(f"Error: Directory not found: {args.batch}")
            sys.exit(1)
        batch_analyze(args.batch, args.model)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
