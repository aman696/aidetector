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


def train_model(real_dir: str = 'data/real', ai_dir: str = 'data/ai_generated',
                model_dir: str = 'models'):
    """Trains the SVM classifier."""
    print_banner()
    print("Training SVM Classifier")
    print("-" * 60)
    
    classifier = AIDetectorClassifier()
    
    start_time = time.time()
    results = classifier.train(real_dir, ai_dir, verbose=True)
    elapsed = time.time() - start_time
    
    # Save the model
    model_path = classifier.save_model(model_dir)
    
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Cross-validation accuracy: {results['cv_accuracy_mean']:.3f} (+/- {results['cv_accuracy_std']:.3f})")
    print(f"Training accuracy: {results['train_accuracy']:.3f}")
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
