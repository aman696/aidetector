"""
SVM Classifier for AI Image Detection.

Combines features from all eight analysis methods (FFT, Eigenvalue, Metadata,
Noise Residual, DCT Block, ELA, Gradient, and PatchCraft)
and uses a Support Vector Machine (SVM) for final classification.

Pipeline:
1. Extract features from all analyzers
2. Normalize features
3. Train SVM with RBF kernel on labeled dataset
4. Predict with confidence scores and explanations
5. Falls back to weighted voting if no trained model available
"""

import os
import tempfile
import numpy as np
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

from src.fft_analyzer import extract_fft_features, fft_score
from src.eigen_analyzer import extract_eigen_features, eigenvalue_score
from src.metadata_extractor import extract_metadata_features, metadata_score
from src.noise_analyzer import extract_noise_features, noise_score
from src.dct_analyzer import extract_dct_features, dct_score
from src.ela_analyzer import extract_ela_features, ela_score
from src.gradient_analyzer import extract_gradient_features, gradient_score
from src.patchcraft_analyzer import extract_patchcraft_features, patchcraft_score
from src.screenshot_detector import detect_screenshot
from src.utils import load_labeled_dataset, validate_image_path


def _extract_one(image_path: str) -> np.ndarray:
    """
    Module-level wrapper for FeatureExtractor.extract().

    Must be at module level (not a method) so that ProcessPoolExecutor
    can pickle it for child processes. Each worker process creates its
    own FeatureExtractor on first call (stored in a process-local cache
    via the function's default mutable argument pattern).
    """
    fe = FeatureExtractor()
    return fe.extract(image_path)


class FeatureExtractor:
    """
    Runs all eight analysis methods and builds a feature vector.

    Feature vector structure (69 features total):
    - FFT features (4): spectral_slope, slope_r_squared, high_freq_ratio, spectral_falloff
    - Eigenvalue features (8): eig_ratio_1_2, eig_ratio_2_3, eig_condition_number, eig_dominance,
                                patch_ratio_mean, patch_ratio_std, patch_dominance_mean, patch_dominance_std
    - Spectral bands (4): band_low_ratio, band_mid_ratio, band_high_ratio, band_mid_high_ratio
    - Metadata features (6): meta_tag_count, meta_camera_tags, meta_has_camera, meta_has_gps,
                              meta_has_timestamps, meta_has_software
    - Noise residual features (11): noise_variance, noise_kurtosis, noise_skewness,
                                    noise_spectral_entropy, noise_autocorrelation, noise_block_var_std,
                                    noise_ms_ratio_1_5, noise_ms_ratio_3_5,
                                    noise_rg_corr, noise_rb_corr, noise_gb_corr
    - DCT block features (8): dct_ac_energy_ratio, dct_high_freq_energy, dct_coeff_kurtosis,
                               dct_coeff_variance, dct_dc_variance, dct_zigzag_decay,
                               dct_boundary_ratio, dct_boundary_var_ratio
    - ELA features (5): ela_mean, ela_variance, ela_max, ela_uniformity, ela_block_inconsistency
    - Gradient features (5): gradient_mean, gradient_variance, gradient_kurtosis,
                              gradient_laplacian_mean, gradient_laplacian_variance
    - PatchCraft features (3): texture_contrast, texture_rich_mean, texture_poor_mean
    - RIGID drift features (15): |original - noise_perturbed| for 15 key features.
                                  Real images are more stable under noise perturbation
                                  than AI images (RIGID, arXiv 2024).
                                  drift_spectral_slope, drift_high_freq_ratio,
                                  drift_noise_variance, drift_noise_spectral_entropy,
                                  drift_noise_autocorrelation,
                                  drift_gradient_variance, drift_gradient_kurtosis,
                                  drift_gradient_laplacian_mean,
                                  drift_dct_coeff_kurtosis, drift_dct_zigzag_decay,
                                  drift_dct_ac_energy_ratio,
                                  drift_texture_contrast, drift_texture_rich_mean,
                                  drift_eig_condition_number, drift_eig_dominance
    """

    # Ordered list of feature names for consistent vector construction
    FEATURE_NAMES = [
        # FFT features (4)
        'spectral_slope', 'slope_r_squared', 'high_freq_ratio', 'spectral_falloff',
        # Eigenvalue features (8)
        'eig_ratio_1_2', 'eig_ratio_2_3', 'eig_condition_number', 'eig_dominance',
        'patch_ratio_mean', 'patch_ratio_std', 'patch_dominance_mean', 'patch_dominance_std',
        # Spectral band features (4)
        'band_low_ratio', 'band_mid_ratio', 'band_high_ratio', 'band_mid_high_ratio',
        # Metadata features (6)
        'meta_tag_count', 'meta_camera_tags', 'meta_has_camera',
        'meta_has_gps', 'meta_has_timestamps', 'meta_has_software',
        # Noise residual features (11)
        'noise_variance', 'noise_kurtosis', 'noise_skewness',
        'noise_spectral_entropy', 'noise_autocorrelation', 'noise_block_var_std',
        'noise_ms_ratio_1_5', 'noise_ms_ratio_3_5',
        'noise_rg_corr', 'noise_rb_corr', 'noise_gb_corr',
        # DCT block features (8)
        'dct_ac_energy_ratio', 'dct_high_freq_energy', 'dct_coeff_kurtosis',
        'dct_coeff_variance', 'dct_dc_variance', 'dct_zigzag_decay',
        'dct_boundary_ratio', 'dct_boundary_var_ratio',
        # ELA features (5)
        'ela_mean', 'ela_variance', 'ela_max', 'ela_uniformity', 'ela_block_inconsistency',
        # Gradient features (5)
        'gradient_mean', 'gradient_variance', 'gradient_kurtosis',
        'gradient_laplacian_mean', 'gradient_laplacian_variance',
        # PatchCraft features (3)
        'texture_contrast', 'texture_rich_mean', 'texture_poor_mean',
        # RIGID-inspired drift features (15) — |original - noise_perturbed|
        # Real images are more stable under tiny noise perturbation (arXiv 2024)
        'drift_spectral_slope', 'drift_high_freq_ratio',
        'drift_noise_variance', 'drift_noise_spectral_entropy', 'drift_noise_autocorrelation',
        'drift_gradient_variance', 'drift_gradient_kurtosis', 'drift_gradient_laplacian_mean',
        'drift_dct_coeff_kurtosis', 'drift_dct_zigzag_decay', 'drift_dct_ac_energy_ratio',
        'drift_texture_contrast', 'drift_texture_rich_mean',
        'drift_eig_condition_number', 'drift_eig_dominance',
    ]
    
    def _compute_drift_features(self, image_path: str) -> Dict[str, float]:
        """
        RIGID-inspired feature drift: extract features from a noise-perturbed
        version of the image and return |original - perturbed| for 15 key features.

        Inspired by RIGID (arXiv 2024): real images are more robust (i.e., their
        feature representation changes less) under tiny noise perturbations than
        AI-generated images when analysed with vision foundation models.
        We approximate this principle classically: large |drift| → AI-like.

        Returns:
            Dict of 15 'drift_*' features.
        """
        import tempfile, cv2 as _cv2
        _zero = {k: 0.0 for k in [
            'drift_spectral_slope', 'drift_high_freq_ratio',
            'drift_noise_variance', 'drift_noise_spectral_entropy',
            'drift_noise_autocorrelation',
            'drift_gradient_variance', 'drift_gradient_kurtosis',
            'drift_gradient_laplacian_mean',
            'drift_dct_coeff_kurtosis', 'drift_dct_zigzag_decay',
            'drift_dct_ac_energy_ratio',
            'drift_texture_contrast', 'drift_texture_rich_mean',
            'drift_eig_condition_number', 'drift_eig_dominance',
        ]}
        try:
            img = _cv2.imread(image_path)
            if img is None:
                return _zero

            # Add tiny Gaussian noise (sigma=2, well within quantization noise)
            noise = np.random.normal(0, 2, img.shape).astype(np.float64)
            noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

            # Save noisy version temporarily
            suffix = os.path.splitext(image_path)[1] or '.jpg'
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            _cv2.imwrite(tmp_path, noisy)

            try:
                orig_fft  = extract_fft_features(image_path)
                noisy_fft = extract_fft_features(tmp_path)

                orig_noise  = extract_noise_features(image_path)
                noisy_noise = extract_noise_features(tmp_path)

                orig_grad  = extract_gradient_features(image_path)
                noisy_grad = extract_gradient_features(tmp_path)

                orig_dct  = extract_dct_features(image_path)
                noisy_dct = extract_dct_features(tmp_path)

                orig_patch  = extract_patchcraft_features(image_path)
                noisy_patch = extract_patchcraft_features(tmp_path)

                orig_eigen  = extract_eigen_features(image_path)
                noisy_eigen = extract_eigen_features(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            def d(a, b, k): return abs(a.get(k, 0.0) - b.get(k, 0.0))

            return {
                'drift_spectral_slope':        d(orig_fft,   noisy_fft,   'spectral_slope'),
                'drift_high_freq_ratio':       d(orig_fft,   noisy_fft,   'high_freq_ratio'),
                'drift_noise_variance':        d(orig_noise, noisy_noise, 'noise_variance'),
                'drift_noise_spectral_entropy':d(orig_noise, noisy_noise, 'noise_spectral_entropy'),
                'drift_noise_autocorrelation': d(orig_noise, noisy_noise, 'noise_autocorrelation'),
                'drift_gradient_variance':     d(orig_grad,  noisy_grad,  'gradient_variance'),
                'drift_gradient_kurtosis':     d(orig_grad,  noisy_grad,  'gradient_kurtosis'),
                'drift_gradient_laplacian_mean':d(orig_grad, noisy_grad,  'gradient_laplacian_mean'),
                'drift_dct_coeff_kurtosis':    d(orig_dct,  noisy_dct,   'dct_coeff_kurtosis'),
                'drift_dct_zigzag_decay':      d(orig_dct,  noisy_dct,   'dct_zigzag_decay'),
                'drift_dct_ac_energy_ratio':   d(orig_dct,  noisy_dct,   'dct_ac_energy_ratio'),
                'drift_texture_contrast':      d(orig_patch, noisy_patch, 'texture_contrast'),
                'drift_texture_rich_mean':     d(orig_patch, noisy_patch, 'texture_rich_mean'),
                'drift_eig_condition_number':  d(orig_eigen, noisy_eigen, 'eig_condition_number'),
                'drift_eig_dominance':         d(orig_eigen, noisy_eigen, 'eig_dominance'),
            }
        except Exception:
            return _zero

    def extract(self, image_path: str) -> np.ndarray:
        """
        Extracts a 69-dimensional feature vector from a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            1D numpy array of features.
        """
        # Run all analyzers
        fft_features      = extract_fft_features(image_path)
        eigen_features    = extract_eigen_features(image_path)
        meta_features     = extract_metadata_features(image_path)
        noise_features    = extract_noise_features(image_path)
        dct_features      = extract_dct_features(image_path)
        ela_features      = extract_ela_features(image_path)
        gradient_features = extract_gradient_features(image_path)
        patchcraft_features = extract_patchcraft_features(image_path)
        drift_features    = self._compute_drift_features(image_path)

        # Merge all features
        all_features = {}
        all_features.update(fft_features)
        all_features.update(eigen_features)
        all_features.update(meta_features)
        all_features.update(noise_features)
        all_features.update(dct_features)
        all_features.update(ela_features)
        all_features.update(gradient_features)
        all_features.update(patchcraft_features)
        all_features.update(drift_features)

        # Build vector in consistent order
        vector = np.array([all_features.get(name, 0.0) for name in self.FEATURE_NAMES])
        return vector
    
    def extract_batch(
        self,
        image_paths: List[str],
        verbose: bool = False,
        n_workers: int = 0,
    ) -> np.ndarray:
        """
        Extracts feature vectors from multiple images in parallel.

        Each image is processed independently, so we use
        ProcessPoolExecutor to spread the work across CPU cores.
        This is the primary bottleneck during training and is
        embarrassingly parallel.

        Args:
            image_paths: List of image file paths.
            verbose:     If True, print progress.
            n_workers:   Number of worker processes.
                         0 = auto (os.cpu_count()).
                         1 = serial (useful for debugging).

        Returns:
            2D numpy array of shape (n_images, n_features).
        """
        import os as _os
        workers = n_workers if n_workers > 0 else (_os.cpu_count() or 4)
        n = len(image_paths)
        results: Dict[int, np.ndarray] = {}
        n_feat = len(self.FEATURE_NAMES)

        if workers == 1 or n <= workers:
            # Serial path (debugging or tiny batches)
            for i, path in enumerate(image_paths):
                if verbose and (i + 1) % 10 == 0:
                    print(f"  Processing {i + 1}/{n}: {_os.path.basename(path)}")
                try:
                    results[i] = self.extract(path)
                except Exception as e:
                    print(f"  Warning: {path}: {e}")
                    results[i] = np.zeros(n_feat)
        else:
            # Parallel path
            if verbose:
                print(f"  Parallel extraction: {workers} workers, {n} images")

            completed = 0
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_extract_one, path): idx
                    for idx, path in enumerate(image_paths)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"  Warning: {image_paths[idx]}: {e}")
                        results[idx] = np.zeros(n_feat)
                    completed += 1
                    if verbose and completed % 20 == 0:
                        print(f"  Processed {completed}/{n}")

        return np.array([results[i] for i in range(n)])
    
    def extract_individual_scores(self, image_path: str) -> Dict[str, float]:
        """
        Gets individual analyzer scores (not the feature vector, but the simple 0-1 scores).
        
        Args:
            image_path: Path to image file.
            
        Returns:
            Dictionary of analyzer_name → score.
        """
        return {
            'fft_score': fft_score(image_path),
            'eigenvalue_score': eigenvalue_score(image_path),
            'metadata_score': metadata_score(image_path),
            'noise_score': noise_score(image_path),
            'dct_score': dct_score(image_path),
            'ela_score': ela_score(image_path),
            'gradient_score': gradient_score(image_path),
            'patchcraft_score': patchcraft_score(image_path),
        }


class AIDetectorClassifier:
    """
    AI Image Detector using SVM classifier.
    
    Can be used in two modes:
    1. Trained mode: uses SVM with extracted features
    2. Fallback mode: uses weighted voting of individual scores
    """
    
    def __init__(self):
        self.svm = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
    
    def train(self, real_dir: str, ai_dir: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Trains the SVM classifier on labeled image directories.
        
        Args:
            real_dir: Directory of real images (label 0).
            ai_dir: Directory of AI-generated images (label 1).
            verbose: If True, print progress.
            
        Returns:
            Dictionary with training results (accuracy, cross-val scores).
        """
        if verbose:
            print("Loading labeled dataset...")
        
        paths, labels = load_labeled_dataset(real_dir, ai_dir)
        labels = np.array(labels)
        
        if verbose:
            print(f"  Real images: {np.sum(labels == 0)}")
            print(f"  AI images: {np.sum(labels == 1)}")
            print(f"  Total: {len(labels)}")
            print("\nExtracting features...")
        
        # Extract features
        X = self.feature_extractor.extract_batch(paths, verbose=verbose)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation
        if verbose:
            print("\nRunning 5-fold cross-validation...")
        
        cv_scores = cross_val_score(self.svm, X_scaled, labels, cv=min(5, len(labels) // 4))
        
        if verbose:
            print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Train on full dataset
        if verbose:
            print("\nTraining SVM on full dataset...")
        
        self.svm.fit(X_scaled, labels)
        self.is_trained = True
        
        # Training accuracy
        train_pred = self.svm.predict(X_scaled)
        train_acc = accuracy_score(labels, train_pred)
        
        if verbose:
            print(f"  Training Accuracy: {train_acc:.3f}")
            print("\nTraining complete!")
        
        return {
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'train_accuracy': float(train_acc),
            'n_real': int(np.sum(labels == 0)),
            'n_ai': int(np.sum(labels == 1)),
        }
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Classifies a single image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Dictionary with:
                - label: "Real" or "AI-Generated"
                - confidence: confidence score (0.5 to 1.0)
                - scores: individual analyzer scores
                - method: "svm" or "voting"
                - explanation: human-readable explanation
        """
        validate_image_path(image_path)

        # Screenshot pre-detection pass (Priority 3)
        screenshot_info = detect_screenshot(image_path)

        # Get individual scores for all modes
        scores = self.feature_extractor.extract_individual_scores(image_path)

        if self.is_trained:
            # SVM prediction
            features = self.feature_extractor.extract(image_path).reshape(1, -1)
            features_scaled = self.scaler.transform(features)

            prediction = self.svm.predict(features_scaled)[0]
            probabilities = self.svm.predict_proba(features_scaled)[0]

            label = "AI-Generated" if prediction == 1 else "Real"
            confidence = float(max(probabilities))
            method = "svm"
        else:
            # Fallback: weighted voting (8 analyzers)
            weighted_score = (
                0.12 * scores['fft_score'] +
                0.15 * scores['eigenvalue_score'] +
                0.10 * scores['metadata_score'] +
                0.15 * scores['noise_score'] +
                0.13 * scores['dct_score'] +
                0.10 * scores['ela_score'] +
                0.13 * scores['gradient_score'] +
                0.12 * scores['patchcraft_score']
            )

            label = "AI-Generated" if weighted_score > 0.5 else "Real"
            confidence = abs(weighted_score - 0.5) * 2
            confidence = max(0.5, min(1.0, 0.5 + confidence))
            method = "voting"

        # Build explanation
        explanation = self._build_explanation(scores, label, method)

        result = {
            'label': label,
            'confidence': confidence,
            'scores': scores,
            'method': method,
            'explanation': explanation,
        }

        # Attach screenshot warning if detected
        if screenshot_info['is_screenshot']:
            result['screenshot_warning'] = (
                "⚠ Input appears to be a screenshot or screen-rendered image. "
                "Classification may be unreliable: screenshots share signal properties "
                "with AI-generated images (no camera noise, no EXIF, no JPEG grid). "
                f"Screenshot confidence: {screenshot_info['confidence']:.0%}. "
                f"Reasons: {'; '.join(screenshot_info['reasons'])}"
            )
            result['screenshot_confidence'] = screenshot_info['confidence']

        return result
    
    def evaluate(self, real_dir: str, ai_dir: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluates the classifier on a test set.
        
        Args:
            real_dir: Directory of real test images.
            ai_dir: Directory of AI test images.
            verbose: If True, print classification report.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        paths, labels = load_labeled_dataset(real_dir, ai_dir)
        labels = np.array(labels)
        
        if verbose:
            print(f"Evaluating on {len(labels)} images...")
        
        predictions = []
        for path in paths:
            result = self.predict(path)
            pred = 1 if result['label'] == 'AI-Generated' else 0
            predictions.append(pred)
        
        predictions = np.array(predictions)
        acc = accuracy_score(labels, predictions)
        
        if verbose:
            print(f"\nAccuracy: {acc:.3f}")
            print("\nClassification Report:")
            print(classification_report(
                labels, predictions,
                target_names=['Real', 'AI-Generated'],
                zero_division=0
            ))
        
        return {
            'accuracy': float(acc),
            'report': classification_report(
                labels, predictions,
                target_names=['Real', 'AI-Generated'],
                output_dict=True,
                zero_division=0
            ),
        }
    
    def save_model(self, model_dir: str = 'models') -> str:
        """
        Saves the trained SVM model and scaler to disk.
        
        Args:
            model_dir: Directory to save model files.
            
        Returns:
            Path to saved model file.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save: model has not been trained yet.")
        
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'svm_classifier.pkl')
        
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'feature_names': FeatureExtractor.FEATURE_NAMES,
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str = 'models/svm_classifier.pkl') -> None:
        """
        Loads a trained model from disk.
        
        Args:
            model_path: Path to saved model file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.is_trained = True
        print(f"Model loaded from {model_path}")
    
    def _build_explanation(self, scores: Dict[str, float], label: str,
                           method: str) -> str:
        """
        Builds a human-readable explanation of the classification.
        """
        lines = [f"Classification: {label} (method: {method})"]
        lines.append("")
        lines.append("Individual Analysis Scores (0=Real, 1=AI):")
        
        for name, score in scores.items():
            indicator = "→ likely AI" if score > 0.6 else "→ likely Real" if score < 0.4 else "→ uncertain"
            lines.append(f"  {name}: {score:.3f} {indicator}")
        
        # Add insights
        lines.append("")
        insights = {
            'fft_score': (
                "⚠ FFT: High-frequency spectrum shows anomalous drop-off",
                "✓ FFT: Frequency spectrum follows natural 1/f power law"),
            'eigenvalue_score': (
                "⚠ Eigenvalue: Color/texture statistics deviate from natural images",
                "✓ Eigenvalue: Color/texture statistics consistent with real images"),
            'metadata_score': (
                "⚠ Metadata: Missing or suspicious EXIF data",
                "✓ Metadata: Rich camera EXIF data present"),
            'noise_score': (
                "⚠ Noise: Residual noise pattern inconsistent with camera sensor",
                "✓ Noise: Natural sensor noise pattern detected"),
            'dct_score': (
                "⚠ DCT: Block coefficient distribution shows synthetic patterns",
                "✓ DCT: Coefficient distribution consistent with natural images"),
            'ela_score': (
                "⚠ ELA: Compression artifacts suggest non-camera origin",
                "✓ ELA: Compression history consistent with camera output"),
            'gradient_score': (
                "⚠ Gradient: Edge distribution too smooth/regular for a real photo",
                "✓ Gradient: Heavy-tailed edge distribution consistent with real scene"),
            'patchcraft_score': (
                "⚠ PatchCraft: Rich/poor texture contrast elevated (generative artifact)",
                "✓ PatchCraft: Texture richness contrast within natural range"),
        }
        for name, (ai_msg, real_msg) in insights.items():
            if name in scores:
                if scores[name] > 0.6:
                    lines.append(ai_msg)
                elif scores[name] < 0.4:
                    lines.append(real_msg)
        
        return "\n".join(lines)


def classify_image(image_path: str, model_path: str = 'models/svm_classifier.pkl') -> Dict[str, Any]:
    """
    Convenience function: classifies a single image.
    Loads SVM model if available, otherwise uses weighted voting.
    
    Args:
        image_path: Path to image file.
        model_path: Path to trained model file.
        
    Returns:
        Classification result dictionary.
    """
    classifier = AIDetectorClassifier()
    
    if os.path.exists(model_path):
        try:
            classifier.load_model(model_path)
        except Exception as e:
            print(f"Warning: Could not load model ({e}), using fallback voting.")
    
    return classifier.predict(image_path)
