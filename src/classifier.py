"""
SVM Classifier for AI Image Detection.

Combines features from all three analysis methods (FFT, Eigenvalue, Metadata)
and uses a Support Vector Machine (SVM) for final classification.

Pipeline:
1. Extract features from all three analyzers
2. Normalize features
3. Train SVM with RBF kernel on labeled dataset
4. Predict with confidence scores and explanations
5. Falls back to weighted voting if no trained model available
"""

import os
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional, Any
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

from src.fft_analyzer import extract_fft_features, fft_score
from src.eigen_analyzer import extract_eigen_features, eigenvalue_score
from src.metadata_extractor import extract_metadata_features, metadata_score
from src.utils import load_labeled_dataset, validate_image_path


class FeatureExtractor:
    """
    Runs all three analysis methods and builds a feature vector.
    
    Feature vector structure (14 features total):
    - FFT features (4): spectral_slope, slope_r_squared, high_freq_ratio, spectral_falloff
    - Eigenvalue features (8): eig_ratio_1_2, eig_ratio_2_3, eig_condition_number, eig_dominance,
                                patch_ratio_mean, patch_ratio_std, patch_dominance_mean, patch_dominance_std
    - Spectral bands (4): band_low_ratio, band_mid_ratio, band_high_ratio, band_mid_high_ratio
    - Metadata features (6): meta_tag_count, meta_camera_tags, meta_has_camera, meta_has_gps,
                              meta_has_timestamps, meta_has_software
    """
    
    # Ordered list of feature names for consistent vector construction
    FEATURE_NAMES = [
        # FFT features
        'spectral_slope', 'slope_r_squared', 'high_freq_ratio', 'spectral_falloff',
        # Eigenvalue features
        'eig_ratio_1_2', 'eig_ratio_2_3', 'eig_condition_number', 'eig_dominance',
        'patch_ratio_mean', 'patch_ratio_std', 'patch_dominance_mean', 'patch_dominance_std',
        # Spectral band features
        'band_low_ratio', 'band_mid_ratio', 'band_high_ratio', 'band_mid_high_ratio',
        # Metadata features
        'meta_tag_count', 'meta_camera_tags', 'meta_has_camera',
        'meta_has_gps', 'meta_has_timestamps', 'meta_has_software',
    ]
    
    def extract(self, image_path: str) -> np.ndarray:
        """
        Extracts a feature vector from a single image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            1D numpy array of features.
        """
        # Run all analyzers
        fft_features = extract_fft_features(image_path)
        eigen_features = extract_eigen_features(image_path)
        meta_features = extract_metadata_features(image_path)
        
        # Merge all features
        all_features = {}
        all_features.update(fft_features)
        all_features.update(eigen_features)
        all_features.update(meta_features)
        
        # Build vector in consistent order
        vector = np.array([all_features.get(name, 0.0) for name in self.FEATURE_NAMES])
        return vector
    
    def extract_batch(self, image_paths: List[str], verbose: bool = False) -> np.ndarray:
        """
        Extracts feature vectors from multiple images.
        
        Args:
            image_paths: List of image file paths.
            verbose: If True, print progress.
            
        Returns:
            2D numpy array of shape (n_images, n_features).
        """
        vectors = []
        for i, path in enumerate(image_paths):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processing {i + 1}/{len(image_paths)}: {os.path.basename(path)}")
            try:
                vec = self.extract(path)
                vectors.append(vec)
            except Exception as e:
                print(f"  Warning: Failed to process {path}: {e}")
                vectors.append(np.zeros(len(self.FEATURE_NAMES)))
        
        return np.array(vectors)
    
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
            # Fallback: weighted voting
            weighted_score = (
                0.35 * scores['fft_score'] +
                0.35 * scores['eigenvalue_score'] +
                0.30 * scores['metadata_score']
            )
            
            label = "AI-Generated" if weighted_score > 0.5 else "Real"
            confidence = abs(weighted_score - 0.5) * 2  # Map distance from 0.5 to 0-1
            confidence = max(0.5, min(1.0, 0.5 + confidence))
            method = "voting"
        
        # Build explanation
        explanation = self._build_explanation(scores, label, method)
        
        return {
            'label': label,
            'confidence': confidence,
            'scores': scores,
            'method': method,
            'explanation': explanation,
        }
    
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
        if scores['fft_score'] > 0.6:
            lines.append("⚠ FFT: High-frequency spectrum shows anomalous drop-off (AI artifact)")
        elif scores['fft_score'] < 0.4:
            lines.append("✓ FFT: Frequency spectrum follows natural 1/f power law")
        
        if scores['eigenvalue_score'] > 0.6:
            lines.append("⚠ Eigenvalue: Color/texture statistics deviate from natural images")
        elif scores['eigenvalue_score'] < 0.4:
            lines.append("✓ Eigenvalue: Color/texture statistics consistent with real images")
        
        if scores['metadata_score'] > 0.6:
            lines.append("⚠ Metadata: Missing or suspicious EXIF data")
        elif scores['metadata_score'] < 0.4:
            lines.append("✓ Metadata: Rich camera EXIF data present")
        
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
