"""
Unit tests for src/classifier.py

Tests cover:
- FeatureExtractor.extract() returns correct-length vector
- FeatureExtractor.extract_individual_scores() returns 3 scores in [0,1]
- AIDetectorClassifier.predict() works in fallback (voting) mode
- classify_image() convenience function works without trained model
- AIDetectorClassifier train/predict/save/load round-trip (integration)
"""

import pytest
import numpy as np
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classifier import (
    FeatureExtractor,
    AIDetectorClassifier,
    classify_image,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
REAL_IMG = os.path.join(DATA_DIR, 'real', 'real_001.jpg')
AI_IMG = os.path.join(DATA_DIR, 'ai_generated', 'ai_gemini_img_001.png')
REAL_DIR = os.path.join(DATA_DIR, 'real')
AI_DIR = os.path.join(DATA_DIR, 'ai_generated')


# =====================================================================
# FeatureExtractor
# =====================================================================

class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_extract_returns_ndarray(self):
        fe = FeatureExtractor()
        vec = fe.extract(REAL_IMG)
        assert isinstance(vec, np.ndarray)

    def test_extract_vector_length(self):
        """Feature vector should have 22 elements (4 FFT + 12 eigen + 6 metadata)."""
        fe = FeatureExtractor()
        vec = fe.extract(REAL_IMG)
        assert len(vec) == len(FeatureExtractor.FEATURE_NAMES)
        assert len(vec) == 22

    def test_extract_no_nans(self):
        fe = FeatureExtractor()
        vec = fe.extract(REAL_IMG)
        assert not np.any(np.isnan(vec)), "Feature vector contains NaN"

    def test_extract_deterministic(self):
        fe = FeatureExtractor()
        v1 = fe.extract(REAL_IMG)
        v2 = fe.extract(REAL_IMG)
        np.testing.assert_array_equal(v1, v2)

    def test_extract_individual_scores_keys(self):
        fe = FeatureExtractor()
        scores = fe.extract_individual_scores(REAL_IMG)
        assert set(scores.keys()) == {'fft_score', 'eigenvalue_score', 'metadata_score'}

    def test_extract_individual_scores_range(self):
        fe = FeatureExtractor()
        scores = fe.extract_individual_scores(REAL_IMG)
        for name, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{name} = {val} out of range"


# =====================================================================
# AIDetectorClassifier — voting fallback
# =====================================================================

class TestAIDetectorClassifierVoting:
    """Tests for AIDetectorClassifier in fallback (voting) mode."""

    def test_predict_returns_dict(self):
        clf = AIDetectorClassifier()
        result = clf.predict(REAL_IMG)
        assert isinstance(result, dict)

    def test_predict_output_keys(self):
        clf = AIDetectorClassifier()
        result = clf.predict(REAL_IMG)
        expected = {'label', 'confidence', 'scores', 'method', 'explanation'}
        assert result.keys() == expected

    def test_predict_label_type(self):
        clf = AIDetectorClassifier()
        result = clf.predict(REAL_IMG)
        assert result['label'] in ('Real', 'AI-Generated')

    def test_predict_method_is_voting(self):
        """Without training, should use voting fallback."""
        clf = AIDetectorClassifier()
        result = clf.predict(REAL_IMG)
        assert result['method'] == 'voting'

    def test_predict_confidence_range(self):
        clf = AIDetectorClassifier()
        result = clf.predict(REAL_IMG)
        assert 0.5 <= result['confidence'] <= 1.0

    def test_predict_explanation_is_string(self):
        clf = AIDetectorClassifier()
        result = clf.predict(REAL_IMG)
        assert isinstance(result['explanation'], str)
        assert len(result['explanation']) > 0


# =====================================================================
# classify_image — convenience function
# =====================================================================

class TestClassifyImage:
    """Tests for classify_image() convenience function."""

    def test_returns_result_without_model(self):
        """Should work even when no model file exists."""
        result = classify_image(REAL_IMG, model_path='/nonexistent/model.pkl')
        assert result['label'] in ('Real', 'AI-Generated')
        assert result['method'] == 'voting'


# =====================================================================
# AIDetectorClassifier — train/save/load round-trip (integration)
# =====================================================================

class TestClassifierTrainIntegration:
    """Integration test for train → save → load → predict cycle.

    Uses only a small subset of images (first 5 from each class) to keep
    tests fast. Marks the test as slow so it can be skipped in quick runs.
    """

    @pytest.fixture
    def small_dirs(self, tmp_path):
        """Create temp dirs with 5 real + 5 AI symlinks for fast training."""
        real_dir = tmp_path / 'real'
        ai_dir = tmp_path / 'ai'
        real_dir.mkdir()
        ai_dir.mkdir()

        # Symlink first 5 images from each class
        real_files = sorted(os.listdir(REAL_DIR))[:5]
        ai_files = sorted(os.listdir(AI_DIR))[:5]

        for f in real_files:
            os.symlink(os.path.join(REAL_DIR, f), str(real_dir / f))
        for f in ai_files:
            os.symlink(os.path.join(AI_DIR, f), str(ai_dir / f))

        return str(real_dir), str(ai_dir)

    def test_train_and_predict(self, small_dirs):
        real_dir, ai_dir = small_dirs
        clf = AIDetectorClassifier()

        # Train
        results = clf.train(real_dir, ai_dir, verbose=False)
        assert 'train_accuracy' in results
        assert results['train_accuracy'] > 0  # Should learn something

        # Predict in SVM mode
        result = clf.predict(REAL_IMG)
        assert result['method'] == 'svm'
        assert result['label'] in ('Real', 'AI-Generated')

    def test_save_and_load(self, small_dirs, tmp_path):
        real_dir, ai_dir = small_dirs
        model_dir = str(tmp_path / 'models')

        # Train and save
        clf = AIDetectorClassifier()
        clf.train(real_dir, ai_dir, verbose=False)
        model_path = clf.save_model(model_dir)
        assert os.path.exists(model_path)

        # Load into new classifier
        clf2 = AIDetectorClassifier()
        clf2.load_model(model_path)
        assert clf2.is_trained

        # Predictions should match
        r1 = clf.predict(REAL_IMG)
        r2 = clf2.predict(REAL_IMG)
        assert r1['label'] == r2['label']

    def test_save_without_training_raises(self):
        clf = AIDetectorClassifier()
        with pytest.raises(RuntimeError):
            clf.save_model('/tmp/test_model')

    def test_load_nonexistent_model_raises(self):
        clf = AIDetectorClassifier()
        with pytest.raises(FileNotFoundError):
            clf.load_model('/nonexistent/model.pkl')
