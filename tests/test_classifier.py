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
# Find valid image paths from the base manifest
_MANIFEST_PATH = os.path.join(DATA_DIR, 'manifests', 'base_manifest.json')
if os.path.exists(_MANIFEST_PATH):
    import json as _json
    with open(_MANIFEST_PATH) as _f:
        _manifest = _json.load(_f)
    _real_candidates = [r['path'] for r in _manifest if r['label'] == 0 and os.path.exists(r['path'])]
    _ai_candidates = [r['path'] for r in _manifest if r['label'] == 1 and os.path.exists(r['path'])]
    REAL_IMG = _real_candidates[0] if _real_candidates else ''
    AI_IMG = _ai_candidates[0] if _ai_candidates else ''
else:
    REAL_IMG = ''
    AI_IMG = ''
# For training integration tests, use the root real dir (contains subdirs coco, openfake)
REAL_DIR = os.path.join(DATA_DIR, 'real')
# AI images are scattered across Gemini, GPT, mixed subdirs — symlink from a temp dir in test
AI_DIR = os.path.join(DATA_DIR, 'ai_generated')  # will be None unless it exists


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
        """Feature vector should have 85 elements (4+8+4+6+11+8+5+5+3+6+10+15)."""
        fe = FeatureExtractor()
        vec = fe.extract(REAL_IMG)
        assert len(vec) == len(FeatureExtractor.FEATURE_NAMES)
        assert len(vec) == 85

    def test_extract_no_nans(self):
        fe = FeatureExtractor()
        vec = fe.extract(REAL_IMG)
        assert not np.any(np.isnan(vec)), "Feature vector contains NaN"

    def test_extract_deterministic(self):
        """Feature vector should be approximately deterministic.
        RIGID drift features use random noise, so we allow small tolerance."""
        fe = FeatureExtractor()
        v1 = fe.extract(REAL_IMG)
        v2 = fe.extract(REAL_IMG)
        # Base 70 features (non-drift) must be exactly equal
        np.testing.assert_array_equal(v1[:70], v2[:70])
        # Drift features (last 15) may differ slightly due to random noise perturbation.
        # Gradient variance drift can be ~200–300 so use loose atol.
        assert np.allclose(v1[70:], v2[70:], atol=50.0), \
            "Drift features differ too much between two calls"

    def test_extract_individual_scores_keys(self):
        fe = FeatureExtractor()
        scores = fe.extract_individual_scores(REAL_IMG)
        assert set(scores.keys()) == {
            'fft_score', 'eigenvalue_score', 'metadata_score',
            'noise_score', 'dct_score', 'ela_score',
            'gradient_score', 'patchcraft_score', 'npr_score',
            'screenshot_img_score',
        }

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
        # These keys must always be present
        required = {'label', 'confidence', 'scores', 'method', 'explanation'}
        assert required <= result.keys(), \
            f"Missing required keys: {required - result.keys()}"
        # screenshot_warning / screenshot_confidence are optional (only added when
        # the image triggers the screenshot detector heuristics)
        allowed_extra = {'screenshot_warning', 'screenshot_confidence'}
        unexpected = result.keys() - required - allowed_extra
        assert not unexpected, f"Unexpected extra keys in result: {unexpected}"


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
        """Create temp dirs with 5 real + 5 AI symlinks for fast training.
        Uses manifest to find existing images since AI images are scattered
        across many subdirectories."""
        import json as _json
        _manifest_path = os.path.join(DATA_DIR, 'manifests', 'base_manifest.json')
        if not os.path.exists(_manifest_path):
            pytest.skip("No manifest available for train integration test")
        with open(_manifest_path) as _f:
            _m = _json.load(_f)
        _real_candidates = [r for r in _m if r['label'] == 0 and os.path.exists(r['path'])][:5]
        _ai_candidates = [r for r in _m if r['label'] == 1 and os.path.exists(r['path'])][:5]

        real_dir = tmp_path / 'real'
        ai_dir = tmp_path / 'ai'
        real_dir.mkdir()
        ai_dir.mkdir()

        for r in _real_candidates:
            os.symlink(r['path'], str(real_dir / os.path.basename(r['path'])))
        for r in _ai_candidates:
            os.symlink(r['path'], str(ai_dir / os.path.basename(r['path'])))

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
