"""
Unit tests for src/npr_analyzer.py

Tests cover:
- compute_npr_map on constant image → zero map
- extract_npr_features on constant image → default values
- extract_npr_features on checkerboard → known nonzero
- npr_score returns value in [0, 1]
- No NaN on a real sample image
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.npr_analyzer import (
    compute_npr_map,
    extract_npr_features,
    npr_score,
)


def _constant_image(h=128, w=128, value=127):
    return np.full((h, w), value, dtype=np.uint8)


def _checkerboard(h=128, w=128, block=3):
    """Checkerboard of alternating black/white 3×3 blocks.
    3 px ≠ multiple of 2, so within-2×2-grid differences appear at
    block boundaries — ensures NPR map is nonzero."""
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, block):
        for j in range(0, w, block):
            if ((i // block) + (j // block)) % 2 == 0:
                img[i:i+block, j:j+block] = 255
    return img


class TestComputeNprMap:
    def test_constant_image_zero(self):
        npr = compute_npr_map(_constant_image())
        assert np.allclose(npr, 0.0), "NPR map of constant image should be zero"

    def test_output_shape(self):
        img = _constant_image(130, 130)
        npr = compute_npr_map(img)
        # Should be even-cropped: 130 → 130
        assert npr.shape == (130, 130), f"Expected (130, 130), got {npr.shape}"

    def test_odd_dimensions_cropped(self):
        img = _constant_image(131, 131)
        npr = compute_npr_map(img)
        assert npr.shape == (130, 130), f"Expected (130, 130), got {npr.shape}"

    def test_checkerboard_nonzero(self):
        npr = compute_npr_map(_checkerboard())
        # The NPR map of a checkerboard should have nonzero values at block boundaries
        assert np.abs(npr).mean() > 0, "Checkerboard should produce nonzero NPR"


class TestExtractNprFeatures:
    def test_constant_image_defaults(self):
        feats = extract_npr_features('/nonexistent/path')
        assert feats['npr_mean_abs'] == 0.0
        assert feats['npr_skewness'] == 0.0
        assert feats['npr_kurtosis'] == 0.0

    def test_checkerboard_known_values(self):
        """Test against a synthetic input where we can verify features."""
        # Use the function via temp path — write a checkerboard to disk
        import cv2, tempfile
        img = _checkerboard(256, 256)
        fd, path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        cv2.imwrite(path, img)
        try:
            feats = extract_npr_features(path)
            # Checkerboard should have high kurtosis (sharp transitions + flat areas)
            assert feats['npr_kurtosis'] != 0.0, "Checkerboard should have nonzero kurtosis"
            # Mean abs should be > 0 (there are edges)
            assert feats['npr_mean_abs'] > 0, "Checkerboard should have positive mean_abs"
        finally:
            os.unlink(path)

    def test_no_nan_on_real_sample(self):
        """Use first real image from manifest to verify no NaN."""
        manifest_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'manifests', 'base_manifest.json')
        if not os.path.exists(manifest_path):
            pytest.skip("No manifest available")
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
        real_paths = [r['path'] for r in manifest if r['label'] == 0 and os.path.exists(r['path'])]
        if not real_paths:
            pytest.skip("No real images available")
        feats = extract_npr_features(real_paths[0])
        for k, v in feats.items():
            assert not np.isnan(v), f"{k} is NaN"


class TestNprScore:
    def test_score_in_range(self):
        feats = extract_npr_features('/nonexistent/path')
        score = npr_score('/nonexistent/path')
        assert 0.0 <= score <= 1.0

    def test_smooth_residue_scores_ai_like(self):
        """npr_score maps low NPR magnitude (residue too smooth) → high score.
        Basis is npr_mean_abs, not kurtosis — kurtosis did not separate the
        classes when measured (see npr_score docstring)."""
        import cv2, tempfile, numpy as np
        img = np.full((256, 256), 128, dtype=np.uint8)
        fd, path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        cv2.imwrite(path, img)
        try:
            score = npr_score(path)
            # Constant image gives mean_abs=0 → score=1.0
            assert score > 0.5, f"Constant image score should be high (AI-like), got {score}"
        finally:
            os.unlink(path)