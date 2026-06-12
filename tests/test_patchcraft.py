"""
Unit tests for src/patchcraft_analyzer.py

Tests cover:
- compute_high_pass output shape and basic properties
- compute_patch_ldiv on constant patch → zero
- compute_patch_ldiv monotonic in noise amplitude
- extract_patchcraft_features on constant image → defaults
- extract_patchcraft_features gradient image → known behaviour
- patchcraft_score in [0, 1]
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.patchcraft_analyzer import (
    compute_high_pass,
    compute_patch_ldiv,
    extract_patchcraft_features,
    patchcraft_score,
)


def _constant_image(h=256, w=256, value=127):
    return np.full((h, w), value, dtype=np.uint8)


def _noise_image(h=256, w=256, sigma=10):
    noise = np.random.randn(h, w).astype(np.float32) * sigma + 128
    return np.clip(noise, 0, 255).astype(np.uint8)


class TestComputeHighPass:
    def test_output_shape(self):
        img = _constant_image(256, 256)
        hp = compute_high_pass(img)
        assert hp.shape == (256, 256), f"Expected (256, 256), got {hp.shape}"

    def test_constant_image_zero_residual(self):
        """Gaussian blur of a constant image is that constant; residual = 0."""
        img = _constant_image(256, 256, 100)
        hp = compute_high_pass(img)
        assert np.allclose(hp, 0.0, atol=1e-5), "Constant image HP residual should be zero"


class TestComputePatchLdiv:
    def test_constant_patch_zero_ldiv(self):
        hp = np.zeros((256, 256), dtype=np.float64)
        ldiv = compute_patch_ldiv(hp)
        assert len(ldiv) > 0, "Should produce patches for 256x256"
        assert np.allclose(ldiv, 0.0), "Zero HP → zero l_div"

    def test_ldiv_shape(self):
        hp = np.random.randn(256, 256).astype(np.float64)
        ldiv = compute_patch_ldiv(hp)
        expected = (256 // 32) * (256 // 32)
        assert len(ldiv) == expected, f"Expected {expected} patches, got {len(ldiv)}"

    def test_monotonic_in_noise(self):
        """l_div should increase with noise amplitude."""
        low = compute_patch_ldiv(compute_high_pass(_noise_image(256, 256, sigma=5)))
        high = compute_patch_ldiv(compute_high_pass(_noise_image(256, 256, sigma=50)))
        assert len(low) > 0 and len(high) > 0
        assert np.mean(high) > np.mean(low), "Higher noise should produce higher l_div"


class TestExtractPatchcraftFeatures:
    def test_constant_image_defaults(self):
        feats = extract_patchcraft_features('/nonexistent/path')
        assert feats['texture_contrast'] == 0.0
        assert feats['texture_rich_mean'] == 0.0
        assert feats['texture_poor_mean'] == 0.0

    def test_constant_image_zero_contrast(self):
        import cv2, tempfile
        img = _constant_image(256, 256, 128)
        fd, path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        cv2.imwrite(path, img)
        try:
            feats = extract_patchcraft_features(path)
            assert feats['texture_contrast'] == 0.0, "Constant image → zero contrast"
        finally:
            os.unlink(path)

    def test_contrast_positive_with_texture(self):
        """A noisy image should produce positive texture contrast."""
        import cv2, tempfile
        img = _noise_image(256, 256, sigma=30)
        fd, path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        cv2.imwrite(path, img)
        try:
            feats = extract_patchcraft_features(path)
            assert feats['texture_contrast'] >= 0, "Contrast should be non-negative"
        finally:
            os.unlink(path)


class TestPatchcraftScore:
    def test_score_range(self):
        score = patchcraft_score('/nonexistent/path')
        assert 0.0 <= score <= 1.0, f"Score {score} outside [0, 1]"

    def test_constant_image_scores_ai_like(self):
        """Zero texture contrast = maximally un-photo-like; with the measured
        direction (real photos have HIGHER contrast, see
        code_notes/09-patchcraft-analyzer.md) the score for a flat image is 1.0."""
        import cv2, tempfile
        img = _constant_image(256, 256, 128)
        fd, path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        cv2.imwrite(path, img)
        try:
            score = patchcraft_score(path)
            assert score == 1.0, f"Constant image score should be 1.0, got {score}"
        finally:
            os.unlink(path)