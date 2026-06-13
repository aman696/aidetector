"""Tests for src/gradient_analyzer.py — focus on the NaN guard."""

import tempfile
import os

import cv2
import numpy as np

from src.gradient_analyzer import extract_gradient_features


def _write(img: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, img)
    return path


class TestGradientKurtosisGuard:
    def test_constant_image_kurtosis_is_finite(self):
        # A flat image has a zero-variance gradient -> scipy kurtosis = NaN.
        # The analyzer must return a finite neutral value, not propagate NaN.
        path = _write(np.full((128, 128), 128, dtype=np.uint8))
        try:
            feats = extract_gradient_features(path)
        finally:
            os.unlink(path)
        assert np.isfinite(feats["gradient_kurtosis"])
        assert feats["gradient_kurtosis"] == 0.0

    def test_textured_image_all_finite(self):
        rng = np.random.default_rng(0)
        path = _write(rng.integers(0, 256, (128, 128), dtype=np.uint8))
        try:
            feats = extract_gradient_features(path)
        finally:
            os.unlink(path)
        assert all(np.isfinite(v) for v in feats.values())
