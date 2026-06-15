"""Tests for src/gradient_analyzer.py — focus on the NaN guard."""

import tempfile
import os

import cv2
import numpy as np
import pytest

from src.gradient_analyzer import (
    extract_gradient_features,
    compute_laplacian,
    compute_sobel_gradient,
    gradient_score,
)


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


class TestSobelGradient:
    def test_constant_image_zero_gradient(self):
        assert np.allclose(compute_sobel_gradient(np.full((64, 64), 100, np.uint8)), 0.0)

    def test_edge_has_nonzero_gradient(self):
        img = np.zeros((64, 64), np.uint8)
        img[:, 32:] = 255  # vertical step edge
        assert compute_sobel_gradient(img).max() > 0.0


class TestGradientScore:
    def test_score_in_range_constant_and_textured(self):
        rng = np.random.default_rng(2)
        for img in (np.full((96, 96), 128, np.uint8),
                    rng.integers(0, 256, (96, 96), dtype=np.uint8)):
            p = _write(img)
            try:
                s = gradient_score(p)
            finally:
                os.unlink(p)
            assert 0.0 <= s <= 1.0

    def test_constant_image_features_finite_and_zeroish(self):
        p = _write(np.full((96, 96), 60, np.uint8))
        try:
            f = extract_gradient_features(p)
        finally:
            os.unlink(p)
        assert all(np.isfinite(v) for v in f.values())
        assert f["gradient_variance"] == pytest.approx(0.0, abs=1e-6)
        assert f["gradient_laplacian_variance"] == pytest.approx(0.0, abs=1e-6)


class TestLaplacianVarianceIsSigned:
    """gradient_laplacian_variance must be the variance of the SIGNED Laplacian
    (the canonical Pech-Pacheco focus measure), not var(|Laplacian|)."""

    def test_matches_signed_variance(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        path = _write(img)
        try:
            feats = extract_gradient_features(path)
            # Replicate the analyzer's loader exactly for a bit-identical compare.
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        finally:
            os.unlink(path)
        signed = compute_laplacian(gray)
        assert abs(feats["gradient_laplacian_variance"] - float(np.var(signed))) < 1e-6
        # And it must differ from the buggy var(|L|), which is strictly smaller.
        assert float(np.var(signed)) > float(np.var(np.abs(signed)))
