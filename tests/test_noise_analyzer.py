"""Tests for src/noise_analyzer.py (PRNU-style noise-residual features).

Goal: no noise feature silently returns nonsense. Constant inputs must hit the
guards (zeros/neutral defaults, never NaN); white noise must look broadband and
uncorrelated; identical channels must give chroma correlation ~1.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from src.noise_analyzer import (
    compute_noise_residual,
    residual_variance,
    residual_kurtosis,
    residual_skewness,
    residual_spectral_entropy,
    residual_spatial_autocorrelation,
    residual_block_variance_std,
    multi_scale_noise_ratio_1_5,
    multi_scale_noise_ratio_3_5,
    chroma_noise_correlation,
    extract_noise_features,
    noise_score,
)

NOISE_KEYS = {
    "noise_variance", "noise_kurtosis", "noise_skewness",
    "noise_spectral_entropy", "noise_autocorrelation", "noise_block_var_std",
    "noise_ms_ratio_1_5", "noise_ms_ratio_3_5",
    "noise_rg_corr", "noise_rb_corr", "noise_gb_corr",
}


def _write(img, suffix=".png"):
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    cv2.imwrite(p, img)
    return p


class TestComputeNoiseResidual:
    def test_constant_image_zero_residual(self):
        p = _write(np.full((128, 128), 100, dtype=np.uint8))
        try:
            r = compute_noise_residual(p)
        finally:
            os.remove(p)
        assert np.allclose(r, 0.0, atol=1e-6)  # blur of a flat image == itself


class TestResidualStats:
    def test_variance_zero_on_flat(self):
        assert residual_variance(np.zeros((64, 64))) == 0.0

    def test_kurtosis_and_skew_guarded_on_flat(self):
        flat = np.zeros((64, 64))
        assert residual_kurtosis(flat) == 0.0   # std < eps guard, not NaN
        assert residual_skewness(flat) == 0.0

    def test_kurtosis_finite_on_noise(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 1, (128, 128))
        assert np.isfinite(residual_kurtosis(r))
        assert np.isfinite(residual_skewness(r))

    def test_spectral_entropy_zero_on_flat(self):
        assert residual_spectral_entropy(np.zeros((64, 64))) == 0.0

    def test_white_noise_has_high_spectral_entropy(self):
        rng = np.random.default_rng(1)
        white = rng.normal(0, 1, (128, 128))
        # broadband -> energy spread across frequencies -> normalized entropy ~1
        assert residual_spectral_entropy(white) > 0.8

    def test_structured_has_lower_entropy_than_noise(self):
        rng = np.random.default_rng(2)
        white = rng.normal(0, 1, (128, 128))
        yy, xx = np.mgrid[0:128, 0:128]
        sinusoid = np.sin(2 * np.pi * xx / 8.0)  # single frequency
        assert residual_spectral_entropy(sinusoid) < residual_spectral_entropy(white)

    def test_autocorr_low_for_independent_noise(self):
        rng = np.random.default_rng(3)
        white = rng.normal(0, 1, (256, 256))
        assert abs(residual_spatial_autocorrelation(white)) < 0.2

    def test_autocorr_high_for_smoothed_noise(self):
        rng = np.random.default_rng(4)
        smooth = cv2.GaussianBlur(rng.normal(0, 1, (256, 256)), (0, 0), sigmaX=3.0)
        assert residual_spatial_autocorrelation(smooth) > 0.5

    def test_autocorr_guarded_on_tiny_and_flat(self):
        assert residual_spatial_autocorrelation(np.zeros((1, 1))) == 0.0  # size<4
        assert residual_spatial_autocorrelation(np.zeros((64, 64))) == 0.0  # nan guard

    def test_block_variance_std_nonneg_and_zero_on_flat(self):
        assert residual_block_variance_std(np.zeros((128, 128))) == 0.0
        rng = np.random.default_rng(5)
        assert residual_block_variance_std(rng.normal(0, 1, (128, 128))) >= 0.0


class TestMultiScale:
    def test_flat_returns_neutral_one(self):
        flat = np.zeros((128, 128))
        assert multi_scale_noise_ratio_1_5(flat) == 1.0
        assert multi_scale_noise_ratio_3_5(flat) == 1.0

    def test_noise_ratios_finite_and_clipped(self):
        rng = np.random.default_rng(6)
        img = rng.normal(128, 20, (128, 128))
        for fn in (multi_scale_noise_ratio_1_5, multi_scale_noise_ratio_3_5):
            v = fn(img)
            assert np.isfinite(v) and 0.0 <= v <= 50.0


class TestChroma:
    def test_identical_channels_correlate_near_one(self):
        rng = np.random.default_rng(7)
        gray = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)  # R == G == B
        p = _write(rgb)
        try:
            c = chroma_noise_correlation(p)
        finally:
            os.remove(p)
        assert c["noise_rg_corr"] == pytest.approx(1.0, abs=1e-6)
        assert c["noise_rb_corr"] == pytest.approx(1.0, abs=1e-6)
        assert c["noise_gb_corr"] == pytest.approx(1.0, abs=1e-6)

    def test_independent_channels_low_correlation(self):
        rng = np.random.default_rng(8)
        rgb = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        p = _write(rgb)
        try:
            c = chroma_noise_correlation(p)
        finally:
            os.remove(p)
        assert all(abs(v) < 0.2 for v in c.values())

    def test_bad_path_returns_defaults(self):
        c = chroma_noise_correlation("/nonexistent/x.png")
        assert c == {"noise_rg_corr": 0.0, "noise_rb_corr": 0.0, "noise_gb_corr": 0.0}


class TestExtractNoiseFeatures:
    def test_keys_and_finite_on_random(self):
        rng = np.random.default_rng(9)
        p = _write(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))
        try:
            f = extract_noise_features(p)
        finally:
            os.remove(p)
        assert set(f) == NOISE_KEYS
        assert all(np.isfinite(v) for v in f.values())

    def test_constant_image_neutral_and_finite(self):
        p = _write(np.full((128, 128, 3), 70, dtype=np.uint8))
        try:
            f = extract_noise_features(p)
        finally:
            os.remove(p)
        assert all(np.isfinite(v) for v in f.values())
        assert f["noise_variance"] == pytest.approx(0.0, abs=1e-6)
        assert f["noise_ms_ratio_1_5"] == 1.0 and f["noise_ms_ratio_3_5"] == 1.0

    def test_bad_path_returns_defaults(self):
        f = extract_noise_features("/nonexistent/x.png")
        assert set(f) == NOISE_KEYS
        assert f["noise_spectral_entropy"] == 0.5  # documented neutral default
        assert f["noise_ms_ratio_1_5"] == 1.0


class TestNoiseScore:
    def test_in_range(self):
        rng = np.random.default_rng(10)
        for img in (np.full((128, 128, 3), 50, dtype=np.uint8),
                    rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)):
            p = _write(img)
            try:
                assert 0.0 <= noise_score(p) <= 1.0
            finally:
                os.remove(p)
