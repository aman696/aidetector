"""Tests for src/screenshot_image_analyzer.py (10 screenshot-forensics features).

Goal: every feature stays finite and inside its documented clip range on all
inputs (constant / smooth / high-frequency / noise / tiny / bad path), and the
texture features move in the right direction (a high-frequency "UI-like" image
has more GLCM contrast and diagonal wavelet energy than a smooth gradient).
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from src.screenshot_image_analyzer import (
    extract_screenshot_image_features,
    screenshot_image_score,
    _ZERO_FEATURES,
)

# Documented clip ranges from extract_screenshot_image_features.
RANGES = {
    "fft_periodic_score": (0, 10),
    "fft_peak_to_bg_ratio": (0, 100),
    "glcm_homogeneity": (0, 1),
    "glcm_contrast": (0, 1000),
    "glcm_energy": (0, 1),
    "lbp_entropy": (0, 1),
    "wavelet_hh_energy": (0, 20),
    "wavelet_ratio_hh_ll": (0, 1),
    "chroma_std_ratio": (0, 5),
    "tone_step_density": (0, 1),
}


def _write(img):
    fd, p = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(p, img)
    return p


def _checkerboard(block=4, size=128):
    idx = np.indices((size, size))
    g = (((idx[0] // block) + (idx[1] // block)) % 2 * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _smooth(size=128):
    ramp = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    return np.stack([ramp, ramp, ramp], axis=-1)


def _extract(img):
    p = _write(img)
    try:
        return extract_screenshot_image_features(p)
    finally:
        os.remove(p)


def _assert_in_ranges(feats):
    assert set(feats) == set(RANGES)
    for k, v in feats.items():
        assert np.isfinite(v), f"{k} not finite"
        lo, hi = RANGES[k]
        assert lo <= v <= hi, f"{k}={v} outside [{lo}, {hi}]"


class TestExtractScreenshotFeatures:
    def test_textured_image_keys_and_ranges(self):
        _assert_in_ranges(_extract(_checkerboard()))

    def test_constant_image_finite_in_range(self):
        _assert_in_ranges(_extract(np.full((128, 128, 3), 120, dtype=np.uint8)))

    def test_noise_image_finite_in_range(self):
        rng = np.random.default_rng(0)
        _assert_in_ranges(_extract(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)))

    def test_small_image_returns_zero_features(self):
        # min side < 64 -> the resolution guard returns the zero vector.
        assert _extract(np.full((32, 32, 3), 200, dtype=np.uint8)) == _ZERO_FEATURES

    def test_bad_path_returns_zero_features(self):
        assert extract_screenshot_image_features("/nonexistent/x.png") == _ZERO_FEATURES


class TestScreenshotDirections:
    def _noise(self, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)

    def test_high_frequency_has_more_contrast_and_wavelet_energy(self):
        # A high-frequency image (noise) must have more GLCM contrast and more
        # diagonal wavelet energy than a smooth gradient. (A block checkerboard
        # is avoided here: even blocks align with the Haar 2x downsampling and
        # cancel in HH -- a real property of the transform, not a bug.)
        hi = _extract(self._noise())
        smooth = _extract(_smooth())
        assert hi["glcm_contrast"] > smooth["glcm_contrast"]
        assert hi["wavelet_hh_energy"] > smooth["wavelet_hh_energy"]

    def test_smooth_image_is_more_homogeneous(self):
        # GLCM homogeneity should be higher for the smooth image than for noise.
        assert _extract(_smooth())["glcm_homogeneity"] > \
            _extract(self._noise())["glcm_homogeneity"]


class TestScreenshotScore:
    def test_score_in_range(self):
        rng = np.random.default_rng(1)
        for img in (_checkerboard(), _smooth(),
                    rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)):
            p = _write(img)
            try:
                assert 0.0 <= screenshot_image_score(p) <= 1.0
            finally:
                os.remove(p)

    def test_score_zero_on_bad_path(self):
        # All-zero features -> score is a finite value in range (no crash).
        assert 0.0 <= screenshot_image_score("/nonexistent/x.png") <= 1.0
