"""Tests for src/ela_analyzer.py.

ELA must behave correctly across the three compression histories it is meant to
tell apart -- never-compressed PNG, fresh JPEG, and recompressed JPEG -- and must
never return NaN/Inf or out-of-range values ("no feature silently returns
nonsense").

Verified directions (empirical, recorded here so the assertions are principled):
a never-compressed PNG re-saved at q95 shows much more error than an image
already at q95 (which sits at the recompression fixed point); a low-quality JPEG
re-saved at q95 sits in between. A constant image has zero ELA.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from src.ela_analyzer import (
    compute_ela_map,
    extract_ela_features,
    ela_uniformity,
    ela_score,
)

ELA_KEYS = {
    "ela_mean", "ela_variance", "ela_max", "ela_uniformity",
    "ela_block_inconsistency",
}


def _textured(size: int = 256, seed: int = 0) -> np.ndarray:
    """A smooth gradient + mild noise: compresses like a natural image (unlike
    pure noise, which JPEG mangles), so ELA behaves the way the method intends."""
    rng = np.random.default_rng(seed)
    ramp = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    img = np.stack([ramp, ramp.T, (ramp // 2 + 30).astype(np.uint8)], axis=-1)
    noise = rng.integers(-8, 8, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def _write(tmp_path, img, name, quality=None) -> str:
    p = str(tmp_path / name)
    if quality is None:
        cv2.imwrite(p, img)
    else:
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return p


def _assert_sane(feats: dict):
    assert set(feats) == ELA_KEYS
    for k, v in feats.items():
        assert np.isfinite(v), f"{k} is not finite ({v})"
    assert feats["ela_mean"] >= 0.0
    assert feats["ela_variance"] >= 0.0
    assert feats["ela_max"] >= 0.0
    assert 0.0 <= feats["ela_uniformity"] <= 1.0
    assert feats["ela_block_inconsistency"] >= 0.0


class TestElaPNG:
    def test_png_features_sane_and_nonzero(self, tmp_path):
        # A never-compressed PNG re-saved at q95 must show real, finite error.
        feats = extract_ela_features(_write(tmp_path, _textured(), "a.png"))
        _assert_sane(feats)
        assert feats["ela_mean"] > 0.0
        assert feats["ela_max"] > 0.0


class TestElaJPEG:
    def test_jpeg_q95_has_lower_error_than_png(self, tmp_path):
        # The core ELA signal: an image already at q95 sits at the recompression
        # fixed point, so its ELA is far lower than the same content as PNG.
        img = _textured()
        png = extract_ela_features(_write(tmp_path, img, "a.png"))
        jpg = extract_ela_features(_write(tmp_path, img, "a.jpg", quality=95))
        _assert_sane(jpg)
        assert jpg["ela_mean"] < png["ela_mean"]


class TestElaRecompressedJPEG:
    def test_recompressed_low_quality_between_png_and_q95(self, tmp_path):
        # A low-quality JPEG re-saved at q95 shows more error than the q95 image
        # but less than the never-compressed PNG: png > q30 > q95.
        img = _textured()
        png = extract_ela_features(_write(tmp_path, img, "a.png"))
        q95 = extract_ela_features(_write(tmp_path, img, "q95.jpg", quality=95))
        q30 = extract_ela_features(_write(tmp_path, img, "q30.jpg", quality=30))
        _assert_sane(q30)
        assert q95["ela_mean"] < q30["ela_mean"] < png["ela_mean"]


class TestElaDegenerate:
    def test_constant_image_zero_ela_and_uniformity_one(self, tmp_path):
        const = np.full((128, 128, 3), 120, dtype=np.uint8)
        feats = extract_ela_features(_write(tmp_path, const, "c.png"))
        _assert_sane(feats)
        assert feats["ela_mean"] == pytest.approx(0.0, abs=1e-6)
        assert feats["ela_max"] == pytest.approx(0.0, abs=1e-6)
        # max < eps -> the guarded uniformity returns 1.0 (not a 0/0 NaN).
        assert feats["ela_uniformity"] == 1.0

    def test_uniformity_guard_on_zero_map(self):
        assert ela_uniformity(np.zeros((32, 32))) == 1.0

    def test_compute_ela_map_is_nonnegative(self, tmp_path):
        m = compute_ela_map(_write(tmp_path, _textured(), "a.png"))
        assert np.all(m >= 0.0)  # ELA is an absolute difference

    def test_bad_path_returns_neutral_defaults(self):
        # Must not raise; returns the documented neutral defaults.
        feats = extract_ela_features("/nonexistent/nope.png")
        assert feats == {
            "ela_mean": 0.0, "ela_variance": 0.0, "ela_max": 0.0,
            "ela_uniformity": 0.5, "ela_block_inconsistency": 0.0,
        }


class TestElaScore:
    def test_score_in_range_png_and_jpeg(self, tmp_path):
        img = _textured()
        for name, q in (("a.png", None), ("a.jpg", 95)):
            s = ela_score(_write(tmp_path, img, name, quality=q))
            assert 0.0 <= s <= 1.0
