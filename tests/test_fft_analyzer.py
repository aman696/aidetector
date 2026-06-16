"""
Unit tests for src/fft_analyzer.py

Tests cover:
- fft_score() returns float in [0, 1] on real and AI images
- extract_fft_features() returns dict with expected keys
- azimuthal_average() on synthetic arrays
- compute_spectral_slope() on known 1/f spectrum
- compute_high_freq_ratio() and compute_spectral_falloff() edge cases
- Error handling for missing/unsupported files
"""

import pytest
import numpy as np
import cv2
import os
import sys
import tempfile

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fft_analyzer import (
    fft_score,
    extract_fft_features,
    azimuthal_average,
    compute_power_spectrum,
    compute_spectral_slope,
    compute_high_freq_ratio,
    compute_spectral_falloff,
)

# --- Paths to sample images ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
REAL_IMG = os.path.join(DATA_DIR, 'real', 'coco', 'img_00001.jpg')
AI_IMG = os.path.join(DATA_DIR, 'Gemini', 'ai_gemini_img_001.png')


# =====================================================================
# fft_score — end-to-end
# =====================================================================

class TestFFTScore:
    """Tests for fft_score() function."""

    def test_returns_float(self):
        score = fft_score(REAL_IMG)
        assert isinstance(score, float)

    def test_score_range_real_image(self):
        score = fft_score(REAL_IMG)
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] range"

    def test_score_range_ai_image(self):
        score = fft_score(AI_IMG)
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] range"

    def test_deterministic(self):
        """Same image should give same score."""
        s1 = fft_score(REAL_IMG)
        s2 = fft_score(REAL_IMG)
        assert s1 == s2


# =====================================================================
# extract_fft_features — feature dict
# =====================================================================

class TestExtractFFTFeatures:
    """Tests for extract_fft_features()."""

    EXPECTED_KEYS = {'spectral_slope', 'slope_r_squared', 'high_freq_ratio', 'spectral_falloff'}

    def test_returns_dict(self):
        features = extract_fft_features(REAL_IMG)
        assert isinstance(features, dict)

    def test_expected_keys(self):
        features = extract_fft_features(REAL_IMG)
        assert features.keys() == self.EXPECTED_KEYS

    def test_all_values_are_floats(self):
        features = extract_fft_features(REAL_IMG)
        for key, val in features.items():
            assert isinstance(val, (float, np.floating)), f"{key} is {type(val)}"

    def test_high_freq_ratio_range(self):
        features = extract_fft_features(REAL_IMG)
        assert 0.0 <= features['high_freq_ratio'] <= 1.0

    def test_slope_r_squared_range(self):
        features = extract_fft_features(REAL_IMG)
        assert 0.0 <= features['slope_r_squared'] <= 1.0


# =====================================================================
# azimuthal_average — internal helper
# =====================================================================

class TestAzimuthalAverage:
    """Tests for azimuthal_average()."""

    def test_uniform_spectrum(self):
        """Uniform spectrum should give constant radial profile."""
        spectrum = np.ones((128, 128))
        profile = azimuthal_average(spectrum)
        # All values should be ~1.0
        np.testing.assert_allclose(profile, 1.0, atol=0.01)

    def test_output_length(self):
        """Output length should be min(h, w) // 2."""
        spectrum = np.ones((100, 200))
        profile = azimuthal_average(spectrum)
        assert len(profile) == 50  # min(100, 200) // 2

    def test_returns_1d_array(self):
        spectrum = np.random.rand(64, 64)
        profile = azimuthal_average(spectrum)
        assert profile.ndim == 1

    def test_dc_component_is_center_value(self):
        """The radius-0 bin should correspond to the center pixel value."""
        spectrum = np.zeros((64, 64))
        spectrum[32, 32] = 100.0  # DC component at center
        profile = azimuthal_average(spectrum)
        assert profile[0] > 0  # DC bin should be nonzero


# =====================================================================
# compute_power_spectrum — must be power |F|^2, not magnitude or log
# =====================================================================

class TestPowerSpectrumIsPower:
    """compute_power_spectrum must return the power spectrum |F|^2.

    Regression guard for the prior bug where the radial profile was the azimuthal
    average of log-magnitude; combined with the log inside compute_spectral_slope
    that was a DOUBLE log, making the slope uninterpretable as a 1/f^beta exponent
    (it collapsed to ~ -0.1).
    """

    def test_power_scales_quadratically(self):
        # |F[2x]|^2 = 4 |F[x]|^2. Magnitude would scale x2 and log1p(magnitude)
        # would not scale cleanly, so the x4 ratio uniquely identifies power.
        rng = np.random.default_rng(0)
        img = rng.random((128, 128)) * 100.0
        _, r1 = compute_power_spectrum(img)
        _, r2 = compute_power_spectrum(img * 2.0)
        ratio = r2.sum() / r1.sum()
        assert abs(ratio - 4.0) < 0.05, f"power should scale x4, got x{ratio:.3f}"

    def test_power_is_nonnegative(self):
        rng = np.random.default_rng(1)
        _, r = compute_power_spectrum(rng.random((96, 96)))
        assert np.all(r >= 0.0)

    def test_real_image_recovers_power_law_slope(self):
        # Natural images follow ~1/f^beta; on true power the slope IS the exponent
        # (about -2 to -3) with high R^2. The double-log bug gave slope ~ -0.1,
        # which this range rejects.
        if not os.path.exists(REAL_IMG):
            pytest.skip("real sample image not present")
        f = extract_fft_features(REAL_IMG)
        assert -4.0 < f['spectral_slope'] < -1.0, \
            f"slope {f['spectral_slope']} is not a power-law exponent (double-log regression?)"
        assert f['slope_r_squared'] > 0.8

    def test_high_freq_ratio_direction(self):
        # Backs the fft_score direction claim: a broadband (high-freq-rich) image
        # has a larger high-frequency energy fraction than a smooth (low-freq) one.
        # fft_score maps lower hf_ratio -> higher AI score.
        rng = np.random.default_rng(3)
        noise = rng.random((256, 256)) * 255.0                 # broadband
        ramp = np.tile(np.linspace(0, 255, 256), (256, 1))     # smooth, low-freq
        _, r_noise = compute_power_spectrum(noise)
        _, r_ramp = compute_power_spectrum(ramp)
        assert compute_high_freq_ratio(r_noise) > compute_high_freq_ratio(r_ramp)


# =====================================================================
# compute_spectral_slope — linear fit
# =====================================================================

class TestComputeSpectralSlope:
    """Tests for compute_spectral_slope()."""

    def test_perfect_power_law(self):
        """A perfect 1/f spectrum should have slope ≈ -1 and high R²."""
        freqs = np.arange(1, 100)
        spectrum = np.concatenate([[100.0], 100.0 / freqs])  # 1/f
        slope, r_sq = compute_spectral_slope(spectrum)
        assert -1.5 < slope < -0.5, f"Slope {slope} unexpected for 1/f"
        assert r_sq > 0.9, f"R² {r_sq} too low for perfect power law"

    def test_flat_spectrum(self):
        """Flat spectrum should have slope ≈ 0."""
        spectrum = np.ones(50) * 10
        slope, r_sq = compute_spectral_slope(spectrum)
        assert abs(slope) < 0.3, f"Slope {slope} too far from 0 for flat spectrum"

    def test_too_short_spectrum(self):
        """Spectrum with < 5 points should return (0, 0)."""
        spectrum = np.array([1, 2, 3])
        slope, r_sq = compute_spectral_slope(spectrum)
        assert slope == 0.0
        assert r_sq == 0.0


# =====================================================================
# compute_high_freq_ratio
# =====================================================================

class TestComputeHighFreqRatio:
    """Tests for compute_high_freq_ratio()."""

    def test_uniform_spectrum(self):
        """Uniform spectrum with cutoff 0.5 → ratio ≈ 0.5."""
        spectrum = np.ones(100)
        ratio = compute_high_freq_ratio(spectrum, cutoff_fraction=0.5)
        assert abs(ratio - 0.5) < 0.01

    def test_empty_spectrum(self):
        ratio = compute_high_freq_ratio(np.array([]))
        assert ratio == 0.0

    def test_zero_energy(self):
        ratio = compute_high_freq_ratio(np.zeros(50))
        assert ratio == 0.0

    def test_range(self):
        spectrum = np.random.rand(100)
        ratio = compute_high_freq_ratio(spectrum)
        assert 0.0 <= ratio <= 1.0


# =====================================================================
# compute_spectral_falloff
# =====================================================================

class TestComputeSpectralFalloff:
    """Tests for compute_spectral_falloff()."""

    def test_uniform_spectrum(self):
        """Uniform spectrum → falloff = 1.0."""
        spectrum = np.ones(100)
        falloff = compute_spectral_falloff(spectrum)
        assert abs(falloff - 1.0) < 0.01

    def test_short_spectrum(self):
        """Spectrum with < 8 points returns 0."""
        falloff = compute_spectral_falloff(np.ones(4))
        assert falloff == 0.0

    def test_decreasing_spectrum(self):
        """Decreasing spectrum should have falloff < 1."""
        spectrum = np.linspace(100, 1, 100)
        falloff = compute_spectral_falloff(spectrum)
        assert falloff < 1.0


# =====================================================================
# Standard diagnostic inputs: constant / checkerboard / noise / smooth
# =====================================================================

def _write_gray(img: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, img)
    return path


def _checkerboard(block: int = 2, size: int = 256) -> np.ndarray:
    idx = np.indices((size, size))
    return (((idx[0] // block) + (idx[1] // block)) % 2 * 255).astype(np.uint8)


def _smooth_ramp(size: int = 256) -> np.ndarray:
    return np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))


class TestFftStandardInputs:
    """Spectral-slope and high-frequency sanity on canonical inputs. The point is
    that no input -- not even degenerate ones -- yields NaN/Inf or a nonsense
    feature; and that the features move in the physically correct direction."""

    def _feats(self, img):
        p = _write_gray(img)
        try:
            return extract_fft_features(p)
        finally:
            os.remove(p)

    def test_constant_image_is_finite_and_degenerate(self):
        # A flat image has all energy at DC (skipped), so the radial features
        # collapse to finite zeros -- never NaN.
        f = self._feats(np.full((256, 256), 128, dtype=np.uint8))
        assert all(np.isfinite(v) for v in f.values())
        assert f["spectral_slope"] == 0.0
        assert f["slope_r_squared"] == 0.0
        assert f["high_freq_ratio"] == 0.0

    def test_checkerboard_has_more_high_freq_than_smooth(self):
        # A high-frequency texture must carry far more high-frequency energy than
        # a smooth gradient (the whole point of high_freq_ratio).
        ck = self._feats(_checkerboard(block=2))
        smooth = self._feats(_smooth_ramp())
        assert all(np.isfinite(v) for v in ck.values())
        assert ck["high_freq_ratio"] > 100.0 * smooth["high_freq_ratio"]

    def test_white_noise_has_flat_slope(self):
        # White noise has a (near) flat power spectrum -> slope ~ 0 and a poor
        # power-law fit; a smooth image has a steep negative slope. This is the
        # spectral-slope sanity check.
        rng = np.random.default_rng(0)
        noise = self._feats(rng.integers(0, 256, (256, 256), dtype=np.uint8))
        smooth = self._feats(_smooth_ramp())
        assert all(np.isfinite(v) for v in noise.values())
        assert abs(noise["spectral_slope"]) < 0.3          # flat
        assert noise["slope_r_squared"] < 0.3              # not a power law
        assert noise["spectral_slope"] > smooth["spectral_slope"] + 1.0  # less negative

    def test_all_standard_inputs_finite(self):
        rng = np.random.default_rng(1)
        inputs = {
            "constant": np.full((256, 256), 200, dtype=np.uint8),
            "checkerboard": _checkerboard(block=4),
            "noise": rng.integers(0, 256, (256, 256), dtype=np.uint8),
            "smooth": _smooth_ramp(),
        }
        for name, img in inputs.items():
            f = self._feats(img)
            assert all(np.isfinite(v) for v in f.values()), f"{name} produced non-finite"
            assert 0.0 <= f["high_freq_ratio"] <= 1.0
            assert 0.0 <= f["slope_r_squared"] <= 1.0


# =====================================================================
# Error handling
# =====================================================================

class TestFFTErrors:
    """Tests for error handling."""

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            fft_score('/nonexistent/image.jpg')

    def test_unsupported_extension(self):
        # Create a temp file with unsupported extension
        tmp_path = '/tmp/test_fft_bad.txt'
        with open(tmp_path, 'w') as f:
            f.write("not an image")
        try:
            with pytest.raises(ValueError):
                fft_score(tmp_path)
        finally:
            os.remove(tmp_path)
