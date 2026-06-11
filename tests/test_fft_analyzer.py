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
import os
import sys

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
REAL_IMG = os.path.join(DATA_DIR, 'real', 'real_001.jpg')
AI_IMG = os.path.join(DATA_DIR, 'ai_generated', 'ai_gemini_img_001.png')


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
