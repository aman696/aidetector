"""
Unit tests for src/eigen_analyzer.py

Tests cover:
- eigenvalue_score() returns float in [0, 1] on real and AI images
- extract_eigen_features() returns dict with expected keys
- compute_rgb_covariance() shape and symmetry
- extract_eigenvalues() on known covariance matrix
- compute_eigenvalue_ratios() correctness
- patch_eigenvalue_analysis() output structure
- spectral_band_analysis() energy sums to ~1
- Error handling
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eigen_analyzer import (
    eigenvalue_score,
    extract_eigen_features,
    compute_rgb_covariance,
    extract_eigenvalues,
    compute_eigenvalue_ratios,
    patch_eigenvalue_analysis,
    spectral_band_analysis,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
REAL_IMG = os.path.join(DATA_DIR, 'real', 'real_001.jpg')
AI_IMG = os.path.join(DATA_DIR, 'ai_generated', 'ai_gemini_img_001.png')


# =====================================================================
# eigenvalue_score — end-to-end
# =====================================================================

class TestEigenvalueScore:
    """Tests for eigenvalue_score()."""

    def test_returns_float(self):
        score = eigenvalue_score(REAL_IMG)
        assert isinstance(score, float)

    def test_score_range_real(self):
        score = eigenvalue_score(REAL_IMG)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_score_range_ai(self):
        score = eigenvalue_score(AI_IMG)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_deterministic(self):
        s1 = eigenvalue_score(REAL_IMG)
        s2 = eigenvalue_score(REAL_IMG)
        assert s1 == s2


# =====================================================================
# extract_eigen_features — feature dict
# =====================================================================

class TestExtractEigenFeatures:
    """Tests for extract_eigen_features()."""

    EXPECTED_KEYS = {
        'eig_ratio_1_2', 'eig_ratio_2_3', 'eig_condition_number', 'eig_dominance',
        'patch_ratio_mean', 'patch_ratio_std', 'patch_dominance_mean', 'patch_dominance_std',
        'band_low_ratio', 'band_mid_ratio', 'band_high_ratio', 'band_mid_high_ratio',
    }

    def test_returns_dict(self):
        features = extract_eigen_features(REAL_IMG)
        assert isinstance(features, dict)

    def test_expected_keys(self):
        features = extract_eigen_features(REAL_IMG)
        assert features.keys() == self.EXPECTED_KEYS

    def test_all_values_numeric(self):
        features = extract_eigen_features(REAL_IMG)
        for key, val in features.items():
            assert isinstance(val, (float, int, np.floating)), f"{key} is {type(val)}"


# =====================================================================
# compute_rgb_covariance — 3x3 matrix
# =====================================================================

class TestComputeRGBCovariance:
    """Tests for compute_rgb_covariance()."""

    def test_shape(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cov = compute_rgb_covariance(img)
        assert cov.shape == (3, 3)

    def test_symmetric(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cov = compute_rgb_covariance(img)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_positive_diagonal(self):
        """Diagonal elements (variances) should be non-negative."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cov = compute_rgb_covariance(img)
        assert np.all(np.diag(cov) >= 0)

    def test_uniform_image(self):
        """Uniform image should have near-zero covariance."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        cov = compute_rgb_covariance(img)
        assert np.allclose(cov, 0.0, atol=1e-10)


# =====================================================================
# extract_eigenvalues
# =====================================================================

class TestExtractEigenvalues:
    """Tests for extract_eigenvalues()."""

    def test_known_matrix(self):
        """Known diagonal matrix → eigenvalues are the diagonal elements."""
        cov = np.diag([100.0, 50.0, 10.0])
        eigs = extract_eigenvalues(cov)
        np.testing.assert_allclose(eigs, [100.0, 50.0, 10.0], atol=1e-8)

    def test_sorted_descending(self):
        cov = np.diag([5.0, 20.0, 10.0])
        eigs = extract_eigenvalues(cov)
        assert eigs[0] >= eigs[1] >= eigs[2]

    def test_length_matches_input(self):
        cov = np.eye(3) * 10
        eigs = extract_eigenvalues(cov)
        assert len(eigs) == 3


# =====================================================================
# compute_eigenvalue_ratios
# =====================================================================

class TestComputeEigenvalueRatios:
    """Tests for compute_eigenvalue_ratios()."""

    def test_known_values(self):
        eigs = np.array([100.0, 50.0, 10.0])
        ratios = compute_eigenvalue_ratios(eigs)
        assert abs(ratios['eig_ratio_1_2'] - 2.0) < 0.01
        assert abs(ratios['eig_ratio_2_3'] - 5.0) < 0.01

    def test_dominance(self):
        eigs = np.array([100.0, 0.0, 0.0])
        ratios = compute_eigenvalue_ratios(eigs)
        assert abs(ratios['eig_dominance'] - 1.0) < 0.01


# =====================================================================
# patch_eigenvalue_analysis
# =====================================================================

class TestPatchEigenvalueAnalysis:
    """Tests for patch_eigenvalue_analysis()."""

    def test_output_keys(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = patch_eigenvalue_analysis(img, patch_size=64)
        expected = {'patch_ratio_mean', 'patch_ratio_std', 'patch_dominance_mean', 'patch_dominance_std'}
        assert result.keys() == expected

    def test_uniform_image_returns_zeros(self):
        """Uniform image patches are skipped, returning zero stats."""
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        result = patch_eigenvalue_analysis(img, patch_size=64)
        assert result['patch_ratio_mean'] == 0.0


# =====================================================================
# spectral_band_analysis
# =====================================================================

class TestSpectralBandAnalysis:
    """Tests for spectral_band_analysis()."""

    def test_output_keys(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = spectral_band_analysis(img)
        expected = {'band_low_ratio', 'band_mid_ratio', 'band_high_ratio', 'band_mid_high_ratio'}
        assert result.keys() == expected

    def test_band_ratios_sum_close_to_one(self):
        """Low + mid + high ratios should roughly sum to (close to but not exactly) 1."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = spectral_band_analysis(img)
        total = result['band_low_ratio'] + result['band_mid_ratio'] + result['band_high_ratio']
        # May not be exactly 1 due to pixels outside max_radius, but should be close
        assert 0.5 < total <= 1.0

    def test_mid_high_ratio_consistency(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = spectral_band_analysis(img)
        expected = result['band_mid_ratio'] + result['band_high_ratio']
        assert abs(result['band_mid_high_ratio'] - expected) < 1e-10


# =====================================================================
# Error handling
# =====================================================================

class TestEigenErrors:
    """Tests for error handling."""

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            eigenvalue_score('/nonexistent/image.jpg')

    def test_unsupported_extension(self):
        tmp_path = '/tmp/test_eigen_bad.txt'
        with open(tmp_path, 'w') as f:
            f.write("not an image")
        try:
            with pytest.raises(ValueError):
                eigenvalue_score(tmp_path)
        finally:
            os.remove(tmp_path)
