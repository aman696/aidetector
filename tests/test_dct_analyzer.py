"""Tests for src/dct_analyzer.py.

Focused on dct_coefficient_variance, which must compute the variance over AC
coefficients ONLY (DC dropped), not over all 64 slots with DC zeroed in place.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from src.dct_analyzer import (
    dct_coefficient_variance,
    dct_ac_energy_ratio,
    dct_high_freq_energy,
    dct_zigzag_energy_decay,
    _zigzag_order,
    block_dct,
    extract_dct_features,
)


def _blocks(n=4):
    """n zero 8x8 DCT blocks to fill with known coefficients."""
    return np.zeros((n, 8, 8), dtype=np.float64)


class TestDctCoefficientVariance:
    def test_variance_is_ac_only(self):
        # Variance must equal var over the 63 AC coefficients per block.
        rng = np.random.default_rng(0)
        blocks = rng.standard_normal((12, 8, 8)) * 5.0 + 3.0  # nonzero-mean AC
        blocks[:, 0, 0] = 500.0                               # large DC term
        ac = blocks.reshape(12, -1)[:, 1:]
        got = dct_coefficient_variance(blocks)
        assert abs(got - float(np.var(ac))) < 1e-9

    def test_differs_from_buggy_zeroed_dc_variance(self):
        # The prior version zeroed DC then took var over all 64 slots; the one
        # zeroed cell per block was still counted in N and the mean. That value
        # is measurably different from the correct AC-only variance.
        rng = np.random.default_rng(1)
        blocks = rng.standard_normal((16, 8, 8)) * 5.0 + 3.0
        blocks[:, 0, 0] = 500.0
        got = dct_coefficient_variance(blocks)
        buggy = blocks.copy()
        buggy[:, 0, 0] = 0.0
        assert abs(got - float(np.var(buggy))) > 1e-6

    def test_dc_magnitude_does_not_affect_result(self):
        rng = np.random.default_rng(2)
        blocks = rng.standard_normal((8, 8, 8))
        v1 = dct_coefficient_variance(blocks.copy())
        blocks[:, 0, 0] += 10_000.0  # change only the DC term
        v2 = dct_coefficient_variance(blocks)
        assert abs(v1 - v2) < 1e-9

    def test_empty_blocks_returns_zero(self):
        assert dct_coefficient_variance(np.zeros((0, 8, 8))) == 0.0


class TestAcDcSeparation:
    """dct_ac_energy_ratio must split DC (the [0,0] term) from AC correctly."""

    def test_dc_only_block_has_zero_ac_ratio(self):
        b = _blocks()
        b[:, 0, 0] = 100.0  # pure DC
        assert dct_ac_energy_ratio(b) == pytest.approx(0.0, abs=1e-9)

    def test_ac_only_block_has_unit_ac_ratio(self):
        b = _blocks()
        b[:, 3, 4] = 50.0  # a single AC coefficient, no DC
        assert dct_ac_energy_ratio(b) == pytest.approx(1.0, abs=1e-9)

    def test_half_and_half(self):
        b = _blocks()
        b[:, 0, 0] = 10.0   # DC energy 100
        b[:, 2, 1] = 10.0   # AC energy 100
        assert dct_ac_energy_ratio(b) == pytest.approx(0.5, abs=1e-9)

    def test_zero_blocks_guarded(self):
        assert dct_ac_energy_ratio(_blocks()) == 0.0


class TestHighFreqMask:
    """dct_high_freq_energy must read only the bottom-right (high-freq) quadrant."""

    def test_energy_in_high_quadrant_is_one(self):
        b = _blocks()
        b[:, 7, 7] = 10.0  # bottom-right corner (highest frequency)
        assert dct_high_freq_energy(b) == pytest.approx(1.0, abs=1e-9)

    def test_energy_in_low_quadrant_is_zero(self):
        b = _blocks()
        b[:, 1, 1] = 10.0  # low-frequency cell, outside [4:, 4:]
        assert dct_high_freq_energy(b) == pytest.approx(0.0, abs=1e-9)

    def test_boundary_cell_4_4_counts_as_high(self):
        # half = 8 // 2 = 4, so index 4 is the first high row/col.
        b = _blocks()
        b[:, 4, 4] = 10.0
        assert dct_high_freq_energy(b) == pytest.approx(1.0, abs=1e-9)

    def test_zero_blocks_guarded(self):
        assert dct_high_freq_energy(_blocks()) == 0.0


class TestZigzagOrdering:
    def test_is_a_permutation_of_all_indices(self):
        z = _zigzag_order(8)
        assert sorted(z.tolist()) == list(range(64))

    def test_standard_jpeg_zigzag_prefix(self):
        # Canonical JPEG zigzag (row-major flat indices): 0,1,8,16,9,2,3,10,...
        z = _zigzag_order(8)
        assert z[:6].tolist() == [0, 1, 8, 16, 9, 2]

    def test_decay_larger_when_energy_is_high_frequency(self):
        z = _zigzag_order(8)
        # Low-frequency-heavy: most energy in the first zigzag position.
        low = _blocks(1)
        low.reshape(1, 64)[0, z[0]] = 10.0
        low.reshape(1, 64)[0, z[-1]] = 1.0
        # High-frequency-heavy: most energy in the last zigzag position.
        high = _blocks(1)
        high.reshape(1, 64)[0, z[0]] = 1.0
        high.reshape(1, 64)[0, z[-1]] = 10.0
        assert dct_zigzag_energy_decay(high) > dct_zigzag_energy_decay(low)


class TestDctOnImages:
    def _write(self, img):
        fd, p = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(p, img)
        return p

    def test_constant_image_is_all_dc(self):
        # A flat image -> every 8x8 block is constant -> only DC is nonzero, so
        # AC and high-frequency energy are ~0 (clean AC/DC separation, no NaN).
        p = self._write(np.full((128, 128), 90, dtype=np.uint8))
        try:
            blocks = block_dct(p)
            feats = extract_dct_features(p)
        finally:
            os.remove(p)
        assert dct_ac_energy_ratio(blocks) == pytest.approx(0.0, abs=1e-6)
        assert dct_high_freq_energy(blocks) == pytest.approx(0.0, abs=1e-6)
        assert all(np.isfinite(v) for v in feats.values())

    def test_noise_image_features_finite(self):
        rng = np.random.default_rng(0)
        p = self._write(rng.integers(0, 256, (128, 128), dtype=np.uint8))
        try:
            feats = extract_dct_features(p)
        finally:
            os.remove(p)
        assert all(np.isfinite(v) for v in feats.values())
