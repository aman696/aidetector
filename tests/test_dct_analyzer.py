"""Tests for src/dct_analyzer.py.

Focused on dct_coefficient_variance, which must compute the variance over AC
coefficients ONLY (DC dropped), not over all 64 slots with DC zeroed in place.
"""

import numpy as np

from src.dct_analyzer import dct_coefficient_variance


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
