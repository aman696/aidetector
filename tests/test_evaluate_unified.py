"""
Tests for src/evaluate_unified.py — the metric core and gate logic on
synthetic data (no model, cache, or torch needed).
"""

import numpy as np
import pytest

from src import evaluate_unified as ev


class TestBinaryMetrics:
    def test_perfect_separation(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        m = ev.binary_metrics(y, s)
        assert m["auc"] == pytest.approx(1.0)
        assert m["ap"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["n_ai"] == 2 and m["n_real"] == 2

    def test_single_class_has_no_auc(self):
        y = np.array([1, 1, 1])
        s = np.array([0.6, 0.7, 0.9])
        m = ev.binary_metrics(y, s)
        assert m["auc"] is None and m["ap"] is None
        assert m["accuracy"] == pytest.approx(1.0)  # all predicted AI at 0.5


class TestPdAtFar:
    def test_perfect_detector(self):
        real = np.array([0.0, 0.1, 0.2, 0.3])
        ai = np.array([0.9, 0.95, 0.99])
        out = ev.pd_at_far(real, ai, target_far=0.05)
        assert out["pd"] == pytest.approx(1.0)

    def test_threshold_is_real_quantile(self):
        real = np.linspace(0, 1, 101)  # quantile(0.95) = 0.95
        ai = np.array([0.94, 0.96, 0.97])
        out = ev.pd_at_far(real, ai, target_far=0.05)
        assert out["threshold"] == pytest.approx(0.95, abs=1e-6)
        assert out["pd"] == pytest.approx(2 / 3, abs=1e-6)  # 0.96, 0.97 pass

    def test_empty_inputs_return_none(self):
        assert ev.pd_at_far([], [0.5])["pd"] is None
        assert ev.pd_at_far([0.5], [])["pd"] is None


class TestResolutionBucket:
    def test_boundaries(self):
        assert ev.resolution_bucket(399) == "<400"
        assert ev.resolution_bucket(400) == "400-800"
        assert ev.resolution_bucket(800) == "400-800"
        assert ev.resolution_bucket(801) == ">800"


class TestGroupReport:
    def test_uses_external_reals_when_single_class(self):
        # All-AI group; reference reals provide the FAR threshold.
        y = np.array([1, 1, 1])
        s = np.array([0.8, 0.9, 0.95])
        ref = np.array([0.1, 0.2, 0.3, 0.4])
        rep = ev.group_report(y, s, ref)
        assert rep["auc"] is None
        assert rep["pd_at_5far"] == pytest.approx(1.0)


class TestGates:
    def test_pass_and_fail_detection(self):
        results = {
            "by_condition": {
                "clean": {"auc": 0.97, "accuracy": 0.92},
                "screenshot": {"accuracy": 0.80},          # fails 0.85
                "facebook": {"auc": 0.90},
            },
            "holdout_overall": {"auc": 0.86, "pd_at_5far": 0.55},  # Pd fails
            "unified_overall": {"auc": 0.95},
            "classical_overall": {"auc": 0.93},
        }
        gates = {g["name"]: g for g in ev.check_gates(results)}
        assert gates["clean AUC >= 0.95"]["passed"] is True
        assert gates["clean accuracy >= 0.90"]["passed"] is True
        assert gates["screenshot accuracy >= 0.85"]["passed"] is False
        assert gates["facebook AUC >= 0.88"]["passed"] is True
        assert gates["holdout AUC >= 0.85"]["passed"] is True
        assert gates["holdout Pd@5%FAR >= 0.60"]["passed"] is False
        assert gates["unified AUC >= classical-only AUC"]["passed"] is True

    def test_unified_below_classical_fails(self):
        results = {"unified_overall": {"auc": 0.90},
                   "classical_overall": {"auc": 0.94}}
        gates = {g["name"]: g for g in ev.check_gates(results)}
        assert gates["unified AUC >= classical-only AUC"]["passed"] is False

    def test_undefined_metric_is_na_not_fail(self):
        # The all-AI holdout has no AUC (single class); that gate must read n/a
        # (passed is None), NOT FAIL. A defined sibling metric still evaluates.
        results = {"holdout_overall": {"auc": None, "pd_at_5far": 0.7}}
        gates = {g["name"]: g for g in ev.check_gates(results)}
        assert gates["holdout AUC >= 0.85"]["passed"] is None
        assert gates["holdout Pd@5%FAR >= 0.60"]["passed"] is True


class TestRecordMinSide:
    def test_uses_record_own_dimensions_not_base(self, tmp_path):
        from PIL import Image
        # The derived variant file is small; the base manifest says it's large.
        # The bucket must reflect the variant's actual pixels.
        variant = tmp_path / "variant.jpg"
        Image.new("RGB", (300, 120)).save(variant)  # (w, h) -> min_side 120
        rec = {"id": "b__facebook", "base_id": "b", "path": str(variant)}
        assert ev._record_min_side(rec, {"b": 1024}) == 120

    def test_falls_back_to_base_when_file_missing(self):
        rec = {"id": "b__x", "base_id": "b", "path": "/nonexistent/x.jpg"}
        assert ev._record_min_side(rec, {"b": 512}) == 512


class TestRenderMarkdown:
    def test_produces_sections(self):
        results = {
            "meta": {"evaluated_at": "now", "n_test": 10, "n_holdout": 5,
                     "unified_features": 855},
            "gates": [{"name": "clean AUC >= 0.95", "target": 0.95,
                       "value": 0.97, "passed": True}],
            "unified_overall": {"n": 10, "auc": 0.95, "accuracy": 0.9,
                                "ap": 0.96, "pd_at_5far": 0.8},
            "by_condition": {"clean": {"n": 5, "auc": 0.97, "accuracy": 0.92,
                                       "ap": 0.98, "pd_at_5far": 0.85}},
            "by_resolution": {}, "by_family": {},
            "exif_permutation": {"auc_base": 0.95, "auc_metadata_permuted": 0.95,
                                 "auc_drop": 0.0, "n_metadata_features": 6},
        }
        md = ev.render_markdown(results)
        assert "## Gates" in md
        assert "PASS" in md
        assert "By condition" in md
        assert "EXIF permutation check" in md
