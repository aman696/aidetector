"""
Tests for src/train_unified.py.

The training core (train_from_matrix) is exercised on synthetic grouped data,
so these tests need neither torch nor the feature cache. They verify the grid
search runs with group-aware CV, the saved bundle round-trips, and the final
model produces calibrated probabilities.
"""

import sys

import joblib
import numpy as np
import pytest

from src import train_unified as tu


def _grouped_blobs(n_groups=40, per_group=6, n_features=20, seed=0):
    """Two linearly separable classes; each base_id group is one class."""
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for g in range(n_groups):
        label = g % 2
        center = np.full(n_features, 2.0 if label else -2.0)
        X.append(center + rng.normal(0, 1.0, (per_group, n_features)))
        y.extend([label] * per_group)
        groups.extend([f"base_{g}"] * per_group)
    return np.vstack(X), np.array(y), np.array(groups)


class TestParamGrid:
    def test_has_rbf_and_linear(self):
        grid = tu._param_grid([1, 10], ["scale"])
        kernels = {k for block in grid for k in block["svc__kernel"]}
        assert kernels == {"rbf", "linear"}


class TestTrainFromMatrix:
    def test_bundle_structure_and_separable_data(self):
        X, y, groups = _grouped_blobs()
        bundle, summary = tu.train_from_matrix(
            X, y, groups, [f"f{i}" for i in range(X.shape[1])],
            embed_model_name="dummy_model", classical_only=False,
            n_splits=3, c_values=[1.0, 10.0], gamma_values=["scale"],
            n_jobs=1, verbose=False,
        )
        assert bundle["version"] == 2
        assert bundle["embed_model_name"] == "dummy_model"
        assert bundle["classical_only"] is False
        assert bundle["n_features"] == X.shape[1]
        assert len(bundle["feature_names"]) == X.shape[1]
        # Separable data -> the grid should find a strong boundary.
        assert bundle["cv_roc_auc"] > 0.9
        assert summary["cv_roc_auc"] == bundle["cv_roc_auc"]

    def test_final_model_predicts_and_calibrates(self):
        X, y, groups = _grouped_blobs()
        bundle, _ = tu.train_from_matrix(
            X, y, groups, [f"f{i}" for i in range(X.shape[1])],
            embed_model_name=None, classical_only=True,
            n_splits=3, c_values=[1.0], gamma_values=["scale"],
            n_jobs=1, verbose=False,
        )
        pipe = bundle["pipeline"]
        preds = pipe.predict(X)
        proba = pipe.predict_proba(X)
        assert set(np.unique(preds)).issubset({0, 1})
        assert proba.shape == (X.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_bundle_roundtrips_through_joblib(self, tmp_path):
        X, y, groups = _grouped_blobs()
        bundle, _ = tu.train_from_matrix(
            X, y, groups, [f"f{i}" for i in range(X.shape[1])],
            embed_model_name="m", classical_only=False,
            n_splits=3, c_values=[1.0], gamma_values=["scale"],
            n_jobs=1, verbose=False,
        )
        path = tmp_path / "bundle.pkl"
        joblib.dump(bundle, path)
        loaded = joblib.load(path)
        np.testing.assert_array_equal(
            loaded["pipeline"].predict(X), bundle["pipeline"].predict(X))
        assert loaded["feature_names"] == bundle["feature_names"]


class TestCalibrationIsGroupAware:
    """The final CalibratedClassifierCV must calibrate with group-aware folds, or
    base_id variants leak into the sigmoid fit and bias threshold metrics."""

    def test_no_base_id_straddles_a_calibration_fold(self, monkeypatch):
        X, y, groups = _grouped_blobs(n_groups=24, per_group=6)
        captured = {}
        real_ccv = tu.CalibratedClassifierCV

        def spy(*args, **kwargs):
            captured["cv"] = kwargs.get("cv")
            return real_ccv(*args, **kwargs)

        monkeypatch.setattr(tu, "CalibratedClassifierCV", spy)
        tu.train_from_matrix(
            X, y, groups, [f"f{i}" for i in range(X.shape[1])],
            embed_model_name=None, classical_only=True,
            n_splits=3, c_values=[1.0], gamma_values=["scale"],
            n_jobs=1, verbose=False,
        )
        cv = captured["cv"]
        # Must be explicit index splits, not an int (which would silently become
        # plain, non-group-aware StratifiedKFold).
        assert isinstance(cv, list) and len(cv) >= 2
        for train_idx, test_idx in cv:
            assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), \
                "a base_id straddles a calibration fold -> leakage"


class TestEmbeddingAblation:
    def test_returns_auc_in_unit_range(self):
        X, y, groups = _grouped_blobs()
        auc = tu._embedding_only_ablation(X, y, groups, n_splits=3)
        assert 0.0 <= auc <= 1.0
        assert auc > 0.9  # separable


class TestOrchestrator:
    def test_saves_both_bundles_and_summary(self, tmp_path, monkeypatch):
        # n_features > N_CLASSICAL so the embedding slice is non-empty.
        X, y, groups = _grouped_blobs(n_groups=30, per_group=6,
                                      n_features=tu.N_CLASSICAL + 10)
        names = [f"f{i}" for i in range(X.shape[1])]

        import src.feature_pipeline as fp
        monkeypatch.setattr(fp, "assemble_split",
                            lambda *a, **k: (X, y, groups, names, []))

        summary = tu.train_unified(
            out_dir=str(tmp_path / "models"),
            reports_dir=str(tmp_path / "reports"),
            n_splits=3, n_jobs=1, verbose=False)

        assert (tmp_path / "models" / "unified_v2.pkl").exists()
        assert (tmp_path / "models" / "classical_v2.pkl").exists()
        assert (tmp_path / "reports" / "train_v2_summary.json").exists()
        assert summary["unified"]["n_features"] == tu.N_CLASSICAL + 10
        assert summary["classical_only"]["n_features"] == tu.N_CLASSICAL

        loaded = joblib.load(tmp_path / "models" / "classical_v2.pkl")
        assert loaded["classical_only"] is True
        assert loaded["embed_model_name"] is None


class TestGpuFlag:
    """Wiring only — enable_gpu is mocked so cuML is never imported here."""

    def test_gpu_flag_passes_gpu_true_keeps_n_jobs(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(tu, "enable_gpu", lambda: True)
        monkeypatch.setattr(tu, "train_unified", lambda **kw: captured.update(kw))
        monkeypatch.setattr(sys, "argv", ["prog", "--gpu", "--n-jobs", "8"])
        tu.main()
        assert captured["gpu"] is True
        assert captured["n_jobs"] == 8  # CPU parallelism preserved for the rest

    def test_cpu_default_gpu_false(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(tu, "train_unified", lambda **kw: captured.update(kw))
        monkeypatch.setattr(sys, "argv", ["prog", "--n-jobs", "8"])
        tu.main()
        assert captured["gpu"] is False
        assert captured["n_jobs"] == 8

    def test_gpu_flag_unavailable_falls_back(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(tu, "enable_gpu", lambda: False)  # cuML absent
        monkeypatch.setattr(tu, "train_unified", lambda **kw: captured.update(kw))
        monkeypatch.setattr(sys, "argv", ["prog", "--gpu", "--n-jobs", "4"])
        tu.main()
        assert captured["gpu"] is False  # falls back to CPU
        assert captured["n_jobs"] == 4
