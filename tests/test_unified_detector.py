"""
Tests for src/unified_detector.py.

The classical-fallback path is exercised with a tiny synthetic classical
bundle on a real manifest image (no torch needed). The unified path is
skipped unless torch + DINOv2 weights are available.
"""

import json
import os

import joblib
import numpy as np
import pytest

from src import train_unified as tu
from src import unified_detector as ud
from src.feature_pipeline import CLASSICAL_NAMES, FEATURE_NAMES, N_CLASSICAL

_HAS_MANIFEST = os.path.exists("data/manifests/base_manifest.json")
pytestmark = pytest.mark.skipif(not _HAS_MANIFEST, reason="dataset manifest absent")


def _real_image_path():
    with open("data/manifests/base_manifest.json") as f:
        mani = json.load(f)
    return next(r["path"] for r in mani if r["label"] == 0)


def _synthetic_bundle(n_features, names, classical_only, path):
    rng = np.random.default_rng(0)
    X, y, g = [], [], []
    for k in range(24):
        lab = k % 2
        X.append(np.full(n_features, 1.0 if lab else -1.0)
                 + rng.normal(0, 0.5, (6, n_features)))
        y += [lab] * 6
        g += [f"b{k}"] * 6
    bundle, _ = tu.train_from_matrix(
        np.vstack(X).astype(np.float32), np.array(y), np.array(g), names,
        embed_model_name=None if classical_only else "m",
        classical_only=classical_only, n_splits=3,
        c_values=[1.0], gamma_values=["scale"], n_jobs=1, verbose=False)
    joblib.dump(bundle, path)
    return path


class TestClassicalFallback:
    def test_predict_with_classical_only(self, tmp_path):
        cls_path = _synthetic_bundle(N_CLASSICAL, CLASSICAL_NAMES, True,
                                     str(tmp_path / "classical_v2.pkl"))
        det = ud.UnifiedDetector(unified_path=str(tmp_path / "nope.pkl"),
                                 classical_path=cls_path)
        res = det.predict(_real_image_path())
        assert res["label"] in ("Real", "AI-Generated")
        assert res["method"] == "classical_v2"
        assert res["fallback"] is True
        assert 0.0 <= res["confidence"] <= 1.0
        assert 0.0 <= res["probability_ai"] <= 1.0

    def test_no_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ud.UnifiedDetector(unified_path=str(tmp_path / "a.pkl"),
                               classical_path=str(tmp_path / "b.pkl"))


class TestUnifiedPath:
    def test_predict_with_unified_model(self, tmp_path):
        pytest.importorskip("torch")
        pytest.importorskip("timm")
        try:
            from src.embedding_extractor import DinoEmbedder
            DinoEmbedder(device="cpu", half=False)
        except Exception as exc:
            pytest.skip(f"DINOv2 weights unavailable: {exc}")
        uni_path = _synthetic_bundle(len(FEATURE_NAMES), FEATURE_NAMES, False,
                                     str(tmp_path / "unified_v2.pkl"))
        det = ud.UnifiedDetector(unified_path=uni_path,
                                 classical_path=str(tmp_path / "nope.pkl"))
        res = det.predict(_real_image_path())
        assert res["method"] == "unified_v2"
        assert res["fallback"] is False
        assert res["label"] in ("Real", "AI-Generated")
