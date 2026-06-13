"""
Tests for src/feature_pipeline.py.

Caches are redirected to a tmp dir so tests never touch the real cache.
The classical-only path needs no torch; the full 855-dim assembly skips
cleanly when torch/timm or the DINOv2 weights are unavailable.
"""

import json
import os

import numpy as np
import pytest

from src import feature_pipeline as fp

_HAS_MANIFEST = os.path.exists("data/manifests/base_manifest.json")
pytestmark = pytest.mark.skipif(not _HAS_MANIFEST, reason="dataset manifest absent")


def _sample_records(n=3, label=0):
    with open("data/manifests/base_manifest.json") as f:
        mani = json.load(f)
    recs = [r for r in mani if r["label"] == label][:n]
    out = []
    for r in recs:
        rr = dict(r)
        rr["base_id"] = r["id"]
        rr["condition"] = "clean"
        out.append(rr)
    return out


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "CLASSICAL_DIR", str(tmp_path / "classical_v2"))
    monkeypatch.setattr(fp, "EMB_DIR", str(tmp_path / "emb_vitb14"))
    monkeypatch.setattr(fp, "CACHE_META_PATH", str(tmp_path / "cache_meta.json"))
    return tmp_path


class TestNames:
    def test_dimensions(self):
        assert fp.N_CLASSICAL == 85
        assert fp.N_EMB == 770
        assert len(fp.FEATURE_NAMES) == 855
        assert len(set(fp.FEATURE_NAMES)) == 855  # all unique

    def test_block_layout(self):
        assert fp.FEATURE_NAMES[:85] == fp.CLASSICAL_NAMES
        assert fp.FEATURE_NAMES[-2:] == ["drift_emb_2_255", "drift_emb_6_255"]


class TestSeed:
    def test_deterministic_and_distinct(self):
        assert fp.seed_for_id("abc") == fp.seed_for_id("abc")
        assert fp.seed_for_id("abc") != fp.seed_for_id("abd")
        assert 0 <= fp.seed_for_id("x") < 2 ** 63


class TestClassicalAssembly:
    def test_shape_and_cache_roundtrip(self, tmp_cache):
        recs = _sample_records(3)
        X, names = fp.assemble_features(recs, skip_embeddings=True, verbose=False)
        assert X.shape == (3, 85)
        assert names == fp.CLASSICAL_NAMES
        for r in recs:
            assert os.path.exists(fp._classical_path(r["id"]))
        # Second call loads from cache and is identical.
        X2, _ = fp.assemble_features(recs, skip_embeddings=True, verbose=False)
        np.testing.assert_array_equal(X, X2)

    def test_row_order_follows_record_order(self, tmp_cache):
        recs = _sample_records(3)
        X, _ = fp.assemble_features(recs, skip_embeddings=True, verbose=False)
        Xr, _ = fp.assemble_features(list(reversed(recs)), skip_embeddings=True,
                                     verbose=False)
        np.testing.assert_array_equal(Xr, X[::-1])

    def test_drift_seeding_makes_classical_reproducible(self, tmp_cache):
        # Same image extracted twice (cache cleared between) must match, since
        # the RIGID drift noise is now seeded from image content.
        rec = _sample_records(1)[0]
        ex = fp.FeatureExtractor()
        v1 = ex.extract(rec["path"])
        v2 = ex.extract(rec["path"])
        np.testing.assert_allclose(v1, v2, rtol=0, atol=1e-9)

    def test_meta_mismatch_raises(self, tmp_cache):
        os.makedirs(fp.CLASSICAL_DIR, exist_ok=True)
        os.makedirs(fp.EMB_DIR, exist_ok=True)
        with open(fp.CACHE_META_PATH, "w") as f:
            json.dump({"classical_version": "deadbeef", "embed_model": None}, f)
        with pytest.raises(RuntimeError):
            fp.assemble_features(_sample_records(2), skip_embeddings=True,
                                 verbose=False)


class TestFullAssembly:
    def test_855_and_blocks_match(self, tmp_cache):
        pytest.importorskip("torch")
        pytest.importorskip("timm")
        from src.embedding_extractor import DinoEmbedder
        try:
            emb = DinoEmbedder(device="cpu", half=False)
        except Exception as exc:
            pytest.skip(f"DINOv2 weights unavailable: {exc}")

        recs = _sample_records(2)
        X, names = fp.assemble_features(recs, embedder=emb, verbose=False)
        assert X.shape == (2, 855)
        assert len(names) == 855

        # Classical block equals the classical-only assembly (same cache).
        Xc, _ = fp.assemble_features(recs, skip_embeddings=True, verbose=False)
        np.testing.assert_allclose(X[:, :85], Xc, rtol=0, atol=1e-5)

        # Embedding block equals a direct seeded embed of the same image.
        direct = emb.embed_with_drift(
            recs[0]["path"], seed=fp.seed_for_id(recs[0]["id"])
        )
        np.testing.assert_allclose(X[0, 85:], direct, rtol=0, atol=1e-4)


class TestExtractUnified:
    def test_classical_only_shape(self, tmp_cache):
        rec = _sample_records(1)[0]
        vec, names = fp.extract_unified(rec["path"], skip_embeddings=True)
        assert vec.shape == (85,)
        assert names == fp.CLASSICAL_NAMES

    def test_full_vector_matches_cached_assembly(self, tmp_cache):
        pytest.importorskip("torch")
        pytest.importorskip("timm")
        from src.embedding_extractor import DinoEmbedder
        try:
            emb = DinoEmbedder(device="cpu", half=False)
        except Exception as exc:
            pytest.skip(f"DINOv2 weights unavailable: {exc}")
        rec = _sample_records(1)[0]
        vec, names = fp.extract_unified(
            rec["path"], embedder=emb, seed=fp.seed_for_id(rec["id"])
        )
        assert vec.shape == (855,) and len(names) == 855
        X, _ = fp.assemble_features([rec], embedder=emb, verbose=False)
        np.testing.assert_allclose(vec, X[0], rtol=0, atol=1e-4)

    def test_bad_image_isolated_and_reported(self, tmp_cache):
        pytest.importorskip("torch")
        pytest.importorskip("timm")
        from src.embedding_extractor import DinoEmbedder
        try:
            emb = DinoEmbedder(device="cpu", half=False)
        except Exception as exc:
            pytest.skip(f"DINOv2 weights unavailable: {exc}")
        good = _sample_records(1)[0]
        bad = dict(good)
        bad["id"] = "nonexistent_image_xyz"
        bad["path"] = "data/this_path_does_not_exist.png"
        with pytest.raises(RuntimeError, match="failed to embed"):
            fp.assemble_features([good, bad], embedder=emb, verbose=False)
        # The good image still got cached; the bad one was left uncached.
        assert os.path.exists(fp._emb_path(good["id"]))
        assert not os.path.exists(fp._emb_path(bad["id"]))
