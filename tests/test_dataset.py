"""
Tests for src/dataset.py — the manifest and the leak-proofing invariants.

The split invariants are the most important tests in the repository: if they
fail, every accuracy number the pipeline produces is untrustworthy.
"""

import os

import pytest

from src.dataset import (
    DEFAULT_HOLDOUT_FAMILIES,
    build_manifest,
    build_splits,
)

DATA_ROOT = 'data'
needs_data = pytest.mark.skipif(
    not os.path.isdir(os.path.join(DATA_ROOT, 'mixed')),
    reason="dataset not present on this machine",
)


@pytest.fixture(scope='module')
def manifest():
    return build_manifest(DATA_ROOT)


@pytest.fixture(scope='module')
def assignments(manifest):
    return build_splits(manifest)


@needs_data
class TestManifest:

    def test_total_and_class_counts(self, manifest):
        assert len(manifest) > 4000
        n_real = sum(1 for r in manifest if r['label'] == 0)
        n_ai = len(manifest) - n_real
        assert n_real >= 1900
        assert n_ai >= 2200

    def test_families_present(self, manifest):
        families = {r['family'] for r in manifest}
        # 32 mixed families + coco + openfake + gemini + gpt
        assert len(families) >= 34
        for fam in DEFAULT_HOLDOUT_FAMILIES:
            assert fam in families

    def test_ids_unique(self, manifest):
        ids = [r['id'] for r in manifest]
        assert len(ids) == len(set(ids))

    def test_ids_cache_safe(self, manifest):
        for rec in manifest:
            assert rec['id'] == rec['id'].lower()
            assert all(c.isalnum() or c == '_' for c in rec['id'])

    def test_paths_exist(self, manifest):
        for rec in manifest[::97]:  # sample, full check is slow
            assert os.path.exists(rec['path']), rec['path']

    def test_dimensions_read(self, manifest):
        readable = sum(1 for r in manifest if r['width'] > 0 and r['height'] > 0)
        assert readable / len(manifest) > 0.99


@needs_data
class TestSplitInvariants:

    def test_every_id_assigned_exactly_once(self, manifest, assignments):
        assert set(assignments) == {r['id'] for r in manifest}

    def test_holdout_families_fully_held_out(self, manifest, assignments):
        for rec in manifest:
            if rec['family'] in DEFAULT_HOLDOUT_FAMILIES:
                assert assignments[rec['id']] == 'test_holdout', rec['id']
            else:
                assert assignments[rec['id']] != 'test_holdout', rec['id']

    def test_split_fractions_roughly_honored(self, manifest, assignments):
        non_holdout = [r for r in manifest
                       if r['family'] not in DEFAULT_HOLDOUT_FAMILIES]
        n = len(non_holdout)
        n_train = sum(1 for r in non_holdout if assignments[r['id']] == 'train')
        n_val = sum(1 for r in non_holdout if assignments[r['id']] == 'val')
        assert abs(n_train / n - 0.7) < 0.03
        assert abs(n_val / n - 0.1) < 0.03

    def test_each_family_stratified(self, manifest, assignments):
        by_family = {}
        for rec in manifest:
            if rec['family'] not in DEFAULT_HOLDOUT_FAMILIES:
                by_family.setdefault(rec['family'], []).append(assignments[rec['id']])
        for family, splits in by_family.items():
            assert 'train' in splits, f"{family} has no training images"
            assert 'test' in splits, f"{family} has no test images"

    def test_deterministic(self, manifest):
        a = build_splits(manifest, seed=42)
        b = build_splits(manifest, seed=42)
        assert a == b
        c = build_splits(manifest, seed=43)
        assert a != c

    def test_unknown_holdout_family_rejected(self, manifest):
        with pytest.raises(ValueError):
            build_splits(manifest, holdout_families=('not_a_family',))
