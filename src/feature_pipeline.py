"""
Two-phase feature assembly for the unified v2 detector (Stage 4).

Builds the 855-dim hybrid matrix per image:
    [ 85 classical features | 768 DINOv2 embedding | 2 RIGID drift ]

Phase A (CPU, multiprocess): classical features via FeatureExtractor's
existing ProcessPoolExecutor path, cached per id at
`data/derived/cache/classical_v2/<id>.npy`.

Phase B (GPU, single process): DINOv2 + drift via DinoEmbedder, cached per id
at `data/derived/cache/emb_vitb14/<id>.npy`. Embeddings must NOT go through
the ProcessPoolExecutor (one CUDA context; classical workers stay torch-free).

Both phases cache per id so the expensive first run (~2-3 h classical,
< 1 h embeddings over ~28k images) resumes for free. The final matrix is
assembled by loading both caches in RECORD ORDER, so X[i] always corresponds
to records[i] — the join is positional, never a re-sort. `cache_meta.json`
records the classical feature-name hash and the embedding model so a change
to either invalidates the right cache instead of silently mixing versions.

Importable without torch: `skip_embeddings=True` yields the 85-dim classical
matrix only (the fallback model's training/prediction path).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.classifier import FeatureExtractor
from src.embedding_extractor import DEFAULT_MODEL, OUT_DIM as EMB_OUT_DIM

CACHE_ROOT = os.path.join("data", "derived", "cache")
CLASSICAL_DIR = os.path.join(CACHE_ROOT, "classical_v2")
EMB_DIR = os.path.join(CACHE_ROOT, "emb_vitb14")
CACHE_META_PATH = os.path.join(CACHE_ROOT, "cache_meta.json")

CLASSICAL_NAMES: List[str] = list(FeatureExtractor.FEATURE_NAMES)      # 85
EMB_NAMES: List[str] = (
    [f"emb_{i:03d}" for i in range(EMB_OUT_DIM - 2)]                    # 768
    + ["drift_emb_2_255", "drift_emb_6_255"]                           # 2
)
FEATURE_NAMES: List[str] = CLASSICAL_NAMES + EMB_NAMES                 # 855

N_CLASSICAL = len(CLASSICAL_NAMES)
N_EMB = EMB_OUT_DIM


# --------------------------------------------------------------------------- #
# Paths, seeds, cache metadata
# --------------------------------------------------------------------------- #

def _sanitize(X: np.ndarray) -> np.ndarray:
    """Last-resort guard: replace any non-finite cell with 0.0 so a single
    degenerate feature (e.g. kurtosis of a zero-variance image) can never put
    NaN/inf into the SVM. The analyzers should return neutral values
    themselves; this catches anything that slips through."""
    if not np.isfinite(X).all():
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _classical_path(rec_id: str) -> str:
    return os.path.join(CLASSICAL_DIR, f"{rec_id}.npy")


def _emb_path(rec_id: str) -> str:
    return os.path.join(EMB_DIR, f"{rec_id}.npy")


def seed_for_id(rec_id: str) -> int:
    """Deterministic 63-bit noise seed from the record id (stable across runs)."""
    digest = hashlib.md5(rec_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


# Bump when a classical feature's COMPUTATION changes but its NAME does not.
# _classical_version hashes the names, so a value-only formula fix (e.g. FFT power
# spectrum, eigen band power/partition, signed Laplacian variance, DCT AC-only
# variance, in-memory drift) would otherwise leave the cached .npy files valid
# and silently reuse stale features. Bumping this forces re-extraction.
#   "1" -> original; "2" -> 2026-06-14 audit formula fixes.
_CLASSICAL_FORMULA_EPOCH = "2"


def _classical_version() -> str:
    payload = "|".join(CLASSICAL_NAMES) + "#epoch=" + _CLASSICAL_FORMULA_EPOCH
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def classical_cache_current() -> bool:
    """True if the on-disk classical cache matches the current feature version.

    Lets end-to-end callers/tests SKIP (rather than hard-error) when a feature
    formula change has invalidated the cache and it has not been re-extracted
    yet. Returns False when no cache exists or the version/epoch differs.
    """
    if not os.path.exists(CACHE_META_PATH):
        return False
    try:
        with open(CACHE_META_PATH) as f:
            old = json.load(f)
    except Exception:
        return False
    return old.get("classical_version") == _classical_version()


def _check_and_write_meta(skip_embeddings: bool) -> None:
    """
    Guard against silently mixing cache versions. Raises if the on-disk cache
    was built with a different classical feature set or embedding model.
    """
    current = {
        "classical_version": _classical_version(),
        "n_classical": N_CLASSICAL,
        "embed_model": None if skip_embeddings else DEFAULT_MODEL,
        "n_emb": N_EMB,
    }
    if os.path.exists(CACHE_META_PATH):
        with open(CACHE_META_PATH) as f:
            old = json.load(f)
        if old.get("classical_version") != current["classical_version"]:
            raise RuntimeError(
                f"Classical feature set changed (cache "
                f"{old.get('classical_version')} vs current "
                f"{current['classical_version']}). Delete {CLASSICAL_DIR} and re-run."
            )
        if (not skip_embeddings
                and old.get("embed_model") not in (None, current["embed_model"])):
            raise RuntimeError(
                f"Embedding model changed (cache {old.get('embed_model')} vs "
                f"current {current['embed_model']}). Delete {EMB_DIR} and re-run."
            )
        # Preserve a previously recorded embed model on classical-only runs.
        current["embed_model"] = current["embed_model"] or old.get("embed_model")
    with open(CACHE_META_PATH, "w") as f:
        json.dump(current, f, indent=1)


# --------------------------------------------------------------------------- #
# Phase A: classical features
# --------------------------------------------------------------------------- #

def _ensure_classical(ids: Sequence[str], paths: Sequence[str],
                      n_workers: int, verbose: bool, chunk: int = 512) -> None:
    missing = [(i, p) for i, p in zip(ids, paths)
               if not os.path.exists(_classical_path(i))]
    if not missing:
        return
    if verbose:
        print(f"[classical] extracting {len(missing)} / {len(ids)} "
              f"(cached: {len(ids) - len(missing)})")
    extractor = FeatureExtractor()
    mids = [m[0] for m in missing]
    mpaths = [m[1] for m in missing]
    # Extract and SAVE in chunks so an interruption keeps completed work —
    # extract_batch returns only at the end of its batch, so a single
    # all-rows call would cache nothing until the whole multi-hour pass done.
    for k in range(0, len(missing), chunk):
        c_ids = mids[k:k + chunk]
        c_paths = mpaths[k:k + chunk]
        matrix = extractor.extract_batch(c_paths, verbose=verbose,
                                         n_workers=n_workers)
        for rec_id, vec in zip(c_ids, matrix):
            np.save(_classical_path(rec_id), vec.astype(np.float32))
        if verbose:
            print(f"  [classical] {min(k + chunk, len(missing))}/{len(missing)}")


# --------------------------------------------------------------------------- #
# Phase B: embeddings
# --------------------------------------------------------------------------- #

def _ensure_embeddings(ids: Sequence[str], paths: Sequence[str],
                       embedder, batch_size: int, chunk: int,
                       verbose: bool) -> None:
    missing = [(i, p) for i, p in zip(ids, paths)
               if not os.path.exists(_emb_path(i))]
    if not missing:
        return
    if embedder is None:
        from src.embedding_extractor import DinoEmbedder
        embedder = DinoEmbedder()
    if verbose:
        print(f"[embed] extracting {len(missing)} / {len(ids)} "
              f"(cached: {len(ids) - len(missing)})")
    mids = [m[0] for m in missing]
    mpaths = [m[1] for m in missing]
    failures: List[Tuple[str, str]] = []
    for k in range(0, len(missing), chunk):
        c_ids = mids[k:k + chunk]
        c_paths = mpaths[k:k + chunk]
        c_seeds = [seed_for_id(i) for i in c_ids]
        try:
            embs = embedder.embed_batch_with_drift(c_paths, c_seeds,
                                                   batch_size=batch_size)
            rows = list(zip(c_ids, embs))
        except Exception:
            # One bad image must not lose the whole chunk: retry per image.
            rows = []
            for rec_id, path, sd in zip(c_ids, c_paths, c_seeds):
                try:
                    vec = embedder.embed_batch_with_drift([path], [sd],
                                                          batch_size=batch_size)[0]
                    rows.append((rec_id, vec))
                except Exception as exc:
                    failures.append((rec_id, str(exc)))
        # Only write successful rows — a failed image is left uncached so a
        # rerun retries it, rather than freezing a fake zero vector.
        for rec_id, vec in rows:
            np.save(_emb_path(rec_id), vec.astype(np.float32))
        if verbose:
            print(f"  [embed] {min(k + chunk, len(missing))}/{len(missing)}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} image(s) failed to embed (left uncached; fix or "
            f"exclude them and re-run). First few: {failures[:10]}"
        )


# --------------------------------------------------------------------------- #
# Public assembly
# --------------------------------------------------------------------------- #

def assemble_features(
    records: Sequence[dict],
    embedder=None,
    n_workers: int = 0,
    batch_size: int = 32,
    chunk: int = 256,
    skip_embeddings: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return (X, feature_names) for the given manifest records.

    X has shape (len(records), 855) — or (len(records), 85) when
    skip_embeddings is True. Row i corresponds to records[i] (positional
    join). Each record must carry 'id' (cache key) and 'path'.
    """
    os.makedirs(CLASSICAL_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)
    _check_and_write_meta(skip_embeddings)

    ids = [r["id"] for r in records]
    paths = [r["path"] for r in records]

    _ensure_classical(ids, paths, n_workers, verbose)
    if not skip_embeddings:
        _ensure_embeddings(ids, paths, embedder, batch_size, chunk, verbose)

    n = len(records)
    if skip_embeddings:
        X = np.empty((n, N_CLASSICAL), dtype=np.float32)
        for i, rec_id in enumerate(ids):
            X[i] = np.load(_classical_path(rec_id))
        return _sanitize(X), CLASSICAL_NAMES

    X = np.empty((n, N_CLASSICAL + N_EMB), dtype=np.float32)
    for i, rec_id in enumerate(ids):
        X[i, :N_CLASSICAL] = np.load(_classical_path(rec_id))
        X[i, N_CLASSICAL:] = np.load(_emb_path(rec_id))
    return _sanitize(X), FEATURE_NAMES


def extract_unified(
    image_path: str,
    embedder=None,
    seed: Optional[int] = None,
    skip_embeddings: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the feature vector for ONE image, in process and uncached — the
    prediction path (web/CLI). Returns (vector, names): 855-dim normally,
    85-dim when skip_embeddings is True (the torch-less fallback). The
    embedding runs in-process (never via the ProcessPoolExecutor) to keep
    single-request latency low.
    """
    classical = FeatureExtractor().extract(image_path).astype(np.float32)
    if skip_embeddings:
        return classical, CLASSICAL_NAMES
    if embedder is None:
        from src.embedding_extractor import DinoEmbedder
        embedder = DinoEmbedder()
    emb = embedder.embed_with_drift(image_path, seed=seed).astype(np.float32)
    return np.concatenate([classical, emb]), FEATURE_NAMES


def assemble_split(
    split_name: str,
    conditions: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[dict]]:
    """
    Convenience wrapper: load a split and return everything training needs.

    Returns (X, y, groups, feature_names, records) where `groups` is the
    per-row base_id for StratifiedGroupKFold (the second leakage gate).
    """
    from src.dataset import load_split
    records = load_split(split_name, conditions=conditions)
    X, names = assemble_features(records, **kwargs)
    y = np.array([r["label"] for r in records], dtype=np.int64)
    groups = np.array([r["base_id"] for r in records])
    return X, y, groups, names, records
