"""
Stage 5 — train the unified 855-dim detector and the 85-dim classical
fallback.

Two models are produced from the SAME assembled training matrix:
    models/unified_v2.pkl    full 855-dim hybrid (classical + embedding)
    models/classical_v2.pkl  85-dim classical only (torch-less fallback +
                             the headline ablation)

The old models/*.pkl are never touched — they were trained on different,
now-removed data and serve only as a reference baseline (see CLAUDE.md).

Leakage gates (two of the three in the project; the third is the base-level
split in dataset.py):
    - StratifiedGroupKFold(groups=base_id): variants of one base never
      straddle a CV fold, so the grid cannot pick parameters that overfit to
      near-duplicate leakage.

Probability handling: the grid scores with ROC-AUC over decision_function
(probability=False, since Platt scaling = an inner CV that multiplies cost
and is deprecated in sklearn >= 1.9). The saved model is calibrated once via
CalibratedClassifierCV(ensemble=False), and that calibration CV is ALSO
group-aware (explicit StratifiedGroupKFold index splits on base_id) so the
sigmoid is not fit on near-duplicate variants of its own held-out rows -- which
would optimistically bias accuracy@0.5 and the clean-accuracy gate.

Run: `python -m src.train_unified` (see --help). Bundle rationale and the
row budget live in code_notes/18-training.md.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.embedding_extractor import DEFAULT_MODEL
from src.feature_pipeline import N_CLASSICAL

DEFAULT_C = (1.0, 10.0, 100.0)
DEFAULT_GAMMA = ("scale", 1e-3, 1e-4)


DEFAULT_MAX_ITER = 5000  # caps each SVC fit so a hard config can't run unbounded


def _param_grid(c_values: Sequence[float], gamma_values: Sequence) -> List[dict]:
    """RBF grid (C x gamma) plus one linear sweep (C only)."""
    return [
        {"svc__kernel": ["rbf"], "svc__C": list(c_values),
         "svc__gamma": list(gamma_values)},
        {"svc__kernel": ["linear"], "svc__C": list(c_values)},
    ]


def _configs(c_values: Sequence[float], gamma_values: Sequence) -> List[dict]:
    """Flat list of SVC kwargs: RBF (C x gamma) + linear (C)."""
    rbf = [{"kernel": "rbf", "C": float(c), "gamma": g}
           for c in c_values for g in gamma_values]
    linear = [{"kernel": "linear", "C": float(c)} for c in c_values]
    return rbf + linear


def _gpu_grid_search(X, y, groups, splits, c_values, gamma_values,
                     max_iter, verbose):
    """
    Group-aware CV grid search using cuML's GPU SVC. Returns (best_kwargs,
    best_auc). Each fold scales on its own train rows (no leakage), fits on
    GPU, and scores ROC-AUC over decision_function (numpy out, no cupy).
    """
    from cuml.svm import SVC as cuSVC

    cv = StratifiedGroupKFold(n_splits=splits)
    Xf = np.ascontiguousarray(X, dtype=np.float32)
    yf = y.astype(np.float32)
    best, best_auc = None, -1.0
    for cfg in _configs(c_values, gamma_values):
        aucs = []
        for tr, va in cv.split(Xf, y, groups):
            scaler = StandardScaler().fit(Xf[tr])
            Xtr = scaler.transform(Xf[tr]).astype(np.float32)
            Xva = scaler.transform(Xf[va]).astype(np.float32)
            svc = cuSVC(class_weight="balanced", max_iter=max_iter, **cfg)
            svc.fit(Xtr, yf[tr])
            scores = np.asarray(svc.decision_function(Xva)).ravel()
            aucs.append(roc_auc_score(y[va], scores))
        mean_auc = float(np.mean(aucs))
        if verbose:
            print(f"  [gpu] {cfg} -> CV AUC {mean_auc:.4f}")
        if mean_auc > best_auc:
            best_auc, best = mean_auc, cfg
    return best, best_auc


def train_from_matrix(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    embed_model_name: Optional[str],
    classical_only: bool,
    n_splits: int = 5,
    c_values: Sequence[float] = DEFAULT_C,
    gamma_values: Sequence = DEFAULT_GAMMA,
    n_jobs: int = -1,
    gpu: bool = False,
    max_iter: int = DEFAULT_MAX_ITER,
    verbose: bool = True,
) -> Tuple[dict, dict]:
    """
    Grid-search an RBF/linear SVM with group-aware CV, then fit a calibrated
    final model on all rows. Returns (bundle, summary).

    gpu=True runs the grid search on the GPU via cuML SVC (fast); the FINAL
    saved model is always a sklearn pipeline, so the .pkl loads and predicts
    on CPU with no cuML/GPU dependency. The scaler lives inside the pipeline
    so it is refit within every fold (no leakage); `groups` (base_id) drives
    StratifiedGroupKFold. `max_iter` caps each SVC fit so a hard config cannot
    run unbounded.
    """
    n_groups = len(np.unique(groups))
    splits = max(2, min(n_splits, n_groups))

    if gpu:
        best, best_auc = _gpu_grid_search(
            X, y, groups, splits, c_values, gamma_values, max_iter, verbose)
    else:
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(class_weight="balanced", max_iter=max_iter)),
        ])
        grid = GridSearchCV(
            base, _param_grid(c_values, gamma_values),
            scoring="roc_auc", cv=StratifiedGroupKFold(n_splits=splits),
            n_jobs=n_jobs, refit=False,
        )
        grid.fit(X, y, groups=groups)
        best = {k.replace("svc__", ""): v for k, v in grid.best_params_.items()}
        best_auc = float(grid.best_score_)
    if verbose:
        print(f"  best params: {best}  (CV ROC-AUC {best_auc:.4f})")

    # Final model is ALWAYS sklearn so the pickle is portable (CPU eval/web,
    # no cuML). Calibrated once for predict_proba.
    #
    # The calibration CV MUST be group-aware: variants of one base_id straddling
    # a calibration fold leak near-duplicates into the sigmoid fit and
    # optimistically bias every threshold-dependent metric (accuracy@0.5 and the
    # clean-accuracy gate). cv=<int> would silently use plain StratifiedKFold, so
    # we pass explicit StratifiedGroupKFold index splits. The scaler also lives
    # INSIDE the calibrated estimator so it is refit on each calibration fold's
    # train rows only (no scaler leakage into the held-out calibration data).
    base_est = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(class_weight="balanced", max_iter=max_iter, **best)),
    ])
    cal_splits = list(StratifiedGroupKFold(n_splits=splits).split(X, y, groups))
    final = CalibratedClassifierCV(
        base_est, method="sigmoid", cv=cal_splits, ensemble=False)
    final.fit(X, y)

    bundle = {
        "version": 2,
        "pipeline": final,
        "feature_names": list(feature_names),
        "embed_model_name": embed_model_name,
        "classical_only": classical_only,
        "best_params": best,
        "cv_roc_auc": float(best_auc),
        "trained_on_gpu": bool(gpu),
        "n_train": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary = {
        "classical_only": classical_only,
        "n_train": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "best_params": best,
        "cv_roc_auc": float(best_auc),
    }
    return bundle, summary


def _embedding_only_ablation(X_emb: np.ndarray, y: np.ndarray,
                             groups: np.ndarray, n_splits: int) -> float:
    """Quick linear-probe CV AUC on the embedding block (reported, not saved)."""
    splits = max(2, min(n_splits, len(np.unique(groups))))
    cv = StratifiedGroupKFold(n_splits=splits)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000,
                                               class_weight="balanced"))])
    scores = cross_val_score(pipe, X_emb, y, groups=groups, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores))


def train_unified(
    out_dir: str = "models",
    reports_dir: str = "reports",
    n_splits: int = 5,
    max_rows: int = 0,
    n_jobs: int = -1,
    gpu: bool = False,
    embedder=None,
    verbose: bool = True,
) -> dict:
    """
    Assemble the train split, then fit and save both models. The derived
    manifest already encodes the train condition policy (clean + 2-of-3
    platforms + 1 chain + screenshots per base), so the full train split is
    used as-is.
    """
    from src.feature_pipeline import assemble_split

    if verbose:
        print("Assembling training features (this populates the caches on "
              "first run; ~2-3 h classical + <1 h embeddings)...")
    X, y, groups, names, _ = assemble_split(
        "train", embedder=embedder, n_workers=n_jobs if n_jobs > 0 else 0,
        verbose=verbose)

    if max_rows and X.shape[0] > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_rows, replace=False)
        X, y, groups = X[idx], y[idx], groups[idx]
        if verbose:
            print(f"  subsampled to {max_rows} rows for a quick run")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    if verbose:
        print(f"Training unified model on {X.shape}...")
    unified_bundle, unified_sum = train_from_matrix(
        X, y, groups, names, DEFAULT_MODEL, classical_only=False,
        n_splits=n_splits, n_jobs=n_jobs, gpu=gpu, verbose=verbose)
    joblib.dump(unified_bundle, os.path.join(out_dir, "unified_v2.pkl"))

    if verbose:
        print(f"Training classical fallback on {X[:, :N_CLASSICAL].shape}...")
    classical_bundle, classical_sum = train_from_matrix(
        X[:, :N_CLASSICAL], y, groups, names[:N_CLASSICAL], None,
        classical_only=True, n_splits=n_splits, n_jobs=n_jobs, gpu=gpu,
        verbose=verbose)
    joblib.dump(classical_bundle, os.path.join(out_dir, "classical_v2.pkl"))

    emb_auc = _embedding_only_ablation(X[:, N_CLASSICAL:], y, groups, n_splits)

    summary = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train_rows": int(X.shape[0]),
        "unified": unified_sum,
        "classical_only": classical_sum,
        "embedding_only_cv_roc_auc": emb_auc,
    }
    with open(os.path.join(reports_dir, "train_v2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"\nCV ROC-AUC -- unified: {unified_sum['cv_roc_auc']:.4f} | "
              f"classical: {classical_sum['cv_roc_auc']:.4f} | "
              f"embedding-only: {emb_auc:.4f}")
        print(f"Saved unified_v2.pkl + classical_v2.pkl to {out_dir}/")
    return summary


def enable_gpu() -> bool:
    """
    Check that cuML's GPU SVC is importable (the grid search uses it directly
    via `_gpu_grid_search` — NOT the cuml.accel transparent accelerator, which
    does not intercept SVC inside a Pipeline/GridSearchCV). Returns False when
    cuML is absent so the caller stays on the CPU sklearn path.
    """
    try:
        from cuml.svm import SVC  # noqa: F401
        return True
    except Exception as exc:  # cuML not installed / GPU unavailable
        print(f"[gpu] cuML GPU SVC unavailable, using CPU: {exc}")
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Train the unified v2 detector")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--max-rows", type=int, default=0,
                   help="subsample train rows (0 = all) for a quick run")
    p.add_argument("--gpu", action="store_true",
                   help="run the SVM grid search on GPU via cuML SVC")
    args = p.parse_args()

    gpu = args.gpu and enable_gpu()
    if gpu:
        print("[gpu] cuML GPU SVC active for the grid search")

    train_unified(out_dir=args.out_dir, reports_dir=args.reports_dir,
                  n_splits=args.folds, max_rows=args.max_rows,
                  n_jobs=args.n_jobs, gpu=gpu)


if __name__ == "__main__":
    main()
