"""
Stage 6 — evaluate the unified v2 detector against the absolute acceptance
gates, with per-condition / per-family / per-resolution breakdowns.

Metrics per group: accuracy (at the calibrated 0.5 threshold), ROC-AUC,
average precision, and Pd@5%FAR — "allowed 5 false accusations per 100 real
images, how many fakes do you catch". AUC flatters; Pd@FAR is the operational
number (see paper_notes/04). Held-out generator families have NO real images,
so their Pd uses the clean-condition reals as the false-alarm reference.

Compares the unified model to its classical-only ablation (gate 5: the hybrid
must not be worse than classical-only anywhere, else the embedding
integration has a bug). The old v1 models are NOT run here: they were trained
on 79-dim features, incompatible with the current 85-dim classical vector, so
a same-rows comparison is not meaningful (they are a reference baseline only,
per CLAUDE.md).

Run: `python -m src.evaluate_unified`. Writes reports/eval_v2_<date>.{json,md}.
Rationale and the gate values live in code_notes/19-evaluation.md.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.feature_pipeline import N_CLASSICAL

RES_BUCKETS = ("<400", "400-800", ">800")


# --------------------------------------------------------------------------- #
# Metric core (no model / cache needed — unit tested)
# --------------------------------------------------------------------------- #

def binary_metrics(y_true: Sequence[int], scores: Sequence[float],
                   threshold: float = 0.5) -> dict:
    """
    Accuracy/AUC/AP for a group. AUC and AP require both classes present;
    they are None for single-class groups (e.g. all-AI holdout), where
    accuracy at the threshold is still meaningful.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "n_ai": int((y_true == 1).sum()),
        "n_real": int((y_true == 0).sum()),
        "accuracy": float((preds == y_true).mean()) if len(y_true) else None,
        "auc": None,
        "ap": None,
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, scores))
        out["ap"] = float(average_precision_score(y_true, scores))
    return out


def pd_at_far(real_scores: Sequence[float], ai_scores: Sequence[float],
              target_far: float = 0.05) -> dict:
    """
    Detection rate at a fixed false-alarm rate. Threshold = the score above
    which only `target_far` of REAL images fall; Pd = fraction of AI images
    at or above it. real_scores is the false-alarm reference (may come from a
    different group, e.g. clean reals for the all-AI holdout).
    """
    real_scores = np.asarray(real_scores, dtype=float)
    ai_scores = np.asarray(ai_scores, dtype=float)
    if len(real_scores) == 0 or len(ai_scores) == 0:
        return {"pd": None, "threshold": None, "far_target": target_far}
    threshold = float(np.quantile(real_scores, 1.0 - target_far))
    return {
        "pd": float(np.mean(ai_scores >= threshold)),
        "threshold": threshold,
        "far_target": target_far,
    }


def resolution_bucket(min_side: int) -> str:
    if min_side < 400:
        return "<400"
    if min_side <= 800:
        return "400-800"
    return ">800"


def group_report(y: np.ndarray, scores: np.ndarray,
                 real_ref_scores: np.ndarray, threshold: float = 0.5) -> dict:
    """binary_metrics + Pd@5%FAR for one subset, using its own reals when
    present, else the provided reference reals."""
    rep = binary_metrics(y, scores, threshold)
    own_real = scores[y == 0]
    ref = own_real if len(own_real) > 0 else real_ref_scores
    rep.update({"pd_at_5far": pd_at_far(ref, scores[y == 1])["pd"]})
    return rep


def check_gates(results: dict) -> List[dict]:
    """
    Absolute acceptance gates (CLAUDE.md / plan). Each returns
    {name, target, value, passed}. Missing inputs -> passed None (not run).
    """
    gates: List[dict] = []

    def add(name, target, value, ok):
        gates.append({"name": name, "target": target,
                      "value": value, "passed": ok})

    cond = results.get("by_condition", {})
    uni = results.get("unified_overall", {})
    cls = results.get("classical_overall", {})

    clean = cond.get("clean", {})
    if clean:
        add("clean AUC >= 0.95", 0.95, clean.get("auc"),
            clean.get("auc") is not None and clean["auc"] >= 0.95)
        add("clean accuracy >= 0.90", 0.90, clean.get("accuracy"),
            clean.get("accuracy") is not None and clean["accuracy"] >= 0.90)

    ss = cond.get("screenshot", {})
    if ss:
        add("screenshot accuracy >= 0.85", 0.85, ss.get("accuracy"),
            ss.get("accuracy") is not None and ss["accuracy"] >= 0.85)

    ho = results.get("holdout_overall", {})
    if ho:
        add("holdout AUC >= 0.85", 0.85, ho.get("auc"),
            ho.get("auc") is not None and ho["auc"] >= 0.85)
        add("holdout Pd@5%FAR >= 0.60", 0.60, ho.get("pd_at_5far"),
            ho.get("pd_at_5far") is not None and ho["pd_at_5far"] >= 0.60)

    for platform in ("facebook", "x", "telegram"):
        g = cond.get(platform, {})
        if g:
            add(f"{platform} AUC >= 0.88", 0.88, g.get("auc"),
                g.get("auc") is not None and g["auc"] >= 0.88)
    for chain in ("chain_fb_x", "chain_ss_tg"):
        g = cond.get(chain, {})
        if g:
            add(f"{chain} accuracy >= 0.75", 0.75, g.get("accuracy"),
                g.get("accuracy") is not None and g["accuracy"] >= 0.75)

    if uni.get("auc") is not None and cls.get("auc") is not None:
        add("unified AUC >= classical-only AUC", cls["auc"], uni["auc"],
            uni["auc"] >= cls["auc"] - 1e-9)

    return gates


def exif_permutation_check(pipeline, X: np.ndarray, y: np.ndarray,
                           feature_names: Sequence[str], seed: int = 0) -> dict:
    """
    Shuffle the metadata feature columns and measure the AUC drop. A small
    drop means the model is not leaning on EXIF (it must not — the real
    images carry no EXIF, so metadata features would be a class shortcut).
    """
    idx = [i for i, n in enumerate(feature_names) if n.startswith("meta_")]
    base = roc_auc_score(y, pipeline.predict_proba(X)[:, 1])
    if not idx:
        return {"auc_base": float(base), "auc_metadata_permuted": float(base),
                "auc_drop": 0.0, "n_metadata_features": 0}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    Xp = X.copy()
    Xp[:, idx] = X[perm][:, idx]
    permuted = roc_auc_score(y, pipeline.predict_proba(Xp)[:, 1])
    return {
        "auc_base": float(base),
        "auc_metadata_permuted": float(permuted),
        "auc_drop": float(base - permuted),
        "n_metadata_features": len(idx),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _min_side_lookup() -> Dict[str, int]:
    """base_id -> min(width, height) from the base manifest (for resolution
    buckets; derived records carry base_id but not dimensions)."""
    from src.dataset import load_manifest
    out = {}
    for rec in load_manifest():
        w, h = rec.get("width", 0), rec.get("height", 0)
        out[rec["id"]] = min(w, h) if w and h else 0
    return out


def _indices_by(records: Sequence[dict], key) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, rec in enumerate(records):
        groups.setdefault(key(rec), []).append(i)
    return groups


def evaluate_unified(
    models_dir: str = "models",
    reports_dir: str = "reports",
    embedder=None,
    verbose: bool = True,
) -> dict:
    """Load the v2 bundles, assemble test + holdout, compute all breakdowns
    and gates, and write the report."""
    from src.feature_pipeline import assemble_split

    unified_path = os.path.join(models_dir, "unified_v2.pkl")
    classical_path = os.path.join(models_dir, "classical_v2.pkl")
    if not os.path.exists(unified_path):
        raise FileNotFoundError(
            f"{unified_path} not found — run `python -m src.train_unified` first.")
    unified = joblib.load(unified_path)
    classical = joblib.load(classical_path) if os.path.exists(classical_path) else None
    names = unified["feature_names"]

    if verbose:
        print("Assembling test + holdout features...")
    Xte, yte, _, _, recs = assemble_split("test", embedder=embedder, verbose=verbose)
    Xho, yho, _, _, recs_ho = assemble_split("test_holdout", embedder=embedder,
                                             verbose=verbose)

    uni_pipe = unified["pipeline"]
    s_uni = uni_pipe.predict_proba(Xte)[:, 1]
    s_uni_ho = uni_pipe.predict_proba(Xho)[:, 1]
    clean_real_ref = s_uni[(yte == 0) &
                           np.array([r["condition"] == "clean" for r in recs])]

    results: dict = {
        "unified_overall": group_report(yte, s_uni, clean_real_ref),
        "holdout_overall": {
            **binary_metrics(yho, s_uni_ho),
            "pd_at_5far": pd_at_far(clean_real_ref, s_uni_ho[yho == 1])["pd"],
        },
        "by_condition": {}, "by_family": {}, "by_resolution": {},
    }
    if classical is not None:
        s_cls = classical["pipeline"].predict_proba(Xte[:, :N_CLASSICAL])[:, 1]
        results["classical_overall"] = group_report(yte, s_cls, clean_real_ref)

    for cond, idx in _indices_by(recs, lambda r: r["condition"]).items():
        idx = np.array(idx)
        results["by_condition"][cond] = group_report(
            yte[idx], s_uni[idx], clean_real_ref)

    for fam, idx in _indices_by(recs, lambda r: r["family"]).items():
        idx = np.array(idx)
        results["by_family"][fam] = group_report(yte[idx], s_uni[idx], clean_real_ref)
    # Holdout families (all-AI) get their own block, flagged.
    for fam, idx in _indices_by(recs_ho, lambda r: r["family"]).items():
        idx = np.array(idx)
        rep = binary_metrics(yho[idx], s_uni_ho[idx])
        rep["pd_at_5far"] = pd_at_far(clean_real_ref, s_uni_ho[idx][yho[idx] == 1])["pd"]
        rep["holdout"] = True
        results["by_family"][fam] = rep

    min_side = _min_side_lookup()
    bucket_of = lambda r: resolution_bucket(min_side.get(r.get("base_id", r["id"]), 0))
    for bucket, idx in _indices_by(recs, bucket_of).items():
        idx = np.array(idx)
        results["by_resolution"][bucket] = group_report(
            yte[idx], s_uni[idx], clean_real_ref)

    results["exif_permutation"] = exif_permutation_check(uni_pipe, Xte, yte, names)
    results["gates"] = check_gates(results)
    results["meta"] = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_test": int(len(yte)), "n_holdout": int(len(yho)),
        "unified_features": int(Xte.shape[1]),
    }

    os.makedirs(reports_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    with open(os.path.join(reports_dir, f"eval_v2_{stamp}.json"), "w") as f:
        json.dump(results, f, indent=2)
    md = render_markdown(results)
    with open(os.path.join(reports_dir, f"eval_v2_{stamp}.md"), "w") as f:
        f.write(md)
    if verbose:
        print(md)
    return results


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _table(title: str, rows: Dict[str, dict]) -> str:
    lines = [f"### {title}", "",
             "| group | n | AUC | acc | AP | Pd@5%FAR |",
             "|---|---|---|---|---|---|"]
    for name, m in sorted(rows.items()):
        flag = " (holdout)" if m.get("holdout") else ""
        lines.append(f"| {name}{flag} | {m.get('n','')} | {_fmt(m.get('auc'))} "
                     f"| {_fmt(m.get('accuracy'))} | {_fmt(m.get('ap'))} "
                     f"| {_fmt(m.get('pd_at_5far'))} |")
    return "\n".join(lines) + "\n"


def render_markdown(results: dict) -> str:
    meta = results.get("meta", {})
    out = [f"# Unified v2 Evaluation ({meta.get('evaluated_at','')})", "",
           f"Test rows: {meta.get('n_test','?')} | holdout rows: "
           f"{meta.get('n_holdout','?')} | features: "
           f"{meta.get('unified_features','?')}", ""]

    out.append("## Gates")
    out.append("| gate | target | value | result |")
    out.append("|---|---|---|---|")
    for g in results.get("gates", []):
        res = "PASS" if g["passed"] else ("n/a" if g["passed"] is None else "FAIL")
        tgt = g["target"] if isinstance(g["target"], str) else _fmt(g["target"])
        out.append(f"| {g['name']} | {tgt} | {_fmt(g['value'])} | {res} |")
    out.append("")

    uni = results.get("unified_overall", {})
    cls = results.get("classical_overall", {})
    out.append("## Overall")
    out.append("| model | n | AUC | acc | AP | Pd@5%FAR |")
    out.append("|---|---|---|---|---|---|")
    out.append(f"| unified (855) | {uni.get('n','')} | {_fmt(uni.get('auc'))} "
               f"| {_fmt(uni.get('accuracy'))} | {_fmt(uni.get('ap'))} "
               f"| {_fmt(uni.get('pd_at_5far'))} |")
    if cls:
        out.append(f"| classical-only (85) | {cls.get('n','')} | {_fmt(cls.get('auc'))} "
                   f"| {_fmt(cls.get('accuracy'))} | {_fmt(cls.get('ap'))} "
                   f"| {_fmt(cls.get('pd_at_5far'))} |")
    out.append("")

    out.append(_table("By condition", results.get("by_condition", {})))
    out.append(_table("By resolution bucket", results.get("by_resolution", {})))
    out.append(_table("By family", results.get("by_family", {})))

    exif = results.get("exif_permutation", {})
    out.append("## EXIF permutation check")
    out.append(f"Base AUC {_fmt(exif.get('auc_base'))}, metadata-permuted "
               f"{_fmt(exif.get('auc_metadata_permuted'))}, drop "
               f"{_fmt(exif.get('auc_drop'))} over "
               f"{exif.get('n_metadata_features','?')} metadata features "
               f"(small drop = not leaning on EXIF).")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the unified v2 detector")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--reports-dir", default="reports")
    args = p.parse_args()
    evaluate_unified(models_dir=args.models_dir, reports_dir=args.reports_dir)


if __name__ == "__main__":
    main()
