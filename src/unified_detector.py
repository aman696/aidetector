"""
Stage 7 — unified v2 predictor.

Loads `models/unified_v2.pkl` (855-dim classical + DINOv2) and predicts a
single image, falling back to `models/classical_v2.pkl` (85-dim) when torch
is unavailable or the embedding step fails. The DINOv2 embedder is a lazy
singleton built on first predict (never in __init__), so importing this and
running the classical fallback never touches torch.

`detect_screenshot` is kept ONLY as an informational warning, not a router —
the unified model is trained to handle screenshots directly (the v1
screenshot specialist is retired once gate 2 passes).

Bundle format (from src.train_unified): {version, pipeline, feature_names,
embed_model_name, classical_only, ...}; `pipeline.predict_proba` is the only
entry point used here.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import joblib
import numpy as np

from src.feature_pipeline import extract_unified

UNIFIED_PATH = os.path.join("models", "unified_v2.pkl")
CLASSICAL_PATH = os.path.join("models", "classical_v2.pkl")


class UnifiedDetector:
    """Predicts Real vs AI-Generated with the v2 hybrid model and a classical
    fallback. Construct once and reuse (loading + the embedder are the costly
    steps)."""

    def __init__(self, unified_path: str = UNIFIED_PATH,
                 classical_path: str = CLASSICAL_PATH) -> None:
        self.unified = joblib.load(unified_path) if os.path.exists(unified_path) else None
        self.classical = (joblib.load(classical_path)
                          if os.path.exists(classical_path) else None)
        if self.unified is None and self.classical is None:
            raise FileNotFoundError(
                "No v2 model found. Train one with `python -m src.train_unified` "
                f"(looked for {unified_path} and {classical_path}).")
        self._embedder = None
        self._torch_ok: Optional[bool] = None

    def _torch_available(self) -> bool:
        if self._torch_ok is None:
            try:
                import torch  # noqa: F401
                self._torch_ok = True
            except Exception:
                self._torch_ok = False
        return self._torch_ok

    def _get_embedder(self):
        if self._embedder is None:
            from src.embedding_extractor import DinoEmbedder
            self._embedder = DinoEmbedder()
        return self._embedder

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Returns {label, confidence, probability_ai, method, fallback,
        explanation, screenshot_warning?}. Uses the unified model when torch
        is present and the embedding succeeds; otherwise the classical model.
        """
        bundle, vec, fallback = None, None, False
        if self.unified is not None and self._torch_available():
            try:
                vec, _ = extract_unified(image_path, embedder=self._get_embedder())
                bundle = self.unified
            except Exception:
                bundle = None  # fall through to classical
        if bundle is None:
            if self.classical is None:
                raise RuntimeError(
                    "Unified model requires torch and none of the fallback "
                    "(classical_v2.pkl) is available.")
            vec, _ = extract_unified(image_path, skip_embeddings=True)
            bundle = self.classical
            fallback = True

        proba = bundle["pipeline"].predict_proba(vec.reshape(1, -1))[0]
        pred = int(np.argmax(proba))
        label = "AI-Generated" if pred == 1 else "Real"
        method = "classical_v2" if fallback else "unified_v2"

        result: Dict[str, Any] = {
            "label": label,
            "confidence": float(np.max(proba)),
            "probability_ai": float(proba[1]),
            "method": method,
            "fallback": fallback,
            "explanation": (
                f"{label} (p(AI)={proba[1]:.1%}) via the "
                f"{'classical-only fallback' if fallback else 'unified hybrid'} model."),
        }

        warning = self._screenshot_warning(image_path)
        if warning:
            result["screenshot_warning"] = warning
        return result

    @staticmethod
    def _screenshot_warning(image_path: str) -> Optional[str]:
        """Informational only — the unified model handles screenshots; this
        just flags the input so a user knows the condition."""
        try:
            from src.screenshot_detector import detect_screenshot
            info = detect_screenshot(image_path)
        except Exception:
            return None
        if info.get("is_screenshot"):
            return ("Input looks screen-rendered (screenshot). The unified model "
                    "is trained on screenshots, but treat borderline scores with "
                    f"care. Screenshot confidence: {info.get('confidence', 0):.0%}.")
        return None
