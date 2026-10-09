"""Bounded, single-process inference for a free hosted demonstration.

Only the repository's classical v2 bundle is loaded. Image bytes live in a
temporary directory owned by the inference thread, including after a client
disconnect or timeout. No image, filename, EXIF value, or result is logged.
"""

from __future__ import annotations

import asyncio
from collections import deque
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import warnings

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from starlette.responses import JSONResponse

from src.feature_pipeline import CLASSICAL_NAMES, extract_unified

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1024 * 1024
MAX_PIXELS = 1024 * 1024
MAX_DIMENSION = 4096
MAX_BODY = MAX_BYTES + 8192
TIMEOUT = 120
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "")
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
logger = logging.getLogger(__name__)

record = json.loads((ROOT / "experiment_v1.json").read_text())["model"]["classical_only"]
model_path = ROOT / record["file"]
if hashlib.sha256(model_path.read_bytes()).hexdigest() != record["sha256"]:
    raise RuntimeError("Classical model hash differs from the experiment record.")
bundle = joblib.load(model_path)
if list(bundle["feature_names"]) != list(CLASSICAL_NAMES) or len(CLASSICAL_NAMES) != 85:
    raise RuntimeError("Classical feature schema does not match the trained model.")


class BodyLimit:
    """Reject an oversized request before the multipart parser can spool it."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_BODY:
                return await JSONResponse({"detail": "too_large"}, status_code=413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, bounded_receive, send)


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(BodyLimit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN else [],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)
active: asyncio.Task | None = None
recent: deque[float] = deque()


@app.middleware("http")
async def headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def validate_image(contents: bytes, filename: str) -> str:
    """Validate format, dimensions, and complete decoding without resizing."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(400, "invalid_type")
    if len(contents) > MAX_BYTES:
        raise HTTPException(413, "too_large")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise HTTPException(400, "invalid_type")
                if getattr(image, "n_frames", 1) != 1:
                    raise HTTPException(400, "animated_image")
                if image.width * image.height > MAX_PIXELS or max(image.size) > MAX_DIMENSION:
                    raise HTTPException(413, "dimensions")
                image.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(413, "dimensions") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "invalid_image") from None
    return suffix


def analyze(contents: bytes, suffix: str) -> dict:
    """Use the unmodified repository feature pipeline and trained calibration."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="humanorai-demo-") as directory:
        image_path = Path(directory) / ("image" + suffix)
        image_path.write_bytes(contents)
        vector, names = extract_unified(str(image_path), skip_embeddings=True)
        if not np.isfinite(vector).all():
            raise RuntimeError("Non-finite features.")
        probabilities = bundle["pipeline"].predict_proba(vector.reshape(1, -1))[0]
        prediction = int(np.argmax(probabilities))
        return {
            "label": "AI-Generated" if prediction == 1 else "Real",
            "probability_ai": float(probabilities[1]),
            "confidence": float(probabilities[prediction]),
            "method": "classical_v2",
            "fallback": True,
            "features": [{"name": name, "value": float(value)} for name, value in zip(names, vector)],
            "analysis_time": round(time.monotonic() - started, 2),
        }


async def run_scan(contents: bytes, suffix: str) -> dict:
    return await asyncio.to_thread(analyze, contents, suffix)


def observe_completion(task: asyncio.Task) -> None:
    """Retrieve late failures without printing upload details or results."""
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Hosted demo inference failed.")


@app.get("/healthz")
async def health():
    return {"status": "ok", "model_loaded": True, "method": "classical_v2", "features": 85,
            "max_bytes": MAX_BYTES, "max_pixels": MAX_PIXELS, "max_dimension": MAX_DIMENSION}


@app.post("/api/detect")
async def detect(request: Request, file: UploadFile):
    """Limit admission and preserve the running job's slot after a timeout."""
    global active
    origin = request.headers.get("origin")
    if origin and origin != ALLOWED_ORIGIN:
        raise HTTPException(403, "origin")
    if active is not None and not active.done():
        raise HTTPException(503, "busy")
    now = time.monotonic()
    while recent and recent[0] <= now - 60:
        recent.popleft()
    if len(recent) >= 3:
        raise HTTPException(429, "rate_limit", headers={"Retry-After": "60"})
    recent.append(now)
    try:
        contents = await file.read(MAX_BYTES + 1)
        suffix = validate_image(contents, file.filename or "")
    finally:
        await file.close()
    # No await between this final admission check and claiming the slot.
    if active is not None and not active.done():
        raise HTTPException(503, "busy")
    active = asyncio.create_task(run_scan(contents, suffix))
    active.add_done_callback(observe_completion)
    try:
        return await asyncio.wait_for(asyncio.shield(active), TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(503, "timeout") from None
    except asyncio.CancelledError:
        raise
    except Exception:
        raise HTTPException(500, "analysis_failed") from None
