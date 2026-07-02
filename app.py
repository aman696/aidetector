"""
Human or AI? — Web Interface (v2 unified model), production-hardened.

FastAPI drag-and-drop UI. Uses the unified v2 detector (classical + DINOv2) with
automatic classical fallback when torch is absent. Uploaded images are written to
a per-request temp file and deleted immediately after the scan — never stored,
logged, or used for training.

Hardening (see SECURITY.md for the rationale of each control):
  - per-IP rate limiting (slowapi) + a global concurrency cap on the expensive
    detection so the box can't be trivially DoS'd;
  - request-body size limit enforced BEFORE the body is read into memory;
  - real-image + dimension validation (blocks decompression / pixel bombs);
  - blocking CPU work runs in a threadpool, never on the async event loop;
  - security headers (CSP, nosniff, frame-deny, referrer, permissions);
  - trusted-host validation; generic client errors (details logged server-side);
  - API docs disabled unless DEBUG=1.

Run locally:  python app.py            (http://localhost:8000)
Serve (prod): uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
"""

import hashlib
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from PIL import Image, UnidentifiedImageError
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
STATIC_DIR = os.path.join(WEB_DIR, "static")

# --------------------------------------------------------------------------- #
# Config (overridable via environment for deployment)
# --------------------------------------------------------------------------- #
DEBUG = os.getenv("DEBUG", "0") == "1"
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10 MB
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(50_000_000)))             # 50 MP
MAX_DIMENSION = int(os.getenv("MAX_DIMENSION", "12000"))              # per side
# TEMP (LOAD TEST): defaults relaxed so a load test measures the HARDWARE, not
# these guards. Revert to MAX_CONCURRENT=2 / MAX_QUEUE=10 / RATE_LIMIT=20/minute
# before real/public use. (Env vars still override these if you ever set them.)
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "6"))               # heavy scans at once
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "100"))                       # extra requests allowed to wait
SCAN_TIMEOUT = float(os.getenv("SCAN_TIMEOUT", "60"))               # seconds per scan before giving up
RATE_LIMIT = os.getenv("RATE_LIMIT", "100000/minute")                # per IP, /api/detect
ALLOWED_HOSTS = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "*").split(",") if h.strip()]

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# --- Contribution / self-labeling (testing branch only) ---------------------- #
# Opt-in: users can donate an image plus THEIR OWN label ("real" / "ai" /
# "unsure") to grow the training set. This is deliberately separate from the
# /api/detect path, which stays privacy-preserving (deleted after scan). Storing
# only happens on an explicit second action with a clear on-screen notice.
CONTRIB_ENABLED = os.getenv("CONTRIB_ENABLED", "1") == "1"
CONTRIB_DIR = os.getenv("CONTRIB_DIR", "contributions")
CONTRIB_RATE_LIMIT = os.getenv("CONTRIB_RATE_LIMIT", "30/minute")
CONTRIB_LABELS = {"real", "ai", "unsure"}

# --- Metrics (see MONITORING.md) --------------------------------------------- #
# /metrics is meant for PRIVATE scraping only (Tailscale/localhost) — never give
# it a public tunnel hostname. METRICS_TOKEN is an optional second layer: if set,
# the route 404s (not 401, so it doesn't confirm the route exists) without it.
METRICS_TOKEN = os.getenv("METRICS_TOKEN", "")

# Refuse to decode absurdly large images (Pillow raises above this).
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("humanorai")

# --------------------------------------------------------------------------- #
# App + middleware
# --------------------------------------------------------------------------- #
def _client_ip(request: Request) -> str:
    """Real client IP for rate limiting. Behind our reverse proxy (Caddy) the
    client IP is the first hop of X-Forwarded-For; otherwise the socket peer.
    Only trust XFF because the app is meant to sit behind a trusted proxy."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)


# Per-route limits only (no constructor default_limits: slowapi evaluates those
# with a zero-arg key_func, which breaks a request-aware key function).
limiter = Limiter(key_func=_client_ip)

app = FastAPI(
    title="Human or AI?",
    version="2.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
    openapi_url="/openapi.json" if DEBUG else None,
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limited(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429,
                        content={"detail": "Too many requests — please slow down and try again shortly."})


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), interest-cohort=()",
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; "
        "img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; connect-src 'self'; form-action 'self'"
    ),
}


class SecurityHeadersMiddleware:
    """Pure-ASGI header injection. (BaseHTTPMiddleware is avoided: it wraps the
    request stream and breaks multipart uploads and exception responses.)"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).extend(
                    (k.encode(), v.encode()) for k, v in SECURITY_HEADERS.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)


app.add_middleware(GZipMiddleware, minimum_size=600)
app.add_middleware(SecurityHeadersMiddleware)
if ALLOWED_HOSTS and ALLOWED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

# Bound the number of simultaneous heavy detections (each is CPU-intensive).
# _inflight = requests currently running OR waiting for a slot; used to cap the
# queue so a flood is turned away (503) instead of piling up unbounded.
import asyncio
_detect_sem = asyncio.Semaphore(MAX_CONCURRENT)
_inflight = 0

# --------------------------------------------------------------------------- #
# Prometheus metrics (see MONITORING.md — scraped by the monitoring stack,
# never exposed on the public tunnel). Mirrors the concurrency/overload state
# already tracked above so "how many users right now" is a real metric, not
# inferred from host CPU.
# --------------------------------------------------------------------------- #
METRIC_DETECT_REQUESTS = Counter(
    "detect_requests_total", "Total /api/detect requests by outcome.",
    ["outcome"])  # ok | client_error | busy | timeout | server_error
METRIC_DETECT_DURATION = Histogram(
    "detect_duration_seconds", "Time spent running one detection (scan only).",
    buckets=(0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60))
METRIC_INFLIGHT = Gauge(
    "detect_inflight", "Requests currently running or queued for a scan slot.")
METRIC_INFLIGHT_CAPACITY = Gauge(
    "detect_inflight_capacity",
    "Max requests allowed in flight before 503 (MAX_CONCURRENT + MAX_QUEUE).")
METRIC_INFLIGHT_CAPACITY.set(MAX_CONCURRENT + MAX_QUEUE)
METRIC_FALLBACK = Counter(
    "detect_fallback_total",
    "Scans served by the classical-only fallback (torch unavailable).")
METRIC_CONTRIBUTIONS = Counter(
    "contributions_total", "Opt-in image contributions by label.", ["label"])

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
detector = None
try:
    from src.unified_detector import UnifiedDetector
    detector = UnifiedDetector()
    logger.info("Loaded unified v2 detector.")
except Exception as exc:  # pragma: no cover - depends on trained model presence
    logger.warning("Unified v2 model not loaded (%s).", exc)

from src.classifier import FeatureExtractor
from src.fft_analyzer import extract_fft_features
from src.eigen_analyzer import extract_eigen_features
from src.metadata_extractor import extract_metadata_features
from src.noise_analyzer import extract_noise_features
from src.dct_analyzer import extract_dct_features
from src.ela_analyzer import extract_ela_features
from src.gradient_analyzer import extract_gradient_features
from src.patchcraft_analyzer import extract_patchcraft_features

_scorer = FeatureExtractor()


def _validate_image_bytes(contents: bytes) -> None:
    """Confirm the bytes are a real, sane image (magic + format + dimensions),
    rejecting non-images and decompression/pixel bombs before any heavy work."""
    try:
        with Image.open(io.BytesIO(contents)) as im:
            fmt = (im.format or "").upper()
            w, h = im.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        raise HTTPException(400, "That doesn't look like a valid image. Use a JPEG, PNG, or WebP.")
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(400, "Unsupported image format. Use a JPEG, PNG, or WebP.")
    if w * h > MAX_PIXELS or max(w, h) > MAX_DIMENSION:
        raise HTTPException(400, "Image dimensions are too large.")


def _run_detection(tmp_path: str) -> dict:
    """Synchronous, CPU-bound detection. Runs in a threadpool (never on the event
    loop) under a concurrency semaphore."""
    result = detector.predict(tmp_path)
    scores = _scorer.extract_individual_scores(tmp_path)
    details = {
        "fft": {k: round(v, 4) for k, v in extract_fft_features(tmp_path).items()},
        "eigenvalue": {k: round(v, 4) for k, v in extract_eigen_features(tmp_path).items()},
        "metadata": {k: round(v, 1) for k, v in extract_metadata_features(tmp_path).items()},
        "noise": {k: round(v, 4) for k, v in extract_noise_features(tmp_path).items()},
        "dct": {k: round(v, 4) for k, v in extract_dct_features(tmp_path).items()},
        "ela": {k: round(v, 4) for k, v in extract_ela_features(tmp_path).items()},
        "gradient": {k: round(v, 4) for k, v in extract_gradient_features(tmp_path).items()},
        "patchcraft": {k: round(v, 4) for k, v in extract_patchcraft_features(tmp_path).items()},
    }
    return {
        "label": result["label"],
        "confidence": round(result["confidence"] * 100, 1),
        "probability_ai": round(result["probability_ai"] * 100, 1),
        "method": result["method"],
        "fallback": result["fallback"],
        "scores": {
            "fft": round(scores["fft_score"], 3),
            "eigenvalue": round(scores["eigenvalue_score"], 3),
            "metadata": round(scores["metadata_score"], 3),
            "noise": round(scores["noise_score"], 3),
            "dct": round(scores["dct_score"], 3),
            "ela": round(scores["ela_score"], 3),
            "gradient": round(scores["gradient_score"], 3),
            "patchcraft": round(scores["patchcraft_score"], 3),
            "npr": round(scores["npr_score"], 3),
            "screenshot_img": round(scores["screenshot_img_score"], 3),
        },
        "details": details,
        "explanation": result["explanation"],
        "screenshot_warning": result.get("screenshot_warning"),
    }


@app.post("/api/detect")
@limiter.limit(RATE_LIMIT)
async def detect_image(request: Request, file: UploadFile, mode: str = Form("normal")):
    """Classify an uploaded image. `mode` is accepted for backward compatibility
    but ignored. Returns the verdict, confidence, per-analyzer signals, and a
    plain-language explanation."""
    if detector is None:
        raise HTTPException(503, "The detector isn't available right now. Please try again shortly.")

    # Reject oversized uploads via Content-Length before reading the body into
    # memory (the reverse proxy also caps this — defense in depth).
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_FILE_SIZE + 4096:
        raise HTTPException(413, "File too large. Maximum size is 10 MB.")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type. Use JPEG, PNG, or WebP.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large. Maximum size is 10 MB.")
    _validate_image_bytes(contents)

    # Overload guard: at most MAX_CONCURRENT running + MAX_QUEUE waiting. Beyond
    # that, turn requests away immediately instead of letting them pile up.
    # This is the precise "too many users at once" signal for monitoring —
    # tracked as a metric rather than inferred from host CPU.
    global _inflight
    if _inflight >= MAX_CONCURRENT + MAX_QUEUE:
        METRIC_DETECT_REQUESTS.labels(outcome="busy").inc()
        raise HTTPException(503, "The server is busy right now — please try again in a moment.")
    _inflight += 1
    METRIC_INFLIGHT.set(_inflight)

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"upload{ext}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)
        start = time.time()
        async with _detect_sem:
            # Per-scan timeout so a stuck image can't hold a slot forever. (The
            # worker thread can't be force-killed, but the slot is freed and the
            # client gets a clean response.)
            payload = await asyncio.wait_for(
                run_in_threadpool(_run_detection, tmp_path), timeout=SCAN_TIMEOUT)
        duration = time.time() - start
        payload["analysis_time"] = round(duration, 2)
        METRIC_DETECT_DURATION.observe(duration)
        METRIC_DETECT_REQUESTS.labels(outcome="ok").inc()
        if payload.get("fallback"):
            METRIC_FALLBACK.inc()
        return payload
    except asyncio.TimeoutError:
        METRIC_DETECT_REQUESTS.labels(outcome="timeout").inc()
        logger.warning("Scan exceeded %ss timeout", SCAN_TIMEOUT)
        raise HTTPException(503, "That image took too long to scan — please try again.")
    except HTTPException:
        METRIC_DETECT_REQUESTS.labels(outcome="server_error").inc()
        raise
    except Exception:
        METRIC_DETECT_REQUESTS.labels(outcome="server_error").inc()
        logger.exception("Detection failed")          # full detail server-side only
        raise HTTPException(500, "Analysis failed. Please try a different image.")
    finally:
        _inflight -= 1
        METRIC_INFLIGHT.set(_inflight)
        shutil.rmtree(tmp_dir, ignore_errors=True)     # PRIVACY: upload deleted now


# --------------------------------------------------------------------------- #
# Contribution / self-labeling endpoint (opt-in dataset growth)
# --------------------------------------------------------------------------- #
def _store_contribution(contents: bytes, label: str, ext: str,
                        model_p: "float | None") -> dict:
    """Persist a donated image under contributions/<label>/<sha256><ext> and
    append a JSONL metadata record. Content-addressed by sha256, so identical
    resubmissions collapse to one file (dedup) while every event is still
    logged. Returns the record."""
    label_dir = os.path.join(CONTRIB_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    digest = hashlib.sha256(contents).hexdigest()
    fpath = os.path.join(label_dir, f"{digest}{ext}")
    is_new = not os.path.exists(fpath)
    if is_new:
        with open(fpath, "wb") as f:
            f.write(contents)
    rec = {
        "id": digest,
        "label": label,
        "ext": ext,
        "bytes": len(contents),
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_probability_ai": model_p,   # user label vs model score -> hard cases
        "new_file": is_new,
    }
    with open(os.path.join(CONTRIB_DIR, "contributions.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


@app.post("/api/contribute")
@limiter.limit(CONTRIB_RATE_LIMIT)
async def contribute_image(request: Request, file: UploadFile,
                           label: str = Form(...),
                           model_probability_ai: str = Form("")):
    """Opt-in: store a donated image plus the user's own real/ai/unsure label.
    Reuses the same validation as /api/detect. Unlike /api/detect, this DOES
    persist the image (by design), so it is only ever called from an explicit
    'contribute' button with an on-screen storage notice."""
    if not CONTRIB_ENABLED:
        raise HTTPException(404, "Contributions are not enabled on this server.")

    label = (label or "").strip().lower()
    if label not in CONTRIB_LABELS:
        raise HTTPException(400, "Label must be one of: real, ai, unsure.")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type. Use JPEG, PNG, or WebP.")

    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_FILE_SIZE + 4096:
        raise HTTPException(413, "File too large. Maximum size is 10 MB.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large. Maximum size is 10 MB.")
    _validate_image_bytes(contents)   # magic/format/dimension + bomb guard

    model_p = None
    try:
        model_p = float(model_probability_ai) if model_probability_ai != "" else None
    except ValueError:
        model_p = None

    try:
        rec = await run_in_threadpool(_store_contribution, contents, label, ext, model_p)
    except Exception:
        logger.exception("Contribution store failed")
        raise HTTPException(500, "Could not save your contribution. Please try again.")

    METRIC_CONTRIBUTIONS.labels(label=label).inc()
    return {"status": "ok", "id": rec["id"], "stored_new": rec["new_file"], "label": label}


# --------------------------------------------------------------------------- #
# Metrics scrape endpoint (Prometheus text format)
# --------------------------------------------------------------------------- #
@app.get("/metrics")
async def metrics(request: Request):
    """Prometheus scrape endpoint — request/latency/inflight/contribution
    counters defined above. PRIVATE ONLY: the monitoring stack scrapes this over
    Tailscale/localhost; it must never be given a public tunnel hostname (see
    MONITORING.md). If METRICS_TOKEN is set, requires it as `Authorization:
    Bearer <token>` or `?token=`."""
    if METRICS_TOKEN:
        auth = request.headers.get("authorization", "")
        supplied = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        supplied = supplied or request.query_params.get("token", "")
        if supplied != METRICS_TOKEN:
            raise HTTPException(404)  # 404, not 401 -- don't confirm the route exists
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --------------------------------------------------------------------------- #
# Static + content routes
# --------------------------------------------------------------------------- #
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/healthz")
async def healthz():
    """Liveness/readiness probe for the host platform."""
    return {"status": "ok", "model_loaded": detector is not None}


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(WEB_DIR, "index.html"), "r") as f:
        return f.read()


@app.get("/robots.txt")
async def robots():
    return FileResponse(os.path.join(STATIC_DIR, "robots.txt"), media_type="text/plain")


@app.get("/sitemap.xml")
async def sitemap():
    return FileResponse(os.path.join(STATIC_DIR, "sitemap.xml"), media_type="application/xml")


@app.get("/llms.txt")
async def llms():
    return FileResponse(os.path.join(STATIC_DIR, "llms.txt"), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    print("\n  Human or AI? — Web Interface")
    print("  Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
