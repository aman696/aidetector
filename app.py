"""
AI Image Detector — Web Interface (v2 unified model).

FastAPI drag-and-drop UI. Uses the unified v2 detector (classical + DINOv2)
with automatic classical fallback when torch is absent. The old screenshot
toggle is retired — the unified model handles screenshots directly; a
screenshot is now only flagged as an informational note.

Usage:
    python app.py     # http://localhost:8000
"""

import os
import sys
import time
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
STATIC_DIR = os.path.join(WEB_DIR, "static")

from src.classifier import FeatureExtractor
from src.fft_analyzer import extract_fft_features
from src.eigen_analyzer import extract_eigen_features
from src.metadata_extractor import extract_metadata_features
from src.noise_analyzer import extract_noise_features
from src.dct_analyzer import extract_dct_features
from src.ela_analyzer import extract_ela_features
from src.gradient_analyzer import extract_gradient_features
from src.patchcraft_analyzer import extract_patchcraft_features

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(title="AI Image Detector", version="2.0")

# Load the unified detector once at startup; None if no v2 model is trained yet.
detector = None
try:
    from src.unified_detector import UnifiedDetector
    detector = UnifiedDetector()
    print("Loaded unified v2 detector.")
except Exception as exc:
    print(f"Warning: unified v2 model not loaded ({exc}). "
          f"Train it with: python main.py --train --gpu")

_scorer = FeatureExtractor()


@app.post("/api/detect")
async def detect_image(file: UploadFile, mode: str = Form("normal")):
    """Classify an uploaded image. `mode` is accepted for backward
    compatibility but ignored — the unified model handles screenshots."""
    if detector is None:
        raise HTTPException(503, "No trained model available. Run: python main.py --train --gpu")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use JPEG, PNG, or WebP.")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large ({len(contents) // 1024}KB). Max 10MB.")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"upload{ext}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)

        start = time.time()
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
        elapsed = time.time() - start

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
            "analysis_time": round(elapsed, 2),
            "explanation": result["explanation"],
            "screenshot_warning": result.get("screenshot_warning"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# Static assets (OG image, etc.) under /static.
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(WEB_DIR, "index.html"), "r") as f:
        return f.read()


# Discoverability files must live at the site root, not under /static.
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
    print("\n  AI Image Detector — Web Interface")
    print("  Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
