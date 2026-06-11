"""
AI Image Detector — Web Interface

FastAPI-based web app for testing the AI image detector.
Provides drag-and-drop image upload with visual results.

Usage:
    python app.py
    # Open http://localhost:8000 in your browser
"""

import os
import sys
import time
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.classifier import AIDetectorClassifier
from src.screenshot_classifier import ScreenshotClassifier
from src.fft_analyzer import extract_fft_features
from src.eigen_analyzer import extract_eigen_features
from src.metadata_extractor import extract_metadata_features
from src.noise_analyzer import extract_noise_features
from src.dct_analyzer import extract_dct_features
from src.ela_analyzer import extract_ela_features
from src.gradient_analyzer import extract_gradient_features
from src.patchcraft_analyzer import extract_patchcraft_features

# --- Config ---
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_PATH            = "models/svm_classifier.pkl"
SCREENSHOT_MODEL_PATH = "models/screenshot_classifier.pkl"

# --- Initialize app and classifiers ---
app = FastAPI(title="AI Image Detector", version="2.0")

classifier = AIDetectorClassifier()
if os.path.exists(MODEL_PATH):
    classifier.load_model(MODEL_PATH)
else:
    print("Warning: No trained model found. Using voting fallback.")

screenshot_clf = ScreenshotClassifier()
if os.path.exists(SCREENSHOT_MODEL_PATH):
    screenshot_clf.load(SCREENSHOT_MODEL_PATH)
else:
    print("Warning: No screenshot classifier found. Run --train-screenshot first.")


@app.post("/api/detect")
async def detect_image(file: UploadFile, mode: str = Form("normal")):
    """Analyze an uploaded image for AI generation.
    
    mode: 'normal' (auto-detect screenshot) or 'screenshot' (force screenshot context).
    """

    # Validate file type
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use JPEG, PNG, or WebP.")

    # Read and validate size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large ({len(contents) // 1024}KB). Max 10MB.")

    # Save to temp file for processing
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"upload{ext}")

    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)

        start = time.time()

        # ── Routing: use dedicated screenshot classifier if mode=screenshot
        # and the screenshot model has been trained.  Otherwise fall back to
        # the main SVM with the old confidence-threshold override.
        if mode == "screenshot" and screenshot_clf.is_trained:
            ss_result = screenshot_clf.predict(tmp_path)
            elapsed   = time.time() - start

            response = {
                "label":      ss_result["label"],
                "confidence": round(ss_result["confidence"] * 100, 1),
                "method":     ss_result["method"],
                "scores": {
                    "fft":            0,
                    "eigenvalue":     0,
                    "metadata":       0,
                    "noise":          0,
                    "dct":            0,
                    "ela":            0,
                    "gradient":       0,
                    "patchcraft":     0,
                    "screenshot_img": round(ss_result["scores"]["screenshot_svm_score"], 3),
                },
                "details": {
                    "screenshot_features": {
                        k: round(v, 4)
                        for k, v in ss_result["features"].items()
                    },
                },
                "analysis_time":         round(elapsed, 2),
                "screenshot_warning":    (
                    "🔍 Dedicated screenshot classifier used. "
                    "Trained specifically on AI-generated vs real screenshots "
                    "using GLCM/wavelet/LBP texture and FFT spectral features."
                ),
                "screenshot_confidence": 1.0,
                "explanation":           ss_result["explanation"],
            }
            return response

        # ── Normal path: main 79-feature SVM ──────────────────────────────
        result = classifier.predict(tmp_path)

        # Get detailed feature dicts for the "expert details" panel
        fft_features      = extract_fft_features(tmp_path)
        eigen_features    = extract_eigen_features(tmp_path)
        meta_features     = extract_metadata_features(tmp_path)
        noise_features    = extract_noise_features(tmp_path)
        dct_features      = extract_dct_features(tmp_path)
        ela_features      = extract_ela_features(tmp_path)
        gradient_features = extract_gradient_features(tmp_path)
        patchcraft_features = extract_patchcraft_features(tmp_path)

        elapsed = time.time() - start

        # Legacy screenshot override: user said screenshot but no dedicated
        # model available — apply confidence-threshold heuristic.
        if mode == "screenshot":
            result["screenshot_warning"] = (
                "⚠ You indicated this is a screenshot. Screenshots of AI images "
                "often fool the detector: your monitor's rendering pipeline adds "
                "camera-like noise, so some AI signals are erased. "
                "Result reliability is reduced — especially for 'Real' verdicts."
            )
            result["screenshot_confidence"] = 1.0

            # Override: "Real" at < 80% confidence → "Uncertain (Screenshot)"
            if result["label"] == "Real" and result["confidence"] < 0.80:
                result["label"] = "Uncertain"
                result["confidence"] = round(1.0 - result["confidence"], 3)
                result["screenshot_warning"] += (
                    " ⚡ Confidence too low to call Real — overridden to Uncertain."
                )


        response = {
            "label": result["label"],
            "confidence": round(result["confidence"] * 100, 1),
            "method": result["method"],
            "scores": {
                "fft":            round(result["scores"]["fft_score"], 3),
                "eigenvalue":     round(result["scores"]["eigenvalue_score"], 3),
                "metadata":       round(result["scores"]["metadata_score"], 3),
                "noise":          round(result["scores"]["noise_score"], 3),
                "dct":            round(result["scores"]["dct_score"], 3),
                "ela":            round(result["scores"]["ela_score"], 3),
                "gradient":       round(result["scores"]["gradient_score"], 3),
                "patchcraft":     round(result["scores"]["patchcraft_score"], 3),
                "screenshot_img": round(result["scores"]["screenshot_img_score"], 3),
            },
            "details": {
                "fft":        {k: round(v, 4) for k, v in fft_features.items()},
                "eigenvalue": {k: round(v, 4) for k, v in eigen_features.items()},
                "metadata":   {k: round(v, 1) for k, v in meta_features.items()},
                "noise":      {k: round(v, 4) for k, v in noise_features.items()},
                "dct":        {k: round(v, 4) for k, v in dct_features.items()},
                "ela":        {k: round(v, 4) for k, v in ela_features.items()},
                "gradient":   {k: round(v, 4) for k, v in gradient_features.items()},
                "patchcraft": {k: round(v, 4) for k, v in patchcraft_features.items()},
            },
            "analysis_time": round(elapsed, 2),
            "screenshot_warning": result.get("screenshot_warning"),
            "screenshot_confidence": result.get("screenshot_confidence"),
        }

        return response

    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main web interface."""
    html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    with open(html_path, "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    print("\n  AI Image Detector — Web Interface")
    print("  Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
