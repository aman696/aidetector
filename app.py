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

from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.classifier import AIDetectorClassifier, FeatureExtractor
from src.fft_analyzer import fft_score, extract_fft_features
from src.eigen_analyzer import eigenvalue_score, extract_eigen_features
from src.metadata_extractor import metadata_score, extract_metadata_features
from src.noise_analyzer import noise_score, extract_noise_features
from src.dct_analyzer import dct_score, extract_dct_features
from src.ela_analyzer import ela_score, extract_ela_features

# --- Config ---
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_PATH = "models/svm_classifier.pkl"

# --- Initialize app and classifier ---
app = FastAPI(title="AI Image Detector", version="1.0")

classifier = AIDetectorClassifier()
if os.path.exists(MODEL_PATH):
    classifier.load_model(MODEL_PATH)
else:
    print("Warning: No trained model found. Using voting fallback.")


@app.post("/api/detect")
async def detect_image(file: UploadFile):
    """Analyze an uploaded image for AI generation."""
    
    # Validate file type
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use: {ALLOWED_EXTENSIONS}")
    
    # Read and validate size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large ({len(contents) // 1024}KB). Max: {MAX_FILE_SIZE // 1024}KB")
    
    # Save to temp file for processing
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"upload{ext}")
    
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)
        
        start = time.time()
        
        # Run full classification
        result = classifier.predict(tmp_path)
        
        # Get detailed features for display
        fft_features = extract_fft_features(tmp_path)
        eigen_features = extract_eigen_features(tmp_path)
        meta_features = extract_metadata_features(tmp_path)
        noise_features = extract_noise_features(tmp_path)
        dct_features = extract_dct_features(tmp_path)
        ela_features = extract_ela_features(tmp_path)
        
        elapsed = time.time() - start
        
        return {
            "label": result["label"],
            "confidence": round(result["confidence"] * 100, 1),
            "method": result["method"],
            "scores": {
                "fft": round(result["scores"]["fft_score"], 3),
                "eigenvalue": round(result["scores"]["eigenvalue_score"], 3),
                "metadata": round(result["scores"]["metadata_score"], 3),
                "noise": round(result["scores"]["noise_score"], 3),
                "dct": round(result["scores"]["dct_score"], 3),
                "ela": round(result["scores"]["ela_score"], 3),
            },
            "details": {
                "fft": {k: round(v, 4) for k, v in fft_features.items()},
                "eigenvalue": {k: round(v, 4) for k, v in eigen_features.items()},
                "metadata": {k: round(v, 1) for k, v in meta_features.items()},
                "noise": {k: round(v, 4) for k, v in noise_features.items()},
                "dct": {k: round(v, 4) for k, v in dct_features.items()},
                "ela": {k: round(v, 4) for k, v in ela_features.items()},
            },
            "analysis_time": round(elapsed, 2),
        }
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
