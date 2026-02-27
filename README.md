# AI Image Detector

> **A research-grade, rule-based system for detecting AI-generated images using frequency analysis, eigenvalue decomposition, noise forensics, and metadata inspection.**

---

## ✨ Features

**8 independent detection methods** — all cross-validated with an SVM trained on 500 augmented images:

| Check | What it looks for | Paper |
|---|---|---|
| 🌊 Frequency Spectrum | AI images have unnatural high-frequency drop-off | Durall 2020 |
| 🎨 Color Statistics | Eigenvalue ratios of RGB covariance matrix | Corvi 2023 |
| 🏷️ Camera Metadata | Missing or suspicious EXIF (camera, GPS, timestamp) | — |
| 🌫️ Sensor Noise | Absence of natural camera grain (PRNU) | Gragnaniello 2021 |
| 🧩 JPEG Fingerprint | DCT coefficient block distribution | Frank 2020 |
| 🕵️ Manipulation Check | Error Level Analysis — re-compression residuals | — |
| 📐 Edge Distribution | Heavy-tailed edge histograms break down in AI | Gragnaniello 2023 |
| 🔬 Texture Contrast | Rich/poor patch energy gap is higher in AI | PatchCraft 2023 |

**Additional capabilities:**
- **RIGID drift features** — 15 extra features measuring feature stability under noise perturbation (54→69 total)
- **Screenshot pre-detection** — flags screenshots before classification with an orange warning banner
- **Training data augmentation** — JPEG Q=70/Q=80 + 0.75× resize augmentation (ITW-SM 2025)
- **GPU-accelerated SVM** — cuML RTX 4060, parallel feature extraction across 16 CPU cores
- **Web interface** — drag-and-drop upload, plain-English results, mode toggle for screenshots

---

## 🛠 Tech Stack

| Category | Library |
|---|---|
| Image processing | OpenCV, Pillow |
| Numerical analysis | NumPy, SciPy |
| Machine learning | scikit-learn (SVM), cuML (GPU SVM) |
| Web server | FastAPI, Uvicorn |
| Testing | pytest |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA 12 (optional, for GPU training)

### Installation

```bash
git clone https://github.com/aman696/aidetector.git
cd aidetector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: GPU acceleration
pip install cupy-cuda12x cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

---

## 💻 Usage

### Web Interface (recommended)
```bash
python app.py
# Open http://localhost:8000
```

### CLI
```bash
# Classify a single image
python main.py --image path/to/image.jpg

# Train (GPU + 16-core parallel)
python main.py --train

# Evaluate on test set
python main.py --evaluate

# Batch classify a folder
python main.py --batch path/to/folder/
```

### Run Tests
```bash
python -m pytest tests/ -v --tb=short
# 83 tests, ~2 minutes
```

---

## 📂 Project Structure

```
aidetector/
├── src/
│   ├── fft_analyzer.py           # Frequency spectrum (Durall 2020)
│   ├── eigen_analyzer.py         # Color eigenvalue analysis (Corvi 2023)
│   ├── metadata_extractor.py     # EXIF forensics
│   ├── noise_analyzer.py         # Sensor noise + multi-scale (Gragnaniello 2021)
│   ├── dct_analyzer.py           # DCT block stats (Frank 2020)
│   ├── ela_analyzer.py           # Error Level Analysis
│   ├── gradient_analyzer.py      # Edge distribution (Gragnaniello 2023)
│   ├── patchcraft_analyzer.py    # Texture contrast (PatchCraft 2023)
│   ├── screenshot_detector.py    # Screenshot pre-detection heuristics
│   ├── classifier.py             # 69-feature extractor + SVM
│   └── utils.py                  # Dataset loader + JPEG augmentation
├── data/
│   ├── real/                     # 50 real photos
│   ├── ai_generated/             # 51 AI-generated images
│   ├── screenshots/              # 14 real screenshots
│   ├── ai_generated_screenshots/ # 10 AI screenshots
│   └── test/{real,ai_generated}/
├── models/
│   └── svm_classifier.pkl        # Trained model (84KB, 69 features)
├── tests/                        # 83 pytest tests
├── papers/                       # Durall 2020, Corvi 2023 PDFs
├── web/index.html                # Web UI
├── app.py                        # FastAPI server
├── main.py                       # CLI entry point
├── WORKFLOW.md                   # Architecture, training pipeline, failure modes
├── RESEARCH.md                   # Literature review, paper status
└── requirements.txt
```

---

## 📊 Performance

| Metric | Value |
|---|---|
| Training images | 500 (125 base + 375 augmented) |
| Feature count | 69 |
| Cross-validation accuracy | **84.8%** |
| Training accuracy | 98.4% |
| Test count | 83 pytest tests |
| Training backend | cuML GPU (RTX 4060) + 16-core CPU extraction |

**Known limitations:**
- Screenshots without the screenshot mode toggle may be misclassified
- Video frames from novel generators (e.g. Seedance/ByteDance) may evade detection
- Low-resolution images (<256px) reduce PatchCraft and FFT reliability
- AI images with injected EXIF data may fool metadata analysis

---

## 📚 Research & References

| Paper | arXiv / Venue | Implemented as |
|---|---|---|
| Durall et al. — *Unmasking DeepFakes with simple Features* (2020) | arXiv:1911.00686 | `fft_analyzer.py` |
| Corvi et al. — *Intriguing Properties of Synthetic Images* (2023) | arXiv:2304.06408 | `eigen_analyzer.py` |
| Frank et al. — *Leveraging Frequency Analysis for Deep Fake Image Recognition* (2020) | ICML 2020 | `dct_analyzer.py` |
| Gragnaniello et al. — *Are GAN generated images easy to detect?* (2021) | ICME 2021 | `noise_analyzer.py` |
| Gragnaniello et al. — *Raising the Bar of AI-generated Image Detection* (2023) | CVPR 2023 | `gradient_analyzer.py` |
| Zhong et al. — *PatchCraft: Exploring Texture Patch for Efficient AI-Generated Image Detection* (2023) | arXiv:2311.12397 | `patchcraft_analyzer.py` |
| RIGID — *Training-Free Detection via Feature Drift* (2024) | arXiv 2024 | `_compute_drift_features()` in `classifier.py` |
| Konstantinidou et al. — *In-the-Wild Social Media Images* (2025) | arXiv:2507.10236 | `augment_dataset_with_jpeg()` in `utils.py` |

---

## ⚠️ Limitations & Scope

This is a **research portfolio project** — not a production-ready service yet once fully built it will be a production-ready service.

- Rule-based signal processing, not deep learning
- Explainable results (can show *why* an image was flagged)
- Trained on ~125 images — limited generalization to novel generators
- See `TODO.md` for known failure modes and planned improvements
