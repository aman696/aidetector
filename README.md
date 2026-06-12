# AI Image Detector

> **A research-grade, rule-based system for detecting AI-generated images** using classical
> signal processing (FFT, eigenvalue decomposition, DCT, noise residuals, gradients, texture)
> feeding a dual-SVM architecture. No deep learning — every decision is explainable.

---

## Quick Start

```bash
pip install -r requirements.txt

# Classify a single image
python main.py --image path/to/image.jpg

# Batch classify
python main.py --batch path/to/folder/

# Web interface (drag-and-drop)
python app.py   # → http://localhost:8000
```

For screenshot classification, add `--screenshot-mode` to CLI or toggle the 📱 button in the web UI.

---

## Architecture

**Two specialised SVM models, one router:**

| Model | Scope | Features | Accuracy* |
|---|---|---|---|
| **Main SVM** | Downloadable images, camera photos, raw screenshots | 79 | 82.3% test |
| **Screenshot SVM** | Desktop/app screenshots only | 15 | 96.6% on AI screenshots |

*\*Measured on the previous training dataset. The project is currently in a **retraining phase**:
the dataset has been expanded with a much wider family of image generators and a larger image
count — updated numbers will land after retraining.*

The 79-feature main model analyses 9 forensic signals (frequency, colour covariance, EXIF,
sensor noise, JPEG fingerprints, error levels, edge statistics, texture consistency, and
screenshot-rendering artefacts) plus a RIGID-inspired drift pass. The 15-feature screenshot
model uses GLCM, wavelet, LBP, and histogram features that survive the display pipeline.

**Known limitation:** Instagram Reel / TikTok / YouTube video-frame screenshots are
misclassified by both models — video-codec textures are a separate problem. See
[RESEARCH.md](RESEARCH.md) for details.

---

## Install

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA 12 (optional — CPU fallback automatic)

### Setup
```bash
git clone https://github.com/aman696/aidetector.git
cd aidetector
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: GPU acceleration
pip install cupy-cuda12x cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

---

## Training

> 🔄 **Retraining phase:** the training dataset was recently expanded (more generator families,
> more images) and the shipped models haven't been retrained on it yet. The commands below
> documented the previous run; the pipeline is being updated for the new `data/` layout.

```bash
# Retrain main SVM (~10 min with GPU + 16 cores)
python main.py --train

# Retrain screenshot SVM (~30 s)
python main.py --train-screenshot

# Evaluate on test set
python main.py --evaluate
```

For full training pipeline and data layout, see [WORKFLOW.md](WORKFLOW.md).

---

## Project Structure

```
aidetector/
├── src/              # 9 forensic analyzers + SVM classifiers
├── data/             # Training images (not in the repo — available from the owner on request)
├── models/           # Trained SVM pickles
├── tests/            # Pytest suite
├── web/              # Frontend HTML
├── main.py           # CLI entry point
├── app.py            # FastAPI web server (port 8000)
├── WORKFLOW.md       # Full architecture, commands, training pipelines
├── RESEARCH.md       # Literature review, failure modes, priority queue
├── AGENTS.md         # Rules for AI contributors
├── CLAUDE.md         # Local AI agent instructions (gitignored)
└── requirements.txt
```

---

## Performance Caveats

- Current models predate the dataset expansion — retraining on the enlarged multi-generator
  dataset is in progress (see RESEARCH.md).
- Social-media-recompressed images lose EXIF and some spectral structure — accuracy is lower.
- Low-resolution images (< 256 px) degrade PatchCraft and FFT reliability.
- AI images with injected EXIF + grain filters may partially fool metadata + noise analyzers.
- See [RESEARCH.md](RESEARCH.md) for full failure-mode table and priority queue.

---

## Research Foundation

Every analyzer is grounded in published image-forensics research. A formal references list
will be added once the ongoing research phase concludes.
Full literature review: [RESEARCH.md](RESEARCH.md).

---

## License

Code and trained models: MIT — see [LICENSE](LICENSE).
Datasets (`data/`): NOT redistributed under MIT — available from the owner on request.
Papers (`papers/`): copyright of their authors/publishers.