# AI Image Detector

> **Live demo: [humanorai.online](https://humanorai.online)** — free, private (images are deleted right after scanning), no sign-up.

Detects whether a still image was **fully generated** by a text-to-image model.
It combines classical image-forensics features (FFT power spectrum, colour-covariance
eigenvalues, DCT/JPEG statistics, noise residuals, gradients, texture, error-level
analysis) with a frozen DINOv2 embedding, feeding a calibrated SVM. It outputs a
calibrated probability and a plain-language explanation.

This is a personal / portfolio project, not a forensic or moderation tool. Treat
the output as an advisory estimate, not proof. See [MODEL_CARD.md](MODEL_CARD.md)
for intended use, limitations, and known weaknesses.

---

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Classify a single image
python main.py --image path/to/image.jpg

# Batch classify a directory
python main.py --batch path/to/folder/

# Web interface (drag-and-drop)
python app.py   # -> http://localhost:8000
```

Without PyTorch installed, the detector automatically falls back to the
classical-only model (lower accuracy, no large download). Screenshots,
social-media-recompressed images, and chained re-uploads are handled by the
single unified model — there is no separate mode to toggle.

---

## Architecture

A single **unified detector**: an 855-dimensional feature vector feeding a
calibrated SVM.

| Component | Dimensions | What it captures |
|---|---|---|
| Classical forensic features | 85 | FFT power spectrum, RGB-covariance eigenvalues + spectral bands, EXIF metadata, PRNU-style noise residuals, DCT/JPEG statistics, ELA, gradient statistics, PatchCraft texture, NPR up-sampling residue, screenshot forensics |
| DINOv2 ViT-B/14 embedding (frozen) | 768 | learned visual representation |
| RIGID perturbation drift | 2 | embedding-space change under a small noise perturbation |

The 855-dim vector is standardised, then classified by
`SVC(kernel="rbf", C=10, gamma=0.001)` wrapped in a group-aware
`CalibratedClassifierCV` (sigmoid). An 85-feature classical-only model
(`SVC(rbf)`) is the fallback when PyTorch is unavailable.

**Known limitation:** Instagram Reel / TikTok / YouTube video-frame screenshots
are misclassified — video-codec textures are a separate problem. Rectified-flow
generators (Flux, SD3) are the weakest case. See [MODEL_CARD.md](MODEL_CARD.md)
and [RESEARCH.md](RESEARCH.md).

---

## Performance

Held-out test split (6,414 rows: 3,216 AI / 3,198 real), disjoint from training
at the base-image level. Full breakdowns by condition, generator architecture,
and resolution are in [RESEARCH.md](RESEARCH.md) and [MODEL_CARD.md](MODEL_CARD.md).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Pd@5%FAR |
|---|---|---|---|---|---|---|---|
| Unified (855) | 0.864 | 0.864 | 0.866 | 0.865 | 0.940 | 0.940 | 0.715 |
| Classical-only (85) | 0.781 | 0.745 | 0.857 | 0.797 | 0.863 | 0.856 | 0.436 |

These numbers are reproducible from the record in
[experiment_v1.json](experiment_v1.json) (dataset hash, seeds, hyperparameters,
feature version). The full dataset is public on Hugging Face
([aman213/aidetector-data](https://huggingface.co/datasets/aman213/aidetector-data)),
so the results can be reproduced independently; see [DATASET.md](DATASET.md) for
the data card and access. **Pd@5%FAR** — detection rate at the
threshold where only 5% of real images are false-flagged — is the most
decision-relevant number for a public tool, where real-image false positives are
the binding constraint.

Note: these are point estimates over 6,414 test rows that descend from ~802
held-out base images (so rows are correlated; effective sample ≈ 802), and the
model does not meet three of its own absolute acceptance gates (clean AUC/accuracy
and held-out-generator Pd@5%FAR). The full gate status and the sample-size caveat
are in [RESEARCH.md](RESEARCH.md) and [MODEL_CARD.md](MODEL_CARD.md).

The older `svm_classifier.pkl` / `screenshot_classifier.pkl` models in `models/`
were trained on a different, now-removed dataset and are kept only as a reference
baseline. They are not used by `main.py` or `app.py`.

---

## Install

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12 (optional — CPU fallback is automatic)

### Setup
```bash
git clone https://github.com/aman696/aidetector.git
cd aidetector
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: GPU acceleration for training (cuML)
pip install cupy-cuda12x cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

---

## Training and Evaluation

Training the unified model requires the dataset and derived variants under
`data/` (not in the repo — public on Hugging Face:
[aman213/aidetector-data](https://huggingface.co/datasets/aman213/aidetector-data)).

```bash
# Train the unified detector (GPU optional; CPU fallback automatic)
python main.py --train --gpu

# Evaluate on the test + held-out-generator splits
python main.py --evaluate
```

Full pipeline, data layout, and the stage-by-stage build are in
[WORKFLOW.md](WORKFLOW.md).

---

## Project Structure

```
aidetector/
|-- src/              # forensic analyzers + feature pipeline + unified detector
|-- data/             # dataset (not in the repo - public on HF: aman213/aidetector-data)
|-- models/           # trained model bundles
|-- tests/            # pytest suite
|-- web/              # frontend + static files
|-- main.py           # CLI entry point
|-- app.py            # FastAPI web server (port 8000)
|-- WORKFLOW.md       # full architecture, commands, training/eval pipeline
|-- RESEARCH.md       # literature review, failure modes, analysis, gate status
|-- MODEL_CARD.md     # intended use, metrics, limitations, known weaknesses
|-- DATASET.md        # dataset card: sources, licenses, counts, biases, access
|-- experiment_v1.json # reproducibility record (hashes, seeds, hyperparameters)
|-- SECURITY.md       # web service threat model and controls
|-- AGENTS.md         # contributor rules and licensing
|-- requirements.txt
```

---

## Limitations

- **Fully-AI images only** — not a deepfake / face-swap / photo-edit detector.
- **Rectified-flow generators (Flux, SD3)** are the weakest case; held-out Flux
  ranks well above chance but at low confidence.
- **Video-frame screenshots** (Reels, TikTok, YouTube) are misclassified.
- **Low-resolution images (< ~256 px)** degrade the FFT and PatchCraft features.
- **Social-media recompression** strips EXIF and some spectral structure, lowering
  accuracy (still ~0.93 ROC-AUC under the conditions tested, on COCO/OpenFake-style
  photographic reals).
- **Real-image diversity is narrow.** All "real" training/test images are
  photographs (COCO, OpenFake). False-positive behaviour on non-photographic real
  content — digital art, illustration, screenshots of documents/UIs, scientific
  imagery — is **not characterized**, so the 5%-FAR operating point only holds for
  photographic reals.
- **Generalization to unseen generators is a known weakness, not solved** —
  held-out architectures (notably rectified-flow: Flux, SD3) rank above chance but
  at low confidence.
- Full failure-mode detail: [MODEL_CARD.md](MODEL_CARD.md), [RESEARCH.md](RESEARCH.md).

---

## License

Code and trained models: MIT — see [LICENSE](LICENSE).
Datasets (`data/`): not under MIT — published separately on Hugging Face
([aman213/aidetector-data](https://huggingface.co/datasets/aman213/aidetector-data)).
Papers (`papers/`): copyright of their authors/publishers.
