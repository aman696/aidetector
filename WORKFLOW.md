# AI Image Detector — Workflow Reference

This document describes the complete current workflow: the unified v2 detector,
its data pipeline, training, evaluation, and prediction. It is the single
reference for how to run and rebuild the system.

- **Unified model:** `models/unified_v2.pkl` — 855-dim (85 classical + 768
  DINOv2 + 2 RIGID drift), `SVC(rbf, C=10, gamma=0.001)` with group-aware
  sigmoid calibration.
- **Classical fallback:** `models/classical_v2.pkl` — 85-dim, `SVC(rbf)`, used
  automatically when PyTorch is unavailable.
- Headline metrics are in [README.md](README.md); full breakdowns and the
  acceptance-gate status in [RESEARCH.md](RESEARCH.md) and
  [MODEL_CARD.md](MODEL_CARD.md); the data itself in [DATASET.md](DATASET.md).
  They are not duplicated here. Contributor rules: [CONTRIBUTING.md](CONTRIBUTING.md).

The older `models/svm_classifier.pkl` and `models/screenshot_classifier.pkl`
were trained on a different, now-removed dataset and are kept only as a
reference baseline. They are not loaded by `main.py` or `app.py`.

---

## Public-checkout scope

This public checkout retains six legacy collection scripts and adds the static
company website under `site/`. Existing ignore rules are preserved; no files were untracked or ignored
as a cleanup shortcut. Models and committed evaluation/family-analysis reports
are retained unchanged.

The dataset, private paper/code notes, and original v2 research helpers are not
included. In particular, the build/capture, architecture/AR analysis, external
fetch/evaluation, and experiment-recording scripts referenced by the original
workflow are absent. The sections below distinguish those historical steps
from commands that this checkout actually supplies. Restore the recorded data
and manifests from DATASET.md before training or evaluation.

`requirements.txt` provides the full development dependencies. Optional tools:

- Legacy browser screenshots: `python -m pip install playwright`, followed by
  `python -m playwright install chromium`.
- Legacy Firefox collection: `python -m pip install selenium webdriver-manager`.
- Load testing: `python -m pip install locust`.

Legacy helpers write to earlier layouts such as `data/ai_generated`, `data/test`,
and `data/socialmedia*_ss`; they do not recreate the v2 benchmark. In particular,
`process_local_data.py` clears its old target directories before ingesting data.
Its video-frame extraction is dataset preprocessing, not a supported video
inference capability of the detector. Do not run it to restore the current data.

## Quick Reference

```bash
# Activate venv
source .venv/bin/activate

# Classify a single image
python main.py --image path/to/image.jpg

# Batch classify a directory
python main.py --batch path/to/folder/

# Train the unified detector (GPU optional; CPU fallback automatic)
python main.py --train --gpu          # or: python -m src.train_unified --gpu --n-jobs -1

# Evaluate on the test + held-out-generator splits
python main.py --evaluate             # or: python -m src.evaluate_unified

# Run the web interface
python app.py                         # -> http://localhost:8000

# Run all tests
python -m pytest tests/ -q
```

The unified model handles clean images, social-media-recompressed images,
screenshots, and chained re-uploads directly. There is no separate screenshot
mode; the v1 `--screenshot-mode` / `--train-screenshot` flags are retired and
print a deprecation note if used.

---

## Architecture

A single feature vector feeds a single calibrated classifier.

```
Image
  |
  |-- Classical features (85)            [CPU, src/classifier.py + analyzers]
  |     FFT power spectrum, eigenvalues + spectral bands, EXIF metadata,
  |     noise residuals, DCT/JPEG stats, ELA, gradients, PatchCraft texture,
  |     NPR up-sampling residue, screenshot forensics (GLCM/LBP/wavelet),
  |     and 15 RIGID classical drift features.
  |
  |-- DINOv2 ViT-B/14 embedding (768)    [GPU/CPU, src/embedding_extractor.py]
  |     frozen vit_base_patch14_dinov2.lvd142m
  |
  |-- RIGID embedding drift (2)          [embedding change under perturbation]
  |
  v   855-dim vector  (src/feature_pipeline.extract_unified)
StandardScaler
  v
SVC(rbf, C=10, gamma=0.001), group-aware sigmoid calibration
  v
"Real" / "AI-Generated" + calibrated p(AI) + explanation
```

When PyTorch is absent or the embedding step fails, `UnifiedDetector` falls back
to the 85-dim classical model (`extract_unified(..., skip_embeddings=True)`),
which loads no backbone. `detect_screenshot` runs only to attach an
informational warning — it does not route to a separate model.

### Classical feature groups (85 total)

| Group | Module | Count |
|---|---|---|
| FFT power spectrum + spectral slope | `fft_analyzer.py` | 4 |
| Eigenvalues + spectral band energy | `eigen_analyzer.py` | 12 |
| EXIF metadata | `metadata_extractor.py` | 6 |
| Noise residual + chroma correlation | `noise_analyzer.py` | 11 |
| DCT block + JPEG boundary | `dct_analyzer.py` | 8 |
| Error Level Analysis | `ela_analyzer.py` | 5 |
| Gradient statistics | `gradient_analyzer.py` | 5 |
| PatchCraft texture | `patchcraft_analyzer.py` | 3 |
| NPR up-sampling residue | `npr_analyzer.py` | 6 |
| Screenshot forensics (GLCM/LBP/wavelet/chroma/tone) | `screenshot_image_analyzer.py` | 10 |
| RIGID classical drift | `classifier.py` | 15 |

Every analyzer wraps its math in try/except and returns neutral values on
failure (0.0 for energies/differences, 1.0 for ratios, 0.5 for scores) to keep the feature vector finite and allow a batch to continue. These defaults
do not guarantee an unbiased prediction when an analyzer fails.

---

## Data Pipeline

The dataset is not in the repo (public on Hugging Face:
[aman213/aidetector-data](https://huggingface.co/datasets/aman213/aidetector-data)). On disk:

```
data/
|-- Gemini/                  # AI images (Google Gemini)
|-- GPT/                     # AI images (ChatGPT)
|-- mixed/                   # generator-family subdirectories (many generators)
|-- real/                    # real photos (coco/ + openfake/)
|-- ar_external/             # independent out-of-distribution check set
|-- derived/                 # platform/screenshot/chained variants (rebuildable)
|   `-- cache/               # per-id classical + embedding feature caches
`-- manifests/
    |-- base_manifest.json   # every base image: id, label, family, source
    |-- splits.json          # base-level train/val/test + held-out families
    `-- derived_manifest.json # every derived variant, keyed to its base_id
```

**Counts (current dataset, see `experiment_v1.json`):** 4,271 base images
(2,271 AI across 34 generator families + 2,000 real), 23,577 derived records.

### Stage 0 — manifest and leak-safe splits (`src/dataset.py`)

Builds `base_manifest.json` and assigns splits **at the base-image level**
(seed 42, 70% train). A base image and all of its derived variants stay in one
split, so near-duplicates never leak across train/test. Five generator families
(`midjourney_7`, `ideogram_3.0`, `imagen_4.0`, `flux_1`, `recraft_v3`) are held
out of training entirely to measure generalization to unseen generators.

### Stage 1 — derived variants (historical build step)

`src/channels.py` is included, but the original `scripts/build_derived.py` and
`scripts/capture_screenshots.py` orchestration scripts are not. Restore the
published derived data rather than treating those absent helpers as runnable
commands. The original pipeline generated in-the-wild distortions from each base image: Facebook / X / Telegram
recompression, screenshot capture (real browser via Playwright), and two chained
pipelines (e.g. screenshot then Telegram). Each variant inherits its base's split
assignment via `splits[base_id]`. All randomness is seeded from the image's own
id (`channels.rng_for`, md5-based) for determinism.

### Stage 2 — feature extraction and caching (`src/feature_pipeline.py`)

Two phases, both cached per id so the expensive first run resumes for free:

- **Phase A (CPU, multiprocess):** 85 classical features via the
  `ProcessPoolExecutor` path, cached at
  `data/derived/cache/classical_v2/<id>.npy`.
- **Phase B (GPU, single process):** 768 DINOv2 dims + 2 RIGID drift dims,
  cached at `data/derived/cache/emb_vitb14/<id>.npy`. Embeddings never run inside
  the process pool (single CUDA context).

The final matrix is assembled by loading both caches in record order, so `X[i]`
always corresponds to `records[i]` (positional join, never a re-sort).
`cache_meta.json` records a classical feature-name hash and the embedding model
name; changing either invalidates the right cache. The classical cache also
carries a formula epoch so a value-only change to a feature invalidates the
classical cache without touching the embeddings.

---

## Training (`python main.py --train` / `python -m src.train_unified`)

From the same assembled training matrix, two models are produced:

```
Assembled train rows (16,593: clean + a leak-safe subset of derived variants)
        |
        v  grid search: SVC kernels {rbf, linear}, C/gamma grid,
        |  scoring = ROC-AUC over decision_function,
        |  cv = StratifiedGroupKFold(groups=base_id)   [no fold straddles a base]
        v
   best estimator -> Pipeline([StandardScaler, SVC])
        |
        v  CalibratedClassifierCV(method="sigmoid", ensemble=False)
        |  with group-aware StratifiedGroupKFold index splits as cv
        v
   models/unified_v2.pkl      (855-dim hybrid)
   models/classical_v2.pkl    (85-dim classical-only fallback / ablation)
```

The grid scores with ROC-AUC over `decision_function` (no inner Platt CV); the
saved model is calibrated once, and the calibration CV is **also** group-aware so
the sigmoid is never fit on near-duplicate variants of its own held-out rows.
GPU (`--gpu`) uses cuML's SVC for the grid search and converts to sklearn before
pickling; the sklearn path is always available as a fallback.

The bundle is a dict: `{version, pipeline, feature_names, embed_model_name,
classical_only, best_params, cv_roc_auc, trained_at, ...}`.

---

## Evaluation (`python main.py --evaluate` / `python -m src.evaluate_unified`)

Scores the unified and classical-only models on the held-out test split and the
held-out-generator split, then writes `reports/eval_v2_<date>.{json,md}`.

Reported per run:
- **Threshold metrics @ 0.5:** Accuracy, Precision, Recall, F1 (positive = AI).
- **Ranking metrics:** ROC-AUC, PR-AUC (threshold-free).
- **Operating point:** Pd@5%FAR — detection rate at the threshold where 5% of
  real images are false-flagged.
- **Breakdowns:** by distribution condition, by generator architecture, by
  resolution bucket, plus an EXIF-permutation leakage check.
- **Gates:** absolute acceptance targets on the current data (e.g. clean
  ROC-AUC, screenshot accuracy, holdout AUC/Pd, unified >= classical-only).

A metric that is undefined for a single-class slice (e.g. an all-AI holdout has
no AUC) is reported as n/a, not as a failure.

---

## Prediction (`src/unified_detector.py`)

```
UnifiedDetector.predict(image_path)
    |
    |-- torch available and unified model present?
    |     yes -> extract_unified() -> 855-dim -> unified_v2 pipeline.predict_proba
    |     no / embedding failed -> extract_unified(skip_embeddings=True)
    |                              -> 85-dim -> classical_v2 pipeline.predict_proba
    |
    `-- {label, confidence, probability_ai, method, fallback, explanation,
         screenshot_warning?}
```

`probability_ai` is the calibrated p(AI); `label` is `argmax` of the calibrated
probabilities. `screenshot_warning` is informational only.

---

## Web Interface (`app.py`)

FastAPI drag-and-drop UI on port 8000, using `UnifiedDetector` with automatic
classical fallback. Uploaded images go to a per-request temp file and are deleted
immediately after the scan — never stored, logged, or used for training. The
server is hardened (rate limiting, concurrency cap, request-size and pixel-bomb
guards, security headers); see [SECURITY.md](SECURITY.md) for each control.
Run `python app.py` and open `http://localhost:8000` on your laptop.
For the smaller classical-only installation, use `requirements-runtime.txt`;
setup is documented in [README.md](README.md). Server-hosting configurations
and guides have been removed; CLI, local UI, training, and evaluation remain.

---

## Research helpers and committed results

The original helpers for family analysis, autoregressive experiments, artifact
probes, external evaluation, and experiment recording are absent from this
public checkout. Commands using those `scripts.*` modules are therefore not
available here. Evaluation and family-analysis reports are retained under
`reports/`; other original research reports are absent. RESEARCH.md preserves
the dated findings and identifies that evidence limitation.

---

## Reproducibility

`experiment_v1.json` records the dataset content hash, generator list, image and
split counts, all random seeds, the feature version, the model hyperparameters,
and the reported metrics. To recreate a reported number: restore `data/` to the
recorded dataset hash, run `python -m src.train_unified --gpu --n-jobs -1`, then
`python -m src.evaluate_unified`, and compare against the record. Regenerate the
record after a retrain only with the original recording helper (not shipped
here) or a separately reviewed replacement. Do not relabel the committed record
as a new run merely because a model was retrained.


## Company website (`site/`)

The public checkout contains the Human or AI static website. It presents product
direction separately from shipped detector capabilities. The full model remains available on your laptop. The optional hosted demo uses
the existing 85-feature classical model on a free Render service; Cloudflare
Pages still has no Python endpoint or Pages Functions.

All visitor copy is in `site/content.mjs`; `site/config.mjs` is the single source
for `DEMO_ENABLED` (defaults false, enabled through the Pages build environment after verification). Build from `site/` with Node 24.19.0:

```bash
SITE_URL=https://your-domain.example npm run build
npm run check
node verify.mjs https://your-domain.example
```

Cloudflare Pages: GitHub repo `aman696/aidetector`, branch `master`, root `site`,
build command `node build.mjs`, output `dist`. Set `SITE_URL` to the chosen HTTPS
origin and `NODE_VERSION=24.19.0`. The website's custom domain is humanorai.online.
No domain is embedded in the site renderer.

The build copies public evaluation records and selected documentation from their
existing source files, generates model metadata without copying model weights,
and adds security headers, hashed assets, sitemap, and robots.txt. It validates
Cloudflare's 25 MiB per-file and 20,000-file limits. Full configuration, asset URL
inventory, and post-deploy checks: [site/README.md](site/README.md).


### Hosted image-testing demonstration

`demo/api.py` serves the existing `models/classical_v2.pkl` with the original
`extract_unified(..., skip_embeddings=True)` pipeline. It verifies the model
hash and feature schema before starting. It never installs or invokes DINOv2 or
PyTorch; the website labels this distinction and shows the classical benchmark.
The full unified model and retired v1 bundles are not loaded by this service.

`render.yaml` fixes the service to the free Python plan: build with
`python -m pip install -r requirements-runtime.txt`, start with
`python -m uvicorn demo.api:app --host 0.0.0.0 --port $PORT --workers 1 --no-access-log`,
and use `/healthz`. Set `ALLOWED_ORIGIN` to the company website's HTTPS origin,
`PYTHON_VERSION=3.12.12`, and BLAS/OMP/MKL thread counts to 1. No local server or
Docker deployment is required. Free capacity may sleep, become busy, or suspend
at provider limits; it is a demonstration, not a production inference service.

The service caps files at 1 MiB, decoded images at 1,048,576 pixels and 4,096
pixels per side, rejects animations, permits one running scan and three requests
per minute globally, and rejects oversized bodies before multipart parsing.
Uploads use temporary directories owned by the inference thread. A client
timeout or disconnect does not free a running scan's slot or delete its image
prematurely; cleanup happens after the thread exits. No upload bytes, filenames,
EXIF values, or results are logged or stored by the application.

Enable `DEMO_ENABLED=true` and set `DEMO_API_ORIGIN` in the Pages build environment
only after a real hosted scan succeeds. The build checks that the configured
service reports a loaded classical model. `/test/` then offers image uploads,
estimated AI probability, all 85 measured features grouped into 11 analyzers,
and a JSON result download. The site's CSP and the API's CORS policy permit
only the configured origins. All website copy remains in `site/content.mjs`.

Run `python -m pytest tests/test_demo_api.py -q` for model parity, upload boundary,
origin/rate-limit, and timeout/cleanup checks. These test the actual model; no
trained parameters or analyzer calculations were changed.
