# AI Image Detector — Workflow Reference

> **This document describes the v1 workflow.** A v2 upgrade is in progress (one unified
> classical + DINOv2 detector, 855-dim, covering social-media/screenshot/chained images),
> adding `src/dataset.py`, `src/channels.py`, `src/feature_pipeline.py`,
> `src/embedding_extractor.py`, `src/train_unified.py`. This file gets its full v2 rewrite when
> the v2 path is wired into `main.py`/`app.py` (Stage 7). Until then, the live v2 state and
> commands are in `V2_PROGRESS.md` (local only).

> **Last updated:** June 11, 2026
> **Main model:** `models/svm_classifier.pkl` (sklearn SVM, 79 features, GPU-trained then converted) — CV 74%, test 82.3%
> **Screenshot model:** `models/screenshot_classifier.pkl` (sklearn SVM, 15 features) — CV 98% (likely overfit; test 96.6% AI / ~85% real)

---

## Quick Reference

```bash
# Activate venv
source .venv/bin/activate

# Classify a single image (main SVM)
python main.py --image path/to/image.jpg

# Classify a screenshot specifically
python main.py --image path/to/screenshot.png
# → use --screenshot-mode in batch; in the web UI toggle "📱 Screenshot"

# Batch classify — main SVM
python main.py --batch path/to/folder/

# Batch classify — dedicated screenshot classifier
python main.py --batch path/to/screenshots/ --screenshot-mode

# Retrain main SVM (79-feature, all image types, ~10 min)
python main.py --train

# Retrain screenshot SVM (15-feature, screenshots only, ~30s)
python main.py --train-screenshot

# Evaluate main SVM on data/test/
python main.py --evaluate

# Diagnose detectability by generator architecture (read-only; reuses the
# feature cache, no retrain). Writes reports/family_analysis_<date>.{json,md}.
# Method + findings: code_notes/20-family-analysis.md, RESEARCH.md.
python -m scripts.analyze_families

# Probe autoregressive (AR) image artifacts: token-grid periodicity + raster-scan
# anisotropy on sampled images (read-only). Writes reports/ar_artifact_probe_<date>.
# Findings: code_notes/21-ar-artifacts.md, paper_notes/ar-detection.md.
python -m scripts.ar_artifact_probe --per-family 40

# Generate real browser screenshots for training data
python scripts/generate_real_screenshots.py --mix   --count 80 --out data/screenshots
python scripts/generate_real_screenshots.py --humans --count 50 --out data/screenshots

# Run the web interface
python app.py   # → http://localhost:8000

# Run all tests
python -m pytest tests/ -v --tb=short
```

### Two Ways to Use AIDetector

**Way 1: The Web App (drag-and-drop)**
- Start the server: `python app.py` → open `http://localhost:5001`
- Switch between "Normal" mode and "Screenshot" mode
- Drag-and-drop an image → see a verdict card, 9 analyzer cards (green/yellow/red), and an expert details panel with raw numbers

**Way 2: The Command Line (for automation/scripts)**
- `python main.py --image photo.jpg` — analyze a single image
- `python main.py --train` — train a new SVM model
- `python main.py --train-screenshot` — train the screenshot detector
- `python main.py --evaluate` — test the model's accuracy
- `python main.py --batch folder/` — analyze all images in a folder

---

## Architecture: Dual-Model System

> Think of AIDetector's architecture as 9 detectives (analyzers) each producing a score from 0–1
> (0 = definitely real, 1 = definitely AI), and a judge (the classifier) that combines all scores
> into a final verdict. There are actually **two judges** — one for normal images, one for
> screenshots.

### Judge A: The SVM Classifier (Main Mode)
The SVM (Support Vector Machine) is trained on a mix of real and AI images (the dataset is
currently being expanded — see the retraining note in Training Pipelines). It
learned a formula: given the 9 scores plus additional features, it outputs a final verdict with
a confidence percentage (e.g., 94.5%) and an explanation of which detective gave the strongest
evidence.

### Judge B: The Voting Classifier (Fallback)
If the SVM model file is missing or broken, AIDetector uses democracy: each detective gets a
vote, and the majority wins. Confidence becomes the average score of all 9 detectives.

### Screenshot Mode: A Separate Judge
If you're checking whether something is a screenshot, AIDetector routes to a ScreenshotClassifier
— a separate model trained specifically on screenshots vs. non-screenshots. It uses a completely
different set of features (GLCM, wavelet, LBP, histogram) that survive the display pipeline.

### Model A — Main SVM (downloaded images + raw screenshots)

```
Image
  │
  ├── fft_analyzer.py              →  4 features  (Durall 2020, arXiv:1911.00686)
  ├── eigen_analyzer.py            → 12 features  (Corvi 2023, arXiv:2304.06408)
  ├── metadata_extractor.py        →  6 features  (EXIF forensics)
  ├── noise_analyzer.py            → 11 features  (PRNU + multi-scale + chroma corr.)
  ├── dct_analyzer.py              →  8 features  (Frank 2020 + JPEG boundary)
  ├── ela_analyzer.py              →  5 features  (Error Level Analysis)
  ├── gradient_analyzer.py         →  5 features  (Gragnaniello CVPR 2023)
  ├── patchcraft_analyzer.py       →  3 features  (Zhong 2023, arXiv:2311.12397)
  ├── screenshot_image_analyzer.py → 10 features  (Ng et al., Thongkamwitoon et al.)
  └── [RIGID drift]                → 15 features  (|original − noise_perturbed|)
                                    ──────────────
                                    79 features total
                                        │
                                  StandardScaler
                                        │
                              cuML SVC (RTX 4060)  [sklearn fallback if no GPU]
                                        │
                          "Real" / "AI-Generated" + confidence + explanation
```

### Model B — Screenshot SVM (when --screenshot-mode or web UI toggle is active)

```
Screenshot image
  │
  ├── GLCM homogeneity + contrast @ distances 1, 3, 7  →  6 features
  ├── Haar wavelet HH energy @ pyramid levels 1, 2, 3  →  3 features
  ├── LBP entropy @ radii 1, 2                          →  2 features
  ├── FFT radial power slope (VAE attenuation signal)   →  1 feature
  ├── Histogram bimodality + peak/valley ratio           →  2 features
  └── Chroma std ratio                                  →  1 feature
                                    ──────────────
                                    15 features total
                                        │
                                  StandardScaler
                                        │
                              sklearn SVC (C=2, class_weight='balanced')
                              threshold: P(AI) ≥ 0.55 to call AI
                                        │
                          "Real" / "AI-Generated" + confidence + feature breakdown
```

### Routing Logic

```
Incoming image
  │
  ├─ mode=screenshot  AND  screenshot_classifier.pkl exists?
  │     └─ YES → ScreenshotClassifier.predict()     (15-feature SVM)
  │
  └─ otherwise  → AIDetectorClassifier.predict()    (79-feature SVM)
                    + legacy confidence override if screenshot warning fires
```

---

## Training Pipelines

### Main SVM (`python main.py --train`)

> 🔄 **RETRAINING PHASE:** The dataset has been expanded — many more generator families and a much
> larger image count than the run described below. The old layout (`data/real/`,
> `data/ai_generated/`, `data/screenshots/`, …) was replaced by `data/{Gemini, GPT, mixed, real}`,
> and the models have not yet been retrained on it. The diagram below documents the **previous**
> training run; the pipeline and this section will be rewritten when retraining is done.

```
data/real/                    (~50 images)   ─┐
data/ai_generated/            (~51 images)   ─┤
data/screenshots/             (~28 real SS)  ─┤
data/ai_generated_screenshots/ (~14 AI SS)  ─┤  → base images (~504 total)
data/socialmediareal_ss/      (109 real SS)  ─┤
data/socialmediaai_ss/        (252 AI SS)    ─┘
         │
         ▼ augment_dataset_with_jpeg()
         │   └─ Q=70 copy + Q=80 copy + 0.75× resize copy
         ▼
    ~2016 augmented images
         │
         ▼ extract_batch(n_workers=16)   ← ProcessPoolExecutor (16 CPU cores)
         │   Each worker: 9 analyzers (original) + RIGID drift (2nd pass with σ=2 noise)
         ▼
    79-feature matrix  (~2016 × 79)
         │
         ▼ StandardScaler.fit_transform()
         ▼ cuML SVC.fit() → converted to sklearn before pickle (portability fix)
         ▼
    models/svm_classifier.pkl
```

**Training time:** ~620s total — feature extraction dominates (~615s, 2× passes per image for RIGID drift)

### Screenshot SVM (`python main.py --train-screenshot`)

> 🔄 **RETRAINING PHASE:** Same situation — these screenshot directories belong to the previous
> dataset layout and no longer exist. The diagram documents the previous training run; to be
> rewritten after retraining on the expanded dataset.

```
data/screenshots/              (~28 real SS)   ─┐
data/ai_generated_screenshots/  (~14 AI SS)    ─┤
data/socialmediareal_ss/        (109 real SS)  ─┤  extra_real_dirs
data/socialmediaai_ss/          (252 AI SS)    ─┘  extra_ai_dirs
         │
         ▼ extract_screenshot_features()   — 15 features each, no augmentation
         ▼
    ~403-image matrix
         │
         ▼ StandardScaler + sklearn SVC (C=2, class_weight='balanced', threshold 0.55)
         ▼
    models/screenshot_classifier.pkl
```

**Training time:** ~30 seconds. No JPEG augmentation — screenshots are already the "compressed form".

### Generating Real Screenshots (`scripts/generate_real_screenshots.py`)

Uses headless Firefox via Playwright. 80+ URL pool: Wikipedia biographies, Unsplash/Pexels portraits,
Reuters/AP photojournalism, GitHub, StackOverflow, MDN, Reddit, arxiv, etc. Scrolls to different
offsets per site for visual diversity. ~4s per screenshot.

```bash
python scripts/generate_real_screenshots.py --mix    --count 80 --out data/screenshots  # 50/50 UI+human
python scripts/generate_real_screenshots.py --humans --count 50 --out data/screenshots  # human-photo focus
python scripts/generate_real_screenshots.py          --count 60 --out data/screenshots  # UI-only (default)
```

---

## Prediction Pipeline (Main SVM)

```
AIDetectorClassifier.predict(image_path)
    │
    ├── detect_screenshot()    ← 3 heuristics: noise var < 3.0, histogram entropy < 6.8, screen dims
    │     └─ if ≥2 fire → adds 'screenshot_warning' key to result dict
    │
    ├── extract_individual_scores()   ← 9 analog scores (for fallback voting + explanation)
    │
    ├── extract()    ← 79-feature vector
    │     ├─ 9 analyzers (original image)
    │     └─ _compute_drift_features()
    │           ├─ add Gaussian noise (σ=2) → save temp file
    │           ├─ re-run FFT, Noise, Gradient, DCT, PatchCraft, Eigen, Screenshot
    │           └─ return |original − perturbed| for 15 key features
    │
    └── scaler.transform() → svm.predict_proba()
          │
          └─ {'label', 'confidence', 'scores', 'method', 'explanation', 'screenshot_warning'?}
```

**Fallback (no trained model):** Weighted voting over 9 individual scores:
```
score = 0.12·fft + 0.14·eigen + 0.09·meta + 0.13·noise + 0.12·dct
       + 0.10·ela + 0.12·gradient + 0.10·patchcraft + 0.08·screenshot_forensics
label = "AI-Generated" if score > 0.5 else "Real"
```

---

## Method Summaries

Each method below has two sections: a **plain-English explanation** (from PROJECT_EXPLAINED.md)
followed by the **technical details** (features, limitations).

### Method 1: FFT Frequency Analysis — The Frequency Listener (`fft_analyzer.py`, 4 features)

> **Plain English:** Every image is made of patterns — big patterns (like the shape of a face)
> and tiny patterns (like the texture of a shirt). The FFT detective listens to ALL the patterns,
> like separating a song into bass, drums, and vocals. AI-generated images have an **unnatural
> mix of patterns** — the AI draws big things well but messes up the tiny details. Think of
> looking at a painting that was copied by a printer: up close, the printer's dots are arranged
> too perfectly — a human artist's brush strokes would be messier.

Natural images follow ~1/f power law. AI images deviate at high frequencies.
`spectral_slope`, `slope_r_squared`, `high_freq_ratio`, `spectral_falloff`
**Limitation:** Weak on modern diffusion models (DALL-E 3, Flux); degrades on < 512px.

### Method 2: Eigenvalue / Spectral Analysis — The Color Math Genius (`eigen_analyzer.py`, 12 features)

> **Plain English:** It looks at how the red, green, and blue colors in the image relate to each
> other mathematically. In real photos, the relationship between colors is complex and messy —
> like real human relationships. In AI images, the color relationships are **too simple**, like
> the AI took a shortcut. Imagine asking "how are red and green connected in this photo?" Real
> photos give a long, complicated answer. AI images give a one-word answer.

Synthetic images have different covariance structure. Global RGB covariance → eigenvalues; patch-level stats; spectral band energy ratios.
**Limitation:** Chroma subsampling (4:2:0) on social media partially destroys eigenvalue structure.

### Method 3: Metadata Forensics — The Hidden Tag Reader (`metadata_extractor.py`, 6 features)

> **Plain English:** Every image file can contain hidden information — like your phone's camera
> model, the date, GPS location, and editing history. These are called metadata or EXIF data.
> AI-generated images usually have **no metadata at all** (no camera info, no date, nothing —
> like a person with no ID). When you take a photo with your phone, it writes "who, what, when,
> where" in the file. AI images are amnesiac — they have no memory of being taken.

Real camera photos carry EXIF (Make, Model, GPS, FocalLength). AI images have none. PNG-aware scoring.
**Limitation:** Social media strips all EXIF — every social-media image looks AI to this analyzer.

### Method 4: Noise Residual Analysis — The Grain Inspector (`noise_analyzer.py`, 11 features)

> **Plain English:** Every real photo has tiny noise — random color dots, like static on a TV.
> Phones and cameras add this noise naturally. AI images often have **no noise at all** (too
> clean) or the noise is **too uniform** (same fake grain everywhere). Real photos have natural,
> varied noise patterns. Look at the ceiling of a room: real photos show tiny imperfections. AI
> images often show a perfect, smooth ceiling that's too good to be true.

Gaussian denoising residual (σ=3). 6 base stats + 2 multi-scale ratios (σ1/σ5, σ3/σ5) + 3 Bayer chroma correlations (RG, RB, GB).
**Limitation:** Screenshots and display-rendered images have no Bayer correlation → score as AI.

### Method 5: DCT Block Analysis — The JPEG Checker (`dct_analyzer.py`, 8 features)

> **Plain English:** Most images are compressed (squished smaller) using JPEG — it splits the
> image into 8×8 pixel blocks and checks how clean the block boundaries are. AI-generated images
> often get saved and re-saved many times, which **smears the block boundaries** in a weird way.
> Real photos have clean, consistent block patterns. Imagine a chessboard where the squares are
> supposed to have clear edges between them — AI images sometimes look like someone smudged the
> edges of the chess squares with their finger.

8×8 block DCT stats + JPEG block boundary consistency (`dct_boundary_ratio`, `dct_boundary_var_ratio`). Vectorized via `np.diff` + boolean mask.
**Limitation:** Screenshots are PNG→JPEG like AI images so `dct_boundary_ratio` ≈ 1.0 for both.

### Method 6: Error Level Analysis — The Re-Save Checker (`ela_analyzer.py`, 5 features)

> **Plain English:** Error Level Analysis (ELA) re-saves the image at a low quality and compares
> "before" and "after." It marks what changed. Different parts of an AI image were compressed at
> different qualities, so when you re-save, some parts change a lot and some don't change at all.
> Real photos change **evenly** everywhere. Imagine photocopying a piece of paper, then
> photocopying the photocopy, then doing it again: real photos degrade evenly each time. AI images
> degrade in patches.

Re-compress at Q=95 → pixel-wise diff. Already-compressed images show uniform error; fresh AI PNGs show high first-compression response.
**Limitation:** PNGs always show high ELA by definition.

### Method 7: Gradient Statistics — The Edge Inspector (`gradient_analyzer.py`, 5 features)

> **Plain English:** It looks at edges — where one object ends and another begins. It uses filters
> (Sobel, Laplacian) to find edges. AI images have edges that look different from real photos —
> some are too sharp, some are too blurry. Real photos have a "depth of field" — things near the
> camera are sharp, far things are blurry. AI images often get this wrong, making everything
> equally sharp or blurring things that shouldn't be.

Sobel + Laplacian stats. Real photos have heavy-tailed edge distributions; AI images are smoother. Survives JPEG recompression (relative measure).

### Method 8: PatchCraft Texture Analysis — The Consistency Checker (`patchcraft_analyzer.py`, 3 features)

> **Plain English:** It splits the image into tiny patches and compares each patch's "fingerprint."
> In a real photo, every small patch should have a similar statistical fingerprint (because the
> same camera took the whole thing). In AI images, different patches may have **different
> fingerprints** because the AI stitched together different patterns. Imagine a mosaic made of
> tiles from different bathrooms: a real photo is like a mosaic where all tiles are from the same
> box; AI images sometimes mix tiles from different boxes.

High-pass filter → 32×32 patch variances → median-split rich/poor → `texture_contrast` (KEY), `texture_rich_mean`, `texture_poor_mean`.
**Limitation:** Images < 256px produce < 36 patches → unreliable. Resolution guard zeroes output.

### Method 9: Screenshot Image Forensics (`screenshot_image_analyzer.py`, 10 features)

> **Plain English:** This detective adds screenshot-aware signals to the main SVM. Screenshots
> have unique patterns — text edges, UI elements, gradient bands, scroll bars, consistent
> background colors. If you take a picture of your computer screen with your phone, the image
> will have lines (scanlines) and a weird pattern. Screenshots from phones have similar tells.

Multi-scale GLCM, LBP, wavelet energy for the main SVM's screenshot-aware pass. Basis: Ng et al. (Imperial), Thongkamwitoon et al.

### RIGID Drift Features (`_compute_drift_features()` in `classifier.py`, 15 features)

Perturb with Gaussian noise (σ=2) → re-extract 6 analyzers → |original − perturbed| for 15 selected features. Real images: low drift. AI: higher drift. Classical approximation — no DINOv2 needed.

---

## Key Design Decisions

1. **Dual-model routing** — screenshots go to 15-feature specialist; everything else to 79-feature generalist. Prevents ELA/DCT/metadata noise from poisoning screenshot detection.
2. **`class_weight='balanced'`** in screenshot SVM — compensates 28 real vs 21 AI imbalance.
3. **Decision threshold 0.55** in screenshot SVM — reduces false AI calls on real screenshots.
4. **FFT radial slope in screenshot SVM** — VAE bottleneck attenuates diffusion model high frequencies; this signal persists through the screenshot display pipeline.
5. **Histogram bimodality in screenshot SVM** — real desktop UI = dark bg + bright text → bimodal. AI image screenshot = bell-curve → unimodal. This single feature is very discriminative.
6. **JPEG augmentation (ITW-SM)** — every training image gets Q=70, Q=80, 0.75× copies. Prevents "JPEG = Real" bias. Files cleaned up after training.
7. **RIGID drift without DINOv2** — classical feature-drift adds 15 generalization features; works on any image without loading a 300MB backbone.
8. **Center-crop (not resize) for FFT** — resizing distorts frequency content.
9. **Vectorized DCT boundary** — `np.diff` + boolean mask; no Python loops over blocks.
10. **Playwright headless screenshot generator** — generates real browser UI screenshots automatically; `--humans` flag targets portrait/people pages to counter-balance AI-face images.

---

## Known Failure Modes

| Situation | Model | Result | Root Cause | Fix Status |
|---|---|---|---|---|
| Screenshot of real desktop UI | Main SVM | ❌ Flagged AI | No EXIF, no Bayer noise, no JPEG grid | ✅ Fixed via --screenshot-mode |
| Screenshot of AI-generated image | Main SVM | ✅ Correct (often) | Same signals fire → AI call happens to be right | ⚠️ Right answer, wrong reason |
| Screenshot of real desktop UI | Screenshot SVM | ✅ ~85% correct | 15-feature specialist ignores ELA/DCT/metadata | ✅ Dedicated model |
| Screenshot of AI image | Screenshot SVM | ✅ 96.6% correct | Spectral + texture signals catch AI even through screenshot | ✅ Dedicated model |
| Instagram Reel / video frame | Both models | ❌ Flagged AI | H.264 smooth frames = low wavelet, unimodal histogram = identical to AI screenshot | ⚠️ Open — 4th category problem |
| Low-res image (< 256px) | Both | ⚠️ Uncertain | PatchCraft needs ≥ 36 patches; FFT needs ≥ 512px | ⚠️ Resolution guards partially help |
| AI image + injected EXIF + grain filter | Main SVM | ❌ May call Real | Noise spoof + metadata spoof together | ⚠️ RIGID drift partially helps |
| Social media recompressed photo | Main SVM | ⚠️ ~77% | EXIF stripped, JPEG recompressed 2× | ✅ JPEG augmentation training helps |

### Root Cause: Why Screenshots Fool the Main SVM

`metadata_score` = 0 (no EXIF) + `noise_rg/rb/gb_corr` ≈ 0 (no Bayer) + `dct_boundary_ratio` ≈ 1.0 (PNG→JPEG) + `texture_contrast` high (sharp icons + flat bg) → 4 of 9 analyzers fire AI with high confidence.

---

## File Structure

```
aidetector/
├── src/
│   ├── fft_analyzer.py              # FFT 4 features
│   ├── eigen_analyzer.py            # Eigenvalue 12 features
│   ├── metadata_extractor.py        # EXIF 6 features
│   ├── noise_analyzer.py            # Noise residual 11 features
│   ├── dct_analyzer.py              # DCT block 8 features
│   ├── ela_analyzer.py              # ELA 5 features
│   ├── gradient_analyzer.py         # Gradient 5 features
│   ├── patchcraft_analyzer.py       # Texture contrast 3 features
│   ├── screenshot_image_analyzer.py # Screenshot forensics 10 features (main SVM)
│   ├── screenshot_classifier.py     # Dedicated 15-feature screenshot SVM
│   ├── classifier.py                # 79-feature extractor + cuML/sklearn GPU SVM
│   └── utils.py                     # Image loader, JPEG augmentation, dataset utilities
├── scripts/
│   ├── generate_real_screenshots.py # Playwright headless Firefox screenshot generator
│   └── download_real_images.py      # Unsplash image downloader
├── data/
│   ├── Gemini/                      # 209 AI images from Google Gemini
│   ├── GPT/                         # 62 AI images from ChatGPT
│   ├── mixed/                       # 32 model-family subdirectories (various generators)
│   └── real/                        # 2000+ real photos (coco/ + openfake/ subdirs)
├── models/
│   ├── svm_classifier.pkl           # Main 79-feature SVM (~84KB)
│   └── screenshot_classifier.pkl    # Screenshot 15-feature SVM
├── tests/                           # pytest suite (83+ tests)
├── papers/                          # Research PDFs
├── main.py                          # CLI entry point
├── app.py                           # FastAPI web interface (port 8000)
├── WORKFLOW.md                      # This file
├── RESEARCH.md                      # Literature review + paper status
├── AGENTS.md                        # Rules for AI contributors
├── CLAUDE.md                        # Local AI agent instructions (gitignored)
└── requirements.txt
```

---

## Research Papers Implemented

| Paper | arXiv / Venue | Implemented in |
|---|---|---|
| Durall et al. 2020 — "Unmasking DeepFakes with simple Features" | arXiv:1911.00686 | `fft_analyzer.py` |
| Corvi et al. 2023 — "Intriguing Properties of Synthetic Images" | arXiv:2304.06408 | `eigen_analyzer.py` |
| Frank et al. 2020 — "Leveraging Frequency Analysis for Deep Fake Image Recognition" | ICML 2020 | `dct_analyzer.py` |
| Gragnaniello et al. 2021/2023 | IEEE ICME + CVPR | `noise_analyzer.py`, `gradient_analyzer.py` |
| Zhong et al. 2023 — "PatchCraft" | arXiv:2311.12397 | `patchcraft_analyzer.py` |
| RIGID — 2024 | arXiv 2024 | `_compute_drift_features()` in `classifier.py` |
| Konstantinidou et al. 2025 — ITW-SM | arXiv:2507.10236 | `augment_dataset_with_jpeg()` in `utils.py` |
| Ng et al. (Imperial) + Thongkamwitoon et al. — recaptured image forensics | Various | `screenshot_image_analyzer.py`, `screenshot_classifier.py` |
| "Any-Resolution AI Detection by Spectral Learning" (Nov 2024) | arXiv Nov 2024 | `fft_radial_slope` in `screenshot_classifier.py` |

---

## Performance

### Main SVM
| Metric | Value |
|---|---|
| CV accuracy | **74%** on ~2016 augmented images |
| Test accuracy | **82.3%** on held-out set |
| Training set | ~50 real + ~51 AI + ~28 real SS + ~14 AI SS + 109 real social + 252 AI social → 504 base → 2016 augmented |
| Total features | **79** (54 base + 10 screenshot-forensics + 15 RIGID drift) |
| Model format | sklearn (converted from cuML before pickle for portability) |

### Screenshot SVM
| Metric | Value |
|---|---|
| CV accuracy | **~98%** — likely overfitting (small dataset before social media integration) |
| Test — AI screenshots (`data/ai_test/`, 29 images) | **96.6%** (28/29 ✅) |
| Test — Real screenshots (`data/real_test/`, 35 clean images) | **~77–85%** |
| Total features | **15** |
| Training time | ~30 seconds |

> ⚠️ **Real screenshot accuracy:** ~6 misclassified files in `data/real_test/` are Instagram Reel video-frame captures. Remove those and real accuracy approaches 97%.