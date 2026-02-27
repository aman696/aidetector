# AI Image Detector — Workflow Reference

> **Last updated:** February 27, 2026  
> **Model:** `models/svm_classifier.pkl` (cuML GPU SVM, 69 features)  
> **CV Accuracy:** 84.8% on 500 augmented images (RTX 4060)

---

## Quick Reference

```bash
# Activate venv
source .venv/bin/activate

# Classify a single image
python main.py --image path/to/image.jpg

# Retrain (GPU + parallel CPU extraction)
python main.py --train

# Evaluate on test set
python main.py --evaluate

# Run all tests
python -m pytest tests/ -v --tb=short

# Batch classify a folder
python main.py --batch path/to/folder/
```

---

## Architecture: 8 Analyzers → 69 Features → GPU SVM

```
Image
  │
  ├── fft_analyzer.py         →  4 features  (Durall 2020, arXiv:1911.00686)
  ├── eigen_analyzer.py       → 12 features  (Corvi 2023, arXiv:2304.06408)
  ├── metadata_extractor.py   →  6 features  (EXIF forensics)
  ├── noise_analyzer.py       → 11 features  (PRNU + multi-scale + chroma corr.)
  ├── dct_analyzer.py         →  8 features  (Frank 2020 + JPEG boundary)
  ├── ela_analyzer.py         →  5 features  (Error Level Analysis)
  ├── gradient_analyzer.py    →  5 features  (Gragnaniello CVPR 2023)
  ├── patchcraft_analyzer.py  →  3 features  (Zhong 2023, arXiv:2311.12397)
  └── [RIGID drift]           → 15 features  (|original − noise_perturbed|, arXiv 2024)
                                ──────────
                                69 features total
                                    │
                              StandardScaler
                                    │
                              cuML SVC (RTX 4060)
                              [sklearn fallback if no GPU]
                                    │
                              "Real" / "AI-Generated"
                              + confidence + explanation
                              + screenshot_warning (if applicable)
```

---

## Training Pipeline

```
data/real/             (50 images)   ─┐
data/ai_generated/     (51 images)   ─┤→ 125 base images
data/screenshots/      (14 real SS)  ─┤
data/ai_screenshots/   (10 AI SS)   ─┘
         │
         ▼ augment_dataset_with_jpeg()
         │   └─ Q=70 copy + Q=80 copy + 0.75x resize copy
         ▼
    500 augmented images  (ITW-SM finding: data composition > model complexity)
         │
         ▼ extract_batch(n_workers=16)   ← ProcessPoolExecutor (16 CPU cores)
         │   Each worker: all 8 analyzers + RIGID drift (2× per image)
         ▼
    69-feature matrix  (500 × 69)
         │
         ▼ StandardScaler.fit_transform()
         │
         ▼ cuML SVC.fit()    ← GPU (RTX 4060, ~seconds)
         │
         ▼ models/svm_classifier.pkl   (84KB)
```

**Training time:** ~620s total (~10 min)  
- Feature extraction: ~615s (bottleneck — 500 images × 2× for RIGID drift, parallelized across 16 cores)  
- SVM training: < 5s (GPU)

---

## Prediction Pipeline

```
predict(image_path)
    │
    ├── detect_screenshot()    ← 3 heuristics: noise var, histogram entropy, screen dims
    │     └─ if is_screenshot → adds 'screenshot_warning' to result
    │
    ├── extract_individual_scores()   ← 8 analog scores (for fallback voting + explanation)
    │
    ├── extract()    ← 69-feature vector
    │     ├─ 8 analyzers (original image)
    │     └─ _compute_drift_features()
    │           ├─ add Gaussian noise (σ=2) → save temp file
    │           ├─ re-run FFT, Noise, Gradient, DCT, PatchCraft, Eigen
    │           └─ return |original − perturbed| for 15 key features
    │
    └── scaler.transform() → svm.predict_proba()
          │
          └─ {'label', 'confidence', 'scores', 'method', 'explanation',
               'screenshot_warning'?}
```

---

## Key Design Decisions

### Resolution Guards
- **PatchCraft:** returns zeroed defaults if `min(h, w) < 256` (too few 32×32 patches)
- **Noise multi-scale:** skips sigma=5 ratios if `min(h, w) < 64` (kernel too large for image)

### JPEG Augmentation (ITW-SM Finding)
- Every training image gets 3 augmented copies: Q=70, Q=80, 0.75× downscale
- These are written to `tempfile.mkdtemp()` and deleted after training
- Teaches the SVM that "JPEG-compressed Real photo ≠ AI"

### RIGID Drift Features
- Perturb image with Gaussian noise (σ=2, imperceptible)
- |original_features − perturbed_features| for 15 selected features
- Real images: low drift (stable). AI images: higher drift
- Implemented classically without DINOv2/ViT — no extra dependencies

### Screenshot Pre-detection
- Runs **before** SVM on every `predict()` call
- Three heuristics: noise variance < 3.0, histogram entropy < 6.8, screen-resolution dimensions
- If ≥2 fire → adds `screenshot_warning` key to result dict
- Does NOT skip classification — user still gets a result, just with the warning

---

## Known Failure Modes

| Situation | Expected output | Why |
|---|---|---|
| Screenshot of real content | Likely "AI-Generated" + `screenshot_warning` | No EXIF, no camera noise, no JPEG grid |
| Social media screenshot of AI art | "AI-Generated" — generally correct | |
| Low-res AI image (<256px) | Less confident / may miss | PatchCraft and FFT degrade below ~400px |
| AI image with grain filter + injected EXIF | May classify as "Real" | Noise looks camera-like, metadata passes |
| Lazy JPEG (Q < 60, heavy artifact) | May classify as "AI" | DCT/ELA score elevated |

---

## File Structure

```
aidetector/
├── src/
│   ├── fft_analyzer.py          # FFT 4 features
│   ├── eigen_analyzer.py        # Eigenvalue 12 features
│   ├── metadata_extractor.py    # EXIF 6 features
│   ├── noise_analyzer.py        # Noise residual 11 features
│   ├── dct_analyzer.py          # DCT block 8 features
│   ├── ela_analyzer.py          # ELA 5 features
│   ├── gradient_analyzer.py     # Gradient 5 features
│   ├── patchcraft_analyzer.py   # Texture contrast 3 features
│   ├── screenshot_detector.py   # Screenshot pre-pass (NEW)
│   ├── classifier.py            # 69-feature extractor + GPU SVM
│   └── utils.py                 # Loader + JPEG augmentation (NEW)
├── data/
│   ├── real/                    # 50 real photos
│   ├── ai_generated/            # 51 AI images
│   ├── screenshots/             # 14 real screenshots
│   ├── ai_generated_screenshots/# 10 AI screenshots
│   └── test/{real,ai_generated}/
├── models/
│   └── svm_classifier.pkl       # Trained model (84KB, 69 features)
├── tests/                       # pytest suite (83+ tests)
├── papers/                      # Durall 2020, Corvi 2023 PDFs
├── main.py                      # CLI entry point
├── WORKFLOW.md                  # This file
├── RESEARCH.md                  # Literature review + paper status
└── requirements.txt
```

---

## Research Papers Implemented

| Paper | arXiv | Implemented as |
|---|---|---|
| Durall et al. 2020 | 1911.00686 | `fft_analyzer.py` |
| Corvi et al. 2023 | 2304.06408 | `eigen_analyzer.py` |
| Frank et al. 2020 | ICML 2020 | `dct_analyzer.py` |
| Gragnaniello 2021/2023 | ICME + CVPR | `noise_analyzer.py`, `gradient_analyzer.py` |
| PatchCraft — Zhong 2023 | 2311.12397 | `patchcraft_analyzer.py` |
| RIGID — anon 2024 | arXiv 2024 | `_compute_drift_features()` in classifier |
| ITW-SM — Konstantinidou 2025 | 2507.10236 | `augment_dataset_with_jpeg()` in utils |


---

## Architecture Overview

```
Image → [FFT Analyzer]          →  4 features ─┐
       → [Eigen Analyzer]        → 12 features  │
       → [Metadata Extractor]    →  6 features  │
       → [Noise Analyzer]        → 11 features  ├─ [54-feature vector] → [SVM Classifier] → Real / AI-Generated
       → [DCT Block Analyzer]    →  8 features  │
       → [ELA Analyzer]          →  5 features  │
       → [Gradient Analyzer]     →  5 features  │
       → [PatchCraft Analyzer]   →  3 features ─┘
```

Eight independent analysis methods extract a **54-dimensional feature vector** from each image. A trained SVM with RBF kernel makes the final binary classification. If no trained model is available, a weighted voting fallback uses individual analyzer scores (one 0–1 score per method).

---

## Research Foundation

| Paper | What We Use It For |
|---|---|
| Durall et al. 2020 — *"Unmasking DeepFakes with simple Features"* (arXiv: 1911.00686) | FFT spectral slope analysis (Method 1) |
| Corvi et al. 2023 — *"Intriguing Properties of Synthetic Images"* (arXiv: 2304.06408) | Eigenvalue + spectral band features (Method 2) |
| Frank et al. 2020 — *"Leveraging Frequency Analysis for Deep Fake Image Recognition"* | DCT block coefficient analysis (Method 5) |
| Gragnaniello et al. CVPR 2023 | Gradient statistics — compression-resilient edge features (Method 7) |
| PatchCraft (arXiv 2024) — *"Towards Universal Fake Image Detection by Detecting Closest Real Image"* | Rich/Poor texture contrast — JPEG-resilient (Method 8) |
| ITW-SM dataset paper — *"Navigating the Challenges of AI-Generated Image Detection in the Wild"* (arXiv 2024) | Key reference for social-media-sourced image challenges and failure modes |

---

## Method 1: FFT Frequency Analysis (`src/fft_analyzer.py`)

**Key Insight:** Natural images follow a ~1/f power law in the frequency domain. AI-generated images deviate, showing a characteristic drop-off at high frequencies.

**Pipeline:**
1. Load image as grayscale, center-crop to square
2. Apply 2D FFT (`np.fft.fft2`) → shift DC to center (`fftshift`)
3. Compute log-magnitude spectrum
4. **Azimuthal average**: average magnitude over concentric rings → 1D radial power spectrum
5. Extract 4 features: `spectral_slope`, `slope_r_squared`, `high_freq_ratio`, `spectral_falloff`

**Known limitation:** Modern diffusion models (Gemini, DALL-E 3, Flux) have improved high-frequency reproduction — this feature discriminates weakly on current generators. Also degrades on images <512px (insufficient frequency bins for reliable slope fitting).

---

## Method 2: Eigenvalue/Spectral Analysis (`src/eigen_analyzer.py`)

**Key Insight:** Synthetic images have different covariance structure and frequency band distributions than natural images.

**Pipeline:**
1. Load image in BGR (OpenCV default)
2. **Global covariance**: reshape to (N, 3) RGB pixels → compute 3×3 covariance → eigenvalues → ratios
3. **Patch analysis**: divide into 64×64 patches → compute eigenvalue stats per patch → aggregate
4. **Spectral bands**: 2D FFT → split into low/mid/high frequency bands → energy ratios
5. Extract 12 features total (4 ratio + 4 patch + 4 band)

**Known limitation:** Chroma subsampling (4:2:0) from social media platforms partially destroys the eigenvalue structure.

---

## Method 3: Metadata Forensics (`src/metadata_extractor.py`)

**Key Insight:** Real camera photos contain rich EXIF metadata (Make, Model, GPS, FocalLength, etc.). AI-generated images have none.

**Pipeline:**
1. Extract EXIF data using `Pillow`
2. Categorize tags: camera tags, software tags, context tags (GPS, timestamps)
3. Extract 6 features: tag count, camera tag count, boolean flags for camera/GPS/timestamps/software
4. Score based on presence/absence of metadata, with PNG-aware leniency

**Known limitation (major):** Social media platforms (Twitter/X, Instagram, Facebook) strip all EXIF on upload. This makes any social-media-sourced image (real or AI) look AI-generated to this analyzer. Screenshots also have no EXIF.

---

## Method 4: Noise Residual Analysis (`src/noise_analyzer.py`)

**Inspired by:** PRNU (Photo Response Non-Uniformity) noise analysis

**Key Insight:** Camera sensors leave structured, spatially correlated noise via the Bayer filter mosaic. AI images generate synthetic or uniform noise.

**Pipeline:**
1. Grayscale Gaussian denoising (sigma=3) → residual = original − denoised
2. Extract 6 base features: `noise_variance`, `noise_kurtosis`, `noise_skewness`, `noise_spectral_entropy`, `noise_autocorrelation`, `noise_block_var_std`
3. **Multi-scale ratios** (social-media-resilient — relative measures survive JPEG recompression):
   - `noise_ms_ratio_1_5`: var(sigma=1 residual) / var(sigma=5 residual) — fine vs coarse noise distribution
   - `noise_ms_ratio_3_5`: var(sigma=3 residual) / var(sigma=5 residual)
4. **Chroma inter-channel correlations** (Bayer fingerprint):
   - `noise_rg_corr`, `noise_rb_corr`, `noise_gb_corr`: Pearsonr of noise residuals in R, G, B channels

**Total: 11 features**

**Known limitation:** Calibrated for standard daylight consumer camera sensor noise. Screenshots, scanned photos, and display-rendered images have completely different noise structure — all score as AI-like.

---

## Method 5: DCT Block Analysis (`src/dct_analyzer.py`)

**Key Insight:** AI generators leave systematic artifacts in 8×8 DCT block coefficient distributions. JPEG block boundary consistency reveals whether an image originated from a camera JPEG or from an AI PNG.

**Pipeline:**
1. Divide grayscale image into 8×8 non-overlapping blocks → apply 2D DCT to each
2. Extract 6 coefficient features: `dct_ac_energy_ratio`, `dct_high_freq_energy`, `dct_coeff_kurtosis`, `dct_coeff_variance`, `dct_dc_variance`, `dct_zigzag_decay`
3. **JPEG block boundary features** (vectorized numpy — social-media-resilient):
   - `dct_boundary_ratio`: mean |diff| at 8×8 block boundaries / mean |diff| interior. Real re-compressed JPEGs have stronger boundaries (>1). AI PNG→JPEG does not.
   - `dct_boundary_var_ratio`: variance ratio of boundary vs interior differences

**Total: 8 features**

**Known limitation:** Screenshots are PNG→JPEG like AI images, so `dct_boundary_ratio` ≈ 1.0 for both — this is correct mechanically but confounds real screenshots with AI.

---

## Method 6: Error Level Analysis (`src/ela_analyzer.py`)

**Key Insight:** Re-compressing an image that was already JPEG-compressed shows uniform error levels. A never-compressed image (typical AI PNG) shows a strong first-compression response.

**Pipeline:**
1. Re-save image at JPEG quality 95 into a temp file
2. Compute pixel-wise absolute difference between original and re-compressed
3. Extract 5 features: `ela_mean`, `ela_variance`, `ela_max`, `ela_uniformity`, `ela_block_inconsistency`

**PNG-awareness:** PNGs show high ELA by definition (first compression). Score weights adjusted accordingly.

---

## Method 7: Gradient Statistics (`src/gradient_analyzer.py`)

**Based on:** Gragnaniello et al. CVPR 2023

**Key Insight:** Real photos have heavy-tailed edge distributions (sharp, unpredictable real-world edges). AI images have smoother, more regularized gradient distributions. These survive JPEG recompression because they measure *relative* edge structure, not absolute pixel values.

**Pipeline:**
1. Compute Sobel gradient magnitude map
2. Compute Laplacian (2nd-order derivative) map
3. Extract 5 features: `gradient_mean`, `gradient_variance`, `gradient_kurtosis`, `gradient_laplacian_mean`, `gradient_laplacian_variance`

**Score logic:** Low kurtosis (smooth/Gaussian edge distribution) → AI. High kurtosis (heavy-tailed) → Real.

---

## Method 8: PatchCraft Texture Analysis (`src/patchcraft_analyzer.py`)

**Based on:** *"Towards Universal Fake Image Detection by Detecting Closest Real Image"* (arXiv 2024)

**Key Insight:** AI generators struggle to reproduce natural rich textures faithfully. High-pass filtering and comparing rich-texture vs poor-texture patches reveals a characteristic elevated contrast in AI images. This survives JPEG recompression because it's a relative (ratio) measure.

**Pipeline:**
1. Apply high-pass filter: `hp = image − gaussian_blur(image, sigma=3)`
2. Divide into 32×32 patches, compute per-patch variance of high-pass image
3. Split patches at median variance into "rich" (top 50%) and "poor" (bottom 50%)
4. Extract 3 features: `texture_contrast` (rich_mean − poor_mean — KEY), `texture_rich_mean`, `texture_poor_mean`

**Known limitation:** Images <256px may have fewer than 36 patches total → statistically unreliable.

---

## SVM Classifier (`src/classifier.py`)

**Pipeline:**
1. `FeatureExtractor` runs all 8 analyzers → builds 54-feature vector in fixed order
2. `StandardScaler` z-score normalizes each feature
3. `SVC(kernel='rbf', C=10, probability=True)` makes binary classification (0=Real, 1=AI)
4. Returns label, confidence (probability), 8 individual scores, human-readable explanation

**Fallback (no trained model):** Weighted voting over 8 individual scores:
```
score = 0.12·fft + 0.15·eigen + 0.10·meta + 0.15·noise + 0.13·dct + 0.10·ela + 0.13·gradient + 0.12·patchcraft
label = "AI-Generated" if score > 0.5 else "Real"
```

---

## Known Failure Modes (Confirmed by Testing)

These are real failure modes found through actual testing, not just theoretical:

| Image Type | Failure | Root Cause |
|---|---|---|
| **Screenshot of real social media** (X, Instagram) | ❌ Flagged AI at 80%+ | No EXIF + no camera noise + no JPEG grid + high PatchCraft contrast from UI elements |
| **Screenshot of AI image from social media** | ✅ Flagged AI — but for wrong reasons | Same as above — right answer, wrong reasoning (brittle) |
| **Low-resolution AI image** (~256px) | ❌ Flagged Real at ~50% (uncertain) | PatchCraft needs ≥36 patches; multi-scale ratios collapse on small images; FFT lacks frequency resolution |
| **Digital art / illustrations** | ❌ False positive (flagged AI) | Synthetic-looking noise, no camera EXIF, no Bayer correlation |
| **AI image with added grain + injected EXIF** | ❌ False negative (misses AI) | Noise features see grain as camera-like; metadata score is low |
| **Heavy Lightroom / noise-reduction edits** | ⚠️ Unreliable | Noise reduction kills camera noise pattern we're looking for |

### Root Cause of Screenshot Problem (Specific to Our Code)

Screenshot → 6 of 8 analyzers fire "AI":
- `metadata_score` — no EXIF at all
- `noise_rg_corr`, `noise_rb_corr`, `noise_gb_corr` — display-rendered channels, no Bayer correlation
- `noise_ms_ratio_1_5/3_5` — no sensor noise, ratios fall outside camera range
- `dct_boundary_ratio` ≈ 1.0 — PNG→JPEG same as AI images
- `texture_contrast` — UI elements (sharp icons + flat backgrounds) → high rich/poor contrast

---

## Directory Structure

```
aidetector/
├── src/
│   ├── fft_analyzer.py        # FFT frequency analysis (Durall 2020)         [4 features]
│   ├── eigen_analyzer.py      # Eigenvalue + spectral band analysis (Corvi 2023) [12 features]
│   ├── metadata_extractor.py  # EXIF metadata forensics                       [6 features]
│   ├── noise_analyzer.py      # Noise residual + multi-scale + chroma corr    [11 features]
│   ├── dct_analyzer.py        # DCT blocks + JPEG boundary analysis            [8 features]
│   ├── ela_analyzer.py        # Error Level Analysis                           [5 features]
│   ├── gradient_analyzer.py   # Sobel/Laplacian gradient statistics            [5 features]
│   ├── patchcraft_analyzer.py # Rich/poor texture contrast (PatchCraft-inspired) [3 features]
│   ├── classifier.py          # SVM classifier + FeatureExtractor (54 features total)
│   └── utils.py               # Image loading, validation, dataset utilities
├── tests/
│   ├── test_fft_analyzer.py
│   ├── test_eigen_analyzer.py
│   ├── test_metadata_extractor.py
│   └── test_classifier.py
├── data/
│   ├── real/                        # 50 real camera images (training)
│   ├── ai_generated/                # 51 AI images (training)
│   ├── screenshots/                 # 14 real screenshots (training)
│   ├── ai_generated_screenshots/    # 10 AI screenshots (training)
│   └── test/
│       ├── real/                    # 10 test real images
│       └── ai_generated/            # 10 test AI images
├── models/
│   └── svm_classifier.pkl           # Trained SVM model
├── papers/                          # PDF copies of research papers
├── main.py                          # CLI entry point
├── app.py                           # Web interface (Flask)
├── WORKFLOW.md                      # This file
├── RESEARCH.md                      # Research notes and future directions
└── TODO.md                          # Known issues and failure modes
```

---

## Usage

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train the SVM

```bash
python main.py --train
```
Extracts features from `data/real/`, `data/ai_generated/`, and screenshot dirs. Trains SVM with 5-fold cross-validation. Saves to `models/svm_classifier.pkl`.

### Evaluate on test set

```bash
python main.py --evaluate
```

### Analyze a single image

```bash
python main.py --image path/to/image.jpg
```
Outputs classification, confidence, all 8 analyzer scores, and a human-readable explanation.

### Batch analysis

```bash
python main.py --batch path/to/directory/
```

### Run tests

```bash
pytest tests/ -v
```

---

## Performance

| Metric | Value |
|--------|-------|
| Cross-validation accuracy | **99.2%** (±1.6%) |
| Training accuracy | 100% |
| Training set | 125 images (50 real + 51 AI + 14 real screenshots + 10 AI screenshots) |
| Test set accuracy (10+10) | ~100% on lab distribution |
| Total features | 54 |
| Total analyzers | 8 |

> ⚠️ **Caveat:** These numbers reflect the training distribution (camera photos vs. AI art). Accuracy drops significantly on social-media-sourced images (metadata stripped, recompressed) and screenshots. See *Known Failure Modes* above.

---

## Key Design Decisions

1. **Center-crop instead of resize** for FFT analysis — resizing distorts frequency content
2. **Azimuthal averaging** reduces 2D spectrum to 1D for robust slope estimation
3. **Patch-based eigenvalue analysis** captures local texture variation, not just global stats
4. **8×8 DCT blocks** align with JPEG encoding grid
5. **ELA is PNG-aware** — PNGs are scored differently since first-compression artifacts are expected
6. **SVM with RBF kernel** — simple, effective for small datasets, no deep learning needed
7. **Fallback voting mode** allows the tool to work even without a trained model
8. **54 features, StandardScaler** — all features normalized before SVM to prevent scale dominance
9. **Multi-scale noise ratios + chroma correlations** use relative measures that survive JPEG recompression
10. **JPEG boundary features vectorized with numpy** — `np.diff` + boolean mask, not Python loops

---

## Relevant Research Papers for Future Work

- **ITW-SM dataset** (arXiv 2024) — *"Navigating the Challenges of AI-Generated Image Detection in the Wild: What Truly Matters?"* — Provides a dataset of real/AI images collected directly from Facebook, Instagram, LinkedIn, X. Key finding: training data composition matters more than model architecture for in-the-wild accuracy.
- **Recaptured image detection** — Research on screenshots as an *anti-forensics* technique (deliberately recapturing manipulated images to erase traces). Opposite of our problem but exposes the same gap: screenshot = forensically ambiguous.
- **JPEG dimples** (Berkeley) — Periodic DCT coefficient artifacts from specific mathematical operations in JPEG encoding. A potential future signal to distinguish original-encoding patterns.
- **RIGID** (arXiv May 2024) — Training-free detection via noise perturbation in vision model embedding space. Could complement our classical approach without needing more training data.
