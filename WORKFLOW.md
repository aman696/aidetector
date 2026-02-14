# AI Image Detector — Complete Workflow

This document explains the complete architecture, algorithms, and usage of the AI Image Detector. If you're coming back to this code after a long time, read this first.

---

## Architecture Overview

```
Image → [FFT Analyzer] → 4 features
      → [Eigen Analyzer] → 12 features   → [22-feature vector] → [SVM Classifier] → Real / AI-Generated
      → [Metadata Extractor] → 6 features
```

Three independent analysis methods extract a 22-dimensional feature vector from each image. A trained SVM with RBF kernel makes the final binary classification. If no trained model is available, a weighted voting fallback uses individual analyzer scores.

---

## Method 1: FFT Frequency Analysis (`src/fft_analyzer.py`)

**Paper:** Durall et al. 2020 — "Unmasking DeepFakes with simple Features" (arXiv: 1911.00686)

**Key Insight:** Natural images follow a ~1/f power law in the frequency domain. AI-generated images deviate from this, showing a characteristic drop-off at high frequencies.

**Pipeline:**
1. Load image as grayscale, center-crop to square
2. Apply 2D FFT (`np.fft.fft2`) → shift DC to center (`fftshift`)
3. Compute log-magnitude spectrum
4. **Azimuthal average**: average magnitude over concentric rings → 1D radial power spectrum
5. Extract 4 features:
   - `spectral_slope`: slope of log-log power spectrum (real images ≈ -1)
   - `slope_r_squared`: goodness of fit to power law
   - `high_freq_ratio`: energy in outer 50% of frequencies / total
   - `spectral_falloff`: ratio of Q4 to Q2 mean energy (steeper = more AI)

**Score:** Weighted combination of slope deviation, high-freq deficit, and falloff.

---

## Method 2: Eigenvalue/Spectral Analysis (`src/eigen_analyzer.py`)

**Paper:** Corvi et al. 2023 — "Intriguing Properties of Synthetic Images" (arXiv: 2304.06408)

**Key Insight:** Synthetic images have different covariance structure and frequency band distributions than natural images.

**Pipeline:**
1. Load image in BGR (OpenCV default)
2. **Global covariance**: reshape to (N, 3) RGB pixels → compute 3×3 covariance → eigenvalues → ratios
3. **Patch analysis**: divide into 64×64 patches → compute eigenvalue stats per patch → aggregate
4. **Spectral bands**: 2D FFT → split into low/mid/high frequency bands → energy ratios
5. Extract 12 features total (4 ratio + 4 patch + 4 band)

**Score:** Weighted combination of dominance deviation, patch uniformity, high-freq deficit, and condition number.

---

## Method 3: Metadata Forensics (`src/metadata_extractor.py`)

**Key Insight:** Real camera photos contain rich EXIF metadata (Make, Model, GPS, FocalLength, etc.). AI-generated images have none.

**Pipeline:**
1. Extract EXIF data using `Pillow`
2. Categorize tags: camera tags, software tags, context tags (GPS, timestamps)
3. Extract 6 features: tag count, camera tag count, boolean flags for camera/GPS/timestamps/software
4. Score based on presence/absence of metadata, with PNG-aware leniency

---

## Method 4: SVM Classifier (`src/classifier.py`)

**Pipeline:**
1. `FeatureExtractor` runs all 3 analyzers → builds 22-feature vector
2. `StandardScaler` normalizes features
3. `SVM(kernel='rbf', C=10)` makes binary classification (0=Real, 1=AI)
4. Returns label, confidence (probability), individual scores, and human-readable explanation

**Fallback:** Without a trained model, uses weighted voting: `0.35*fft + 0.35*eigen + 0.30*metadata > 0.5 → AI`

---

## Directory Structure

```
aidetector/
├── src/
│   ├── fft_analyzer.py        # FFT frequency analysis
│   ├── eigen_analyzer.py      # Eigenvalue/spectral analysis
│   ├── metadata_extractor.py  # EXIF metadata forensics
│   ├── classifier.py          # SVM classifier + feature extraction
│   └── utils.py               # Image loading, validation, dataset utilities
├── tests/
│   ├── test_fft_analyzer.py
│   ├── test_eigen_analyzer.py
│   ├── test_metadata_extractor.py
│   └── test_classifier.py
├── data/
│   ├── real/                  # 50 real images (training)
│   ├── ai_generated/          # 51 AI images (training)
│   └── test/                  # 10+10 test images
│       ├── real/
│       └── ai_generated/
├── models/
│   └── svm_classifier.pkl     # Trained SVM model
├── main.py                    # CLI entry point
├── requirements.txt
└── WORKFLOW.md                # This file
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
Extracts features from `data/real/` and `data/ai_generated/`, trains SVM with 5-fold cross-validation, saves model to `models/svm_classifier.pkl`.

### Evaluate on test set

```bash
python main.py --evaluate
```
Runs on `data/test/real/` and `data/test/ai_generated/`, prints classification report.

### Analyze a single image

```bash
python main.py --image path/to/image.jpg
```
Outputs classification (Real/AI-Generated), confidence, individual analyzer scores, and human-readable explanation.

### Batch analysis

```bash
python main.py --batch path/to/directory/
```
Analyzes all images in a directory and prints summary.

### Run tests

```bash
pytest tests/ -v
```

---

## Performance

| Metric | Value |
|--------|-------|
| Cross-validation accuracy | 98.0% (+/- 2.4%) |
| Training accuracy | 100% |
| Test accuracy (10+10) | 100% |
| Target | >70% |

---

## Key Design Decisions

1. **Center-crop instead of resize** for FFT analysis — resizing distorts frequency content
2. **Azimuthal averaging** reduces 2D spectrum to 1D for robust slope estimation
3. **Patch-based eigenvalue analysis** captures local texture variation, not just global stats
4. **SVM with RBF kernel** — simple, effective for small datasets, no deep learning needed
5. **Fallback voting mode** allows the tool to work even without a trained model
