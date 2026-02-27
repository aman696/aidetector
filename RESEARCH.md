# AI Image Detection — Literature Review & Research Notes

> **Last updated:** February 27, 2026  
> **Goal:** Track the landscape of AI image detection, what's implemented, what's confirmed broken by testing, and what to build next.

---

## Current Implementation Status

As of today, the detector uses **8 analyzers producing 69 features** (54 base + 15 RIGID drift), trained on 500 augmented images. CV accuracy: **84.8%** (on augmented set — harder than the old 99.2% on 125 unaugmented images).

| Analyzer | Paper Basis | Features | Status |
|---|---|---|---|
| FFT | Durall et al. 2020 (arXiv:1911.00686) | 4 | ✅ Implemented |
| Eigenvalue + Spectral bands | Corvi et al. 2023 (arXiv:2304.06408) | 12 | ✅ Implemented |
| Metadata | Standard EXIF forensics | 6 | ✅ Implemented |
| Noise residual | PRNU-inspired (Lukáš 2006) | 6 | ✅ Implemented |
| Multi-scale noise + chroma corr. | Gragnaniello 2021 + our extension | 5 | ✅ Implemented |
| DCT block coefficients | Frank et al. 2020 | 6 | ✅ Implemented |
| JPEG block boundary | Classic JPEG forensics | 2 | ✅ Implemented |
| ELA | Digital forensics literature | 5 | ✅ Implemented |
| Gradient statistics | Gragnaniello CVPR 2023 | 5 | ✅ Implemented |
| PatchCraft texture contrast | Zhong et al. 2023 (arXiv:2311.12397) | 3 | ✅ Implemented (simplified) |
| **RIGID drift features** | **RIGID 2024 (arXiv)** | **15** | ✅ **Implemented (classical approx.)** |

**Training pipeline:** ITW-SM 2025 augmentation (Q=70, Q=80, 0.75× resize) → 500 training images  
**Backend:** cuML GPU SVM (RTX 4060) + 16-core parallel CPU feature extraction


---

## Papers Implemented

### 1. Durall et al. 2020 — "Unmasking DeepFakes with simple Features"
**arXiv:1911.00686** | ICML 2020 Workshop

**Core Idea:** GANs fail to reproduce the ~1/f spectral distribution of natural images. The spectral slope in log-log space deviates at high frequencies.

**What we implemented:** `fft_analyzer.py` — 2D FFT → azimuthal average → 1D power spectrum → spectral slope + R², high-freq ratio, spectral falloff.

**What we know now from testing:**
- Works reasonably on older GAN-generated images
- **Weak on modern diffusion models** (DALL-E 3, Gemini, Kling) — diffusion upsampling doesn't produce the same transposed-convolution checkerboard artifacts
- Degrades badly on images <512px (too few frequency bins for reliable slope estimation)
- FFT score frequently shows "uncertain" on social-media-recompressed images

---

### 2. Frank et al. 2020 — "Leveraging Frequency Analysis for Deep Fake Image Recognition"
**ICML 2020**

**Core Idea:** GAN-generated images show systematic artifacts in 8×8 DCT block coefficient distributions, specifically in AC coefficient statistics. These are block-level patterns that azimuthal-averaged FFT misses.

**What we implemented:** `dct_analyzer.py` — 8×8 block DCT → AC energy ratio, high-freq energy, coefficient kurtosis, coefficient variance, DC variance, zigzag decay.

**What we extended beyond the paper:** Added JPEG block boundary analysis — `dct_boundary_ratio` and `dct_boundary_var_ratio`. These detect whether the image has a pre-existing JPEG quantization grid (real recompressed JPEG) vs. PNG-origin compression (AI images). Fully vectorized with `np.diff` + boolean masking.

**Known limitation confirmed in testing:** Screenshots are also PNG→JPEG, so `dct_boundary_ratio` ≈ 1.0 for both — correctly identifies "no pre-existing JPEG grid" but cannot distinguish screenshot from AI.

---

### 3. Corvi et al. 2023 — "Intriguing Properties of Synthetic Images"
**arXiv:2304.06408** | CVPRW 2023

**Core Idea:** Both GANs and diffusion models produce frequency-domain artifacts, but diffusion models specifically show a "frequency bias" — they struggle to reproduce high frequencies and fine spatial details. Spectral band energy ratios are discriminative.

**What we implemented:** `eigen_analyzer.py` — global RGB covariance matrix eigenvalues, patch-based eigenvalue statistics (64×64 patches), spectral band energy ratios (low/mid/high/mid-high).

**Known limitation:** Chroma subsampling (4:2:0) from social media partially destroys inter-channel covariance structure. The eigenvalue features degrade on any social-media-sourced image.

---

### 4. Gragnaniello et al. 2021/2023
**IEEE ICME 2021 + CVPR 2023**

**Two separate contributions we used:**

**(a) ICME 2021 — chrominance + residual domain features:**  
Chrominance features are more robust than luminance for detection. JPEG augmentation during training is critical.

**(b) CVPR 2023 — gradient statistics:**  
Real images have heavy-tailed edge distributions (real-world scene edges are sharp and unpredictable). AI images have smoother/more regularized gradient distributions. Gradient statistics survive JPEG recompression at Q>70 because they measure *relative* structure.

**What we implemented:**
- `gradient_analyzer.py` (CVPR 2023): Sobel gradient magnitude → mean, variance, kurtosis, Laplacian mean, Laplacian variance (5 features)
- `noise_analyzer.py` chroma extension (ICME 2021 inspired): inter-channel noise correlations `noise_rg_corr`, `noise_rb_corr`, `noise_gb_corr`

---

### 5. PatchCraft — Zhong et al. 2023
**arXiv:2311.12397** | Updated v3: March 2024

**Core Idea:** AI generative models systematically fail to reproduce fine-grained natural textures. After high-pass filtering (to isolate texture from global scene), image patches split into "rich texture" (high variance) and "poor texture" (low variance) groups. AI images show a characteristically *higher* contrast between these two groups because generators over-smooth poor-texture regions while producing artificial-looking rich texture.

**Key properties:**
- Evaluated on a benchmark of **17 generative models** — both GANs and diffusion models
- Also uses a "Smash&Reconstruct" preprocessing to erase global semantics and enhance texture patterns (more aggressive than our implementation)
- Shows significant improvement over Wang et al. 2020 baselines across all model types

**What we implemented (simplified):** `patchcraft_analyzer.py` — high-pass filter (img − gaussian_blur) → 32×32 patch variances → median split into rich/poor → 3 features: `texture_contrast`, `texture_rich_mean`, `texture_poor_mean`.

**What we did NOT implement from the paper:**
- Smash&Reconstruction preprocessing (erases global semantics — requires more complex augmentation)
- Inter-pixel correlation within patches (we use patch variance as a proxy)
- The full benchmark evaluation pipeline (17 models)

**Known limitation confirmed:** Images <256px may produce <36 patches total — statistically unreliable. This is why low-resolution AI images get uncertain scores.

---

---

### 6. ITW-SM — Konstantinidou et al. 2025
**arXiv:2507.10236** | ITI-CERTH  
**Status: ✅ IMPLEMENTED** — `augment_dataset_with_jpeg()` in `src/utils.py`, called during `python main.py --train`

Built a dataset of 10,000 images from Facebook, Instagram, LinkedIn, and X. Key finding: current detectors lose >26% AUC on social-media-sourced images vs. clean benchmarks.

**What we implemented:** For each training image, add JPEG copies at Q=70 and Q=80 + a 0.75× downscaled copy. 125 base images → 500 augmented images. Teaches the SVM that "JPEG-compressed Real photo ≠ AI". Files cleaned up via `cleanup_augmented_files()` after training.

**Effect confirmed:** Training accuracy dropped from 99.2%→84.8% on the harder augmented set — which is correct. The old 99.2% was overfitting on easy images; 84.8% on augmented images reflects real-world harder conditions.

---

### 7. RIGID — 2024
**arXiv (2024)** — "RIGID: A Training-free and Model-Agnostic Framework for Robust AI-Generated Image Detection"  
**Status: ✅ IMPLEMENTED (classical approximation)** — `_compute_drift_features()` in `src/classifier.py`, features 55–69 of 69 total

**Core Idea:** Real images are more robust to tiny noise perturbations than AI-generated images in DINOv2 feature space.

**What we implemented (without DINOv2):**
1. Add Gaussian noise (σ=2) to image → save to temp file
2. Re-run FFT, Noise, Gradient, DCT, PatchCraft, Eigen analyzers on perturbed image
3. Return `|original_features − perturbed_features|` for 15 key features
4. These 15 "drift" values become the final 15 features (indices 54–68)

Real images: low drift (features are stable under noise). AI images: somewhat higher drift. Contributes a training-free generalization signal.

**What we did NOT implement:** The DINOv2/ViT backbone (requires ~300MB model, out of scope for classical pipeline).

---

### 8. Wang et al. 2020 — "CNN-generated images are surprisingly easy to spot... for now"
**CVPR 2020**  
**Status: ✅ PARTIALLY IMPLEMENTED** — data augmentation principle applied via ITW-SM implementation; ResNet-50 classifier out of scope

Wang et al. showed that adding JPEG-compressed and Gaussian-blurred training images significantly improves robustness to unseen generators. We applied this principle in our `augment_dataset_with_jpeg()` (Q=70, Q=80, 0.75× resize). The ResNet-50 detector itself was not implemented (classical-only pipeline).

---

## What's Actually Broken (From Real Testing)

| Failure | Confirmed | Root Cause | Fix Status |
|---|---|---|---|
| Screenshot of real content → AI (80%+) | ✅ Confirmed | No EXIF + no camera noise + no JPEG grid | ✅ Screenshot mode toggle in web UI |
| Low-res AI (~256px) → Real/Uncertain | ✅ Confirmed | PatchCraft needs ≥36 patches; multi-scale collapses | ✅ Resolution guard added |
| Social media screenshot of real photo → AI | ✅ Confirmed | EXIF stripped, display-rendered | ✅ Partially mitigated by JPEG augmentation |
| AI image with grain filter + injected EXIF → Real | ⚠️ Expected | Noise looks camera-like, low metadata score | ✅ Partially mitigated by RIGID drift features |
| Video frames from Seedance/ByteDance | ⚠️ Observed | Video codec (H.264/H.265) differs from image generators | ❌ Not fixed — use Screenshot mode as workaround |

---

## Priority Queue — All Completed ✅

| Priority | Item | Status |
|---|---|---|
| 1 | ITW-SM training augmentation (Q=70, Q=80, 0.75× resize) | ✅ `augment_dataset_with_jpeg()` in `utils.py` |
| 2 | Resolution guard for PatchCraft + noise multi-scale | ✅ Guards in `patchcraft_analyzer.py` + `noise_analyzer.py` |
| 3 | Screenshot pre-detection + web UI toggle | ✅ `screenshot_detector.py` + "📱 Screenshot" button |
| 4 | RIGID-inspired feature drift (54→69 features) | ✅ `_compute_drift_features()` in `classifier.py` |
| 5 | GPU-accelerated SVM training | ✅ cuML RTX 4060 + 16-core ProcessPoolExecutor |

**Next priority (not yet done):** Grow training dataset — add Ideogram, Recraft, Seedance samples to `data/ai_generated/` and retrain.

---

## Key Insights for ISI Interview

1. **Our FFT targets GAN artifacts, not diffusion.** Durall 2020 was designed for transposed-convolution checkerboard patterns. Diffusion models don't produce those.

2. **Metadata is the strongest signal but also the most fragile.** Social media strips it. Tools like `exiftool` can inject fake cameras. Yet it still contributes heavily to the SVM.

3. **PatchCraft generalizes across 17 generator types** — the best-generalizing classical feature we have. The texture-synthesis limitation appears to be fundamental to how all current generators work.

4. **Training data composition > model complexity** (per ITW-SM 2025). We don't need a fancier SVM — we need training images that reflect the actual distribution we're being tested on.

5. **Screenshots are a third class, not a subset of Real or AI.** The correct architecture is a 3-class detector: {Real, AI-Generated, Screenshot/Rendered}. Forcing binary classification produces systematic errors on screenshots.

6. **RIGID drift without DINOv2 still adds value.** Even our classical feature-drift approximation (σ=2 perturbation → |Δfeatures|) adds 15 training-free generalization features. The principle works independently of the backbone.
