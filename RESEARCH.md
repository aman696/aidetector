# AI Image Detection — Literature Review & Research Notes

> **Date:** February 15, 2025  
> **Goal:** Understand the landscape of AI image detection techniques, identify what works for modern diffusion models, and determine which methods to apply to improve our detector.

---

## Papers Read

### 1. Durall et al. 2020 — "Unmasking DeepFakes with simple Features"
**Citation:** arXiv:1911.00686 (ICML 2020 Workshop)

**Core Idea:** GANs fail to correctly reproduce the spectral distributions of natural images. Real images follow a ~1/f power law in the frequency domain; GAN images deviate systematically at high frequencies.

**Method:**
- Compute 2D FFT → azimuthal average → 1D radial power spectrum
- Fit slope in log-log space — natural images have slope ≈ -1
- Use deviation from natural slope as a detection signal
- Simple classifier (even a threshold) achieves 100% on high-res face datasets

**Key Findings:**
- Works well on **GANs** (ProGAN, StyleGAN, DCGAN)
- Achieved 100% accuracy with as few as 20 training samples on Faces-HQ
- 91% accuracy on low-resolution FaceForensics++ video frames

**Limitations (critical for us):**
- **Designed for GANs, not diffusion models.** Modern generators (DALL-E 3, Gemini, Flux) use diffusion, not GAN upsampling
- Diffusion models don't have the same checkerboard artifacts from transposed convolution
- This explains why our FFT analyzer shows "Real" for almost all images — it's detecting a GAN-specific artifact that doesn't exist in diffusion outputs

**Insight for our project:** FFT spectral slope alone is insufficient for diffusion models. Need to look at different frequency features or combine with other approaches.

---

### 2. Frank et al. 2020 — "Leveraging Frequency Analysis for Deep Fake Image Recognition"
**Citation:** ICML 2020

**Core Idea:** Use **Discrete Cosine Transform (DCT)** instead of FFT for frequency analysis. DCT coefficients from image blocks can be linearly separated.

**Method:**
- Divide image into 8×8 blocks (like JPEG encoding)
- Apply 2D DCT to each block
- Aggregate DCT coefficient statistics across blocks
- Use ridge regression classifier on DCT features

**Key Findings:**
- GAN-generated images show systematic artifacts in DCT coefficients
- Artifacts are consistent across GAN architectures, datasets, and resolutions
- Artifacts are caused by upsampling operations (transposed convolutions)
- DCT features can be **linearly separated** — no need for complex classifiers

**Insight for our project:** DCT analysis is more interpretable than raw FFT and aligns with JPEG's native compression domain. We should add DCT-based features — they capture block-level patterns that FFT's azimuthal average misses.

---

### 3. Corvi et al. 2023 — "Intriguing Properties of Synthetic Images"
**Citation:** CVPRW 2023 (arXiv:2304.06408)

**Core Idea:** Both GANs and diffusion models produce visible artifacts in the Fourier domain, but the nature of artifacts differs. Diffusion models have a "frequency bias" — they struggle to reproduce high frequencies and fine details.

**Method:**
- Analyze radial and angular spectral power distributions
- Compare frequency band energy distributions (low/mid/high)
- Train ResNet-50 on frequency-domain representations

**Key Findings:**
- Diffusion model images show distinct mid-high frequency anomalies
- Autocorrelation reveals "artificial fingerprints" in synthetic images
- Training data biases (e.g., JPEG compression in training set) transfer to generated images
- Spectral analysis works for both GANs and diffusion models, but needs **different features** for each

**Insight for our project:** Our eigen_analyzer's spectral band analysis is inspired by this, but we should add:
1. **Autocorrelation analysis** — detect periodic patterns in frequency domain
2. **Angular spectral analysis** — not just radial (azimuthal) averaging
3. **Mid-high frequency energy ratio** as a distinct feature (we have this but need better thresholds)

---

### 4. Wang et al. 2020 — "CNN-generated images are surprisingly easy to spot... for now"
**Citation:** CVPR 2020

**Core Idea:** A classifier trained on images from **one** CNN generator (ProGAN) can generalize to detect images from many other unseen generators.

**Method:**
- Train a standard ResNet-50 classifier on ProGAN real/fake pairs
- Test on 11 different generators (StyleGAN, BigGAN, CycleGAN, etc.)
- Apply data augmentation (blur, JPEG compression) during training for robustness

**Key Findings:**
- Surprising cross-generator generalization — CNN generators share common artifacts
- Data augmentation during training is **critical** for generalization
- JPEG compression and Gaussian blur augmentation improved unseen-generator accuracy significantly
- However, **this generalization does NOT extend well to diffusion models** (later studies showed)

**Insight for our project:** We should apply data augmentation (blur, compression, resize) to our training pipeline. Also, training on diverse generators is important — our dataset only has Gemini and Kling images.

---

### 5. Ojha et al. 2023 — "Towards Universal Fake Image Detectors"
**Citation:** CVPR 2023

**Core Idea:** Instead of learning real-vs-fake features, use a **frozen pre-trained CLIP** feature space. CLIP features, even without fine-tuning for detection, can discriminate real from fake.

**Method:**
- Extract features from CLIP's vision encoder (ViT-L/14)
- Use nearest-neighbor or linear probing on frozen features
- Train only on ProGAN images, test across GANs and diffusion models

**Key Findings:**
- CLIP features generalize dramatically better than CNN-specific detectors
- +15 mAP improvement over prior SOTA on unseen diffusion/autoregressive models
- The key is that CLIP learned general "naturalness" features from internet-scale data
- Even a linear probe on CLIP features outperforms complex trained detectors

**Why this matters but is out of scope for us:**
- Requires large pre-trained model (~400MB+ ViT-L)
- We're building a lightweight rule-based system, not a deep learning pipeline
- However, if we ever want to add a neural component, CLIP linear probing is the way to go

---

### 6. Gragnaniello et al. 2021 — "Are GAN Generated Images Easy to Detect?"
**Citation:** IEEE ICME 2021

**Core Idea:** Detection robustness to real-world conditions (JPEG compression, social media upload) is far more important than raw accuracy on clean data.

**Method:**
- Extract chrominance features and residual domain features
- Test under domain mismatch (train on one GAN, test on another)
- Evaluate under JPEG compression at various quality levels

**Key Findings:**
- Chrominance (color channel) features are more robust than luminance for detection
- Residual domain analysis (denoising filter then analyzing the residual) helps
- JPEG compression significantly degrades detection — pixel-level artifacts are destroyed
- Training with compression augmentation vastly improves robustness
- One-class classification (trained on real images only) is feasible using learned features

**Insight for our project:**
1. We should analyze **YCbCr color space** separately — chrominance channels may carry stronger signals
2. **Residual noise analysis** — apply a denoising filter, then analyze the difference (residual)
3. We should augment training with JPEG-compressed versions to improve robustness

---

### 7. Error Level Analysis (ELA)
**Source:** Digital forensics literature, various implementations

**Core Idea:** Re-save a JPEG image at known quality → compare to original → the difference reveals compression inconsistencies.

**Method:**
1. Re-save image at fixed JPEG quality (e.g., 95%)
2. Compute absolute pixel difference between original and re-saved
3. Analyze variance of the ELA map — uniform = consistent compression history

**Key Properties:**
- Originally designed for image tampering detection (spliced regions have different compression levels)
- AI images often have **uniform ELA** (never been JPEG-compressed before) vs real photos (compressed by camera)
- PNG images always have uniform ELA → not useful for AI PNGs
- Works best when comparing JPEG vs JPEG

**Insight for our project:**
- ELA mean/variance as additional features could help distinguish camera-JPEG vs never-compressed AI images
- Only useful for JPEG inputs — we need to handle PNGs differently
- Simple to implement with OpenCV

---

### 8. PRNU (Photo Response Non-Uniformity) Analysis
**Source:** Lukáš et al. 2006, Li 2010

**Core Idea:** Camera sensors have a unique "fingerprint" — a fixed noise pattern from manufacturing defects. AI images lack this.

**Method:**
1. Apply a denoising filter (wavelet, BM3D, or Wiener)
2. Subtract denoised from original → get noise residual
3. Real camera photos have structured sensor noise; AI images have random/structured-differently noise

**Key Properties:**
- PRNU is like a camera fingerprint — each sensor is unique
- AI images have no sensor → no PRNU pattern → noise residual looks different
- Can compute noise residual variance, kurtosis, spectral properties

**Insight for our project:** 
- **Noise residual statistics** (variance, kurtosis, spectral entropy) as new features
- Simple implementation: Gaussian blur → subtract → analyze residual
- This directly addresses the screenshot problem: screenshots have different noise characteristics than both camera photos and AI images

---

### 9. Wavelet/DCT Kurtosis-Based Noise Analysis
**Source:** Mahdian & Saic (CAS Prague), multiple forensics papers

**Core Idea:** In band-pass filtered domains (wavelets, DCT), natural images have kurtosis values that concentrate around a constant. Synthetic/manipulated regions deviate.

**Method:**
1. Apply wavelet decomposition or block DCT
2. Compute kurtosis of coefficients in each sub-band/block
3. Regions with inconsistent kurtosis → likely tampered or synthetic

**Key Properties:**
- PCA can be applied to blocks for noise estimation, even in textured regions
- Wavelet HH1 sub-band captures high-frequency noise characteristics
- Block-based analysis can detect local inconsistencies

**Insight for our project:** Add wavelet-domain kurtosis as features. We already do block-based eigenvalue analysis — extending to wavelet sub-band statistics is straightforward.

---

## Summary: What Works and What Doesn't

| Technique | Works on GANs? | Works on Diffusion? | Our Status | Notes |
|-----------|:-:|:-:|---|---|
| FFT spectral slope (Durall) | ✅ Strong | ❌ Weak | ⚠ Implemented but ineffective | Designed for GAN upsampling artifacts |
| DCT block analysis (Frank) | ✅ Strong | ⚡ Moderate | ❌ Not implemented | Block-level features, linearly separable |
| Spectral band analysis (Corvi) | ✅ Strong | ✅ Strong | ✅ Implemented | Mid-high freq deficit is key |
| CLIP features (Ojha) | ✅ Strong | ✅ Strong | ❌ Not implemented | Requires large model, out of scope |
| ELA (various) | ⚡ Moderate | ⚡ Moderate | ❌ Not implemented | JPEG-only, simple to add |
| PRNU/noise residual | ✅ Strong | ✅ Strong | ❌ Not implemented | Universal, addresses screenshot issue |
| Chrominance analysis (Gragnaniello) | ✅ Strong | ⚡ Moderate | ❌ Not implemented | Color channel separation |
| Wavelet kurtosis | ⚡ Moderate | ⚡ Moderate | ❌ Not implemented | Good for local inconsistency |
| Metadata (ours) | ✅ Strong | ✅ Strong | ✅ Implemented | Effective but trivially fakeable |

---

## What We Will Apply (Prioritized)

### Priority 1 — Noise Residual Analysis (High Impact)
**Why:** Directly addresses screenshot false-positive problem. Camera photos have sensor noise; AI images have structured/uniform noise; screenshots have no sensor noise but different patterns.
**How:**
- Apply Gaussian/median denoising → compute noise residual
- Extract: residual variance, kurtosis, spectral entropy, spatial autocorrelation
- ~4-6 new features for the SVM

### Priority 2 — DCT Block Analysis (High Impact)
**Why:** More appropriate than FFT for modern images. Works at block level (like JPEG), captures local patterns. Frank 2020 showed linear separability.
**How:**
- Divide into 8×8 blocks → 2D DCT per block
- Compute: coefficient statistics (mean, var, kurtosis of AC coefficients), energy distribution
- ~4-6 new features

### Priority 3 — ELA Features (Medium Impact, JPEG-only)
**Why:** Simple to implement, captures compression history differences.
**How:**
- Re-save at quality=95 → compute difference map
- Extract: ELA mean, variance, range, uniformity
- ~3-4 new features (only for JPEG inputs; default values for PNG)

### Priority 4 — Improved FFT Features (Medium Impact)
**Why:** Current FFT features target GAN artifacts. Need to retune for diffusion models.
**How:**
- Add angular (directional) spectral analysis instead of just radial
- Add autocorrelation-based periodicity detection
- Retune spectral slope thresholds based on diffusion-model characteristics

### Priority 5 — YCbCr Chrominance Features (Lower Priority)
**Why:** Gragnaniello showed chrominance is more robust than luminance for detection.
**How:**
- Convert to YCbCr → analyze Cb and Cr channels separately
- Compute variance ratios, correlation between channels

---

## Key Takeaways

1. **Our FFT is targeting the wrong artifact.** Durall's spectral slope works for GANs but diffusion models don't exhibit the same high-frequency dropout. This is why FFT shows "Real" for everything.

2. **Metadata is our strongest feature but also the weakest link.** It's trivially fakeable — anyone can inject EXIF data, and screenshots/social media strip it.

3. **Noise residual analysis is the most promising addition.** It works across all generator types and addresses our screenshot problem.

4. **DCT is better suited than FFT for block-level analysis.** JPEG images are already DCT-encoded, so analyzing DCT coefficients is natural and aligned with how images are stored.

5. **Data augmentation during training is essential.** We should add JPEG-compressed and resized versions of our training images.

6. **We need more diverse training data.** 50+51 images from only 2 generators (Gemini, Kling) is not enough for generalization.
