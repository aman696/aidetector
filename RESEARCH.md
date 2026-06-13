# AI Image Detection — Literature Review & Research Notes

> **Last updated:** June 11, 2026
> **Goal:** Track the landscape of AI image detection, what's implemented, what's confirmed broken, and what to build next.

---

## Current Implementation Status

> 🔄 **RETRAINING PHASE (June 2026):** the training dataset has been expanded — a much wider
> family of generators (`data/mixed/` now holds 32 model-family subdirectories) and a far larger
> image count (`data/{Gemini, GPT, mixed, real}`). All accuracy numbers in this document describe
> models trained on the **previous, smaller dataset** and will be refreshed after retraining.

As of today, the detector uses a **dual-model architecture**:
- **Main SVM:** 9 analyzers producing 79 features (54 base + 10 screenshot-forensics + 15 RIGID drift). Pre-expansion CV accuracy: ~74–77% (sources disagreed; will be re-measured after retraining).
- **Screenshot SVM:** 15 features tuned for screen-rendered content, trained on a small screenshot-only dataset. Pre-expansion CV accuracy: ~88–98% (sources disagreed; will be re-measured after retraining).

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
| Screenshot image forensics (main SVM) | Ng et al. (Imperial), Thongkamwitoon et al. | 10 | ✅ Implemented |
| RIGID drift features | RIGID 2024 (arXiv) | 15 | ✅ Implemented (classical approx.) |
| **Dedicated Screenshot SVM** | **Ng et al + "Any-Resolution Spectral Learning" (Nov 2024)** | **15** | ✅ **`src/screenshot_classifier.py`** |

**Training pipeline:** ITW-SM 2025 augmentation (Q=70, Q=80, 0.75× resize); training-set size in flux — see retraining note above
**Backend:** cuML GPU SVM (RTX 4060) + 16-core parallel CPU feature extraction; screenshot SVM on CPU in ~30s

---

## Generator Architecture vs Detectability (v2 family analysis, 2026-06-13)

Detailed artifact: `reports/family_analysis_20260613.{md,json}`; method and the
architecture map: `code_notes/20-family-analysis.md`, `paper_notes/architectures.md`.
Run: `python -m scripts.analyze_families`. This analysis re-scores the cached
test/holdout matrices with the shipped v2 models and groups the 32 generator
families by *generation architecture* (U-Net diffusion / pixel diffusion /
rectified-flow-DiT / autoregressive / undisclosed), correcting two confounds in
the headline eval: per-family AUC was undefined (each family is all-AI), and the
per-family accuracies mixed architecture with training-exposure and resolution.

**The eval's per-family accuracy was misleading; ranking is robust across
architectures.** Real-anchored AUC (each family's AI scores vs the pooled real
images), clean condition:

| bucket | mean AUC | mean Pd@5%FAR | note |
|---|---|---|---|
| U-Net diffusion (udiff) | 0.954 | 0.773 | 19 families |
| pixel diffusion (GLIDE) | 0.964 | 0.769 | 1 family |
| rectified-flow / DiT | 0.918 (seen 0.918) | 0.512 | flux holdout drags it |
| autoregressive | 0.943 | 0.724 | Aurora/Gemini ~0.99, GPT 0.90 |
| undisclosed | 0.923 | 0.661 | MJ7/recraft_v3 holdout |

Key corrections to the earlier "diffusion is cleared, other architectures are
blind spots" framing:

1. **The dramatic weakness was a threshold/confidence artifact, not blindness.**
   Flux's headline accuracy was 0.522 (≈ chance) at the 0.5 threshold, but its
   *ranking* AUC is 0.820 — it ranks Flux images above reals far better than
   chance; it just assigns them low confidence (mean p(AI) 0.57, right at the
   boundary). Across flow models mean p(AI) clusters 0.75–0.79 and AR-GPT 0.74,
   so they fall just under the 0.5-calibrated threshold → low Pd@5%FAR (flow
   0.51 vs udiff 0.77) despite AUC ~0.90.

2. **The DINOv2 embedding is part of the cause — it carries a diffusion bias.**
   Per-family embedding gain (hybrid AUC − classical-only AUC) is *negative*
   exactly on the flow/AR/holdout families: flux −0.072, gpt −0.049, chroma
   −0.043, hidream −0.032, SD3 −0.031 — while strongly positive on classic
   diffusion: GLIDE +0.308, DALL·E 2 +0.251, SD 1.3 +0.084. The embedding
   encodes diffusion-era cues that pull novel-architecture scores toward the
   boundary. The classical frequency forensics generalize better: a group-aware
   linear probe on the freq-forensic features alone gets flow AUC 0.850 vs udiff
   0.882 (small gap), where the embedding-only probe gets flow 0.802.

3. **Architecture matters modestly; exposure matters too; the worst case is
   both.** Even *seen* flow models (SD3 0.905, chroma 0.882, hidream 0.913) rank
   ~0.04 AUC below seen diffusion. Held-out *diffusion* families generalize well
   (ideogram_3.0 0.952, imagen_4.0 0.932) while held-out flux (flow) is 0.820 —
   held-out-diffusion minus held-out-non-diffusion AUC = +0.082. Flux is both a
   novel architecture and unseen, which is why it is the single weakest family.

4. **Autoregressive is not a ranking blind spot.** Aurora 0.990 and Gemini 0.987
   (both AR) are among the most detectable; GPT (0.896, mean p 0.74) is the
   weakest AR family but still ranks well — its low accuracy is a confidence +
   small-sample effect (only 62 GPT base images), not architectural invisibility.

**Implications (recommendations, evidence-tagged; no retrain done yet):**
- *Calibration / threshold (low effort, high operational gain):* the model
  already ranks flow/AR images correctly, so recalibrating or lowering the
  decision threshold (globally or per-architecture) recovers much of the Pd loss
  without retraining. Supported by finding 1.
- *Training-data composition (medium effort, principled fix):* add rectified-flow
  (Flux, SD3, more chroma/hidream) and AR (more GPT — currently thin) images so
  the embedding's diffusion bias is corrected; held-out flux at 0.820 marks the
  true unseen-flow generalization gap. Supported by findings 2–3.
- *Flow-specific features (low priority):* classical freq-forensics already reach
  ~0.85 on flow, so bespoke flow artifact features are a smaller lever than the
  two above. Supported by finding 2 (probe).

---

## Autoregressive (AR) detection: does it generalize? (2026-06-13)

Artifacts: `reports/ar_experiments_20260613.{md,json}`,
`reports/ar_artifact_probe_20260613.md`; method/literature:
`code_notes/21-ar-artifacts.md`, `paper_notes/ar-detection.md`,
`paper_notes/architectures.md`. Run: `python -m scripts.ar_experiments`,
`python -m scripts.ar_artifact_probe`.

Motivation: every AR family in the data (gpt, gpt_image_1, gemini, aurora, grok)
was *seen* in training, so the strong AR numbers could have been memorization.
Two questions: (1) what artifact makes AR detectable, and (2) does detection
transfer to *unseen* AR?

**What artifact (W2).** Not an AR-specific signature. Image-level probes found
token-grid periodicity **weak/per-model only** (faint in Aurora, absent in
GPT/Gemini) and raster-scan anisotropy **null** (AR 0.134 vs real 0.130). The
operative cue is the **decoder upsampling / super-resolution residue** that
VQ-AR and hybrid-AR (GPT-4o = AR backbone + diffusion head, per GPT-ImgEval,
where NPR scores 99%) share with latent-diffusion VAE decoders. So the real axis
is **"has a learned upsampling decoder vs not"**, not "diffusion vs AR".

**Does it generalize (W4 E2).** Yes. A linear probe trained on the leak-safe
train split with **all AR families removed** still scores **real-anchored AUC
0.905** on the held-out AR families (vs 0.937 with AR seen — generalization gap
only **0.033**). Flow generalizes similarly (0.839 unseen, gap 0.040); removing
diffusion hurts most (udiff gap 0.084), consistent with diffusion supplying the
bulk of the shared-fingerprint signal. **Implication: we do not need new AR data
to claim AR generalization** — the detector catches autoregressive images it
never trained on, because the upsampling-decoder fingerprint transfers from
diffusion/flow. The one untested case is **continuous-token AR** (NextStep/MAR,
no VQ grid) — the predicted genuine blind spot, absent from every dataset family.

**Calibration (W4 E3).** At the shipped 0.5 threshold, detection per architecture
bucket is moderate-good (ar Pd 0.90, flow 0.77, udiff 0.91). The 5%-FAR threshold
sits high (0.916) because the openfake reals score high, so the binding
constraint is **real false positives**, not the AI-side threshold — recalibration
alone will not lift flow; reducing real FPs would.

**External out-of-distribution check (W4 E1).** Scored the shipped model on an
independent OpenFake subset (gpt-image-1, sd-3.5, flux.2, + laion/pexels reals —
collection never in our pipeline; `reports/ar_external_eval_20260613.md`).
Real-anchored AUC: ar (gpt-image-1, n=7) **0.788**, flow (sd-3.5 + flux.2, n=60)
**0.619** (flux.2-dev 0.593 ~ chance), reals mean p(AI) 0.433. **Caveat:** this
run downscaled images to 512px for speed, which smears the native-resolution
upsampling traces the detector relies on, so these are a *lower bound* (a full-res
run would likely score higher) — flagged honestly, not a clean number. Even so:
AR holds up OOD (supports the generalization thesis), while flow — including our
*seen* architecture SD3 on an independent collection (0.625) and the newer flux.2
(~chance) — is the confirmed blind spot. Reals are not over-flagged.

**Draft contribution statement (for a writeup).** *Architecture-stratified
detectability of AI-generated images: a black-box forensic + DINOv2 detector
generalizes across the diffusion family and to unseen autoregressive generators
(leave-AR-out AUC 0.905) because latent-diffusion, VQ-autoregressive, and hybrid
models share a decoder upsampling fingerprint; the residual blind spots are
rectified-flow models with cleaner VAEs and continuous-token AR. Unlike PRADA
(white-box token-probability ratios), this needs no model access.* Open work:
acquire continuous-token AR images to test the predicted blind spot; reduce
real-image false positives (openfake).

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

### 6. ITW-SM — Konstantinidou et al. 2025
**arXiv:2507.10236** | ITI-CERTH  
**Status: ✅ IMPLEMENTED** — `augment_dataset_with_jpeg()` in `src/utils.py`, called during `python main.py --train`

Built a dataset of 10,000 images from Facebook, Instagram, LinkedIn, and X (native compression preserved; memes/screenshots/watermarked images filtered out). Key findings: detectors that excel on curated benchmarks degrade significantly in the wild, and naively scaling training data or model size does not fix it — but optimizing backbone, training-data composition, cropping (not resizing!), and augmentations together recovers **+26.87% average AUC**. (Note: earlier versions of this doc misquoted 26.87% as the *loss*; it is the *improvement*.)

**What we implemented:** For each training image, add JPEG copies at Q=70 and Q=80 + a 0.75× downscaled copy (4× the base set — historical counts varied as the dataset grew; exact numbers will be re-recorded after the current retraining on the expanded dataset). Teaches the SVM that "JPEG-compressed Real photo ≠ AI". Files cleaned up via `cleanup_augmented_files()` after training.

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

### 9. Dedicated Screenshot Classifier — March 2026
**Based on:** Ng et al. (Imperial College) recaptured image forensics + "Any-Resolution AI Detection by Spectral Learning" (arXiv Nov 2024)
**Status: ✅ IMPLEMENTED** — `src/screenshot_classifier.py`, `models/screenshot_classifier.pkl`

**Why a separate model is needed:**
Screenshots go through a display pipeline (monitor gamma, panel quantization, screenshot PNG encoding) that erases the signals the main 79-feature SVM relies on — EXIF is always absent, camera sensor noise is always absent, JPEG grid is always absent. Forcing the main SVM to handle screenshots creates a category-confusion problem.

**Feature vector (15 features):**

| Feature | Basis | Why it survives screenshots |
|---|---|---|
| GLCM homogeneity @ d=1,3,7 | Ng et al. (recaptured image forensics) | Texture regularity survives display pipeline |
| GLCM contrast @ d=1,3,7 | Ng et al. | Real UI: sharp text edges → high contrast |
| Haar wavelet HH energy @ L=1,2,3 | Thongkamwitoon et al. | Diagonal detail energy differs real UI vs AI |
| LBP entropy @ r=1,2 | Recaptured image forensics | Local binary pattern entropy captures grain structure |
| FFT radial power slope | "Any-Resolution Spectral Learning" Nov 2024 | VAE bottleneck attenuates AI high-freq; signal persists through screenshot |
| Histogram bimodality | SCI (Screen Content Image) research | Real UI = dark bg + bright text = bimodal; AI image = bell curve |
| Histogram peak/valley ratio | SCI research | Same discriminant |
| Chroma std ratio | Our extension | Colour channel imbalance |

**Test results (unseen data):**
- AI screenshots (`data/ai_test/`, 29 images): **28/29 correct — 96.6%**
- Real screenshots (`data/real_test/`, 35 clean images): **~85%** (impacted by Instagram Reel video-frame captures accidentally included)

**Shortcoming — Instagram Reel screenshots:** Video-frame captures (from Instagram, TikTok, YouTube) have smooth H.264/H.265 compressed content — low wavelet energy, unimodal histogram, low GLCM contrast — all identical to AI image screenshots. This is a genuinely hard unsolved case requiring a 3-class approach. For now: these should not be included in real_test data.

**How to retrain:**
```bash
# Add more real screenshots to data/screenshots/
python scripts/generate_real_screenshots.py --mix --count 80 --out data/screenshots
python main.py --train-screenshot
```

---

## What's Actually Broken (From Real Testing)

| Failure | Confirmed | Root Cause | Fix Status |
|---|---|---|---|
| Screenshot of real content → AI (80%+) with main SVM | ✅ Confirmed | No EXIF + no camera noise + no JPEG grid | ✅ Fixed — dedicated screenshot SVM |
| AI screenshot → Real with main SVM | ✅ Confirmed | Display pipeline adds real-camera-like noise | ✅ Fixed — dedicated screenshot SVM |
| Instagram Reel screenshot → AI (both models) | ✅ Confirmed | H.264 video codec = smooth frames, no sensor noise, no bimodal histogram | ❌ Open — 3rd/4th category problem |
| Low-res AI (~256px) → Real/Uncertain | ✅ Confirmed | PatchCraft needs ≥36 patches; multi-scale collapses | ✅ Partially mitigated by resolution guard |
| Social media screenshot of real photo → AI | ✅ Confirmed | EXIF stripped, display-rendered | ✅ Partially mitigated by JPEG augmentation |
| AI image with grain filter + injected EXIF → Real | ⚠️ Expected | Noise looks camera-like, low metadata score | ✅ Partially mitigated by RIGID drift features |
| Video frames from Seedance/ByteDance | ⚠️ Observed | Video codec (H.264/H.265) differs from image generators | ❌ Open — use Screenshot mode as workaround |
| AI images with injected / faked EXIF | ✅ Confirmed | Trivial to use ExifTool to inject fake camera metadata | ⚠️ Partial: RIGID drift and PatchCraft not fooled by EXIF |

### Limitations (Not Hard Failures)

**Low-resolution images (<256px):** `patchcraft_analyzer.py` has a resolution guard (`min(h,w) < 256 → return zeros`) — returns neutral features, not AI-biased. FFT slopes become unreliable below ~300px but don't crash. Effect: reduced confidence on small images, not systematic misclassification.

---

## Priority Queue / Still Planned

| Priority | Item | Status |
|---|---|---|
| 1 | ITW-SM training augmentation (Q=70, Q=80, 0.75× resize) | ✅ `augment_dataset_with_jpeg()` |
| 2 | Resolution guard for PatchCraft + noise multi-scale | ✅ Guards added |
| 3 | Screenshot pre-detection + web UI toggle | ✅ `screenshot_detector.py` + web toggle |
| 4 | RIGID-inspired feature drift (54→69 features) | ✅ `_compute_drift_features()` |
| 5 | GPU-accelerated SVM training | ✅ cuML RTX 4060 + ProcessPoolExecutor |
| 6 | Screenshot forensics features for main SVM (10 features) | ✅ `screenshot_image_analyzer.py` |
| 7 | **Dedicated screenshot SVM** | ✅ `src/screenshot_classifier.py` |
| 8 | **Playwright real screenshot generator** | ✅ `scripts/generate_real_screenshots.py` |
| 9 | **Social media dataset integration** (252 AI + 109 real) | ✅ Integrated |
| 10 | **Analyzer calibration fixes** (FFT slope, DCT kurtosis, eigenvalue capping) | ✅ Fixed |

### Still Planned

#### Data
- [ ] Grow AI desktop screenshot training data — current sources unknown; need to assess
- [ ] Grow main training set — target 200 real + 200 AI downloaded images
  - Add: Ideogram, Recraft, Playground, Seedance, Gemini samples
  - Add: Flickr / personal photos in varied lighting
- [ ] Test cross-platform: TikTok, Facebook, YouTube thumbnail crops

#### Accuracy
- [ ] Calibrate SVM probability outputs (Platt scaling not applied on screenshot SVM)
- [ ] Adversarial test: manually JPEG-compress, crop, re-upload known-AI images

#### Web Interface
- [ ] Progress bar / step indicator during analysis (currently just spinner)
- [ ] Side-by-side comparison mode (upload two images)
- [ ] Export results as JSON or PDF report

#### Deployment
- [ ] Docker container for easy homelab deployment
- [ ] Rate limiting (if exposed to internet)
- [ ] HTTPS via nginx reverse proxy

---

## Tips for Best Results

1. **Crop to content only** — remove browser chrome, taskbars, watermarks, overlays
2. **Minimum 400×400 pixels** — smaller images degrade PatchCraft and FFT
3. **No heavy post-processing** — heavy Photoshop filters can fool the detector
4. **Use JPEG or PNG** — avoid heavy WebP compression before uploading
5. **Desktop/app screenshots** — always use `--screenshot-mode` flag or Screenshot toggle in web UI
6. **Video-frame captures (Reels, TikTok)** — neither model handles these reliably; manually inspect

---

## Key Insights for ISI Interview

1. **Our FFT targets GAN artifacts, not diffusion.** Durall 2020 was designed for transposed-convolution checkerboard patterns. Diffusion models don't produce those. But the VAE bottleneck in latent diffusion models creates a high-frequency attenuation signature — which our screenshot SVM's `fft_radial_slope` feature captures.

2. **Metadata is the strongest signal but also the most fragile.** Social media strips it. Tools like `exiftool` can inject fake cameras. Yet it still contributes heavily to the main SVM.

3. **PatchCraft generalizes across 17 generator types** — the best-generalizing classical feature we have. The texture-synthesis limitation appears to be fundamental to how all current generators work.

4. **Training data composition > model complexity** (per ITW-SM 2025). We don't need a fancier SVM — we need training images that reflect the actual distribution we're being tested on.

5. **Screenshots are a third class, not a subset of Real or AI.** The correct architecture is a 3-class detector: {Real, AI-Generated, Screenshot/Rendered}. We now approximate this with dual-model routing: a specialist screenshot SVM handles the screenshot subproblem independently.

6. **RIGID drift without DINOv2 still adds value.** Even our classical feature-drift approximation (σ=2 perturbation → |Δfeatures|) adds 15 training-free generalization features. The principle works independently of the backbone.

7. **Video-frame captures are genuinely unsolvable with 2-class binary classification.** Instagram Reels, TikTok, YouTube screenshots are neither "camera photo" nor "AI image" nor "desktop UI screenshot" — they're a fourth category with video-codec textures. This is an open research problem.

---

## Performance Summary (previous dataset — superseded by retraining phase)

> 🔄 These numbers come from the last training run on the **old, smaller dataset**. The dataset
> has since been expanded (more generator families, more images); retraining is pending and this
> whole section will be replaced with fresh numbers afterwards. Kept for reference only.

### Main SVM (pre-expansion)
| Metric | Value |
|---|---|
| CV accuracy | **74%** on ~2016 augmented images (a 77.0% figure also circulated — re-measure on retrain) |
| Test accuracy | **82.3%** on held-out set |
| Training base images | ~504 (50 real + 51 AI + 28 real SS + ~14 AI SS + 109 real social + 252 AI social) |
| Augmented training | ~2016 |
| Total features | **79** (54 base + 10 screenshot-forensics + 15 RIGID drift) |

### Screenshot SVM (pre-expansion)
| Metric | Value |
|---|---|
| CV accuracy | **~98%** — likely overfit (an 88.6% figure also circulated — re-measure on retrain) |
| Test — AI screenshots (29 images) | **96.6%** (28/29 ✅) |
| Test — Real screenshots (35 clean) | **~77–85%** (impacted by video-frame contamination) |
| Total features | **15** |
| Training time | ~30 seconds |

> ⚠️ The misclassified real screenshots in `data/real_test/` are Instagram Reel captures. Remove those and real accuracy is ~97%.