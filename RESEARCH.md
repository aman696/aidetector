# AI Image Detection — Literature Review & Research Notes

> **Last updated:** June 16, 2026
> **Goal:** Track the landscape of AI image detection, what's implemented, what's confirmed broken, and what to build next.

---

## Current Implementation Status

The detector is a single **unified model**: an 855-dimensional feature vector
(85 classical forensic features + 768 frozen DINOv2 ViT-B/14 dims + 2 RIGID
embedding-drift dims) feeding `SVC(rbf, C=10, gamma=0.001)` with group-aware
sigmoid calibration. An 85-feature classical-only model is the fallback when
PyTorch is unavailable. Both are trained on the current dataset (4,271 base
images across 34 generator families + reals, 23,577 derived variants); headline
metrics are in [README.md](README.md), the held-out evaluation detail is below
and in [MODEL_CARD.md](MODEL_CARD.md).

The classical features below are the same forensic signals used since v1; in the
unified model they are 85 of the 855 dimensions rather than a standalone
classifier. The separate v1 screenshot SVM is retired — the unified model is
trained on screenshots directly.

| Analyzer | Paper Basis | Features | Status |
|---|---|---|---|
| FFT power spectrum + slope | Durall et al. 2020 (arXiv:1911.00686) | 6 | Implemented |
| Eigenvalue + Spectral bands | Corvi et al. 2023 (arXiv:2304.06408) | 8 | Implemented |
| Metadata | Standard EXIF forensics | 6 | Implemented |
| Noise residual + chroma corr. | PRNU-inspired + Gragnaniello 2021 | 11 | Implemented |
| DCT block + JPEG boundary | Frank et al. 2020 + JPEG forensics | 8 | Implemented |
| ELA | Digital forensics literature | 5 | Implemented |
| Gradient statistics | Gragnaniello CVPR 2023 | 5 | Implemented |
| PatchCraft texture | Zhong et al. 2023 (arXiv:2311.12397) | 7 | Implemented (simplified) |
| NPR up-sampling residue | Tan et al. 2024 | 6 | Implemented |
| Screenshot forensics (GLCM/LBP/wavelet) | Ng et al., Thongkamwitoon et al. | 8 | Implemented |
| RIGID classical drift | RIGID 2024 (arXiv) | 15 | Implemented (classical approx.) |
| DINOv2 ViT-B/14 embedding (frozen) | Oquab et al. 2023 | 768 | Implemented |
| RIGID embedding drift | He et al. 2024 | 2 | Implemented |

**Training pipeline:** base-level leak-safe splits (seed 42), derived-variant
augmentation (Facebook/X/Telegram recompression, screenshot capture, chaining),
group-aware calibration.
**Backend:** cuML GPU SVC (optional) for the grid search + multiprocess CPU
feature extraction; per-id feature caches.

### Held-out evaluation (current dataset)

Test split (6,414 rows: 3,216 AI / 3,198 real), disjoint from training at the
base-image level. Full breakdowns by condition / architecture / resolution are in
[MODEL_CARD.md](MODEL_CARD.md) and `reports/eval_v2_<date>.{json,md}`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Pd@5%FAR |
|---|---|---|---|---|---|---|---|
| Unified (855) | 0.864 | 0.864 | 0.866 | 0.865 | 0.940 | 0.940 | 0.715 |
| Classical-only (85) | 0.781 | 0.745 | 0.857 | 0.797 | 0.863 | 0.856 | 0.436 |

Caveat on sample size: the 6,414 test rows descend from ~802 held-out base
images (clean + 7 derived conditions per base), so rows from one base image are
correlated. The effective number of independent units is closer to 802 than
6,414; treat the point metrics accordingly (base-image-level bootstrap
confidence intervals are not yet computed). The per-condition tables below
(n = 802 each) are the cleaner comparison.

### Acceptance gates (status as of the 2026-06-15 run)

These are absolute target gates defined in `src/evaluate_unified.py` (not
relative to the v1 baseline). The current model PASSES the social-media and
robustness gates but does NOT meet three of them; this is recorded here rather
than hidden. Source: `reports/eval_v2_20260615.md`.

| Gate | Target | Value | Result |
|---|---|---|---|
| clean AUC | >= 0.95 | 0.947 | FAIL (narrow) |
| clean accuracy | >= 0.90 | 0.867 | FAIL |
| screenshot accuracy | >= 0.85 | 0.870 | PASS |
| holdout AUC | >= 0.85 | n/a | n/a (holdout is all-AI; AUC undefined) |
| holdout Pd@5%FAR | >= 0.60 | 0.513 | FAIL (held-out generators) |
| facebook / x / telegram AUC | >= 0.88 | 0.930 / 0.945 / 0.939 | PASS |
| chain_fb_x / chain_ss_tg accuracy | >= 0.75 | 0.855 / 0.859 | PASS |
| unified AUC >= classical-only AUC | -- | 0.940 vs 0.863 | PASS |

The two clean-condition misses are small (AUC 0.947 vs 0.95; accuracy 0.867 vs
0.90). The held-out-generator Pd shortfall is the substantive one and is the
same rectified-flow / unseen-architecture weakness analysed below: novel
generators rank above chance but at low confidence, so they fall under a
5%-FAR threshold. Generalization to unseen architectures is a known open
weakness, not a solved problem.

---

## Generator Architecture vs Detectability (v2 family analysis, 2026-06-15)

> Numbers refreshed 2026-06-15 after the code-audit fixes (FFT power spectrum,
> eigen bands, signed-Laplacian variance, DCT AC-only variance, in-memory RIGID
> drift) and the group-aware-calibration retrain. The architecture story is
> unchanged from the 2026-06-13 run; the digits moved by <=0.01 except where
> noted (the freq-forensic probe shifted more, since those feature values
> changed). Prior run: `reports/family_analysis_20260613.{md,json}`.

Detailed artifact: `reports/family_analysis_20260615.{md,json}`.
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
| U-Net diffusion (udiff) | 0.951 | 0.753 | 19 families |
| pixel diffusion (GLIDE) | 0.985 | 0.923 | 1 family |
| rectified-flow / DiT | 0.894 (seen 0.910) | 0.487 | flux holdout drags it |
| autoregressive | 0.940 | 0.693 | Aurora/Gemini ~0.99, GPT ~0.90 |
| undisclosed | 0.922 (seen 0.935) | 0.651 | MJ7/recraft_v3 holdout |

Key corrections to the earlier "diffusion is cleared, other architectures are
blind spots" framing:

1. **The dramatic weakness was a threshold/confidence artifact, not blindness.**
 Flux's headline accuracy was 0.540 (≈ chance) at the 0.5 threshold, but its
 *ranking* AUC is 0.811 — it ranks Flux images above reals far better than
 chance; it just assigns them low confidence (mean p(AI) 0.55, right at the
 boundary). Across flow models mean p(AI) spreads 0.55–0.87 and AR-GPT 0.69,
 so the weaker ones fall just under the 0.5-calibrated threshold → low
 Pd@5%FAR (flow 0.49 vs udiff 0.75) despite AUC ~0.89.

2. **The DINOv2 embedding is part of the cause — it carries a diffusion bias.**
 Per-family embedding gain (hybrid AUC − classical-only AUC) is *negative*
 on the flow and the novel/held-out families: flux −0.031, chroma −0.033,
 SD3 −0.030, recraft_v3 −0.058, gpt −0.062 — while strongly positive on
 classic diffusion: DALL·E 2 +0.311, grok +0.114, SD 1.5 +0.109, SD 1.3
 +0.104, GLIDE +0.080. The embedding encodes diffusion-era cues that pull
 novel-architecture scores toward the boundary. The classical frequency
 forensics generalize better: a group-aware linear probe on the freq-forensic
 features alone gets flow AUC 0.754 vs udiff 0.831, where the embedding-only
 probe gets flow 0.802. (Note: the freq-forensic AUCs dropped from the
 2026-06-13 run — flow 0.850→0.754, udiff 0.882→0.831 — because the FFT and
 eigen feature *values* changed in the audit fix; the udiff>flow ordering
 held.)

3. **Architecture matters modestly; exposure matters too; the worst case is
 both.** Even *seen* flow models (SD3 0.907, chroma 0.878, hidream 0.883) rank
 ~0.04 AUC below seen diffusion (seen-flow mean 0.910 vs seen-udiff 0.953).
 Held-out *diffusion* families generalize well (ideogram_3.0 0.946, imagen_4.0
 0.925) while held-out flux (flow) is 0.811 — held-out-diffusion minus
 held-out-non-diffusion AUC = +0.086. Flux is both a novel architecture and
 unseen, which is why it is the single weakest family.

4. **Autoregressive is not a ranking blind spot.** Aurora 0.990 and Gemini 0.985
 (both AR) are among the most detectable; GPT (0.899, mean p 0.69) is the
 weakest AR family but still ranks well — its low accuracy is a confidence +
 small-sample effect (only 62 GPT base images), not architectural invisibility.

**Implications (recommendations, evidence-tagged; no retrain done yet):**
- *Calibration / threshold (low effort, high operational gain):* the model
 already ranks flow/AR images correctly, so recalibrating or lowering the
 decision threshold (globally or per-architecture) recovers much of the Pd loss
 without retraining. Supported by finding 1.
- *Training-data composition (medium effort, principled fix):* add rectified-flow
 (Flux, SD3, more chroma/hidream) and AR (more GPT — currently thin) images so
 the embedding's diffusion bias is corrected; held-out flux at 0.811 marks the
 true unseen-flow generalization gap. Supported by findings 2–3.
- *Flow-specific features (now a bigger lever):* classical freq-forensics reach
 only ~0.75 on flow after the audit fix (was ~0.85), so the freq features lost
 some flow signal; bespoke flow artifact features are a larger lever than the
 pre-fix numbers suggested. Supported by finding 2 (probe).

---

## Autoregressive (AR) detection: does it generalize? (2026-06-15)

> Refreshed 2026-06-15 on the post-audit-fix retrain (prior run 2026-06-13). The
> generalization and artifact conclusions are unchanged; numbers updated below.

Artifacts: `reports/ar_experiments_20260615.{md,json}`,
`reports/ar_artifact_probe_20260615.md`. Run: `python -m scripts.ar_experiments`,
`python -m scripts.ar_artifact_probe`.

Motivation: every AR family in the data (gpt, gpt_image_1, gemini, aurora, grok)
was *seen* in training, so the strong AR numbers could have been memorization.
Two questions: (1) what artifact makes AR detectable, and (2) does detection
transfer to *unseen* AR?

**What artifact (W2).** Not an AR-specific signature. Image-level probes found
token-grid periodicity **weak/per-model only** (faint in GPT/gpt_image_1,
near-zero or negative in Aurora/Gemini, and actually highest in a *diffusion*
family, ideogram_3.0 +0.0047) and raster-scan anisotropy **null** (AR 0.134 vs
real 0.130). The
operative cue is the **decoder upsampling / super-resolution residue** that
VQ-AR and hybrid-AR (GPT-4o = AR backbone + diffusion head, per GPT-ImgEval,
where NPR scores 99%) share with latent-diffusion VAE decoders. So the real axis
is **"has a learned upsampling decoder vs not"**, not "diffusion vs AR".

**Does it generalize (W4 E2).** Yes. A linear probe trained on the leak-safe
train split with **all AR families removed** still scores **real-anchored AUC
0.892** on the held-out AR families (vs 0.928 with AR seen — generalization gap
only **0.036**). Flow generalizes similarly (0.823 unseen, gap 0.036); removing
diffusion hurts most (udiff gap 0.087), consistent with diffusion supplying the
bulk of the shared-fingerprint signal. **Implication: we do not need new AR data
to claim AR generalization** — the detector catches autoregressive images it
never trained on, because the upsampling-decoder fingerprint transfers from
diffusion/flow. The one untested case is **continuous-token AR** (NextStep/MAR,
no VQ grid) — the predicted genuine blind spot, absent from every dataset family.

**Calibration (W4 E3).** At the shipped 0.5 threshold, detection per architecture
bucket is moderate-good (ar Pd 0.88, flow 0.73, udiff 0.90). The 5%-FAR threshold
is 0.759 (down from 0.916 pre-fix — the retrained model's real scores are lower),
but it still sits well above the flow mass (mean p(AI) ~0.69), so the binding
constraint remains **real false positives**, not the AI-side threshold —
recalibration alone will not lift flow (Pd 0.52 at 5%FAR); reducing real FPs would.

**External out-of-distribution check (W4 E1).** Scored the shipped model on an
independent OpenFake subset (gpt-image-1, sd-3.5, flux.2, + laion/pexels reals —
collection never in our pipeline; `reports/ar_external_eval_20260615.md`).
Real-anchored AUC: ar (gpt-image-1, n=7) **0.777**, flow (sd-3.5 + flux.2, n=60)
**0.627** (flux.2-dev 0.564 ~ chance), reals mean p(AI) 0.468. This is now a
**full-resolution** run: the earlier 2026-06-13 run downscaled to 512px and was
flagged as a possible lower bound, but the full-res numbers are essentially
identical (ar 0.788→0.777, flow 0.619→0.627), so that caveat is resolved —
native-resolution upsampling traces were not the limiting factor. AR holds up
OOD (supports the generalization thesis), while flow — including our *seen*
architecture SD3 on an independent collection (0.640) and the newer flux.2
(~chance) — is the confirmed blind spot. Reals are not over-flagged.

**Draft contribution statement (for a writeup).** *Architecture-stratified
detectability of AI-generated images: a black-box forensic + DINOv2 detector
generalizes across the diffusion family and to unseen autoregressive generators
(leave-AR-out AUC 0.892) because latent-diffusion, VQ-autoregressive, and hybrid
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
**Status: IMPLEMENTED** — `augment_dataset_with_jpeg()` in `src/utils.py`, called during `python main.py --train`

Built a dataset of 10,000 images from Facebook, Instagram, LinkedIn, and X (native compression preserved; memes/screenshots/watermarked images filtered out). Key findings: detectors that excel on curated benchmarks degrade significantly in the wild, and naively scaling training data or model size does not fix it — but optimizing backbone, training-data composition, cropping (not resizing!), and augmentations together recovers **+26.87% average AUC**. (Note: earlier versions of this doc misquoted 26.87% as the *loss*; it is the *improvement*.)

**What we implemented:** For each training image, add JPEG copies at Q=70 and Q=80 + a 0.75× downscaled copy (4× the base set — historical counts varied as the dataset grew; exact numbers will be re-recorded after the current retraining on the expanded dataset). Teaches the SVM that "JPEG-compressed Real photo ≠ AI". Files cleaned up via `cleanup_augmented_files()` after training.

**Effect confirmed:** Training accuracy dropped from 99.2%→84.8% on the harder augmented set — which is correct. The old 99.2% was overfitting on easy images; 84.8% on augmented images reflects real-world harder conditions.

---

### 7. RIGID — 2024
**arXiv (2024)** — "RIGID: A Training-free and Model-Agnostic Framework for Robust AI-Generated Image Detection" 
**Status: Implemented (classical approximation)** — `_compute_drift_features()` in `src/classifier.py`, 15 of the 85 classical features

**Core Idea:** Real images are more robust to tiny noise perturbations than AI-generated images in DINOv2 feature space.

**What we implemented (without DINOv2):**
1. Add Gaussian noise (σ=2) to image → save to temp file
2. Re-run FFT, Noise, Gradient, DCT, PatchCraft, Eigen analyzers on perturbed image
3. Return `|original_features − perturbed_features|` for 15 key features
4. These 15 "drift" values are 15 of the 85 classical features

Real images: low drift (features are stable under noise). AI images: somewhat higher drift. Contributes a training-free generalization signal.

**What we did NOT implement:** The DINOv2/ViT backbone (requires ~300MB model, out of scope for classical pipeline).

---

### 8. Wang et al. 2020 — "CNN-generated images are surprisingly easy to spot... for now"
**CVPR 2020** 
**Status: PARTIALLY IMPLEMENTED** — data augmentation principle applied via ITW-SM implementation; ResNet-50 classifier out of scope

Wang et al. showed that adding JPEG-compressed and Gaussian-blurred training images significantly improves robustness to unseen generators. We applied this principle in our `augment_dataset_with_jpeg()` (Q=70, Q=80, 0.75× resize). The ResNet-50 detector itself was not implemented (classical-only pipeline).

---

### 9. Dedicated Screenshot Classifier — March 2026
**Based on:** Ng et al. (Imperial College) recaptured image forensics + "Any-Resolution AI Detection by Spectral Learning" (arXiv Nov 2024)
**Status: IMPLEMENTED** — `src/screenshot_classifier.py`, `models/screenshot_classifier.pkl`

**Why this mattered (v1 history):**
Screenshots go through a display pipeline (monitor gamma, panel quantization, screenshot PNG encoding) that erases EXIF, camera sensor noise, and the JPEG grid. In v1 this justified a separate screenshot SVM. In v2 these 15 signals are part of the unified model's 85 classical features and screenshots are trained on directly, so no separate model or routing is used.

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

**Shortcoming — Instagram Reel screenshots:** Video-frame captures (from Instagram, TikTok, YouTube) have smooth H.264/H.265 compressed content — low wavelet energy, unimodal histogram, low GLCM contrast — all identical to AI image screenshots. This remains a hard unsolved case.

> **Note (v2):** the standalone screenshot classifier is retired. These 15
> screenshot-forensic signals now live inside the unified model's 85 classical
> features; screenshots are trained on directly rather than routed to a separate
> model.

---

## What's Actually Broken (From Real Testing)

| Failure | Confirmed | Root Cause | Fix Status |
|---|---|---|---|
| Screenshot of real content → AI (80%+) with main SVM | Confirmed | No EXIF + no camera noise + no JPEG grid | Fixed — dedicated screenshot SVM |
| AI screenshot → Real with main SVM | Confirmed | Display pipeline adds real-camera-like noise | Fixed — dedicated screenshot SVM |
| Instagram Reel screenshot → AI (both models) | Confirmed | H.264 video codec = smooth frames, no sensor noise, no bimodal histogram | Open — 3rd/4th category problem |
| Low-res AI (~256px) → Real/Uncertain | Confirmed | PatchCraft needs ≥36 patches; multi-scale collapses | Partially mitigated by resolution guard |
| Social media screenshot of real photo → AI | Confirmed | EXIF stripped, display-rendered | Partially mitigated by JPEG augmentation |
| AI image with grain filter + injected EXIF → Real | Expected | Noise looks camera-like, low metadata score | Partially mitigated by RIGID drift features |
| Video frames from Seedance/ByteDance | Observed | Video codec (H.264/H.265) differs from image generators | Open |
| AI images with injected / faked EXIF | Confirmed | Trivial to use ExifTool to inject fake camera metadata | Partial: RIGID drift and PatchCraft not fooled by EXIF |

### Limitations (Not Hard Failures)

**Low-resolution images (<256px):** `patchcraft_analyzer.py` has a resolution guard (`min(h,w) < 256 → return zeros`) — returns neutral features, not AI-biased. FFT slopes become unreliable below ~300px but don't crash. Effect: reduced confidence on small images, not systematic misclassification.

---

## Priority Queue / Still Planned

| Priority | Item | Status |
|---|---|---|
| 1 | ITW-SM training augmentation (Q=70, Q=80, 0.75× resize) | `augment_dataset_with_jpeg()` |
| 2 | Resolution guard for PatchCraft + noise multi-scale | Guards added |
| 3 | Screenshot pre-detection + web UI toggle | `screenshot_detector.py` + web toggle |
| 4 | RIGID-inspired feature drift (54→69 features) | `_compute_drift_features()` |
| 5 | GPU-accelerated SVM training | cuML RTX 4060 + ProcessPoolExecutor |
| 6 | Screenshot forensics features for main SVM (10 features) | `screenshot_image_analyzer.py` |
| 7 | **Dedicated screenshot SVM** | `src/screenshot_classifier.py` |
| 8 | **Playwright real screenshot generator** | `scripts/generate_real_screenshots.py` |
| 9 | **Social media dataset integration** (252 AI + 109 real) | Integrated |
| 10 | **Analyzer calibration fixes** (FFT slope, DCT kurtosis, eigenvalue capping) | Fixed |

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
5. **Desktop/app screenshots** — handled directly by the unified model; no separate mode needed
6. **Video-frame captures (Reels, TikTok)** — neither model handles these reliably; manually inspect

---

## Key Insights 

1. **Our FFT targets GAN artifacts, not diffusion.** Durall 2020 was designed for transposed-convolution checkerboard patterns. Diffusion models don't produce those. But the VAE bottleneck in latent diffusion models creates a high-frequency attenuation signature — which our screenshot SVM's `fft_radial_slope` feature captures.

2. **Metadata is the strongest signal but also the most fragile.** Social media strips it. Tools like `exiftool` can inject fake cameras. Yet it still contributes heavily to the main SVM.

3. **PatchCraft generalizes across 17 generator types** — the best-generalizing classical feature we have. The texture-synthesis limitation appears to be fundamental to how all current generators work.

4. **Training data composition > model complexity** (per ITW-SM 2025). We don't need a fancier SVM — we need training images that reflect the actual distribution we're being tested on.

5. **Screenshots are effectively a third class, not a subset of Real or AI.** v1 approximated this with a separate screenshot SVM and routing. v2 instead trains the single unified model on screenshot variants directly (the screenshot-forensic signals are part of its classical features), which removed the routing logic while keeping screenshot handling.

6. **RIGID drift without DINOv2 still adds value.** Even our classical feature-drift approximation (σ=2 perturbation → |Δfeatures|) adds 15 training-free generalization features. The principle works independently of the backbone.

7. **Video-frame captures are genuinely unsolvable with 2-class binary classification.** Instagram Reels, TikTok, YouTube screenshots are neither "camera photo" nor "AI image" nor "desktop UI screenshot" — they're a fourth category with video-codec textures. This is an open research problem.

---

## Performance Summary (current dataset)

Held-out test split (6,414 rows: 3,216 AI / 3,198 real), base-level disjoint from
training. Source: `reports/eval_v2_<date>.json`, recorded in `experiment_v1.json`.
Per-condition / per-architecture / per-resolution detail is in
[MODEL_CARD.md](MODEL_CARD.md) and the family-analysis section above.

### Unified model (855-dim)
| Metric | Value |
|---|---|
| Accuracy | 0.864 |
| Precision / Recall / F1 | 0.864 / 0.866 / 0.865 |
| ROC-AUC / PR-AUC | 0.940 / 0.940 |
| Pd@5%FAR | 0.715 |
| Train rows | 16,593 |
| Features | 855 (85 classical + 768 DINOv2 + 2 drift) |

### Classical-only fallback (85-dim)
| Metric | Value |
|---|---|
| Accuracy | 0.781 |
| ROC-AUC / PR-AUC | 0.863 / 0.856 |
| Pd@5%FAR | 0.436 |
| Features | 85 |

The hybrid is well-balanced (precision ~ recall); the classical-only fallback is
weaker and over-flags toward AI. Real-image false positives are the binding
operating constraint, which is why Pd@5%FAR is reported.

### Earlier v1 models (reference baseline only)
The `models/svm_classifier.pkl` (79-feature) and `models/screenshot_classifier.pkl`
(15-feature) models were trained on a different, now-removed dataset and are not
comparable to the numbers above. They are kept only as a historical reference and
are not loaded by `main.py` or `app.py`.