# Model Card — AI Image Detector (unified v2)

Following the structure of Mitchell et al. 2019, "Model Cards for Model
Reporting." All quantitative values are sourced from the reproducibility record
[experiment_v1.json](experiment_v1.json) and the held-out evaluation report; no
numbers are hand-entered. Regenerate the source numbers with
`python -m src.evaluate_unified` and refresh the record with
`python -m scripts.record_experiment`.

> NOTE TO OWNER: the "Intended Use" and "Out of Scope" sections below are
> outward-facing claims about what this model is for. They are drafted from the
> existing README/SECURITY framing — review and edit them before this card is
> merged to master or published.

## Model Details

- **Name / version:** unified v2 (`models/unified_v2.pkl`), with an 85-feature
  classical-only fallback (`models/classical_v2.pkl`).
- **Date:** trained 2026-06-15 (post code-audit fixes).
- **Type:** calibrated Support Vector Machine (RBF kernel) over a fixed,
  855-dimensional feature vector. Not an end-to-end neural network.
- **Feature vector (855 dims):** 85 hand-crafted classical forensic features
  (FFT power spectrum, RGB-covariance eigenvalues + spectral bands, EXIF
  metadata, PRNU-style noise residuals, DCT/JPEG statistics, ELA, gradient
  statistics, PatchCraft texture, NPR up-sampling residue, screenshot
  forensics, RIGID-style perturbation drift) + 768 frozen DINOv2 ViT-B/14
  embedding dims + 2 RIGID embedding-drift dims.
- **Classifier:** `SVC(kernel="rbf", C=10.0, gamma=0.001, class_weight="balanced")`,
  wrapped in `CalibratedClassifierCV(method="sigmoid", ensemble=False)` with a
  **group-aware** calibration CV (StratifiedGroupKFold on `base_id`). The
  classical-only fallback is `SVC(rbf, C=1.0, gamma="scale")`.
- **Embedding model:** `vit_base_patch14_dinov2.lvd142m` (timm), frozen.
- **Input:** a single still image (PNG/JPEG/WebP). **Output:** calibrated
  probability in [0, 1] that the image is AI-generated.
- **Versioning anchors:** dataset `72b88efc0497`, classical feature version
  `01ae6c390125` (formula epoch 2), model bundle version 2. See
  [experiment_v1.json](experiment_v1.json).

## Intended Use

- **Primary use:** a free, research/educational demonstrator (live at
  humanorai.online) that estimates whether a still image was **fully generated**
  by a text-to-image model.
- **Users:** the general public (self-serve web demo) and the project owner for
  research/evaluation.
- **Intended decision support, not adjudication:** the output is an advisory
  probability, not proof. It must not be used as sole evidence for any
  consequential decision about a person or a piece of content.

## Out of Scope (do not use for)

- **Forensic, legal, journalistic, or moderation decisions** where a wrong call
  has real consequences. This is a demo/research tool.
- **Deepfakes / face-swaps / localized photo edits / inpainting.** It detects
  *fully* AI-generated images, not manipulated real photographs.
- **Video / video-frame captures.** Instagram Reel / TikTok / YouTube frame
  screenshots are misclassified (video-codec textures are a separate problem).
- **Attribution** (which generator produced an image). It outputs real-vs-AI
  only.

## Factors

Evaluation is stratified along the factors that move performance:

- **Distribution condition:** clean download, social-media recompression
  (Facebook / X / Telegram), screenshot capture, and chained transforms.
- **Generator architecture:** U-Net diffusion, pixel diffusion, rectified-flow /
  DiT, autoregressive, undisclosed.
- **Resolution:** min-side buckets `<400 / 400-800 / >800` px.
- **Training exposure:** five generator families are held out of training
  entirely (see Evaluation Data).

## Metrics

- **Threshold metrics @ 0.5:** Accuracy, Precision, Recall, F1 (positive class =
  AI). These depend on calibration, so they are reported only after the
  group-aware calibration fix.
- **Ranking metrics:** ROC-AUC and PR-AUC (average precision) — threshold-free.
- **Operating-point metric:** **Pd@5%FAR** — detection rate (fraction of AI
  images flagged) at the threshold where only 5% of real images are
  false-flagged. This is the most decision-relevant number for a public tool,
  where real-image false positives are the binding constraint.

## Training Data

- **Base images:** 4,271 total — 2,271 AI (34 generator families) + 2,000 real
  (COCO, OpenFake).
- **Augmentation / derived records:** 23,577 platform-emulated and screenshot
  variants (Facebook/X/Telegram recompression, screenshot capture, two chained
  pipelines), so the model sees in-the-wild distortions during training.
- **Assembled training rows:** 16,593 (clean + a leak-safe subset of derived
  variants per base image).
- **Splits:** assigned at **base-image level** (seed 42; 70% train) so a base
  image and all its derived variants stay in one split — no near-duplicate
  leakage. CV groups by `base_id`.

## Evaluation Data

- **Test split:** 6,414 rows (3,216 AI / 3,198 real), disjoint from training at
  the base-image level, spanning all distribution conditions.
- **Held-out generators:** `midjourney_7`, `ideogram_3.0`, `imagen_4.0`,
  `flux_1`, `recraft_v3` are absent from training (2,520-row holdout), measuring
  generalization to unseen generators.

## Quantitative Analyses

**Overall (test split, n = 6,414):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Pd@5%FAR |
|---|---|---|---|---|---|---|---|
| Unified (855) | 0.864 | 0.864 | 0.866 | 0.865 | 0.940 | 0.940 | 0.715 |
| Classical-only (85) | 0.781 | 0.745 | 0.857 | 0.797 | 0.863 | 0.856 | 0.436 |

The hybrid is well-balanced (precision ≈ recall). The classical-only fallback is
weaker and over-flags toward AI; the embedding closes most of that gap.

**By distribution condition (unified ROC-AUC):** clean 0.947, screenshot 0.946,
X 0.945, Telegram 0.939, Facebook 0.930, chain (ss→tg) 0.935, chain (fb→x) 0.930.
Performance holds up across social-media recompression.

**By generator architecture (real-anchored ROC-AUC; see RESEARCH.md):** U-Net
diffusion 0.951, pixel diffusion 0.985, autoregressive 0.940, undisclosed 0.922,
rectified-flow/DiT 0.894 (0.910 on seen flow; held-out Flux 0.811). Flow /
flux and the (untested) continuous-token AR are the weak spots.

**By resolution (unified ROC-AUC):** `<400` 0.974, `400-800` 0.926, `>800` 0.947.

**Leakage / shortcut checks:** EXIF-permutation AUC drop ≈ 0.000 (not leaning on
metadata); the RIGID drift was fixed to remove a file-format confound.

## Ethical Considerations

- **False positives harm real creators.** A real photo flagged as AI can damage
  reputation; the tool is tuned and reported at a fixed low false-alarm rate
  (5% FAR) for this reason, and outputs are advisory only.
- **Adversarial fragility.** Detection degrades under heavy recompression, low
  resolution, and novel architectures; a motivated actor can evade it. Do not
  treat a "real" verdict as a guarantee of authenticity.
- **Distribution bias.** Real images are COCO/OpenFake photographs; performance
  on out-of-distribution real content (art, screenshots of documents, scientific
  imagery) is not characterized.
- **Privacy.** The live service deletes uploaded images immediately after
  scanning and never stores or trains on them (see SECURITY.md).

## Caveats and Limitations

- **Rectified-flow models (Flux, SD3) are the headline blind spot** — held-out
  Flux ranks at 0.811 AUC and ~chance Pd; the DINOv2 embedding carries a
  diffusion-era bias that pulls novel-architecture scores toward the boundary.
- **Continuous-token autoregressive** generators (NextStep/MAR) are untested —
  the predicted next blind spot.
- **Below ~256 px** the FFT/PatchCraft features lose reliability.
- **Real-image false positives are the binding operating constraint** at a fixed
  FAR, not the AI-side threshold.
- **AI images with injected EXIF + grain filters** can partially fool the
  metadata/noise features.
- The shipped **v1** `.pkl` models (`svm_classifier.pkl`,
  `screenshot_classifier.pkl`) are a reference baseline trained on a different,
  now-removed dataset; they are not this model and are not compared here.

## Reproducibility

Full inputs, seeds, hashes, hyperparameters, and target metrics are recorded in
[experiment_v1.json](experiment_v1.json). To recreate: restore `data/` to the
recorded dataset hash, run `python -m src.train_unified --gpu --n-jobs -1`, then
`python -m src.evaluate_unified`, and compare against the recorded metrics.
Methodology and failure-mode detail: [RESEARCH.md](RESEARCH.md); per-feature
rationale: `code_notes/` (local).
