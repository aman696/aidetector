# Model Card — AI Image Detector (unified v2)

Following the structure of Mitchell et al. 2019, "Model Cards for Model
Reporting." All quantitative values are sourced from the reproducibility record
[experiment_v1.json](experiment_v1.json) and the held-out evaluation report.
Regenerate evaluation with
`python -m src.evaluate_unified` after restoring the recorded dataset. The
original `scripts.record_experiment` helper is absent from this public checkout;
the committed record describes the shipped experiment, not a new run.

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

Current headline metrics are in [README.md](README.md). Full condition,
generator-family, resolution, and gate analyses are in [RESEARCH.md](RESEARCH.md).
The source measurements are [the June 15 evaluation](reports/eval_v2_20260615.json)
and [experiment_v1.json](experiment_v1.json); they are not independently copied
into another model-card table.

**Three acceptance gates remain unmet:** clean ROC-AUC, clean accuracy, and
held-out-generator detection at the 5% false-alarm reference threshold. Their
measured values and targets are recorded in the evaluation and research document.

The 6,414 test rows derive from 802 base images, so the rows are correlated;
base-image-level confidence intervals are not yet computed. The separate
holdout contains only AI images, so standalone holdout ROC-AUC is undefined;
real-anchored architecture comparisons use a separate reference of real scores.
The 5%-FAR statistic is evaluated on photographic real images and is not a
universal deployment false-positive guarantee or the inference decision rule.

## Ethical Considerations

- **False positives harm real creators.** A real photo flagged as AI can damage
  reputation; the evaluation reports detection at a 5% false-alarm reference
  threshold for this reason. Inference chooses the highest calibrated class
  probability; it does not apply that evaluation threshold. Outputs are advisory.
- **Adversarial fragility.** Detection degrades under heavy recompression, low
  resolution, and novel architectures; a motivated actor can evade it. Do not
  treat a "real" verdict as a guarantee of authenticity.
- **Distribution bias.** Real images are COCO/OpenFake photographs; performance
  on out-of-distribution real content (art, screenshots of documents, scientific
  imagery) is not characterized.
- **Privacy.** `/api/detect` writes a temporary scan file and removes it in
  `finally` after the request. This checkout has no contribution or persistent
  upload endpoint (see SECURITY.md).

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
Methodology and failure-mode detail: [RESEARCH.md](RESEARCH.md).
