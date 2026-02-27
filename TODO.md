# AI Image Detector — TODO & Known Issues

> **Last updated:** February 27, 2026

---

## ✅ Resolved (Implemented)

- [x] Screenshot pre-detection — `src/screenshot_detector.py` (3 heuristics, orange warning banner)
- [x] Web interface mode toggle — separate "Downloaded Image" vs "Screenshot" button
- [x] Rich/Poor texture contrast — `src/patchcraft_analyzer.py` (PatchCraft arXiv:2311.12397)
- [x] Image gradient statistics — `src/gradient_analyzer.py` (Gragnaniello CVPR 2023)
- [x] Multi-scale noise ratios — `src/noise_analyzer.py` (sigma=1,3,5 denoising)
- [x] JPEG block boundary consistency — `src/dct_analyzer.py`
- [x] Chroma noise channel correlation — `src/noise_analyzer.py`
- [x] JPEG / resize augmentation — `src/utils.py` `augment_dataset_with_jpeg()` (ITW-SM 2025)
- [x] Resolution guard for PatchCraft + noise multi-scale
- [x] RIGID-inspired drift features — 15 features, 54→69 total
- [x] GPU-accelerated SVM — cuML RTX 4060; parallel CPU extraction with ProcessPoolExecutor

---

## 🔴 Active Known Failure Modes

### 1. Video frame screenshots (e.g. Seedance / ByteDance)
- Screenshots of AI-generated videos are especially hard because:
  - The motion blur, post-processing, and video codec compression change frequency stats
  - Seedance / ByteDance video generators use techniques different from image generators
  - Video codec (H.264/H.265) introduces its own compression grid that confuses our DCT/ELA checks
- **Workaround:** Use "📱 Screenshot" mode in the web UI — forces the warning banner
- **Future fix:** Train a separate classifier on video-frame screenshots

### 2. Novel AI generators not in training set
- Ideogram, Recraft, Playground v3, Seedance — not in our 51-image AI training set
- Different generators have different artifacts — missing generator = reduced accuracy
- **Fix:** Add samples of each generator to `data/ai_generated/` and retrain

### 3. Screenshots still misclassified
- Some screenshots evade all 3 heuristics (noise var, histogram entropy, screen resolution dims)
- Now mitigated by the "Screenshot" mode toggle — user can force the warning manually
- **Future fix:** Train a dedicated screenshot classifier using image statistics

### 4. Low-resolution images (<256px)
- PatchCraft requires 32×32 patches — images under 256×256 return zeroed features
- FFT slopes become unreliable below ~300px
- **Fix:** Downsample during inference only if image is very large; upscale small images before feature extraction

### 5. AI images with injected / faked EXIF
- Trivial to use ExifTool to inject fake camera metadata
- Our metadata check sees camera data → scores as "Real"
- **Partial mitigation:** RIGID drift and PatchCraft are not fooled by EXIF
- **Fix:** Cross-reference camera model against a lens/image-size consistency check

---

## 🟡 Still Planned (Not Yet Done)

### Accuracy Improvements
- [ ] Grow training set — target 200 real + 200 AI images (currently 50+51)
  - Add: Ideogram, Recraft, Playground, Seedance, Gemini samples to `data/ai_generated/`
  - Add: Flickr / personal photos in varied lighting to `data/real/`
- [ ] Add video-frame screenshot category to training data
- [ ] Feature selection with `SelectKBest` — 69 features on ~125 base images risks overfitting
- [ ] Calibrate SVM probability outputs (Platt scaling is not applied)

### Robustness
- [ ] Adversarial test: manually JPEG-compress, crop, and re-upload known-AI images
- [ ] Test cross-platform: TikTok, Facebook, YouTube thumbnail crops
- [ ] Watermark/overlay robustness — images with social media overlays test frames

### Web Interface
- [ ] Progress bar / step indicator during analysis (currently just spinner)
- [ ] Side-by-side comparison mode (upload two images)
- [ ] Export results as JSON or PDF report

### Deployment
- [ ] Docker container for easy homelab deployment
- [ ] Rate limiting (if exposed to internet)
- [ ] HTTPS via nginx reverse proxy

---

## 🧪 Crop / Image Quality Tips for Best Results

For accurate detection:
1. **Crop to content only** — remove browser chrome, taskbars, watermarks, social media overlays
2. **Minimum 400×400 pixels** — smaller images degrade PatchCraft and FFT analysis
3. **No heavy post-processing** — heavy Photoshop filters can fool the detector
4. **Use JPEG or PNG** — avoid heavy WebP compression before uploading
5. **Screenshots** — always use the "📱 Screenshot" toggle in the web UI

---

## 📊 Current Performance (Feb 2026)

| Metric | Value |
|---|---|
| Base training images | 125 (50 real + 51 AI + 24 screenshots) |
| Augmented training | 500 |
| Features | 69 (54 base + 15 RIGID drift) |
| CV Accuracy | **84.8%** on augmented set |
| Observed real-world acc. | **~80–90%** (manual testing) |
| Hard cases | Seedance video frames, low-res AI, EXIF-injected AI |
