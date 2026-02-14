# AI Image Detector — TODO & Known Issues

## Observed Discrepancies

### 1. Screenshots falsely classified as AI-Generated
- YouTube video screenshots are being classified as AI-Generated (~93% confidence)
- Screenshots are real images but lack camera EXIF metadata → metadata_score inflates
- Compression artifacts from screenshot tools may affect frequency analysis
- **Root cause:** The model is trained on camera photos vs AI art — screenshots are neither
- **Fix needed:** Add a "screenshot" category or reduce metadata weight for PNGs

### 2. FFT Analysis seems ineffective
- FFT score shows "likely Real" for nearly all images (both real AND AI)
- The spectral slope and high-freq ratio don't differentiate well on this dataset
- Possible reasons:
  - Modern AI generators (Gemini, DALL-E 3, Flux) may have improved high-frequency reproduction since the Durall 2020 paper
  - The paper targeted GANs specifically — newer diffusion models may not have the same artifacts
  - Our dataset is too small/homogeneous to expose the differences
- **Fix needed:** Re-examine FFT scoring thresholds, try different frequency features, or retrain on more diverse data

---

## Known Failure Modes (Where This Model WILL Fail)

### High Confidence False Positives (real → flagged as AI)
1. **Screenshots** — No EXIF, flat frequency profile, uniform patches → looks AI to the model
2. **Scanned documents / digital art** — Same issues as screenshots
3. **Heavily compressed images** — JPEG compression at low quality wipes frequency detail
4. **Social media re-uploads** — Twitter/Instagram strip EXIF, recompress, and resize
5. **Edited/composited photos** — Photoshop output has software tags but no camera data
6. **Phone screenshots of real photos** — Metadata is lost, frequency content is altered

### High Confidence False Negatives (AI → passes as real)
1. **AI images with injected EXIF** — Trivial to fake camera metadata
2. **AI images post-processed through camera apps** — Some apps add EXIF
3. **AI images saved as JPEG from editing software** — May gain software tags
4. **Upscaled/enhanced photos** — AI upscalers alter frequency spectrum
5. **Novel generators** — Models not represented in training data (Flux, Ideogram, Recraft, etc.)

### Architectural Limitations
1. **Metadata dependence** — Metadata is the strongest discriminator, but it's trivially fakeable/strippable
2. **Small training set** — 50+51 images is nowhere near enough for generalization
3. **No adversarial robustness** — Model can be fooled by anyone who knows the features
4. **Single-image analysis only** — No cross-referencing with reverse image search or provenance
5. **No diffusion-specific features** — Modern detectors use noise pattern analysis, not just FFT

---

## Future Improvements (Not Yet Implemented)
- [ ] Add screenshot/digital-art handling (detect non-camera sources differently)
- [ ] Tune FFT thresholds or add new frequency features (wavelet analysis, DCT)
- [ ] Train on larger, more diverse dataset (1000+ images, multiple generators)
- [ ] Add noise pattern analysis (Corvi 2023 extended methods)
- [ ] Reduce metadata weight in final score — too easy to game
- [ ] Add JPEG compression artifact detection
- [ ] Consider ensemble with a lightweight CNN for better generalization
