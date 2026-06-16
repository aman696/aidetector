---
name: Misclassification report
about: The detector got an image wrong
title: "[misclassification] "
labels: misclassification
---

> Before filing: this tool detects **fully AI-generated images** only — not
> deepfakes, face-swaps, or edited photos — and all "real" training data is
> photographic. Misses on non-photographic real content (art, UI screenshots,
> documents) and on video-frame screenshots are **known limitations**, not bugs.
> See [MODEL_CARD.md](../../MODEL_CARD.md).

## The image

- True class: Real / AI-generated
- If AI: which generator (if known)?
- If real: is it a photograph, or art / screenshot / document / other?
- Shareable? If so, attach it or describe how it was produced.

## What the detector said

- Label and probability:
- The plain-language explanation it returned:

## Condition

- [ ] Clean / original download
- [ ] Screenshot or screen capture
- [ ] Social-media recompressed (Facebook / X / Telegram / other)
- [ ] Video frame (Reel / TikTok / YouTube) — known weak case
- [ ] Low resolution (< ~256 px)
- [ ] Other (describe)

## Anything else

Resolution, file format, whether EXIF was present, etc.
