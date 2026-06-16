# Dataset Card — AI Image Detector

This card documents the data the v2 models were trained and evaluated on. All
counts are taken from the reproducibility record
[experiment_v1.json](experiment_v1.json) (dataset content hash
`72b88efc0497...`). The image files themselves are not in this Git repository;
they are published on Hugging Face (see Access below).

The intent of this card is honesty about what the data is and is not. The most
important limitation: **every "real" image is a photograph.** Performance on
non-photographic real content is not characterized.

## Composition

### Base images: 4,271 total

| Class | Count | Sources |
|---|---|---|
| Real | 2,000 | COCO (1,000), OpenFake real split (1,000) |
| AI-generated | 2,271 | 34 generator families (below) |

### Real sources

| Source | Count | Origin | License |
|---|---|---|---|
| `coco` | 1,000 | COCO (Common Objects in Context), public photographic dataset | COCO images: Flickr terms / CC-BY 4.0 annotations — verify per-image before redistribution |
| `openfake` | 1,000 | OpenFake dataset, real split | Per the OpenFake dataset license — confirm before redistribution |

### AI generator families (2,271 images, 34 families)

Owner-collected and benchmark-sourced text-to-image outputs. Per-family counts
(from `experiment_v1.json`):

| Family | n | Family | n | Family | n |
|---|---|---|---|---|---|
| aurora_20_1_25 | 64 | gpt | 62 | recraft_v2 | 63 |
| chroma | 64 | gpt_image_1 | 63 | recraft_v3 | 63 |
| dalle_2 | 64 | grok_2_image_1212 | 63 | sd_1.5 | 63 |
| dalle_3 | 63 | halfmoon_4_4_25 | 44 | sd_1.5_dreamshaper | 63 |
| firefly | 63 | hidream_i1_full | 63 | sd_1.5_epicdream | 63 |
| flux_1 | 63 | ideogram_2.0 | 63 | sd_2.1 | 63 |
| frames_23_1_25 | 63 | ideogram_3.0 | 63 | stable_diffusion_1_3 | 63 |
| gemini | 209 | imagen_3.0_002 | 63 | stable_diffusion_1_4 | 63 |
| glide | 63 | imagen_4.0 | 63 | stable_diffusion_2 | 63 |
| | | lumina_17_2_25 | 63 | stable_diffusion_3 | 63 |
| | | midjourney_6 | 63 | stable_diffusion_xl | 63 |
| | | midjourney_7 | 63 | midjourney_v5 | 63 |
| | | mystic | 63 | | |

> Provenance / license TODO (owner): the per-family origin and redistribution
> license of the `data/mixed/` families must be confirmed before any public
> dataset release. Several family names correspond to public generation
> benchmarks; others are owner-generated. Do not redistribute these images until
> each family's source and license are recorded here. This is a known gap and a
> blocker for full external reproducibility.

### Derived (augmented) records: 23,577

Generated from the base images to simulate real-world distribution conditions.
Built by `scripts/build_derived.py` / `scripts/capture_screenshots.py`.

| Condition | Count | What it is |
|---|---|---|
| `screenshot` | 8,530 | screen-capture re-render of the base image |
| `x` | 3,246 | X/Twitter-style recompression |
| `facebook` | 3,210 | Facebook-style recompression |
| `telegram` | 3,203 | Telegram-style recompression |
| `chain_ss_tg` | 2,723 | screenshot then Telegram (chained) |
| `chain_fb_x` | 2,665 | Facebook then X (chained) |

> The exact recompression quality factors and the screenshot capture method are
> defined in `scripts/build_derived.py` and `scripts/capture_screenshots.py`;
> document the concrete parameters here when releasing the data.

## Folder layout (on disk)

```
data/
|-- real/coco/         real photos (COCO)
|-- real/openfake/     real photos (OpenFake real split)
|-- Gemini/            AI (Google Gemini)
|-- GPT/               AI (ChatGPT / DALL.E)
|-- mixed/<family>/    AI, one subdirectory per generator family
|-- ar_external/       independent out-of-distribution check set
|-- manifests/         base_manifest.json, splits.json, derived_manifest.json
|-- derived/           platform/screenshot/chained variants + per-id feature caches
```

## Splits (leak-safe)

Assigned at **base-image level** so a base image and all of its derived variants
stay in one split (no near-duplicate leakage); derived variants inherit their
base's split; cross-validation groups by `base_id`. Defined in `src/dataset.py`.

- Seed: 42 — train 0.70, val 0.10, remainder test.
- Held-out generator families (never in training; measure generalization to
  unseen generators): `midjourney_7`, `ideogram_3.0`, `imagen_4.0`, `flux_1`,
  `recraft_v3`.
- Base-level split counts: train 2,767 / val 387 / test 802 / test_holdout 315.
- Assembled training rows (clean + leak-safe derived subset): 16,593.
- Test rows: 6,414 (3,216 AI / 3,198 real). Holdout rows: 2,520 (all AI).

## Known biases and limitations

- **All real images are photographs** (COCO / OpenFake). Digital art,
  illustration, screenshots of documents/UIs, charts, and scientific imagery are
  absent from the real class, so false-positive behaviour on them is unknown.
- **Generator-family imbalance:** Gemini (209) is over-represented relative to
  the ~63-per-family norm; `halfmoon` (44) is under-represented.
- **Class balance** is roughly even at the base level (2,271 AI / 2,000 real) and
  by construction in the test rows; real-world base rates differ, so accuracy at
  a fixed threshold does not transfer directly to deployment.
- **Real-vs-AI is the only label.** No manipulation masks, no per-generator
  attribution labels, no edited-region annotations.

## Access and reproduction

The dataset is published on Hugging Face:
**[aman213/aidetector-data](https://huggingface.co/datasets/aman213/aidetector-data)**
(full set, ~21 GB, matching the recorded content hash). Code and trained models
are MIT-licensed; the data is not under MIT (see the per-family provenance note
above — license is `other`).

```bash
hf download aman213/aidetector-data --repo-type dataset --local-dir data/
# then restore the packed feature caches:
mkdir -p data/derived/cache && tar -xzf data/derived/cache.tar.gz -C data/derived/cache
```

Because the full dataset is downloadable and pinned by the content hash in
[experiment_v1.json](experiment_v1.json), the headline metrics can be reproduced
independently, not just by the owner. The remaining open item is resolving the
per-family provenance/license TODO above.

## Privacy

The live service (humanorai.online) deletes uploaded images immediately after
scanning and never stores them or trains on them. No user-submitted images are
part of this dataset. See [SECURITY.md](SECURITY.md).
