# Contributing

Thanks for your interest. This is a research prototype and portfolio project, so
contributions are welcome but the bar is correctness and honesty over features.

Please read [AGENTS.md](AGENTS.md) (rules and licensing) and
[WORKFLOW.md](WORKFLOW.md) (full pipeline) before opening a PR.

## Ground rules

- **No overclaiming.** Performance numbers live in one place per kind: the
  headline in [README.md](README.md), the detail in [RESEARCH.md](RESEARCH.md).
  If a change alters behaviour, accuracy, commands, or file layout, update the
  relevant doc in the same PR (the WORKFLOW.md rule).
- **Measured beats asserted.** A score-direction claim in a docstring must be
  backed by a measured probe and a test. If a measurement contradicts a
  docstring or a paper-derived prior, follow the measurement and record the
  discrepancy.
- **No emojis in documentation.**

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # run the test suite
```

- Python 3.10+, PEP 8, type hints on public functions, docstrings stating
  purpose / args / returns.
- Every new analyzer or feature gets pytest coverage **before** it is wired into
  the classifier. Use synthetic ground truths where possible (constant image ->
  zero texture; checkerboard -> known residual). Tests must skip cleanly when
  optional dependencies (torch/timm) or the dataset are absent.
- Heavy dependencies (torch/timm/playwright) are imported lazily; the repo must
  keep working classical-only without them.

## Feature versioning (important)

If you change what a classical feature **computes** (not just its name), bump
`_CLASSICAL_FORMULA_EPOCH` in `src/feature_pipeline.py` so the per-id caches are
rebuilt instead of silently reused, and note in the PR that re-extraction +
retraining is required. Pure refactors that are value-identical do not need a
bump (prove it with a test).

## Pull requests

- One logical change per PR. Explain hyperparameter / threshold choices in the
  description, not just in the diff.
- Note explicitly whether the change requires re-extraction or retraining.
- Confirm `python -m pytest -q` passes.

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/`. For a misclassification
report, the most useful thing you can include is the **condition** (clean /
screenshot / social-media recompression) and, if shareable, the image or how it
was produced. Note that the detector targets **fully AI-generated images** only
(not deepfakes, face-swaps, or photo edits) and that all "real" training data is
photographic — see [MODEL_CARD.md](MODEL_CARD.md) for scope.
