# AGENTS.md — Read This Before Touching the Repo

**Audience: every AI agent and every human.** If you are about to use this repo, extend it, fix a
bug in it, or train anything with it — read this file first. It is the single source of truth for
the rules and the licensing. Everything else (architecture, commands,
research) is linked from here.

---

## 1. What this repo is

An AI-image detector. It is a personal / portfolio project, not a forensic or moderation service;
its output is an advisory estimate, not proof.

The shipped model is a single **unified detector**: 85 classical forensic features (FFT,
eigenvalues, DCT, noise residuals, gradients, texture, NPR, screenshot forensics) plus a frozen
**DINOv2** embedding and RIGID noise-drift, an 855-dim hybrid feeding a calibrated SVM, trained to
hold up on clean, social-media-compressed, screenshotted, and chained images. A classical-only
85-feature model is the automatic fallback when PyTorch is unavailable. (The earlier v1 was
classical-only with a separate screenshot SVM; those `.pkl` files remain in `models/` as a
reference baseline but are not used.)

- Full architecture, commands, training/eval pipeline → **[WORKFLOW.md](WORKFLOW.md)**
- Intended use, metrics, limitations, known weaknesses → **[MODEL_CARD.md](MODEL_CARD.md)**
- Literature review, known failure modes, analysis → **[RESEARCH.md](RESEARCH.md)**
- Reproducibility record (hashes, seeds, hyperparameters) → **[experiment_v1.json](experiment_v1.json)**
- Plain-language explanations of every paper used → `paper_notes/` (**local only, gitignored** —
  if you cloned this from GitHub you will not have it; ask the owner)

## 2. Hard rules (non-negotiable)

1. **Never use `sudo`.** Nothing in this project needs elevated privileges.
2. **Never modify anything outside this project folder.** No global config, no system packages,
   no other repos.
3. **Never delete anything from `data/`, `models/`, or `papers/`** without the owner's explicit
   per-file approval. Datasets are expensive to rebuild.
4. **`WORKFLOW.md` must always describe the complete current workflow in one document.** If your
   change alters commands, file layout, features, or training — update WORKFLOW.md in the same
   change. The owner relies on it to re-learn the project after long gaps.
5. **The owner reviews everything.** Propose first, get approval, then execute. Do not make
   architectural decisions unilaterally.
6. Performance numbers (accuracy, etc.) live in **one** place per kind: headline in README.md,
   details in RESEARCH.md. Do not copy them into other docs — they drift.

## 3. Rules for AI agents specifically

- Generate complete, working code: type hints, docstrings, error handling, PEP 8. No stubs that
  silently swallow exceptions.
- Comment only where the code cannot explain itself (constraints, units, paper equations).
- Every new analyzer or feature gets pytest coverage before it is wired into the classifier.
- Hyperparameter choices must be explained in the PR/commit description, not just set.
- The current data layout is `data/{Gemini, GPT, mixed, real}`. Older docs may reference a
  previous layout (`data/ai_generated`, `data/screenshots`, …) — **verify paths on disk before
  running any training command.**
- If you add a method from a paper, record it in `paper_notes/CITATIONS.md` **and** write a
  plain-language note for it in `paper_notes/` (see its `INDEX.md` for the template). Do not put
  citations in public docs — see §5.

## 4. Licensing

- **Code:** MIT License — see [LICENSE](LICENSE). You may use, modify, and redistribute the code
  with attribution per the license text.
- **Trained models (`models/*.pkl`):** provided as-is under the same MIT terms. They were trained
  on a mixed dataset (see below); no warranty of accuracy. Do not present their output as
  authoritative proof that an image is or is not AI-generated.
- **Datasets (`data/`, not in the repo):** NOT redistributed and NOT covered by the MIT license.
  Sources include Unsplash (Unsplash License — free to use, but mass redistribution of unaltered
  copies is prohibited, which is one reason `data/` is gitignored), personal photos, and outputs
  of various image generators whose terms differ. The dataset is shared by the owner on
  request, on a per-case basis.
- **Papers (`papers/`, not in the repo):** copyright of their authors/publishers; we link to
  arXiv instead of redistributing PDFs.

## 5. Papers & citations

The papers behind each analyzer are **internal for now** — the owner will add public citations
in one pass once the ongoing research concludes. Until then, public docs (README etc.) carry
**no references section**. The full citation ledger — verified arXiv IDs, per-analyzer mapping,
"inspired by vs. implements" wording requirements, and two unresolved attributions — lives in
`paper_notes/CITATIONS.md` (local only). If you add a method from a paper, record it there and
write a note in `paper_notes/` — do not add citations to public docs.

## 6. Where to go next

| You want to… | Read |
|---|---|
| Run or train the detector | [WORKFLOW.md](WORKFLOW.md) |
| Know what the model is for and its limits | [MODEL_CARD.md](MODEL_CARD.md) |
| Understand why a paper/method is here | `paper_notes/INDEX.md` (local only) |
| Know what is broken or planned | [RESEARCH.md](RESEARCH.md) |
| Reproduce a reported number | [experiment_v1.json](experiment_v1.json) |
| Add a feature/analyzer | §2–§3 above, then WORKFLOW.md architecture section |
