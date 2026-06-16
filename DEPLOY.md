# Deploying Human or AI? (humanorai.online)

The app is a FastAPI server (`app.py`) + the v2 model. A **domain only points a
name at a server — it is not hosting.** You still deploy the app somewhere, then
point `humanorai.online` at it via DNS.

## The honest trade-off (free hosting vs custom domain vs the full model)

The full detector needs ~2 GB RAM (PyTorch + DINOv2). That collides with "free +
custom domain":

| Option | Cost | Custom domain? | Model | Notes |
|---|---|---|---|---|
| **Hugging Face Spaces** (Docker, free CPU) | free | only on PRO ($9/mo) | **full** (best accuracy) | 16 GB RAM runs torch fine; free gives a `*.hf.space` URL |
| **Render / Koyeb** (free web service) | free | **yes, free** | **classical-only** (512 MB can't fit torch) | build with `--build-arg INCLUDE_TORCH=0`; sleeps when idle (cold start) |
| **Small VPS / Fly paid** (~$4–5/mo) | cheap | yes | full | best of both; not free |

**Recommendation for getting `humanorai.online` live for free now:** deploy the
**lean (classical-only)** image to **Render or Koyeb** and attach the domain. You
lose the DINOv2 accuracy edge (classical-only ROC-AUC ~0.86 vs unified ~0.94) but it's genuinely free with
your domain and the page is identical. Upgrade to the full model later (HF PRO or
a tiny VPS) when you want the extra accuracy.

## Path A — Render/Koyeb (free, custom domain, classical-only)

1. Push this repo to GitHub.
2. New Web Service → connect the repo → environment **Docker**.
3. Set build arg / env: **`INCLUDE_TORCH=0`** (lean image, fits 512 MB).
4. Deploy. It serves on the host's `*.onrender.com` / `*.koyeb.app` URL first.
5. Add custom domain `humanorai.online` in the host's dashboard; it gives you a
   DNS target.
6. In your registrar (where you bought `.online`), add DNS:
   - apex `humanorai.online` → an **ALIAS/ANAME** (or the host's A record) to the target
   - `www` → **CNAME** to the same target
   HTTPS certificate is issued automatically by the host.

## Path B — Hugging Face Spaces (free, full model, hf.space URL)

1. Create a new **Space** → SDK **Docker**.
2. Add this frontmatter to the **top of the Space's `README.md`** (do not put it
   in this repo's README — it's only for the Space):
   ```
   ---
   title: Human or AI
   emoji: 🔍
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
3. Push the repo (with `Dockerfile`). It builds the **full** image by default and
   serves at `https://<user>-human-or-ai.hf.space`.
4. Custom domain `humanorai.online` requires HF **PRO**; until then use the
   `hf.space` URL (you can still link to it from anywhere).

## Local / self-host (full model)

```
docker build -t humanorai .
docker run -p 7860:7860 humanorai          # full model
docker build --build-arg INCLUDE_TORCH=0 -t humanorai-lean .   # classical-only
```

## Notes
- The Dockerfile bakes the DINOv2 weights at build time so the first scan is fast
  (full build only).
- Free hosts sleep on inactivity — the first request after idle is slow (cold
  start), then fast.
- Privacy is unchanged in any deployment: uploads go to a temp file and are
  deleted immediately after the scan (`app.py`), never stored.
