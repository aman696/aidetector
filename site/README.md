# Human or AI website

Plain HTML, CSS, and JavaScript keep this small static site fast, accessible, and dependency-free. The full Python detector remains available locally; the optional free hosted demo uses its 85-feature classical model.

## Cloudflare Pages

| Setting | Value |
| --- | --- |
| Repository | `aman696/aidetector` (public stripped checkout) |
| Production branch | `master` |
| Project name | `humanorai` |
| Framework | None |
| Root directory | `site` |
| Build command | `node build.mjs` |
| Output directory | `dist` |
| Node version | `24.19.0`, also recorded in `.nvmrc` |
| Environment | `SITE_URL` = production HTTPS origin; `NODE_VERSION=24.19.0` |

The domain is configured through `SITE_URL` and Pages custom domains, never embedded in the renderer. No package installation is needed. Build and check from this directory:

```bash
SITE_URL=https://your-domain.example npm run build
npm run check
node verify.mjs https://your-domain.example
```

To keep generated output outside the checkout, set `OUTPUT_DIR` for both the build and check. The source has no added ignore rules.

## Editing

All visitor copy is in `content.mjs`. Metrics are derived from the committed evaluation and experiment JSON records; the laptop commands are extracted verbatim from the root README. Product development priorities are identified as priorities, not completed capabilities. There are no contacts, application-program references, customer claims, or trackers. The dedicated `/test/` upload form is only rendered when a verified demo is enabled.

`config.mjs` reads the single `DEMO_ENABLED` build flag (false by default). Without it the site says “Live demo offline, run it locally” and renders no upload widget. Set `DEMO_ENABLED=true` with a real HTTPS `DEMO_API_ORIGIN` after a successful scan. The build verifies `/healthz` and rejects an unavailable or incorrect model. The company website remains on Cloudflare; the demo runs on Render’s free Python service described in WORKFLOW.md and `render.yaml`.

## Runtime and public files

`app.py` uses Python FastAPI, NumPy, SciPy, OpenCV, Pillow, scikit-learn, and joblib, with optional PyTorch/timm for the full detector. It cannot execute in Pages Functions' JavaScript/TypeScript Workers runtime. There is no `/functions` directory, inference endpoint, or model weight in the deployment.

The full unified model is 34,366,149 bytes (32.77 MiB), exceeding Pages' 25 MiB per-file limit. Metadata ships instead. The build validates every asset and the free-plan 20,000-file ceiling. The classical-only demo uses a separately hosted Python API on Render. Hosting the full model would require a larger external service or a separately designed Cloudflare Containers service; it remains available on a laptop. A normal Worker alone cannot run the existing Python dependencies.

Public source documents and JSON use repository-relative symlinks under `public/`. The build dereferences them so deployment receives ordinary files, preserving one source of truth:

| Public URL | Source / purpose |
| --- | --- |
| `/data/evaluation.json` | `reports/eval_v2_20260615.json` |
| `/data/experiment.json` | `experiment_v1.json` |
| `/data/family-analysis.json` | `reports/family_analysis_20260615.json` |
| `/data/model-metadata.json` | Generated size/hash metadata checked against model files |
| `/data/api.json` | Generated API availability and runtime explanation, not an inference API |
| `/docs/MODEL_CARD.md` | Root model card |
| `/docs/RESEARCH.md` | Root research record |
| `/docs/DATASET.md` | Root dataset documentation |
| `/docs/SECURITY.md` | Root laptop application's security documentation |
| `/docs/LICENSE.txt` | Root MIT license |

Other routes: `/`, `/resources/`, `/test/`, `/404.html`, `/robots.txt`, `/sitemap.xml`, `/llms.txt`, `/favicon.svg`, `/opengraph.png`, and four content-hashed assets under `/assets/`. `_headers` supplies CSP and security headers, explicit document/data types, and immutable caching for hashed assets. HTML uses `no-transform` to prevent automatic proxy script injection. Keep domain-level Web Analytics disabled. The CSP allows the configured demo origin only when enabled. No redirects are needed. A real `404.html` prevents Pages from pretending unknown endpoints are working SPA routes.

## Post-deploy verification

1. Run `node verify.mjs` against the custom domain: every route, static document/data file, and hashed asset must return 200 with the correct content type and security headers.
2. Confirm an unknown route returns 404 and `POST /api/detect` returns 404 or Pages' native 405 (method not allowed), never a successful inference response.
3. Test theme switching, keyboard navigation, benchmark condition controls, command copying, and responsive layout.
4. Check browser console for errors, CSP violations, failed requests, and mixed content.
5. Run mobile Lighthouse on all three HTML routes. Require at least 95 in performance, accessibility, best practices, and SEO.
6. Verify custom-domain HTTPS, canonical URLs, sitemap, social metadata, and the appropriate enabled/offline demo state. Send a real image to the enabled API and compare its probability with the local classical model; also verify rejection of invalid/oversized images and allowed-origin CORS.
