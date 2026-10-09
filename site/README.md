# Human or AI website

Plain HTML, CSS, and JavaScript keep this small static site fast, accessible, and dependency-free. The Python detector remains a separate laptop application.

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

All visitor copy is in `content.mjs`. Metrics are derived from the committed evaluation and experiment JSON records; the laptop commands are extracted verbatim from the root README. Product development priorities are identified as priorities, not completed capabilities. There are no contacts, application-program references, customer claims, or trackers.

`config.mjs` has the single `DEMO_ENABLED` flag, currently false. The site then says “Live demo offline, run it locally” and renders no upload widget. Enabling it requires an actual external HTTPS demo in `DEMO_URL`; the build rejects a demo pointing at this static origin.

## Runtime and public files

`app.py` uses Python FastAPI, NumPy, SciPy, OpenCV, Pillow, scikit-learn, and joblib, with optional PyTorch/timm for the full detector. It cannot execute in Pages Functions' JavaScript/TypeScript Workers runtime. There is no `/functions` directory, inference endpoint, or model weight in the deployment.

The full unified model is 34,366,149 bytes (32.77 MiB), exceeding Pages' 25 MiB per-file limit. Metadata ships instead. The build validates every asset and the free-plan 20,000-file ceiling. Future inference options are a separately hosted Python API, a separately designed Cloudflare Containers service, or the current laptop-only workflow. A normal Worker alone cannot run the existing Python dependencies.

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

Other routes: `/`, `/resources/`, `/404.html`, `/robots.txt`, `/sitemap.xml`, `/llms.txt`, `/favicon.svg`, `/opengraph.png`, and three content-hashed assets under `/assets/`. `_headers` supplies CSP and security headers, explicit document/data types, and immutable caching for hashed assets. No redirects are needed. A real `404.html` prevents Pages from pretending unknown endpoints are working SPA routes.

## Post-deploy verification

1. Run `node verify.mjs` against the custom domain: every route, static document/data file, and hashed asset must return 200 with the correct content type and security headers.
2. Confirm an unknown route returns 404 and `POST /api/detect` returns 404 or Pages' native 405 (method not allowed), never a successful inference response.
3. Test theme switching, keyboard navigation, benchmark condition controls, command copying, and responsive layout.
4. Check browser console for errors, CSP violations, failed requests, and mixed content.
5. Run mobile Lighthouse on both HTML routes. Require at least 95 in performance, accessibility, best practices, and SEO.
6. Verify custom-domain HTTPS, canonical URLs, sitemap, social metadata, and the offline demo message.
