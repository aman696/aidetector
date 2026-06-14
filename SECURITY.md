# Backend Security & Hardening (Human or AI?)

The app is a **public, unauthenticated** service that accepts **file uploads** and
runs **CPU-heavy** inference per request, with a hard **privacy** promise (uploads
deleted immediately, never stored or used for training). The threat model is
therefore: resource-abuse / DoS, malicious uploads, info disclosure, and anything
that would break the privacy claim. Controls live in `app.py`; the reverse proxy
(Caddy) adds a second layer. Everything here is exercised by
`tests/test_app_security.py` (8 tests).

## Controls

| # | Control | Where (`app.py`) | Protects against |
|---|---|---|---|
| 1 | **Per-IP rate limiting** (`slowapi`, `RATE_LIMIT`, default 20/min on `/api/detect`) | `@limiter.limit(...)` | abuse, brute-force, request floods |
| 2 | **Concurrency cap** on heavy scans (`asyncio.Semaphore(MAX_CONCURRENT)`, default 2) | `async with _detect_sem` | CPU/RAM exhaustion from parallel scans |
| 2b | **Bounded queue** (`MAX_QUEUE`, default 10) — beyond running+waiting, return `503` instead of piling up; **per-scan timeout** (`SCAN_TIMEOUT`, default 60 s) frees a stuck slot | `_inflight` counter + `asyncio.wait_for` | flood pile-up / a hung scan holding a slot |
| 3 | **Non-blocking offload** of the sync model work | `run_in_threadpool(_run_detection, ...)` | event-loop starvation (one slow scan freezing the whole server) |
| 4 | **Request-size limit** before reading body into memory | `Content-Length` precheck + `len(contents)` check vs `MAX_FILE_SIZE` (10 MB) | memory-exhaustion DoS |
| 5 | **Real-image + dimension validation** | `_validate_image_bytes` (PIL magic/format, `MAX_IMAGE_PIXELS`=50 MP, `MAX_DIMENSION`=12000) | decompression / pixel bombs, non-image payloads |
| 6 | **Path-traversal safety** | fixed temp name `upload{ext}` with an allow-listed extension; user filename never used in a path | writing outside the temp dir |
| 7 | **Privacy / data handling** | per-request `tempfile.mkdtemp()` -> `shutil.rmtree` in `finally`; image bytes never logged or persisted | data leakage; broken privacy promise |
| 8 | **Generic client errors + server-side logging** | `except Exception -> logger.exception(...)` + `HTTPException(500, "Analysis failed...")` | internal-detail / path disclosure |
| 9 | **Security headers** (CSP, `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`) | `SecurityHeadersMiddleware` (pure ASGI) | XSS, clickjacking, MIME sniffing |
| 10 | **Trusted-host validation** (`ALLOWED_HOSTS`) | `TrustedHostMiddleware` (enabled when set) | Host-header attacks / cache poisoning |
| 11 | **API docs disabled in prod** (`/docs`, `/redoc`, `/openapi.json` off unless `DEBUG=1`) | `FastAPI(docs_url=None, ...)` | attack-surface reduction |
| 12 | **No auth/cookies/sessions** | stateless design | removes CSRF surface entirely |
| 13 | **Unprivileged container** (uid 1000), slim image, no secrets baked in | `Dockerfile` | blast-radius reduction |
| 14 | **GZip** responses; **`/healthz`** liveness probe | middleware + route | efficiency; host health checks |

Note on #9/#3: middleware is **pure ASGI**, not `BaseHTTPMiddleware` — the latter
wraps the request stream and breaks multipart uploads and exception responses
(this bit us during development; see the commit history).

## Defense in depth — reverse proxy (Caddy)

The app is meant to sit behind Caddy, which adds, independent of the app:
- automatic HTTPS (Let's Encrypt) for `humanorai.online`;
- `request_body { max_size 10MB }` — rejects oversized uploads at the edge;
- an edge rate limit;
- sets `X-Forwarded-For` so the app's per-IP rate limiting (#1) sees the **real**
  client IP. The app only trusts `X-Forwarded-For` because it expects a trusted
  proxy in front — do not expose the app port directly to the internet.

## Configuration (environment variables)

| Var | Default | Purpose |
|---|---|---|
| `DEBUG` | `0` | `1` re-enables `/docs` |
| `MAX_FILE_SIZE` | `10485760` | max upload bytes |
| `MAX_PIXELS` | `50000000` | max decoded pixels (bomb guard) |
| `MAX_DIMENSION` | `12000` | max side length |
| `MAX_CONCURRENT` | `2` | simultaneous heavy scans |
| `MAX_QUEUE` | `10` | extra requests allowed to wait before returning 503 |
| `SCAN_TIMEOUT` | `60` | seconds per scan before giving up (503) |
| `RATE_LIMIT` | `20/minute` | per-IP limit on `/api/detect` |
| `ALLOWED_HOSTS` | `*` | comma-separated; set to `humanorai.online` in prod |
| `PORT` | `8000` | listen port (host platforms inject this) |

## Verification

```
python -m pytest tests/test_app_security.py -q
```
Covers: health check, security headers, docs-disabled, wrong-extension (400),
non-image content (400), oversize (413), generic errors, and rate-limit (429).

## Out of scope (honest)
- Not a deepfake/face-swap or photo-edit detector — only fully-AI-generated images.
- No WAF/bot-management beyond rate limiting; the reverse proxy / host platform is
  expected to provide network-level protection.
