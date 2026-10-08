# Backend Security & Hardening (Human or AI?)

The local browser app accepts **file uploads** without account authentication
and runs **CPU-heavy** inference per request. `/api/detect` writes uploads to a
per-request temporary directory and removes it in `finally`. This checkout has
no contribution endpoint or persistent upload collection. The threat model
includes resource abuse, malicious uploads, disclosure, and data handling.
Controls live in `app.py`; `tests/test_app_security.py` covers selected
controls, not every security claim.

## Controls

| # | Control | Where (`app.py`) | Protects against |
|---|---|---|---|
| 1 | **Per-IP rate limiting** (`slowapi`, `RATE_LIMIT`, default 20/min on `/api/detect`) | `@limiter.limit(...)` | abuse, brute-force, request floods |
| 2 | **Concurrency cap** on heavy scans (`asyncio.Semaphore(MAX_CONCURRENT)`, default 2) | `async with _detect_sem` | CPU/RAM exhaustion from parallel scans |
| 2b | **Bounded queue** (`MAX_QUEUE`, default 10) — beyond running+waiting, return `503` instead of piling up; **per-scan timeout** (`SCAN_TIMEOUT`, default 60 s) ends the request wait | `_inflight` counter + `asyncio.wait_for` | flood pile-up / a hung scan holding a slot |
| 3 | **Non-blocking offload** of the sync model work | `run_in_threadpool(_run_detection, ...)` | event-loop starvation (one slow scan freezing the whole server) |
| 4 | **Route-level upload checks** | `Content-Length` precheck and `len(contents)` against `MAX_FILE_SIZE` (10 MiB); multipart parsing has already occurred | Rejects oversized files at the route; an upstream body cap is still needed |
| 5 | **Real-image + dimension validation** | `_validate_image_bytes` (PIL magic/format, `MAX_IMAGE_PIXELS`=50 MP, `MAX_DIMENSION`=12000) | decompression / pixel bombs, non-image payloads |
| 6 | **Path-traversal safety** | fixed temp name `upload{ext}` with an allow-listed extension; user filename never used in a path | writing outside the temp dir |
| 7 | **Privacy / data handling** | per-request `tempfile.mkdtemp()` -> `shutil.rmtree` in `finally`; image bytes never logged or persisted | data leakage; broken privacy promise |
| 8 | **Generic client errors + server-side logging** | `except Exception -> logger.exception(...)` + `HTTPException(500, "Analysis failed...")` | internal-detail / path disclosure |
| 9 | **Security headers** (CSP, `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`) | `SecurityHeadersMiddleware` (pure ASGI) | XSS, clickjacking, MIME sniffing |
| 10 | **Trusted-host validation** (`ALLOWED_HOSTS`) | `TrustedHostMiddleware` (enabled when set) | Host-header attacks / cache poisoning |
| 11 | **API docs disabled by default** (`/docs`, `/redoc`, `/openapi.json` off unless `DEBUG=1`) | `FastAPI(docs_url=None, ...)` | attack-surface reduction |
| 12 | **No account/session authentication** | Public endpoints do not use login cookies | No account-authenticated actions; public uploads still need abuse controls |
| 14 | **GZip** responses; **`/healthz`** liveness probe | middleware + route | efficiency; host health checks |

Note on #9/#3: middleware is **pure ASGI**, not `BaseHTTPMiddleware` — the latter
wraps the request stream and breaks multipart uploads and exception responses
(this bit us during development; see the commit history).

## Local use and remaining limits

Run `python app.py` and open `http://localhost:8000` on your laptop. The
entry point binds to `0.0.0.0`, so other devices may also reach the app if
your network allows it. Removing the hosting files does not change that
runtime behavior. There are no server-hosting instructions in this checkout.

Multipart parsing precedes the route-level upload checks; those checks are
not a total request-body cap. `_client_ip` trusts the first `X-Forwarded-For`
value, so a supplied header can influence the rate-limit identity.

`asyncio.wait_for` does not forcibly terminate the inference thread. A timeout
may therefore leave CPU work running after the client gets an error; it is not
a worker-kill guarantee. `/healthz` reports bundle loading, not a completed
DINOv2 inference or downloaded embedding weights.

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
| `ALLOWED_HOSTS` | `*` | Optional comma-separated host allow-list for local requests |
| `PORT` | `8000` | Local app listen port for `python app.py` |

## Verification

```
python -m pytest tests/test_app_security.py -q
```
Covers: health check, security headers, docs-disabled, wrong-extension (400),
non-image content (400), oversize (413), generic errors, and rate-limit (429).

## Out of scope (honest)
- Not a deepfake/face-swap or photo-edit detector — only fully-AI-generated images.
- No WAF/bot-management beyond the application rate limiter.
