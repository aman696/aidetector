"""
Security/hardening tests for the web app (app.py), via FastAPI's TestClient
(no real sockets, so no curl/multipart flakiness). Covers the controls added
for deployment: health check, security headers, upload validation, request-size
limit, and per-IP rate limiting.

Skipped cleanly if the v2 model isn't present (the /api/detect guard returns 503
before validation otherwise).
"""

import pytest
from fastapi.testclient import TestClient

import app as appmod

client = TestClient(appmod.app)
needs_model = pytest.mark.skipif(appmod.detector is None, reason="v2 model not loaded")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"  # not enough to be a valid image, but right magic


class TestBasics:
    def test_healthz(self):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_security_headers_present(self):
        r = client.get("/")
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert "content-security-policy" in r.headers
        assert "referrer-policy" in r.headers

    def test_docs_disabled_in_prod(self):
        # DEBUG defaults off -> interactive docs should not be served.
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


@needs_model
class TestUploadValidation:
    def test_wrong_extension_rejected(self):
        r = client.post("/api/detect", files={"file": ("x.txt", b"hello", "text/plain")})
        assert r.status_code == 400

    def test_non_image_content_rejected(self):
        # right extension, but the bytes are not a real image
        r = client.post("/api/detect", files={"file": ("x.png", b"not an image", "image/png")})
        assert r.status_code == 400

    def test_oversize_rejected_413(self):
        big = b"\x00" * (appmod.MAX_FILE_SIZE + 5000)
        r = client.post("/api/detect", files={"file": ("x.jpg", big, "image/jpeg")})
        assert r.status_code == 413

    def test_error_message_is_generic(self):
        # no internal exception text leaks to the client
        r = client.post("/api/detect", files={"file": ("x.png", b"not an image", "image/png")})
        assert "Traceback" not in r.text and "/home/" not in r.text


@needs_model
class TestRateLimiting:
    def test_rate_limit_trips_429(self):
        # fire well past the per-IP limit; cheap because invalid uploads are
        # rejected before any model inference, but each still counts.
        codes = [client.post("/api/detect",
                             files={"file": ("x.png", b"nope", "image/png")}).status_code
                 for _ in range(30)]
        assert 429 in codes, f"expected a 429 among {set(codes)}"


@needs_model
class TestOverloadGuard:
    def test_full_queue_returns_503(self, monkeypatch):
        # With no slots and no queue, a VALID upload is turned away (503) before
        # any processing — the bounded-queue overload guard.
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        monkeypatch.setattr(appmod, "MAX_CONCURRENT", 0)
        monkeypatch.setattr(appmod, "MAX_QUEUE", 0)
        # unique client IP -> fresh rate-limit bucket (other tests exhaust the
        # shared one), so we reach the overload guard rather than a 429.
        r = client.post("/api/detect",
                        files={"file": ("x.png", buf.getvalue(), "image/png")},
                        headers={"X-Forwarded-For": "203.0.113.7"})
        assert r.status_code == 503


def _png_bytes(size=(8, 8)) -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, (123, 200, 90)).save(buf, "PNG")
    return buf.getvalue()


class TestContribution:
    """The opt-in donate path, focused on the verdict correct/wrong feedback
    field. Uses a temp CONTRIB_DIR so nothing touches the real contributions/."""

    def _post(self, monkeypatch, tmp_path, ip, **form):
        monkeypatch.setattr(appmod, "CONTRIB_DIR", str(tmp_path))
        return client.post(
            "/api/contribute",
            data=form,
            files={"file": ("x.png", _png_bytes(), "image/png")},
            headers={"X-Forwarded-For": ip})

    def _records(self, tmp_path):
        import json
        path = tmp_path / "contributions.jsonl"
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]

    def test_feedback_stored_when_valid(self, monkeypatch, tmp_path):
        r = self._post(monkeypatch, tmp_path, "203.0.113.20",
                       label="ai", verdict_feedback="wrong")
        assert r.status_code == 200, r.text
        assert r.json()["verdict_feedback"] == "wrong"
        assert self._records(tmp_path)[-1]["verdict_feedback"] == "wrong"

    def test_bad_feedback_rejected(self, monkeypatch, tmp_path):
        r = self._post(monkeypatch, tmp_path, "203.0.113.21",
                       label="ai", verdict_feedback="maybe")
        assert r.status_code == 400

    def test_feedback_optional(self, monkeypatch, tmp_path):
        r = self._post(monkeypatch, tmp_path, "203.0.113.22", label="real")
        assert r.status_code == 200, r.text
        assert r.json()["verdict_feedback"] is None
        assert self._records(tmp_path)[-1]["verdict_feedback"] is None


class TestMetrics:
    def test_metrics_open_by_default(self, monkeypatch):
        monkeypatch.setattr(appmod, "METRICS_TOKEN", "")
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "detect_requests_total" in r.text
        assert "detect_inflight" in r.text

    def test_metrics_gated_when_token_set(self, monkeypatch):
        monkeypatch.setattr(appmod, "METRICS_TOKEN", "s3cret")
        # no token -> 404, not 401 (don't confirm the route exists)
        assert client.get("/metrics").status_code == 404
        # wrong token -> still 404
        assert client.get("/metrics?token=wrong").status_code == 404
        # correct token via query param
        assert client.get("/metrics?token=s3cret").status_code == 200
        # correct token via Authorization header
        r = client.get("/metrics", headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200

    def test_inflight_capacity_reflects_config(self):
        r = client.get("/metrics")
        assert "detect_inflight_capacity" in r.text
