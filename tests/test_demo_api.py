"""Hosted demo checks against the real classical bundle and upload boundaries."""

import asyncio
import io
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import httpx
from PIL import Image
import pytest

from demo import api
from src.unified_detector import UnifiedDetector


@pytest.fixture(autouse=True)
def reset_admission():
    api.active = None
    api.recent.clear()


def image_bytes():
    stream = io.BytesIO()
    Image.effect_noise((128, 128), 40).convert("RGB").save(stream, "PNG")
    return stream.getvalue()


def test_probability_matches_existing_detector(tmp_path):
    image = image_bytes()
    original = tmp_path / "image.png"
    original.write_bytes(image)
    detector = UnifiedDetector(
        unified_path=str(tmp_path / "no-unified-model.pkl"),
        classical_path=str(api.model_path),
    )
    expected = detector.predict(str(original))
    with TestClient(api.app) as client:
        response = client.post("/api/detect", files={"file": ("image.png", image, "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["method"] == "classical_v2" and result["fallback"] is True
    assert result["label"] == expected["label"]
    assert result["probability_ai"] == pytest.approx(expected["probability_ai"], abs=1e-12)
    assert len(result["features"]) == 85
    assert [feature["name"] for feature in result["features"]] == api.CLASSICAL_NAMES


@pytest.mark.parametrize("name,contents,status,code", [
    ("image.txt", b"text", 400, "invalid_type"),
    ("image.png", b"not an image", 400, "invalid_image"),
    ("image.png", b"x" * (api.MAX_BYTES + 1), 413, "too_large"),
])
def test_invalid_uploads(name, contents, status, code):
    with TestClient(api.app) as client:
        response = client.post("/api/detect", files={"file": (name, contents)})
    assert response.status_code == status
    assert response.json()["detail"] == code


def test_oversized_body_rejected_before_multipart():
    with TestClient(api.app) as client:
        response = client.post("/api/detect", content=b"x" * (api.MAX_BODY + 1),
                               headers={"Content-Type": "multipart/form-data; boundary=bad"})
    assert response.status_code == 413
    assert response.json()["detail"] == "too_large"


def test_large_and_animated_images_rejected():
    stream = io.BytesIO()
    Image.new("RGB", (1025, 1024)).save(stream, "PNG")
    with pytest.raises(api.HTTPException) as exception:
        api.validate_image(stream.getvalue(), "large.png")
    assert exception.value.detail == "dimensions"
    stream = io.BytesIO()
    frames = [Image.new("RGB", (8, 8), color) for color in ["black", "white"]]
    frames[0].save(stream, "PNG", save_all=True, append_images=frames[1:])
    with pytest.raises(api.HTTPException) as exception:
        api.validate_image(stream.getvalue(), "animation.png")
    assert exception.value.detail == "animated_image"


def test_origin_and_global_rate_limit(monkeypatch):
    monkeypatch.setattr(api, "ALLOWED_ORIGIN", "https://website.example")
    with TestClient(api.app) as client:
        response = client.post("/api/detect", files={"file": ("image.png", image_bytes())},
                               headers={"Origin": "https://other.example"})
        assert response.status_code == 403
        for _ in range(3):
            assert client.post("/api/detect", files={"file": ("image.png", b"invalid")}).status_code == 400
        response = client.post("/api/detect", files={"file": ("image.png", b"invalid")})
        assert response.status_code == 429 and response.headers["Retry-After"] == "60"


def test_timeout_retains_slot_and_upload_until_thread_finishes(monkeypatch):
    release = threading.Event()
    paths = []
    original = api.extract_unified

    def slow_extract(path, **kwargs):
        paths.append(Path(path))
        release.wait(5)
        return original(path, **kwargs)

    monkeypatch.setattr(api, "extract_unified", slow_extract)
    monkeypatch.setattr(api, "TIMEOUT", 0.05)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            try:
                response = await client.post("/api/detect", files={"file": ("image.png", image_bytes())})
                assert response.status_code == 503 and response.json()["detail"] == "timeout"
                assert paths and paths[0].exists()
                response = await client.post("/api/detect", files={"file": ("image.png", image_bytes())})
                assert response.status_code == 503 and response.json()["detail"] == "busy"
            finally:
                release.set()
                if api.active:
                    await api.active
            assert not paths[0].exists()

    asyncio.run(scenario())
