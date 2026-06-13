"""
Tests for src/embedding_extractor.py.

The geometry/math helpers (crop selection, RGBA compositing, cosine, seeding)
run without torch. The DinoEmbedder tests skip cleanly when torch/timm or the
pretrained weights (network) are unavailable.
"""

import numpy as np
import pytest

from src.embedding_extractor import (
    DRIFT_SIGMAS,
    INPUT_SIZE,
    OUT_DIM,
    _best_window,
    _cosine_rows,
    _load_rgb,
    _seed_from_array,
    texture_rich_crop,
)


class TestConstants:
    def test_out_dim_is_768_plus_two_drift(self):
        assert OUT_DIM == 768 + len(DRIFT_SIGMAS) == 770


class TestLoadRgb:
    def test_transparent_pixels_become_white(self):
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[..., 0] = 255  # red, but fully transparent (alpha 0)
        rgb = _load_rgb(rgba)
        assert rgb.shape == (2, 2, 3)
        assert np.all(rgb == 255)

    def test_opaque_color_preserved(self):
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[..., 0] = 200
        rgba[..., 3] = 255  # opaque
        rgb = _load_rgb(rgba)
        assert np.all(rgb[..., 0] == 200)
        assert np.all(rgb[..., 1] == 0)

    def test_grayscale_promoted_to_three_channels(self):
        gray = np.full((4, 4), 100, dtype=np.uint8)
        rgb = _load_rgb(gray)
        assert rgb.shape == (4, 4, 3)
        assert np.all(rgb == 100)


class TestBestWindow:
    def test_picks_highest_summed_window(self):
        grid = np.zeros((4, 4), dtype=np.float64)
        grid[2:4, 2:4] = 10.0  # bottom-right hot corner
        top, left = _best_window(grid, win=2)
        assert (top, left) == (2, 2)

    def test_smaller_than_window_falls_back_centered(self):
        grid = np.ones((2, 2), dtype=np.float64)
        top, left = _best_window(grid, win=4)
        assert (top, left) == (0, 0)


class TestTextureRichCrop:
    def test_output_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (600, 600, 3), dtype=np.uint8)
        crop = texture_rich_crop(img)
        assert crop.shape == (INPUT_SIZE, INPUT_SIZE, 3)
        assert crop.dtype == np.uint8

    def test_prefers_textured_region(self):
        # Top half flat, bottom half noisy -> crop should land in the bottom.
        rng = np.random.default_rng(1)
        img = np.full((896, 448, 3), 128, dtype=np.uint8)
        img[448:, :, :] = rng.integers(0, 256, (448, 448, 3), dtype=np.uint8)

        import cv2
        from src.embedding_extractor import (GRID_PATCH, CROP_SIZE,
                                             _ldiv_grid, _best_window)
        from src.patchcraft_analyzer import compute_high_pass
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        grid = _ldiv_grid(compute_high_pass(gray), GRID_PATCH)
        top, _ = _best_window(grid, CROP_SIZE // GRID_PATCH)
        assert top * GRID_PATCH >= 256  # well into the lower (noisy) half

    def test_small_image_uses_resize_center_crop(self):
        img = np.full((100, 150, 3), 64, dtype=np.uint8)
        crop = texture_rich_crop(img)
        assert crop.shape == (INPUT_SIZE, INPUT_SIZE, 3)


class TestCosineRows:
    def test_identical_vectors_give_one(self):
        a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        assert _cosine_rows(a, a.copy())[0] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_give_zero(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        assert _cosine_rows(a, b)[0] == pytest.approx(0.0, abs=1e-6)


class TestSeed:
    def test_seed_deterministic_and_content_sensitive(self):
        a = np.zeros((8, 8, 3), dtype=np.uint8)
        b = a.copy()
        b[0, 0, 0] = 1
        assert _seed_from_array(a) == _seed_from_array(a.copy())
        assert _seed_from_array(a) != _seed_from_array(b)
        assert 0 <= _seed_from_array(a) < 2 ** 63


# --------------------------------------------------------------------------- #
# Embedder integration (needs torch/timm + pretrained weights)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def embedder():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from src.embedding_extractor import DinoEmbedder
    try:
        return DinoEmbedder(device="cpu", half=False)
    except Exception as exc:  # weights download / network unavailable
        pytest.skip(f"DINOv2 weights unavailable: {exc}")


class TestEmbedder:
    def test_output_dim_and_finite(self, embedder):
        rng = np.random.default_rng(2)
        img = rng.integers(0, 256, (500, 500, 3), dtype=np.uint8)
        vec = embedder.embed_with_drift(img, seed=123)
        assert vec.shape == (OUT_DIM,)
        assert np.all(np.isfinite(vec))

    def test_drift_in_valid_range(self, embedder):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, (500, 500, 3), dtype=np.uint8)
        vec = embedder.embed_with_drift(img, seed=7)
        drift = vec[embedder.embed_dim:]
        assert drift.shape == (len(DRIFT_SIGMAS),)
        assert np.all(drift >= 0.0) and np.all(drift <= 2.0)

    def test_seeded_drift_reproducible(self, embedder):
        rng = np.random.default_rng(4)
        img = rng.integers(0, 256, (500, 500, 3), dtype=np.uint8)
        v1 = embedder.embed_with_drift(img, seed=42)
        v2 = embedder.embed_with_drift(img, seed=42)
        np.testing.assert_allclose(v1, v2, rtol=0, atol=1e-5)

    def test_rgba_input_accepted(self, embedder):
        rgba = np.zeros((500, 500, 4), dtype=np.uint8)
        rgba[..., :3] = 100
        rgba[..., 3] = 255
        vec = embedder.embed_with_drift(rgba, seed=1)
        assert vec.shape == (OUT_DIM,)
        assert np.all(np.isfinite(vec))

    def test_batch_matches_single(self, embedder):
        rng = np.random.default_rng(5)
        imgs = [rng.integers(0, 256, (500, 500, 3), dtype=np.uint8) for _ in range(2)]
        seeds = [11, 22]
        batch = embedder.embed_batch_with_drift(imgs, seeds)
        single0 = embedder.embed_with_drift(imgs[0], seed=11)
        assert batch.shape == (2, OUT_DIM)
        np.testing.assert_allclose(batch[0], single0, rtol=0, atol=1e-4)
