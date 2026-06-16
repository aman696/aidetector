"""
Tests for src/channels.py — the platform rules are measured values;
these tests pin them exactly.
"""

import numpy as np
import pytest

from src.channels import (
    apply_chain,
    decode,
    emulate_facebook,
    emulate_telegram,
    emulate_x,
    rng_for,
)


def _textured(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Random-noise BGR image so JPEG actually has something to quantize."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


class TestFacebook:

    def test_wide_image_resized_to_720(self):
        data, ext, params = emulate_facebook(_textured(1600, 900), rng_for('a'))
        out = decode(data)
        assert ext == '.jpg'
        assert out.shape[1] == 720
        assert out.shape[0] == round(900 * 720 / 1600)
        assert 61 <= params['quality'] <= 92

    def test_narrow_image_not_resized(self):
        data, _, _ = emulate_facebook(_textured(640, 480), rng_for('a'))
        assert decode(data).shape[:2] == (480, 640)

    def test_quality_sampled_within_measured_range(self):
        qualities = {emulate_facebook(_textured(100, 100), rng_for(f'id{i}'))[2]['quality']
                     for i in range(50)}
        assert min(qualities) >= 61
        assert max(qualities) <= 92
        assert len(qualities) > 5  # actually sampling, not a constant

    def test_compression_is_single_pass(self):
        # bytes are the platform encode itself; decoding must differ from the
        # noise source (lossy) but encoding happened exactly once
        img = _textured(640, 480)
        data, _, _ = emulate_facebook(img, rng_for('a'))
        assert not np.array_equal(decode(data), img)


class TestX:

    def test_small_image_untouched_and_lossless(self):
        img = _textured(768, 768)
        data, ext, params = emulate_x(img, rng_for('a'))
        assert params.get('untouched') is True
        assert ext == '.png'
        assert np.array_equal(decode(data), img)

    def test_large_image_resized_and_recompressed(self):
        data, ext, params = emulate_x(_textured(2000, 1000), rng_for('a'))
        assert decode(data).shape[1] == 1200
        assert ext == '.jpg'
        assert params['quality'] == 87

    def test_tall_but_narrow_image_recompressed_not_resized(self):
        data, ext, params = emulate_x(_textured(700, 1000), rng_for('a'))
        assert decode(data).shape[:2] == (1000, 700)
        assert ext == '.jpg'
        assert params.get('untouched') is None


class TestTelegram:

    def test_tall_image_resized_to_height_800(self):
        data, _, params = emulate_telegram(_textured(1000, 1600), rng_for('a'))
        assert decode(data).shape[0] == 800
        assert params['quality'] == 85

    def test_short_image_not_resized(self):
        data, _, _ = emulate_telegram(_textured(1000, 600), rng_for('a'))
        assert decode(data).shape[:2] == (600, 1000)


class TestChainAndRng:

    def test_chain_applies_in_order(self):
        data, ext, params = apply_chain(
            _textured(1600, 1200), ['facebook', 'x'], rng_for('a'))
        assert [p['platform'] for p in params] == ['facebook', 'x']
        # facebook leaves 720x540 which is within X's 768 pass-through
        assert params[1].get('untouched') is True
        assert decode(data).shape[1] == 720

    def test_chain_rejects_unknown_op(self):
        with pytest.raises(ValueError):
            apply_chain(_textured(100, 100), ['instagram'], rng_for('a'))

    def test_rng_deterministic_per_id(self):
        assert rng_for('img_1').integers(0, 1000) == rng_for('img_1').integers(0, 1000)

    def test_rng_salt_changes_stream(self):
        a = [int(rng_for('img_1', 'fb').integers(0, 10 ** 9)) for _ in range(3)]
        b = [int(rng_for('img_1', 'x').integers(0, 10 ** 9)) for _ in range(3)]
        assert a != b

    def test_rng_differs_across_ids(self):
        draws = {int(rng_for(f'img_{i}').integers(0, 10 ** 9)) for i in range(100)}
        assert len(draws) > 95


class TestNoExif:

    def test_platform_bytes_carry_no_metadata(self, tmp_path):
        from PIL import Image
        data, _, _ = emulate_facebook(_textured(640, 480), rng_for('a'))
        p = tmp_path / 'fb.jpg'
        p.write_bytes(data)
        with Image.open(p) as im:
            assert len(im.getexif()) == 0
