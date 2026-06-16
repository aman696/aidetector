"""
Platform-channel emulation: pure transforms reproducing what social media
platforms do to uploaded images.

Parameters are measured values from the TrueFake dataset paper, not tuning
choices. Emulators return the
final ENCODED bytes
(plus the parameters applied), so what is written to disk is the single
platform compression pass, never a second re-encode. Nothing here touches
disk.

The JPEG encoding via cv2 carries no EXIF — metadata stripping is therefore
symmetric across classes by construction.
"""

import hashlib
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np


def _encode(img: np.ndarray, ext: str, quality: int = 95) -> bytes:
    flags = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext == '.jpg' else []
    ok, buf = cv2.imencode(ext, img, flags)
    if not ok:
        raise ValueError(f"{ext} encoding failed")
    return buf.tobytes()


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image decoding failed")
    return img


def _downscale_to_width(img: np.ndarray, max_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    new_h = max(1, int(round(h * max_width / w)))
    return cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)


def _downscale_to_height(img: np.ndarray, max_height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= max_height:
        return img
    new_w = max(1, int(round(w * max_height / h)))
    return cv2.resize(img, (new_w, max_height), interpolation=cv2.INTER_AREA)


def emulate_facebook(img: np.ndarray,
                     rng: np.random.Generator) -> Tuple[bytes, str, Dict]:
    """Facebook: resize to width 720 if wider, JPEG QF sampled from 61-92."""
    quality = int(rng.integers(61, 93))
    out = _downscale_to_width(img, 720)
    return (_encode(out, '.jpg', quality), '.jpg',
            {'platform': 'facebook', 'quality': quality, 'max_width': 720})


def emulate_x(img: np.ndarray,
              rng: np.random.Generator) -> Tuple[bytes, str, Dict]:
    """X: images up to 768x768 pass through untouched (saved losslessly);
    else resize to width 1200 if wider and recompress at QF 87."""
    h, w = img.shape[:2]
    if w <= 768 and h <= 768:
        return _encode(img, '.png'), '.png', {'platform': 'x', 'untouched': True}
    out = _downscale_to_width(img, 1200)
    return (_encode(out, '.jpg', 87), '.jpg',
            {'platform': 'x', 'quality': 87, 'max_width': 1200})


def emulate_telegram(img: np.ndarray,
                     rng: np.random.Generator) -> Tuple[bytes, str, Dict]:
    """Telegram: resize to height 800 if taller, JPEG QF fixed at 85."""
    out = _downscale_to_height(img, 800)
    return (_encode(out, '.jpg', 85), '.jpg',
            {'platform': 'telegram', 'quality': 85, 'max_height': 800})


PLATFORMS: Dict[str, Callable] = {
    'facebook': emulate_facebook,
    'x': emulate_x,
    'telegram': emulate_telegram,
}


def apply_chain(img: np.ndarray,
                ops: List[str],
                rng: np.random.Generator) -> Tuple[bytes, str, List[Dict]]:
    """
    Applies a sequence of platform emulations, e.g. ['facebook', 'x'],
    decoding between hops exactly as a re-upload would.

    Screenshot hops are not computed here (they need a real browser, see
    scripts/capture_screenshots.py); chains containing one are assembled by
    scripts/build_derived.py from captured files.
    """
    params: List[Dict] = []
    data, ext = None, None
    current = img
    for op in ops:
        if op not in PLATFORMS:
            raise ValueError(f"Unknown platform op '{op}' (have {sorted(PLATFORMS)})")
        data, ext, p = PLATFORMS[op](current, rng)
        params.append(p)
        current = decode(data)
    return data, ext, params


def rng_for(base_id: str, salt: str = '') -> np.random.Generator:
    """
    Deterministic per-image RNG: rebuilds make identical decisions per image,
    and adding or removing images never shifts the randomness of others.
    """
    digest = hashlib.md5((base_id + ':' + salt).encode('utf-8')).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))
