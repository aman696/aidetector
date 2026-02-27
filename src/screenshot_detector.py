"""
Screenshot Pre-Detection for AI Image Detector.

Detects whether an input image is a screenshot (or other screen-rendered
image) rather than a real photograph or AI-generated art. This is important
because screenshots share many statistical properties with AI-generated
images (no camera EXIF, no sensor noise, no pre-existing JPEG grid) and
therefore trigger systematic false positives in the main detector.

Three independent heuristics, each contributing to a confidence score:

  H1 — Near-zero noise variance
       Real camera photos always have measurable sensor noise.
       Screenshots rendered by a display have essentially zero residual.

  H2 — Low pixel histogram entropy
       Screenshots from UI, text, and solid-color regions cluster tightly
       at specific pixel values. Real photos fill the histogram broadly.

  H3 — Dimension matches common screen resolutions
       Screenshots are taken at known screen sizes. Checking width/height
       against a table of common screen widths is a cheap, reliable signal.

If ≥ 2 heuristics fire, is_screenshot = True.
"""

import numpy as np
import cv2
from typing import Dict, List

from src.utils import validate_image_path


# Common screen widths (px) — covers phones, tablets, laptops, desktops, 4K
_COMMON_SCREEN_WIDTHS = {
    360, 375, 390, 393, 412, 414, 428,   # Mobile portrait
    667, 720, 750, 780, 812, 844, 896,   # Mobile landscape / mid
    768, 800, 834, 1024, 1080, 1112,     # Tablet
    1280, 1366, 1440, 1536, 1600, 1920,  # Laptop / desktop
    2048, 2160, 2560, 2880, 3024, 3840,  # HiDPI / 4K
    4096,                                  # Cinema 4K
}

# Tolerance for "close to a screen resolution" (±4%)
_RES_TOLERANCE = 0.04

# H1: noise variance below this → likely screenshot
_NOISE_VAR_THRESHOLD = 3.0

# H2: pixel entropy below this → likely screenshot  (max possible = 8.0 bits for uint8)
_ENTROPY_THRESHOLD = 6.8


def _noise_variance(img_gray: np.ndarray) -> float:
    """Returns variance of high-pass noise residual (sigma=2 blur)."""
    img_float = img_gray.astype(np.float64)
    blurred = cv2.GaussianBlur(img_float, (13, 13), 2.0)
    residual = img_float - blurred
    return float(np.var(residual))


def _pixel_histogram_entropy(img_gray: np.ndarray) -> float:
    """
    Computes Shannon entropy (bits) of the grayscale pixel histogram.
    Real photos: ~7–8 bits (broad distribution).
    Screenshots: lower (clustered values from rendering).
    """
    hist = np.bincount(img_gray.flatten(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total < 1:
        return 0.0
    p = hist[hist > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _matches_screen_resolution(h: int, w: int) -> bool:
    """
    Returns True if width or height is within ±RES_TOLERANCE of a known
    screen resolution, or if the aspect ratio matches a common screen ratio.
    """
    for sw in _COMMON_SCREEN_WIDTHS:
        if abs(w - sw) / sw <= _RES_TOLERANCE:
            return True
        if abs(h - sw) / sw <= _RES_TOLERANCE:
            return True

    # Check common screen aspect ratios: 16:9, 9:16, 4:3, 3:4, 16:10, 21:9
    common_ratios = [16 / 9, 9 / 16, 4 / 3, 3 / 4, 16 / 10, 10 / 16, 21 / 9]
    if h > 0:
        aspect = w / h
        for r in common_ratios:
            if abs(aspect - r) / r < 0.03:  # within 3%
                return True

    return False


def detect_screenshot(image_path: str) -> Dict:
    """
    Detects whether an image is a screenshot rather than a photograph.

    Args:
        image_path: Path to image file.

    Returns:
        Dict with keys:
            - 'is_screenshot' (bool): True if likely a screenshot
            - 'confidence' (float): 0.0–1.0 confidence in screenshot classification
            - 'reasons' (List[str]): Which heuristics fired
    """
    result = {
        'is_screenshot': False,
        'confidence': 0.0,
        'reasons': [],
    }

    try:
        validate_image_path(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return result

        h, w = img.shape
        heuristics_fired = 0
        confidence_sum = 0.0

        # H1: Near-zero noise variance
        nv = _noise_variance(img)
        if nv < _NOISE_VAR_THRESHOLD:
            heuristics_fired += 1
            # Score inversely proportional to variance (lower variance = more confident)
            h1_conf = float(np.clip(1.0 - nv / _NOISE_VAR_THRESHOLD, 0.0, 1.0))
            confidence_sum += h1_conf
            result['reasons'].append(
                f"H1: Near-zero sensor noise (variance={nv:.2f} < {_NOISE_VAR_THRESHOLD})"
            )

        # H2: Low pixel histogram entropy
        entropy = _pixel_histogram_entropy(img)
        if entropy < _ENTROPY_THRESHOLD:
            heuristics_fired += 1
            h2_conf = float(np.clip(1.0 - entropy / _ENTROPY_THRESHOLD, 0.0, 1.0))
            confidence_sum += h2_conf
            result['reasons'].append(
                f"H2: Low pixel histogram entropy ({entropy:.2f} < {_ENTROPY_THRESHOLD})"
            )

        # H3: Dimension matches screen resolution
        if _matches_screen_resolution(h, w):
            heuristics_fired += 1
            confidence_sum += 0.5  # weaker heuristic — many real photos also happen to be these sizes
            result['reasons'].append(
                f"H3: Dimensions {w}×{h} match common screen resolution or aspect ratio"
            )

        # Classify: 2+ heuristics → screenshot
        if heuristics_fired >= 2:
            result['is_screenshot'] = True
            # Normalize confidence
            result['confidence'] = float(np.clip(confidence_sum / heuristics_fired, 0.0, 1.0))
        elif heuristics_fired == 1:
            result['confidence'] = float(np.clip(confidence_sum * 0.3, 0.0, 1.0))

    except Exception:
        pass  # On any error, return default (not screenshot)

    return result
