"""
PatchCraft-Inspired Texture Contrast Analyzer for AI Image Detection.

Inspired by: "Towards Universal Fake Image Detection by Detecting Closest
Real Image" (arXiv 2024, PatchCraft approach).

Key insight: AI generative models struggle to faithfully reproduce the
fine-grained texture found in real photographs. When you apply a high-pass
filter (original − blurred) and compare patches with rich texture vs poor
texture, AI images show a characteristically HIGHER contrast between these
two patch types.

This is robust to JPEG recompression because it uses relative differences
between patch groups — compression shifts rich and poor patches similarly,
preserving the contrast ratio.

Pipeline:
1. Apply high-pass filter: hp = original − gaussian_blur
2. Compute per-patch variance of the high-pass image
3. Split patches into "rich" (top 50% variance) and "poor" (bottom 50%)
4. Features: mean(rich) − mean(poor), mean(rich), mean(poor)

Features extracted (3 total):
    - texture_contrast: mean(rich_patches) - mean(poor_patches) — KEY FEATURE
    - texture_rich_mean: mean variance of rich-texture patches
    - texture_poor_mean: mean variance of poor-texture patches
"""

import numpy as np
import cv2
from typing import Dict

from src.utils import validate_image_path


def compute_high_pass(img_gray: np.ndarray, blur_sigma: float = 3.0) -> np.ndarray:
    """
    Computes the high-pass filtered image: original − gaussian_blur.

    This isolates fine-grained texture and noise while removing
    low-frequency content (color, lighting, broad shapes).

    Args:
        img_gray: Grayscale image (2D array).
        blur_sigma: Gaussian blur sigma for creating the low-pass reference.

    Returns:
        High-pass residual as float64 array.
    """
    img_float = img_gray.astype(np.float64)
    ksize = int(blur_sigma * 6) | 1  # Must be odd
    blurred = cv2.GaussianBlur(img_float, (ksize, ksize), blur_sigma)
    return img_float - blurred


def compute_patch_variances(
    high_pass: np.ndarray, patch_size: int = 32
) -> np.ndarray:
    """
    Computes variance of high-pass energy within non-overlapping patches.

    Args:
        high_pass: 2D high-pass filtered image (float64).
        patch_size: Side length of each square patch in pixels.

    Returns:
        1D array of per-patch variances.
    """
    h, w = high_pass.shape
    variances = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = high_pass[y : y + patch_size, x : x + patch_size]
            variances.append(float(np.var(patch)))
    return np.array(variances)


def extract_patchcraft_features(image_path: str) -> Dict[str, float]:
    """
    Extracts PatchCraft-inspired rich/poor texture contrast features.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary with 3 features:
            - texture_contrast: mean(rich) - mean(poor); higher = more AI-like
            - texture_rich_mean: average variance in rich-texture patches
            - texture_poor_mean: average variance in poor-texture patches
    """
    _default = {
        'texture_contrast': 0.0,
        'texture_rich_mean': 0.0,
        'texture_poor_mean': 0.0,
    }

    try:
        validate_image_path(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return _default

        h, w = img.shape

        # Resolution guard: need at least 256px on each side to get
        # enough 32×32 patches for reliable rich/poor statistics.
        # Below this threshold, return neutral defaults so the SVM
        # doesn't penalize low-res real images as AI-generated.
        if min(h, w) < 256:
            return _default

        high_pass = compute_high_pass(img)
        variances = compute_patch_variances(high_pass)

        if len(variances) < 4:
            return _default

        # Split at median into rich (top 50%) and poor (bottom 50%)
        median = np.median(variances)
        rich = variances[variances >= median]
        poor = variances[variances < median]

        rich_mean = float(np.mean(rich)) if len(rich) > 0 else 0.0
        poor_mean = float(np.mean(poor)) if len(poor) > 0 else 0.0
        contrast = rich_mean - poor_mean

        return {
            'texture_contrast': contrast,
            'texture_rich_mean': rich_mean,
            'texture_poor_mean': poor_mean,
        }

    except Exception:
        return _default


def patchcraft_score(image_path: str) -> float:
    """
    Computes a PatchCraft-inspired AI detection score in [0.0, 1.0].

    Higher texture contrast between rich and poor patches is the key
    indicator of AI generation.

    Args:
        image_path: Path to image file.

    Returns:
        float: Score in [0, 1]. Higher = more likely AI-generated.
    """
    features = extract_patchcraft_features(image_path)

    contrast = features['texture_contrast']

    # Calibrated loosely: real photos typically have contrast in 0–100,
    # AI images tend toward 100–400+ because generators produce
    # artificially smooth regions alongside over-sharpened detail regions.
    score = float(np.clip(contrast / 300.0, 0.0, 1.0))
    return score
