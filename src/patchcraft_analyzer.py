"""
PatchCraft-Inspired Texture Contrast Analyzer for AI Image Detection.

Inspired by: "Towards Universal Fake Image Detection by Detecting Closest
Real Image" (arXiv 2024, PatchCraft approach).

Key insight: AI generative models struggle to faithfully reproduce the
fine-grained texture found in real photographs. After high-pass filtering
(original − blurred), the contrast between rich-texture and poor-texture
patches separates the classes. Measured on this dataset with the l_div
diversity measure, REAL photos show the larger contrast (chaotic texture
next to smooth regions) while generators produce more uniform statistics —
see patchcraft_score and code_notes/09-patchcraft-analyzer.md for the
direction note.

This is robust to JPEG recompression because it uses relative differences
between patch groups — compression shifts rich and poor patches similarly,
preserving the contrast ratio.

Pipeline:
1. Apply high-pass filter: hp = original − gaussian_blur
2. Compute per-patch texture diversity l_div (4-direction mean absolute
   neighbour difference, PatchCraft Eq. 1 — see code_notes/09-patchcraft-analyzer.md)
3. Split patches into "rich" (top 50% l_div) and "poor" (bottom 50%)
4. Features: mean(rich) − mean(poor), mean(rich), mean(poor)

Features extracted (3 total):
    - texture_contrast: mean(rich_patches) - mean(poor_patches) — KEY FEATURE
    - texture_rich_mean: mean l_div of rich-texture patches
    - texture_poor_mean: mean l_div of poor-texture patches
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


def compute_patch_ldiv(
    high_pass: np.ndarray, patch_size: int = 32
) -> np.ndarray:
    """
    Computes per-patch texture diversity l_div: the mean absolute neighbour
    difference over four directions (horizontal, vertical, diagonal,
    anti-diagonal), per PatchCraft Eq. 1. See code_notes/09-patchcraft-analyzer.md
    for why this replaces patch variance.

    Args:
        high_pass: 2D high-pass filtered image (float64).
        patch_size: Side length of each square patch in pixels.

    Returns:
        1D array of per-patch l_div values (mean over difference terms, so
        the scale is independent of patch size).
    """
    h, w = high_pass.shape
    ph, pw = h // patch_size, w // patch_size
    if ph == 0 or pw == 0:
        return np.array([])
    x = high_pass[:ph * patch_size, :pw * patch_size]

    diffs = (
        np.abs(x[:, 1:] - x[:, :-1]),     # horizontal
        np.abs(x[1:, :] - x[:-1, :]),     # vertical
        np.abs(x[1:, 1:] - x[:-1, :-1]),  # diagonal
        np.abs(x[1:, :-1] - x[:-1, 1:]),  # anti-diagonal
    )

    totals = np.zeros((ph, pw), dtype=np.float64)
    counts = np.zeros((ph, pw), dtype=np.float64)
    for d in diffs:
        dh, dw = d.shape
        gh, gw = dh - dh % patch_size, dw - dw % patch_size
        if gh == 0 or gw == 0:
            continue
        block = d[:gh, :gw].reshape(gh // patch_size, patch_size,
                                    gw // patch_size, patch_size)
        sums = block.sum(axis=(1, 3))
        totals[:sums.shape[0], :sums.shape[1]] += sums
        counts[:sums.shape[0], :sums.shape[1]] += patch_size * patch_size

    valid = counts > 0
    ldiv = np.zeros_like(totals)
    ldiv[valid] = totals[valid] / counts[valid]
    return ldiv.ravel()


def extract_patchcraft_features(image_path: str) -> Dict[str, float]:
    """
    Extracts PatchCraft-inspired rich/poor texture contrast features.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary with 3 features (all built from per-patch l_div, the mean
        absolute neighbour difference; see compute_patch_ldiv):
            - texture_contrast: mean(rich) - mean(poor)
            - texture_rich_mean: mean l_div over the rich-texture patches (top 50%)
            - texture_poor_mean: mean l_div over the poor-texture patches (bottom 50%)
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
        ldiv = compute_patch_ldiv(high_pass)

        if len(ldiv) < 4:
            return _default

        # Split at median into rich (top 50%) and poor (bottom 50%)
        median = np.median(ldiv)
        rich = ldiv[ldiv >= median]
        poor = ldiv[ldiv < median]

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

    Measured direction with l_div (30 real vs 30 AI, June 2026): REAL photos
    show the larger rich/poor contrast (median 10.1 vs 6.8) — genuine scenes
    pair chaotic texture with smooth regions, while generators produce more
    uniform texture statistics. LOW contrast therefore reads as AI-like.
    The SVM uses the raw features; this score is for the explanation panel
    and the voting fallback only. See code_notes/09-patchcraft-analyzer.md.

    Args:
        image_path: Path to image file.

    Returns:
        float: Score in [0, 1]. Higher = more likely AI-generated.
    """
    features = extract_patchcraft_features(image_path)

    contrast = features['texture_contrast']

    # Scale calibrated from a 99-image sample (p95 ≈ 17.3, max ≈ 26.2);
    # divisor 20 keeps the mapping inside [0, 1] for all but extremes.
    score = float(np.clip(1.0 - contrast / 20.0, 0.0, 1.0))
    return score
