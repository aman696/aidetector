"""
DCT (Discrete Cosine Transform) Block Analyzer for AI Image Detection.

Based on Frank et al. 2020 — "Leveraging Frequency Analysis for Deep Fake Image Recognition"
Uses block-level DCT coefficient analysis (aligned with JPEG 8×8 encoding).

Method:
1. Divide image into 8×8 blocks
2. Apply 2D DCT to each block
3. Analyze coefficient statistics — GAN/AI patterns show systematic deviations
"""

import numpy as np
import cv2
from scipy.fft import dctn
from typing import Dict

from src.utils import validate_image_path


def block_dct(image_path: str, block_size: int = 8) -> np.ndarray:
    """
    Computes 2D DCT on non-overlapping blocks of a grayscale image.
    
    Args:
        image_path: Path to image file.
        block_size: Block size (default 8, matching JPEG).
    
    Returns:
        Array of shape (num_blocks, block_size, block_size) with DCT coefficients.
    """
    validate_image_path(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")
    
    img = img.astype(np.float64)
    h, w = img.shape
    
    # Trim to exact block multiples
    h_blocks = h // block_size
    w_blocks = w // block_size
    img = img[:h_blocks * block_size, :w_blocks * block_size]
    
    blocks = []
    for y in range(0, h_blocks * block_size, block_size):
        for x in range(0, w_blocks * block_size, block_size):
            block = img[y:y + block_size, x:x + block_size]
            dct_block = dctn(block, type=2, norm='ortho')
            blocks.append(dct_block)
    
    return np.array(blocks)


def dct_ac_energy_ratio(dct_blocks: np.ndarray) -> float:
    """
    Ratio of AC coefficient energy to total energy.
    AC = all coefficients except DC (top-left [0,0]).
    AI images may have different AC energy distributions.
    """
    dc_energy = np.sum(dct_blocks[:, 0, 0] ** 2)
    total_energy = np.sum(dct_blocks ** 2)
    
    if total_energy < 1e-10:
        return 0.0
    
    ac_energy = total_energy - dc_energy
    return float(ac_energy / total_energy)


def dct_high_freq_energy(dct_blocks: np.ndarray) -> float:
    """
    Proportion of energy in high-frequency DCT coefficients.
    High-frequency = bottom-right quadrant of each block.
    """
    bs = dct_blocks.shape[1]  # block size
    half = bs // 2
    
    hf_energy = np.sum(dct_blocks[:, half:, half:] ** 2)
    total_energy = np.sum(dct_blocks ** 2)
    
    if total_energy < 1e-10:
        return 0.0
    
    return float(hf_energy / total_energy)


def dct_coefficient_kurtosis(dct_blocks: np.ndarray) -> float:
    """
    Kurtosis of AC coefficient distribution across all blocks.
    Natural images tend to have heavy-tailed (high kurtosis) DCT distributions.
    AI images may have lighter tails.
    """
    from scipy import stats as scipy_stats
    
    # Extract all AC coefficients (exclude DC)
    ac_coeffs = []
    for block in dct_blocks:
        # Flatten and skip DC
        flat = block.flatten()
        ac_coeffs.extend(flat[1:])  # Skip [0,0] = DC
    
    ac = np.array(ac_coeffs)
    if np.std(ac) < 1e-10:
        return 0.0
    
    return float(scipy_stats.kurtosis(ac, fisher=True))


def dct_coefficient_variance(dct_blocks: np.ndarray) -> float:
    """Variance of all AC coefficients."""
    ac_coeffs = dct_blocks.copy()
    ac_coeffs[:, 0, 0] = 0.0  # Zero out DC
    return float(np.var(ac_coeffs))


def dct_block_dc_variance(dct_blocks: np.ndarray) -> float:
    """
    Variance of DC coefficients across blocks.
    Measures how much average brightness varies block-to-block.
    """
    dc_values = dct_blocks[:, 0, 0]
    return float(np.var(dc_values))


def dct_zigzag_energy_decay(dct_blocks: np.ndarray) -> float:
    """
    Rate at which energy decays along the zigzag scan order.
    Natural images have a specific decay pattern; AI images may differ.
    Simplified: ratio of energy in first quarter vs last quarter of zigzag.
    """
    bs = dct_blocks.shape[1]
    total_coeffs = bs * bs
    
    # Create zigzag order indices
    zigzag_indices = _zigzag_order(bs)
    
    quarter = total_coeffs // 4
    first_q_indices = zigzag_indices[:quarter]
    last_q_indices = zigzag_indices[-quarter:]
    
    first_energy = 0.0
    last_energy = 0.0
    
    for block in dct_blocks:
        flat = block.flatten()
        first_energy += np.sum(flat[first_q_indices] ** 2)
        last_energy += np.sum(flat[last_q_indices] ** 2)
    
    if first_energy < 1e-10:
        return 0.0
    
    return float(last_energy / first_energy)


def _zigzag_order(n: int) -> np.ndarray:
    """Returns indices for zigzag scan order of an n×n matrix."""
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(min(s, n - 1), max(0, s - n + 1) - 1, -1):
                j = s - i
                indices.append(i * n + j)
        else:
            for j in range(min(s, n - 1), max(0, s - n + 1) - 1, -1):
                i = s - j
                indices.append(i * n + j)
    return np.array(indices)


# ---------------------------------------------------------------------------
# JPEG block boundary consistency (social-media resilient)
# ---------------------------------------------------------------------------

def jpeg_block_boundary_ratio(image_path: str, block_size: int = 8) -> float:
    """
    Ratio of mean absolute pixel differences at JPEG block boundaries vs interior.

    When a real JPEG is re-compressed by social media (Q=75), the original
    8×8 JPEG grid survives — boundaries remain stronger than interior.
    AI images are often generated as PNG first, then JPEG-compressed once,
    producing a *more uniform* discontinuity pattern (ratio closer to 1.0).

    Real re-uploaded JPEG: ratio > 1  (boundaries are stronger)
    AI PNG → JPEG: ratio ≈ 1  (boundaries not meaningfully stronger)

    Returns:
        Boundary-to-interior ratio clamped to [0, 5]. Higher = more real-like.
    """
    try:
        validate_image_path(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 1.0

        img = img.astype(np.float64)
        h, w = img.shape

        # Horizontal differences (axis=1): shape (h, w-1)
        hdiff = np.abs(np.diff(img, axis=1))
        # Columns where (col+1) is a block boundary (col+1) % block_size == 0
        boundary_cols = np.arange(block_size - 1, w - 1, block_size)  # 7, 15, 23 ...
        mask_h = np.zeros(w - 1, dtype=bool)
        mask_h[boundary_cols] = True

        h_boundary = hdiff[:, mask_h].mean() if mask_h.any() else 0.0
        h_interior = hdiff[:, ~mask_h].mean() if (~mask_h).any() else 0.0

        # Vertical differences (axis=0): shape (h-1, w)
        vdiff = np.abs(np.diff(img, axis=0))
        boundary_rows = np.arange(block_size - 1, h - 1, block_size)
        mask_v = np.zeros(h - 1, dtype=bool)
        mask_v[boundary_rows] = True

        v_boundary = vdiff[mask_v, :].mean() if mask_v.any() else 0.0
        v_interior = vdiff[~mask_v, :].mean() if (~mask_v).any() else 0.0

        boundary_mean = (h_boundary + v_boundary) / 2.0
        interior_mean = (h_interior + v_interior) / 2.0

        if interior_mean < 1e-10:
            return 1.0

        return float(np.clip(boundary_mean / interior_mean, 0.0, 5.0))
    except Exception:
        return 1.0


def jpeg_boundary_variance_ratio(image_path: str, block_size: int = 8) -> float:
    """
    Ratio of variance of boundary differences to variance of interior differences.

    A high variance ratio indicates the boundary structure is more
    irregular (as in real re-compressed JPEGs with their original grid).
    AI images show more uniform discontinuity.

    Returns:
        Variance ratio clamped to [0, 10].
    """
    try:
        validate_image_path(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 1.0

        img = img.astype(np.float64)
        h, w = img.shape

        hdiff = np.abs(np.diff(img, axis=1))
        boundary_cols = np.arange(block_size - 1, w - 1, block_size)
        mask_h = np.zeros(w - 1, dtype=bool)
        mask_h[boundary_cols] = True

        b_vals = hdiff[:, mask_h].flatten() if mask_h.any() else np.array([])
        i_vals = hdiff[:, ~mask_h].flatten() if (~mask_h).any() else np.array([])

        if len(b_vals) < 2 or len(i_vals) < 2:
            return 1.0

        var_b = np.var(b_vals)
        var_i = np.var(i_vals)

        if var_i < 1e-10:
            return 1.0

        return float(np.clip(var_b / var_i, 0.0, 10.0))
    except Exception:
        return 1.0


def extract_dct_features(image_path: str) -> Dict[str, float]:
    """
    Extracts all DCT-based features from an image.

    Returns:
        Dict with 8 features (6 original + 2 JPEG boundary).
    """
    try:
        dct_blocks = block_dct(image_path)
    except Exception:
        return {
            'dct_ac_energy_ratio': 0.5,
            'dct_high_freq_energy': 0.0,
            'dct_coeff_kurtosis': 0.0,
            'dct_coeff_variance': 0.0,
            'dct_dc_variance': 0.0,
            'dct_zigzag_decay': 0.0,
            'dct_boundary_ratio': 1.0,
            'dct_boundary_var_ratio': 1.0,
        }

    boundary_ratio = jpeg_block_boundary_ratio(image_path)
    boundary_var_ratio = jpeg_boundary_variance_ratio(image_path)

    return {
        'dct_ac_energy_ratio': dct_ac_energy_ratio(dct_blocks),
        'dct_high_freq_energy': dct_high_freq_energy(dct_blocks),
        'dct_coeff_kurtosis': dct_coefficient_kurtosis(dct_blocks),
        'dct_coeff_variance': dct_coefficient_variance(dct_blocks),
        'dct_dc_variance': dct_block_dc_variance(dct_blocks),
        'dct_zigzag_decay': dct_zigzag_energy_decay(dct_blocks),
        'dct_boundary_ratio': boundary_ratio,
        'dct_boundary_var_ratio': boundary_var_ratio,
    }


def dct_score(image_path: str) -> float:
    """
    Computes a DCT-based score from 0.0 (likely real) to 1.0 (likely AI).
    
    AI images tend to have:
    - Lower AC kurtosis (lighter tails)
    - Different energy decay patterns
    - More uniform block statistics
    """
    features = extract_dct_features(image_path)
    
    # Kurtosis: natural images have high kurtosis, AI images lower
    kurtosis = features['dct_coeff_kurtosis']
    # Typical range: 5-50 for real, 1-10 for AI
    kurt_score = max(0.0, 1.0 - min(1.0, kurtosis / 20.0))
    
    # Energy decay: AI images may have different decay rates
    decay = features['dct_zigzag_decay']
    decay_score = min(1.0, decay * 5.0)  # Higher decay ratio → more AI-like
    
    # High freq energy: AI images often have less high-freq detail
    hf = features['dct_high_freq_energy']
    hf_score = max(0.0, 1.0 - min(1.0, hf / 0.05))  # Less HF = more AI-like
    
    score = 0.40 * kurt_score + 0.30 * decay_score + 0.30 * hf_score
    return float(np.clip(score, 0.0, 1.0))
