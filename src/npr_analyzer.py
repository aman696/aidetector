"""
NPR (Neighboring Pixel Relationships) analyzer.

Based on Tan et al., CVPR 2024: generator up-sampling layers force the
pixels inside each 2x2 grid into artificial mutual correlation. The NPR map
is the residual `image - upsample2(downsample2(image))` with
nearest-neighbour resampling, which is equivalent to within-2x2-grid pixel
differences. Real images leave heavy-tailed, irregular residue; generated
images leave residue that is too uniform or too structured.

Features extracted (6 total):
    - npr_mean_abs: mean absolute value of the NPR map
    - npr_std: standard deviation of the NPR map
    - npr_skewness: distribution asymmetry of NPR values
    - npr_kurtosis: tail weight of NPR values (heavy tails = real-like)
    - npr_diag_axial_ratio: diagonal vs axial neighbour-difference energy
    - npr_energy_ratio: NPR-map energy above half-Nyquist / total energy
"""

from typing import Dict

import cv2
import numpy as np
from scipy import stats

from src.utils import validate_image_path

_DEFAULTS = {
    'npr_mean_abs': 0.0,
    'npr_std': 0.0,
    'npr_skewness': 0.0,
    'npr_kurtosis': 0.0,
    'npr_diag_axial_ratio': 1.0,
    'npr_energy_ratio': 0.0,
}


def compute_npr_map(img_gray: np.ndarray) -> np.ndarray:
    """
    NPR residual: image minus its nearest-neighbour down-up resampled copy.

    Args:
        img_gray: 2D grayscale image.

    Returns:
        float64 residual map with the same even-cropped shape.
    """
    h, w = img_gray.shape
    x = img_gray[:h - h % 2, :w - w % 2].astype(np.float64)
    down = x[::2, ::2]
    up = np.repeat(np.repeat(down, 2, axis=0), 2, axis=1)
    return x - up


def _diag_axial_ratio(npr: np.ndarray) -> float:
    """Energy of diagonal neighbour differences relative to axial ones —
    generators reproduce diagonal detail worst (Corvi finding)."""
    axial = (np.mean(np.abs(npr[:, 1:] - npr[:, :-1])) +
             np.mean(np.abs(npr[1:, :] - npr[:-1, :])))
    diag = (np.mean(np.abs(npr[1:, 1:] - npr[:-1, :-1])) +
            np.mean(np.abs(npr[1:, :-1] - npr[:-1, 1:])))
    if axial < 1e-12:
        return 1.0
    return float(diag / axial)


def _high_band_energy_ratio(npr: np.ndarray) -> float:
    """Fraction of NPR spectral energy beyond half the Nyquist radius."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(npr))) ** 2
    h, w = spectrum.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt(((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2)
    total = float(spectrum.sum())
    if total < 1e-12:
        return 0.0
    return float(spectrum[radius > 0.5].sum() / total)


def extract_npr_features(image_path: str) -> Dict[str, float]:
    """
    Extracts the 6 NPR features; neutral defaults on any failure.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary with the 6 npr_* features.
    """
    try:
        validate_image_path(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None or min(img.shape) < 64:
            return dict(_DEFAULTS)

        npr = compute_npr_map(img)
        values = npr.ravel()
        std = float(np.std(values))

        return {
            'npr_mean_abs': float(np.mean(np.abs(values))),
            'npr_std': std,
            'npr_skewness': float(stats.skew(values)) if std > 1e-9 else 0.0,
            'npr_kurtosis': float(stats.kurtosis(values)) if std > 1e-9 else 0.0,
            'npr_diag_axial_ratio': _diag_axial_ratio(npr),
            'npr_energy_ratio': _high_band_energy_ratio(npr),
        }
    except Exception:
        return dict(_DEFAULTS)


def npr_score(image_path: str) -> float:
    """
    Heuristic [0, 1] score for the explanation panel; the SVM uses the raw
    features. Lower NPR magnitude (residue too smooth) reads as more AI-like.

    Basis measured on 30 real vs 30 AI samples (June 2026): npr_mean_abs
    medians 6.1 (real) vs 4.8 (AI); kurtosis did not separate (27.1 vs 27.8)
    and is not used here.
    """
    features = extract_npr_features(image_path)
    mean_abs = features['npr_mean_abs']
    # 12.0 is a soft scale (≈ 2x the real-class median), not a threshold.
    return float(np.clip(1.0 - mean_abs / 12.0, 0.0, 1.0))
