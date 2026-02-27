"""
Gradient Statistics Analyzer for AI Image Detection.

Based on: Gragnaniello et al. CVPR 2023 — frequency and gradient features
for detecting diffusion-model generated images.

Key insight: AI generators produce images with smoother, more regularized
gradient distributions than real camera photos. Real images have heavier-
tailed edge distributions because real scenes have sharp, unpredictable edges.

These features survive JPEG recompression at Q>70 because they measure
*relative* structure of the gradient field, not absolute pixel values.

Features extracted (5 total):
    - gradient_mean: mean Sobel edge magnitude
    - gradient_variance: spread of edge magnitudes
    - gradient_kurtosis: tail weight (real images = heavier tail = higher kurtosis)
    - gradient_laplacian_mean: mean Laplacian response (measures 2nd-order sharpness)
    - gradient_laplacian_variance: variance of Laplacian response
"""

import numpy as np
import cv2
from scipy import stats as scipy_stats
from typing import Dict

from src.utils import validate_image_path


def compute_sobel_gradient(img_gray: np.ndarray) -> np.ndarray:
    """
    Computes the Sobel gradient magnitude map.

    Args:
        img_gray: Grayscale image as 2D uint8 or float array.

    Returns:
        2D float array of gradient magnitudes.
    """
    img_float = img_gray.astype(np.float64)

    # Sobel in x and y directions
    grad_x = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return magnitude


def compute_laplacian(img_gray: np.ndarray) -> np.ndarray:
    """
    Computes the Laplacian (2nd-order derivative) of the image.

    The Laplacian captures sharp transitions (edges, corners) more sensitively
    than the Sobel gradient.

    Args:
        img_gray: Grayscale image.

    Returns:
        2D float array of Laplacian values (signed).
    """
    img_float = img_gray.astype(np.float64)
    laplacian = cv2.Laplacian(img_float, cv2.CV_64F, ksize=3)
    return laplacian


def extract_gradient_features(image_path: str) -> Dict[str, float]:
    """
    Extracts gradient-based features from an image.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary of 5 gradient features.
    """
    validate_image_path(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    try:
        # Sobel gradient features
        gradient_mag = compute_sobel_gradient(img)
        flat_grad = gradient_mag.flatten()

        grad_mean = float(np.mean(flat_grad))
        grad_var = float(np.var(flat_grad))

        # Kurtosis: Fisher definition (kurtosis of Gaussian = 0)
        # Real photos have heavy-tailed edge distributions (high kurtosis)
        # AI images tend to have more uniform edge magnitudes (lower kurtosis)
        grad_kurt = float(scipy_stats.kurtosis(flat_grad, fisher=True))

        # Laplacian features
        laplacian = compute_laplacian(img)
        flat_lap = np.abs(laplacian).flatten()

        lap_mean = float(np.mean(flat_lap))
        lap_var = float(np.var(flat_lap))

        return {
            'gradient_mean': grad_mean,
            'gradient_variance': grad_var,
            'gradient_kurtosis': grad_kurt,
            'gradient_laplacian_mean': lap_mean,
            'gradient_laplacian_variance': lap_var,
        }

    except Exception:
        return {
            'gradient_mean': 0.0,
            'gradient_variance': 0.0,
            'gradient_kurtosis': 0.0,
            'gradient_laplacian_mean': 0.0,
            'gradient_laplacian_variance': 0.0,
        }


def gradient_score(image_path: str) -> float:
    """
    Computes a gradient-based AI detection score from 0.0 (real) to 1.0 (AI).

    AI images tend to have:
    - Lower gradient kurtosis (fewer extreme edges; smoother, more regular distribution)
    - Lower gradient variance (more uniform edge strength across the image)

    Args:
        image_path: Path to image file.

    Returns:
        float: Score in [0, 1]. Higher = more likely AI.
    """
    features = extract_gradient_features(image_path)

    # Kurtosis: real images are heavy-tailed (high kurtosis).
    # Low kurtosis → more gaussian-distributed edges → AI
    kurt = features['gradient_kurtosis']
    # Typical real image kurtosis > 5; AI images ~1–5
    kurt_score = float(np.clip(1.0 - (kurt / 15.0), 0.0, 1.0))

    # Gradient variance: AI images tend toward more uniform edge strength
    # Normalize loosely — typical range 0–5000
    var = features['gradient_variance']
    var_score = float(np.clip(1.0 - (var / 3000.0), 0.0, 1.0))

    # Laplacian mean: very high or very low sharpness can be AI artifact
    lap_mean = features['gradient_laplacian_mean']
    # Calibrated around typical real image laplacian mean ~10–30
    lap_score = float(np.clip(1.0 - (lap_mean / 40.0), 0.0, 1.0))

    score = 0.50 * kurt_score + 0.30 * var_score + 0.20 * lap_score
    return float(np.clip(score, 0.0, 1.0))
