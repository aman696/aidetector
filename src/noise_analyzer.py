"""
Noise Residual Analyzer for AI Image Detection.

Inspired by PRNU (Photo Response Non-Uniformity) noise analysis.
Camera sensor noise patterns differ from AI-generated noise patterns.

Method:
1. Apply denoising filter (Gaussian blur) to image
2. Compute residual = original - denoised
3. Analyze residual statistics: variance, kurtosis, spectral entropy, autocorrelation

Also includes social-media-resilient features:
- Multi-scale noise ratios (sigma 1 vs 5, sigma 3 vs 5)
- Chroma inter-channel noise correlations (R-G, R-B, G-B)

Real camera photos have structured sensor noise.
AI images have uniform/synthetic noise patterns.
Screenshots have display-rendering noise (different from both).
"""

import numpy as np
import cv2
from scipy import stats as scipy_stats
from typing import Dict

from src.utils import validate_image_path


def compute_noise_residual(image_path: str, blur_sigma: float = 3.0) -> np.ndarray:
    """
    Computes the noise residual by subtracting a denoised version from the original.
    
    Args:
        image_path: Path to image file.
        blur_sigma: Gaussian blur sigma for denoising.
        
    Returns:
        Noise residual as float array.
    """
    validate_image_path(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")
    
    img_float = img.astype(np.float64)
    
    # Denoise with Gaussian blur
    ksize = int(blur_sigma * 6) | 1  # Ensure odd kernel size
    denoised = cv2.GaussianBlur(img_float, (ksize, ksize), blur_sigma)
    
    # Residual = original - denoised
    residual = img_float - denoised
    return residual


def residual_variance(residual: np.ndarray) -> float:
    """Global variance of the noise residual."""
    return float(np.var(residual))


def residual_kurtosis(residual: np.ndarray) -> float:
    """
    Kurtosis of noise residual distribution.
    Real sensor noise tends to be Gaussian (kurtosis ≈ 0).
    AI noise patterns may have different distributions.
    """
    flat = residual.flatten()
    if np.std(flat) < 1e-10:
        return 0.0
    return float(scipy_stats.kurtosis(flat, fisher=True))


def residual_skewness(residual: np.ndarray) -> float:
    """Skewness of noise residual."""
    flat = residual.flatten()
    if np.std(flat) < 1e-10:
        return 0.0
    return float(scipy_stats.skew(flat))


def residual_spectral_entropy(residual: np.ndarray) -> float:
    """
    Spectral entropy of the noise residual.
    Measures how uniformly energy is distributed across frequencies.
    Higher entropy = more random/uniform noise (like sensor noise).
    Lower entropy = more structured patterns (like AI artifacts).
    """
    # 2D FFT of residual
    f_transform = np.fft.fft2(residual)
    magnitude = np.abs(f_transform)
    
    # Compute power spectrum
    power = magnitude ** 2
    total_power = np.sum(power)
    
    if total_power < 1e-10:
        return 0.0
    
    # Normalize to probability distribution
    p = power / total_power
    p = p.flatten()
    p = p[p > 0]  # Remove zeros for log
    
    # Shannon entropy
    entropy = -np.sum(p * np.log2(p))
    
    # Normalize by max possible entropy
    max_entropy = np.log2(len(p))
    if max_entropy > 0:
        entropy /= max_entropy
    
    return float(entropy)


def residual_spatial_autocorrelation(residual: np.ndarray) -> float:
    """
    Spatial autocorrelation at lag-1 (adjacent pixels).
    Real sensor noise has low autocorrelation (independent pixels).
    AI patterns may have higher autocorrelation (structured).
    """
    if residual.size < 4:
        return 0.0
    
    # Horizontal autocorrelation
    h_corr = np.corrcoef(residual[:, :-1].flatten(), residual[:, 1:].flatten())[0, 1]
    # Vertical autocorrelation
    v_corr = np.corrcoef(residual[:-1, :].flatten(), residual[1:, :].flatten())[0, 1]
    
    if np.isnan(h_corr):
        h_corr = 0.0
    if np.isnan(v_corr):
        v_corr = 0.0
    
    return float((h_corr + v_corr) / 2.0)


def residual_block_variance_std(residual: np.ndarray, block_size: int = 32) -> float:
    """
    Standard deviation of local block variances.
    Real images: different textures → high variance of local noise.
    AI images: more uniform noise → low variance of local noise.
    """
    h, w = residual.shape
    variances = []
    
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = residual[y:y + block_size, x:x + block_size]
            variances.append(np.var(block))
    
    if len(variances) < 2:
        return 0.0
    
    return float(np.std(variances))


# ---------------------------------------------------------------------------
# Multi-scale noise ratio (robust to JPEG recompression)
# ---------------------------------------------------------------------------

def _gaussian_residual(img_float: np.ndarray, sigma: float) -> np.ndarray:
    """Returns img - gaussian_blur(img, sigma)."""
    ksize = int(sigma * 6) | 1
    blurred = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)
    return img_float - blurred


def multi_scale_noise_ratio_1_5(img_float: np.ndarray) -> float:
    """
    Ratio of noise variance at sigma=1 vs sigma=5.

    Real camera sensor noise is largely at fine scales (sigma=1 captures it).
    AI images often have a different scale distribution.
    The ratio is a relative measure that survives JPEG recompression.
    """
    r1 = _gaussian_residual(img_float, sigma=1)
    r5 = _gaussian_residual(img_float, sigma=5)
    var5 = float(np.var(r5))
    if var5 < 1e-10:
        return 1.0
    return float(np.clip(np.var(r1) / var5, 0.0, 50.0))


def multi_scale_noise_ratio_3_5(img_float: np.ndarray) -> float:
    """
    Ratio of noise variance at sigma=3 vs sigma=5.
    Captures mid-scale structure in addition to fine-scale.
    """
    r3 = _gaussian_residual(img_float, sigma=3)
    r5 = _gaussian_residual(img_float, sigma=5)
    var5 = float(np.var(r5))
    if var5 < 1e-10:
        return 1.0
    return float(np.clip(np.var(r3) / var5, 0.0, 50.0))


# ---------------------------------------------------------------------------
# Chroma inter-channel noise correlation (Bayer pattern feature)
# ---------------------------------------------------------------------------

def _channel_noise(channel: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Returns high-pass residual for a single image channel."""
    ch_float = channel.astype(np.float64)
    ksize = int(sigma * 6) | 1
    blurred = cv2.GaussianBlur(ch_float, (ksize, ksize), sigma)
    return ch_float - blurred


def chroma_noise_correlation(image_path: str) -> Dict[str, float]:
    """
    Computes inter-channel noise correlations (R-G, R-B, G-B).

    In real cameras, noise in the three channels is correlated in a
    specific way due to the Bayer filter mosaic. AI generators produce
    independent or differently-correlated channel noise.

    Returns dict with 3 float features:
        - noise_rg_corr, noise_rb_corr, noise_gb_corr  (each in [-1, 1])
    """
    _default = {'noise_rg_corr': 0.0, 'noise_rb_corr': 0.0, 'noise_gb_corr': 0.0}
    try:
        img = cv2.imread(image_path)  # BGR
        if img is None:
            return _default

        b_channel, g_channel, r_channel = cv2.split(img)

        noise_r = _channel_noise(r_channel).flatten()
        noise_g = _channel_noise(g_channel).flatten()
        noise_b = _channel_noise(b_channel).flatten()

        def safe_corr(a: np.ndarray, b_arr: np.ndarray) -> float:
            if np.std(a) < 1e-10 or np.std(b_arr) < 1e-10:
                return 0.0
            c = np.corrcoef(a, b_arr)[0, 1]
            return 0.0 if np.isnan(c) else float(c)

        return {
            'noise_rg_corr': safe_corr(noise_r, noise_g),
            'noise_rb_corr': safe_corr(noise_r, noise_b),
            'noise_gb_corr': safe_corr(noise_g, noise_b),
        }
    except Exception:
        return _default


def extract_noise_features(image_path: str) -> Dict[str, float]:
    """
    Extracts all noise residual features from an image.

    Returns:
        Dict with 11 features:
          Base 6: noise_variance, noise_kurtosis, noise_skewness,
                  noise_spectral_entropy, noise_autocorrelation, noise_block_var_std
          Multi-scale 2: noise_ms_ratio_1_5, noise_ms_ratio_3_5
          Chroma 3: noise_rg_corr, noise_rb_corr, noise_gb_corr
    """
    _default_base = {
        'noise_variance': 0.0,
        'noise_kurtosis': 0.0,
        'noise_skewness': 0.0,
        'noise_spectral_entropy': 0.5,
        'noise_autocorrelation': 0.0,
        'noise_block_var_std': 0.0,
        'noise_ms_ratio_1_5': 1.0,
        'noise_ms_ratio_3_5': 1.0,
    }

    try:
        residual = compute_noise_residual(image_path)
    except Exception:
        result = dict(_default_base)
        result.update({'noise_rg_corr': 0.0, 'noise_rb_corr': 0.0, 'noise_gb_corr': 0.0})
        return result

    # Load grayscale float once for multi-scale
    try:
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is not None:
            img_float = img_gray.astype(np.float64)
            # Resolution guard: sigma=5 needs a 31px kernel; anything smaller
            # produces unreliable ratios — clamp to neutral value 1.0 instead.
            if min(img_gray.shape) < 64:
                img_float = None
        else:
            img_float = None
    except Exception:
        img_float = None

    ms_ratio_1_5 = multi_scale_noise_ratio_1_5(img_float) if img_float is not None else 1.0
    ms_ratio_3_5 = multi_scale_noise_ratio_3_5(img_float) if img_float is not None else 1.0

    chroma = chroma_noise_correlation(image_path)

    base = {
        'noise_variance': residual_variance(residual),
        'noise_kurtosis': residual_kurtosis(residual),
        'noise_skewness': residual_skewness(residual),
        'noise_spectral_entropy': residual_spectral_entropy(residual),
        'noise_autocorrelation': residual_spatial_autocorrelation(residual),
        'noise_block_var_std': residual_block_variance_std(residual),
        'noise_ms_ratio_1_5': ms_ratio_1_5,
        'noise_ms_ratio_3_5': ms_ratio_3_5,
    }
    base.update(chroma)
    return base


def noise_score(image_path: str) -> float:
    """
    Computes a noise-based score from 0.0 (likely real) to 1.0 (likely AI).
    
    Based on:
    - Low spectral entropy → more structured noise → likely AI
    - High autocorrelation → correlated noise → likely AI
    - Low block variance std → uniform noise → likely AI
    """
    features = extract_noise_features(image_path)
    
    # Spectral entropy: lower = more AI-like
    entropy = features['noise_spectral_entropy']
    entropy_score = max(0.0, 1.0 - entropy)  # Invert: low entropy → high score
    
    # Autocorrelation: higher = more AI-like
    autocorr = abs(features['noise_autocorrelation'])
    autocorr_score = min(1.0, autocorr * 2.0)  # Scale up
    
    # Block variance std: lower = more uniform = more AI-like
    block_var = features['noise_block_var_std']
    # Normalize roughly: typical values 0-50
    block_score = max(0.0, 1.0 - min(1.0, block_var / 30.0))
    
    score = 0.35 * entropy_score + 0.35 * autocorr_score + 0.30 * block_score
    return float(np.clip(score, 0.0, 1.0))
