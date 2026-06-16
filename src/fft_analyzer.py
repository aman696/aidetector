"""
FFT Frequency Analysis for AI Image Detection.

Based on: Durall et al. 2020 - "Unmasking DeepFakes with simple Features"
(arXiv: 1911.00686)

Key insight: AI-generated images exhibit different frequency distributions than
real images. Specifically, AI images show a characteristic drop-off in high-frequency
components compared to the ~1/f power law that natural images follow.

Pipeline:
1. Convert to grayscale
2. Center-crop to square (avoid resizing to preserve spectrum)
3. Apply 2D DFT, shift to center DC component
4. Compute the power spectrum |F|^2 (linear, NOT log, NOT magnitude)
5. Azimuthal average → 1D radial power spectrum
6. Analyze spectral slope, high-frequency energy, and spectral features

The single log that turns a 1/f^beta power law into a straight line is applied
once, downstream in compute_spectral_slope. See code_notes/02-fft-analyzer.md
for the prior magnitude/double-log bug and the corrected derivation.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from src.utils import load_grayscale, crop_to_square, validate_image_path


def azimuthal_average(spectrum_2d: np.ndarray) -> np.ndarray:
    """
    Computes the azimuthally averaged 1D radial profile from a 2D spectrum.

    This averages the 2D spectrum over concentric rings centered at DC,
    producing a 1D function of spatial frequency (radius from center). Callers
    pass the linear power spectrum |F|^2 so the result is a radial power profile.

    This is the key operation from Durall 2020, Section III.
    Uses vectorized numpy (np.bincount) for performance on large images.

    Args:
        spectrum_2d: 2D spectrum (after fftshift, so DC is centered). Linear,
            not log; pass |F|^2 for a radial power spectrum.

    Returns:
        1D array of ring-averaged values at each radius (from 0 to max_radius).
    """
    h, w = spectrum_2d.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)
    
    # Create distance map from center (vectorized)
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
    
    # Convert to integer bins and clip to max_radius
    radius_bins = distances.astype(np.int64).ravel()
    spectrum_flat = spectrum_2d.ravel()
    
    # Only keep pixels within max_radius
    valid = radius_bins < max_radius
    radius_bins = radius_bins[valid]
    spectrum_flat = spectrum_flat[valid]
    
    # Sum and count per radius bin using np.bincount (fully vectorized)
    bin_sums = np.bincount(radius_bins, weights=spectrum_flat, minlength=max_radius)
    bin_counts = np.bincount(radius_bins, minlength=max_radius).astype(np.float64)
    
    # Avoid division by zero
    bin_counts[bin_counts == 0] = 1.0
    radial_profile = bin_sums / bin_counts
    
    return radial_profile


def compute_power_spectrum(image_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the 2D power spectrum |F|^2 and the azimuthally averaged 1D radial
    power spectrum of a grayscale image.

    The radial profile is the azimuthal average of the LINEAR power |F|^2
    (Durall 2020, Section III). Downstream, compute_spectral_slope takes a single
    log of this power, so a 1/f^beta image yields slope ~= -beta. The earlier
    implementation averaged log-magnitude here and logged again in the slope fit
    (a double log, and magnitude where power was intended), which made the slope
    uninterpretable as a power-law exponent; see code_notes/02-fft-analyzer.md.

    Args:
        image_gray: Grayscale image as 2D numpy array.

    Returns:
        Tuple of (power_spectrum_2d, radial_power_spectrum_1d)
    """
    # Apply 2D FFT
    fft = np.fft.fft2(image_gray.astype(np.float64))
    fft_shifted = np.fft.fftshift(fft)  # Center DC component

    # Power spectral density |F|^2 (linear; no log here, no magnitude). The one
    # log that linearises a 1/f^beta law is applied once, in the slope fit.
    power = np.abs(fft_shifted) ** 2

    # Azimuthally averaged 1D radial power spectrum
    radial_spectrum = azimuthal_average(power)

    return power, radial_spectrum


def compute_spectral_slope(radial_spectrum: np.ndarray) -> Tuple[float, float]:
    """
    Fits a linear regression to the log-log radial power spectrum.
    
    Natural images follow approximately a 1/f^beta power law, where beta ≈ 1.
    AI-generated images deviate from this, often with steeper slopes (higher beta)
    indicating weaker high-frequency content.
    
    Args:
        radial_spectrum: 1D azimuthally averaged spectrum.
        
    Returns:
        Tuple of (slope, r_squared) where:
        - slope: the spectral slope (negative = normal falloff)
        - r_squared: goodness of fit (how well it follows power law)
    """
    # Skip DC component (index 0) to avoid log(0) issues
    n = len(radial_spectrum)
    if n < 5:
        return 0.0, 0.0
    
    # Use frequencies from 1 to n-1
    freqs = np.arange(1, n)
    powers = radial_spectrum[1:]
    
    # Filter out any zero or negative values
    valid = powers > 0
    if np.sum(valid) < 5:
        return 0.0, 0.0
    
    log_freqs = np.log(freqs[valid])
    log_powers = np.log(powers[valid])
    
    # Linear regression: log(power) = slope * log(freq) + intercept
    coeffs = np.polyfit(log_freqs, log_powers, 1)
    slope = coeffs[0]
    
    # Compute R-squared
    predicted = np.polyval(coeffs, log_freqs)
    ss_res = np.sum((log_powers - predicted) ** 2)
    ss_tot = np.sum((log_powers - np.mean(log_powers)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return slope, r_squared


def compute_high_freq_ratio(radial_spectrum: np.ndarray, cutoff_fraction: float = 0.5) -> float:
    """
    Ratio of the high-frequency tail of the radial power profile to the whole.

    NOTE on naming: the input is the azimuthally AVERAGED power |F|^2 (each
    radius is the mean over its ring, already divided by the ring's pixel
    count). Summing those per-ring means is therefore a radial-PROFILE ratio,
    NOT a true energy ratio (which would sum the un-averaged ring power and so
    weight large outer rings by their pixel count). The value is a stable,
    monotone proxy that the SVM consumes; it is not physical energy.

    Args:
        radial_spectrum: 1D azimuthally averaged power profile (from
            compute_power_spectrum).
        cutoff_fraction: Fraction of the profile treated as "high frequency".
                         0.5 means the outer 50% of radii.

    Returns:
        Ratio of the outer-radius profile sum to the full profile sum (0.0-1.0).
    """
    n = len(radial_spectrum)
    if n == 0:
        return 0.0
    
    cutoff_idx = int(n * (1.0 - cutoff_fraction))
    
    total_energy = np.sum(radial_spectrum)
    if total_energy == 0:
        return 0.0
    
    high_freq_energy = np.sum(radial_spectrum[cutoff_idx:])
    return float(high_freq_energy / total_energy)


def compute_spectral_falloff(radial_spectrum: np.ndarray) -> float:
    """
    Measures how sharply the spectrum drops off at high frequencies.
    Compares the last quarter of the spectrum to the second quarter.
    
    AI images typically show a steeper drop-off than real images.
    
    Args:
        radial_spectrum: 1D azimuthally averaged spectrum.
        
    Returns:
        Falloff ratio (higher = steeper drop = more likely AI).
    """
    n = len(radial_spectrum)
    if n < 8:
        return 0.0
    
    q2_start = n // 4
    q2_end = n // 2
    q4_start = 3 * n // 4
    
    mid_energy = np.mean(radial_spectrum[q2_start:q2_end])
    high_energy = np.mean(radial_spectrum[q4_start:])
    
    if mid_energy == 0:
        return 0.0
    
    # Ratio < 1 means high freq is weaker than mid freq (normal)
    # Smaller ratio = steeper falloff = more likely AI
    ratio = high_energy / mid_energy
    return float(ratio)


def extract_fft_features(image_path: str) -> Dict[str, float]:
    """
    Extracts all FFT-based features from an image.
    
    Args:
        image_path: Path to image file.
        
    Returns:
        Dictionary of feature name → value:
            - spectral_slope: slope of log-log power spectrum
            - slope_r_squared: goodness of fit to power law
            - high_freq_ratio: high-frequency / total radial-profile power
              (a profile ratio, not physical energy; see compute_high_freq_ratio)
            - spectral_falloff: sharpness of high-freq drop-off
    """
    validate_image_path(image_path)
    
    # Load and preprocess
    img = load_grayscale(image_path)
    img = crop_to_square(img)  # Square crop, NOT resize (preserves spectrum)
    
    # Compute power spectrum
    _, radial_spectrum = compute_power_spectrum(img)
    
    # Extract features
    slope, r_sq = compute_spectral_slope(radial_spectrum)
    hf_ratio = compute_high_freq_ratio(radial_spectrum, cutoff_fraction=0.5)
    falloff = compute_spectral_falloff(radial_spectrum)
    
    return {
        'spectral_slope': slope,
        'slope_r_squared': r_sq,
        'high_freq_ratio': hf_ratio,
        'spectral_falloff': falloff,
    }


# Reference values for the standalone heuristic below, measured on the current
# data (n = 40 real + 40 AI, 2026-06-14) AFTER the power-spectrum fix. These are
# weak, overlapping signals: only two directions held up across resamples and
# they are recorded as such in code_notes/02-fft-analyzer.md. The shipped
# detector does NOT use this score; it feeds the raw features to the SVM. This
# heuristic only powers the v1 voting fallback and the per-signal display.
_SLOPE_NATURAL = -2.6      # median power-law exponent of natural images (beta ~ 2.6)
_SLOPE_ANOM_TOL = 0.6      # |slope - natural| at which the anomaly term saturates
_HF_RATIO_REF = 5.0e-6     # high-freq energy fraction below which an image looks high-freq-poor


def fft_score(image_path: str) -> float:
    """
    Computes a single FFT-based score for AI detection (weak standalone heuristic).

    Higher score = more likely AI-generated. Built from the two directions that
    were stable across resamples on the current data:
    - high-frequency energy fraction: real images keep MORE high-freq energy, so
      a lower ratio leans AI (the Durall thesis);
    - spectral-slope anomaly: AI images deviate MORE from the natural ~1/f^beta
      power law (real beta clusters tightly near -2.6, AI spreads), so a larger
      |slope - natural| leans AI.

    Spectral falloff is intentionally not used: its direction flipped between
    resamples (overlapping distributions), so it is not a reliable standalone
    signal. See code_notes/02-fft-analyzer.md for the measured numbers.

    Args:
        image_path: Path to image file.

    Returns:
        float: Score between 0.0 and 1.0 (higher = more likely AI).
    """
    features = extract_fft_features(image_path)

    # High-frequency energy fraction: lower => weaker high-freq => more likely AI.
    hf_ratio = features['high_freq_ratio']
    hf_score = np.clip((_HF_RATIO_REF - hf_ratio) / _HF_RATIO_REF, 0.0, 1.0)

    # Slope anomaly: distance from the natural power-law exponent, either side.
    slope = features['spectral_slope']
    slope_anomaly = np.clip(abs(slope - _SLOPE_NATURAL) / _SLOPE_ANOM_TOL, 0.0, 1.0)

    # Weight the more direct high-freq signal higher than the anomaly term.
    score = 0.6 * hf_score + 0.4 * slope_anomaly

    return float(np.clip(score, 0.0, 1.0))
