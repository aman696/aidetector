"""
Eigenvalue/Spectral Analysis for AI Image Detection.

Based on: Corvi et al. 2023 - "Intriguing Properties of Synthetic Images"
(arXiv: 2304.06408)

Key insight: Synthetic images exhibit different covariance structures and
spectral properties than real images. The eigenvalue ratios of RGB covariance
matrices and the energy distribution across frequency bands can discriminate
between real and AI-generated images.

Methods:
1. Global RGB covariance → eigenvalue ratios
2. Patch-based covariance → eigenvalue statistics across patches
3. Spectral band energy analysis → low/mid/high frequency distributions
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
from src.utils import load_image, crop_to_square, validate_image_path


def compute_rgb_covariance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Computes the 3×3 covariance matrix of the RGB channels.
    
    Args:
        img_bgr: BGR image array (H, W, 3).
        
    Returns:
        3×3 covariance matrix.
    """
    # Convert BGR → RGB and reshape to (N, 3)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3).astype(np.float64)
    
    # Compute covariance matrix
    cov_matrix = np.cov(pixels.T)
    return cov_matrix


def extract_eigenvalues(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Extracts eigenvalues from a covariance matrix, sorted descending.
    
    Args:
        cov_matrix: Square covariance matrix.
        
    Returns:
        Array of eigenvalues sorted in descending order.
    """
    eigenvalues = np.linalg.eigvalsh(cov_matrix)  # eigvalsh for symmetric matrices
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # Descending, absolute values
    return eigenvalues


def compute_eigenvalue_ratios(eigenvalues: np.ndarray) -> Dict[str, float]:
    """
    Computes diagnostic ratios from eigenvalues.
    
    For RGB images (3 eigenvalues):
    - ratio_1_2: dominance of first principal component
    - ratio_2_3: spread of second vs third component
    - condition_number: ratio of largest to smallest eigenvalue
    
    Args:
        eigenvalues: Array of eigenvalues in descending order.
        
    Returns:
        Dictionary of ratio features.
    """
    eps = 1e-10  # Avoid division by zero
    
    ratios = {
        'eig_ratio_1_2': float(eigenvalues[0] / (eigenvalues[1] + eps)),
        'eig_ratio_2_3': float(eigenvalues[1] / (eigenvalues[2] + eps)),
        'eig_condition_number': float(eigenvalues[0] / (eigenvalues[-1] + eps)),
        'eig_dominance': float(eigenvalues[0] / (np.sum(eigenvalues) + eps)),
    }
    
    return ratios


def patch_eigenvalue_analysis(img_bgr: np.ndarray, patch_size: int = 64) -> Dict[str, float]:
    """
    Divides the image into patches and computes eigenvalue statistics across all patches.
    
    This captures LOCAL texture properties. AI images tend to have more uniform 
    eigenvalue distributions across patches than natural images, which have 
    spatially varying texture properties.
    
    Args:
        img_bgr: BGR image array.
        patch_size: Size of square patches.
        
    Returns:
        Dictionary of patch-level eigenvalue statistics.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    # Collect eigenvalue ratios from all patches
    patch_ratios_1_2 = []
    patch_dominances = []
    
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = img_rgb[y:y + patch_size, x:x + patch_size]
            pixels = patch.reshape(-1, 3).astype(np.float64)
            
            # Skip near-uniform patches (very low variance)
            if np.std(pixels) < 1.0:
                continue
            
            cov = np.cov(pixels.T)
            eigs = extract_eigenvalues(cov)
            
            eps = 1e-10
            patch_ratios_1_2.append(eigs[0] / (eigs[1] + eps))
            patch_dominances.append(eigs[0] / (np.sum(eigs) + eps))
    
    if len(patch_ratios_1_2) == 0:
        return {
            'patch_ratio_mean': 0.0,
            'patch_ratio_std': 0.0,
            'patch_dominance_mean': 0.0,
            'patch_dominance_std': 0.0,
        }
    
    return {
        'patch_ratio_mean': float(np.mean(patch_ratios_1_2)),
        'patch_ratio_std': float(np.std(patch_ratios_1_2)),
        'patch_dominance_mean': float(np.mean(patch_dominances)),
        'patch_dominance_std': float(np.std(patch_dominances)),
    }


def spectral_band_analysis(img_bgr: np.ndarray) -> Dict[str, float]:
    """
    Analyzes energy distribution across spectral bands (low/mid/high frequency).
    
    From Corvi 2023: synthetic images exhibit significant differences in
    mid-high frequency signal content compared to real images.
    
    Args:
        img_bgr: BGR image array.
        
    Returns:
        Dictionary of spectral band energy features.
    """
    # Convert to grayscale for spectral analysis
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = crop_to_square(gray)
    
    # 2D FFT
    fft = np.fft.fft2(gray.astype(np.float64))
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)
    
    # Distance map from center
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
    
    # Define three bands: low (0-33%), mid (33-66%), high (66-100%)
    r_low = max_radius * 0.33
    r_mid = max_radius * 0.66
    
    low_mask = distances < r_low
    mid_mask = (distances >= r_low) & (distances < r_mid)
    high_mask = (distances >= r_mid) & (distances <= max_radius)
    
    total_energy = np.sum(magnitude) + 1e-10
    low_energy = np.sum(magnitude[low_mask])
    mid_energy = np.sum(magnitude[mid_mask])
    high_energy = np.sum(magnitude[high_mask])
    
    return {
        'band_low_ratio': float(low_energy / total_energy),
        'band_mid_ratio': float(mid_energy / total_energy),
        'band_high_ratio': float(high_energy / total_energy),
        'band_mid_high_ratio': float((mid_energy + high_energy) / total_energy),
    }


def extract_eigen_features(image_path: str) -> Dict[str, float]:
    """
    Extracts all eigenvalue-based and spectral features from an image.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Dictionary of all eigenvalue and spectral features.
    """
    validate_image_path(image_path)
    
    img = load_image(image_path)
    
    # Global covariance analysis
    cov_matrix = compute_rgb_covariance(img)
    eigenvalues = extract_eigenvalues(cov_matrix)
    ratios = compute_eigenvalue_ratios(eigenvalues)
    
    # Patch-based analysis
    patch_features = patch_eigenvalue_analysis(img, patch_size=64)
    
    # Spectral band analysis
    band_features = spectral_band_analysis(img)
    
    # Merge all features
    features = {}
    features.update(ratios)
    features.update(patch_features)
    features.update(band_features)
    
    return features


def eigenvalue_score(image_path: str) -> float:
    """
    Computes a single eigenvalue-based score for AI detection.
    
    Higher score = more likely AI-generated.
    
    Based on:
    - Eigenvalue ratio deviation from natural image statistics
    - Patch eigenvalue uniformity (AI images are more uniform)
    - High-frequency energy deficit (AI images have less high-freq content)
    
    Args:
        image_path: Path to image file.
        
    Returns:
        float: Score between 0.0 and 1.0 (higher = more likely AI).
    """
    features = extract_eigen_features(image_path)
    
    # Score 1: Eigenvalue dominance deviation
    # Real images: eig_dominance typically 0.5-0.8
    # AI images: often higher or lower depending on color distribution
    dominance = features['eig_dominance']
    dominance_score = abs(dominance - 0.65) / 0.35  # Deviation from typical real value
    dominance_score = np.clip(dominance_score, 0.0, 1.0)
    
    # Score 2: Patch uniformity  
    # AI images have lower patch_ratio_std (more uniform texture across patches)
    patch_std = features['patch_ratio_std']
    # Real images have higher variation, so low std → more likely AI
    uniformity_score = np.clip(1.0 - (patch_std / 10.0), 0.0, 1.0)
    
    # Score 3: High-frequency energy deficit
    # AI images have less high-frequency content
    hf_ratio = features['band_high_ratio']
    # Low high-freq energy → more likely AI
    hf_deficit_score = np.clip(1.0 - (hf_ratio / 0.3), 0.0, 1.0)
    
    # Score 4: Eigenvalue condition number
    # Extreme condition numbers suggest synthetic color distribution
    cond = features['eig_condition_number']
    log_cond = np.log1p(cond)
    cond_score = np.clip((log_cond - 3.0) / 5.0, 0.0, 1.0)
    
    # Weighted combination
    score = (0.2 * dominance_score + 
             0.3 * uniformity_score + 
             0.3 * hf_deficit_score + 
             0.2 * cond_score)
    
    return float(np.clip(score, 0.0, 1.0))
