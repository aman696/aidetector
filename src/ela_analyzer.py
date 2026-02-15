"""
Error Level Analysis (ELA) Analyzer for AI Image Detection.

ELA detects compression-level inconsistencies in JPEG images.

Method:
1. Re-save image at known JPEG quality (95%)
2. Compute pixel-wise difference between original and re-saved
3. Analyze the difference map — uniform = never JPEG-compressed (likely AI PNG)

Note: Most useful for distinguishing camera-JPEGs from never-compressed AI PNGs.
For PNG inputs, provides format-aware default features.
"""

import os
import numpy as np
import cv2
import tempfile
from typing import Dict

from src.utils import validate_image_path


def compute_ela_map(image_path: str, quality: int = 95) -> np.ndarray:
    """
    Computes the Error Level Analysis difference map.
    
    Args:
        image_path: Path to image file.
        quality: JPEG quality for re-compression (default 95).
        
    Returns:
        ELA map as float array (absolute difference per pixel).
    """
    validate_image_path(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")
    
    img_float = img.astype(np.float64)
    
    # Re-save as JPEG at specified quality
    tmp_path = tempfile.mktemp(suffix='.jpg')
    try:
        cv2.imwrite(tmp_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        recompressed = cv2.imread(tmp_path).astype(np.float64)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    # ELA = absolute difference
    ela = np.abs(img_float - recompressed)
    
    # Convert to grayscale for analysis
    ela_gray = np.mean(ela, axis=2)
    return ela_gray


def ela_mean(ela_map: np.ndarray) -> float:
    """Mean ELA value. Higher = more compression change = first-time JPEG."""
    return float(np.mean(ela_map))


def ela_variance(ela_map: np.ndarray) -> float:
    """Variance of ELA map. Uniform = consistent compression. Variable = mixed."""
    return float(np.var(ela_map))


def ela_max(ela_map: np.ndarray) -> float:
    """Maximum ELA value. Indicates worst-case compression difference."""
    return float(np.max(ela_map))


def ela_uniformity(ela_map: np.ndarray) -> float:
    """
    Ratio of mean to max ELA. Higher = more uniform compression.
    AI images (never compressed) tend to have very uniform ELA.
    """
    max_val = np.max(ela_map)
    if max_val < 1e-10:
        return 1.0
    return float(np.mean(ela_map) / max_val)


def ela_block_inconsistency(ela_map: np.ndarray, block_size: int = 8) -> float:
    """
    Standard deviation of block-wise ELA means.
    Non-uniform block compression = potential tampering or mixed sources.
    """
    h, w = ela_map.shape
    block_means = []
    
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = ela_map[y:y + block_size, x:x + block_size]
            block_means.append(np.mean(block))
    
    if len(block_means) < 2:
        return 0.0
    
    return float(np.std(block_means))


def extract_ela_features(image_path: str) -> Dict[str, float]:
    """
    Extracts ELA-based features from an image.
    
    For PNG images (never JPEG-compressed), values will reflect
    the characteristic high ELA of first-time compression.
    
    Returns:
        Dict with 5 features.
    """
    try:
        ela_map = compute_ela_map(image_path)
    except Exception:
        return {
            'ela_mean': 0.0,
            'ela_variance': 0.0,
            'ela_max': 0.0,
            'ela_uniformity': 0.5,
            'ela_block_inconsistency': 0.0,
        }
    
    return {
        'ela_mean': ela_mean(ela_map),
        'ela_variance': ela_variance(ela_map),
        'ela_max': ela_max(ela_map),
        'ela_uniformity': ela_uniformity(ela_map),
        'ela_block_inconsistency': ela_block_inconsistency(ela_map),
    }


def ela_score(image_path: str) -> float:
    """
    Computes an ELA-based score from 0.0 (likely real) to 1.0 (likely AI).
    
    Key signals:
    - High mean ELA on PNG = never JPEG-compressed = could be AI
    - Very uniform ELA = no mixed compression = could be AI
    - But also could be a screenshot (PNG with no JPEG history)
    """
    features = extract_ela_features(image_path)
    
    ext = os.path.splitext(image_path)[1].lower()
    is_png = ext == '.png'
    
    # For PNGs, high mean ELA is expected (first compression)
    # For JPEGs, high mean ELA means it's been heavily edited or is fresh
    mean_val = features['ela_mean']
    
    if is_png:
        # PNG: high ELA is normal, focus on uniformity
        mean_score = min(1.0, mean_val / 20.0) * 0.3
    else:
        # JPEG: very low ELA = already compressed multiple times (real camera flow)
        # High ELA = heavily edited or first-time compression
        mean_score = min(1.0, mean_val / 10.0) * 0.5
    
    # Uniformity: very uniform = consistent source (could be AI or screenshot)
    uniformity = features['ela_uniformity']
    uni_score = uniformity * 0.3
    
    # Block inconsistency: low = uniform source, high = mixed
    block_incon = features['ela_block_inconsistency']
    block_score = max(0.0, 1.0 - min(1.0, block_incon / 5.0)) * 0.2
    
    score = mean_score + uni_score + block_score
    return float(np.clip(score, 0.0, 1.0))
