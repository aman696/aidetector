"""
Utility functions for image loading, preprocessing, and dataset management.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional


# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


def validate_image_path(path: str) -> None:
    """
    Validates that a file exists and has a supported image extension.
    
    Args:
        path: Path to the image file.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file extension is not supported.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")


def load_image(path: str) -> np.ndarray:
    """
    Loads an image in BGR color space (OpenCV default).
    
    Args:
        path: Path to the image file.
        
    Returns:
        np.ndarray: BGR image array.
        
    Raises:
        ValueError: If image cannot be loaded.
    """
    validate_image_path(path)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not decode image at {path}")
    return img


def load_grayscale(path: str) -> np.ndarray:
    """
    Loads an image as grayscale.
    
    Args:
        path: Path to the image file.
        
    Returns:
        np.ndarray: Grayscale image array.
    """
    validate_image_path(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image at {path}")
    return img


def resize_to_square(img: np.ndarray, size: int = 512) -> np.ndarray:
    """
    Resizes an image to a square of the given size.
    Note: per Durall 2020 paper notes, resizing should be done AFTER 
    spectral analysis when possible to avoid distorting the frequency spectrum.
    Use this only for eigenvalue/metadata analysis or for visualization.
    
    Args:
        img: Input image array.
        size: Target square size in pixels.
        
    Returns:
        np.ndarray: Resized square image.
    """
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def crop_to_square(img: np.ndarray) -> np.ndarray:
    """
    Center-crops an image to a square (preserves frequency content better than resizing).
    
    Args:
        img: Input image array (2D grayscale or 3D color).
        
    Returns:
        np.ndarray: Square-cropped image.
    """
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    
    side = min(h, w)
    y_start = (h - side) // 2
    x_start = (w - side) // 2
    
    if img.ndim == 2:
        return img[y_start:y_start + side, x_start:x_start + side]
    else:
        return img[y_start:y_start + side, x_start:x_start + side, :]


def get_all_image_paths(directory: str) -> List[str]:
    """
    Collects all image file paths from a directory (non-recursive).
    
    Args:
        directory: Path to directory containing images.
        
    Returns:
        List of absolute image file paths.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")
    
    paths = []
    for fname in sorted(os.listdir(directory)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            paths.append(os.path.join(directory, fname))
    return paths


def load_labeled_dataset(real_dir: str, ai_dir: str) -> Tuple[List[str], List[int]]:
    """
    Loads image paths with labels from real and AI-generated directories.
    
    Args:
        real_dir: Directory of real images.
        ai_dir: Directory of AI-generated images.
        
    Returns:
        Tuple of (image_paths, labels) where label 0 = real, 1 = AI.
    """
    real_paths = get_all_image_paths(real_dir)
    ai_paths = get_all_image_paths(ai_dir)
    
    paths = real_paths + ai_paths
    labels = [0] * len(real_paths) + [1] * len(ai_paths)
    
    return paths, labels
