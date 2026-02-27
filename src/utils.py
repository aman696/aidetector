"""
Utility functions for image loading, preprocessing, and dataset management.
"""

import os
import cv2
import numpy as np
import tempfile
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


def augment_dataset_with_jpeg(
    paths: List[str],
    labels: List[int],
    qualities: Tuple[int, ...] = (70, 80),
    scale_factor: float = 0.75,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Augments a training dataset by adding JPEG-recompressed and
    downscaled copies of each image.

    This implements the ITW-SM (arXiv:2507.10236) finding that training
    data composition — specifically including recompressed variants — is the
    most effective way to improve in-the-wild detection accuracy.

    For each image the function creates:
        - One copy recompressed at each quality level in `qualities`
        - One copy downscaled to `scale_factor` × original size (then
          saved as JPEG Q=85) to simulate low-resolution social-media crops

    All copies carry the same label as the original.
    Copies are written to a temporary directory managed by `tempfile`;
    call `cleanup_augmented_files(tmp_paths)` when training finishes.

    Args:
        paths:        List of original image paths (training set).
        labels:       Corresponding labels (0 = real, 1 = AI).
        qualities:    JPEG quality levels to use for recompression.
        scale_factor: Downscale factor for the resize augmentation.

    Returns:
        (aug_paths, aug_labels, tmp_paths)
            aug_paths   – original + augmented paths (len = original × (1 + len(qualities) + 1))
            aug_labels  – corresponding labels
            tmp_paths   – paths of temp files to delete after training
    """
    aug_paths: List[str] = list(paths)
    aug_labels: List[int] = list(labels)
    tmp_paths: List[str] = []

    tmp_dir = tempfile.mkdtemp(prefix='aidet_aug_')

    for idx, (src, label) in enumerate(zip(paths, labels)):
        try:
            img = cv2.imread(src)
            if img is None:
                continue

            base = f"aug_{idx}"

            # JPEG recompression copies
            for q in qualities:
                dst = os.path.join(tmp_dir, f"{base}_q{q}.jpg")
                cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, q])
                aug_paths.append(dst)
                aug_labels.append(label)
                tmp_paths.append(dst)

            # Downscaled copy
            h, w = img.shape[:2]
            small = cv2.resize(
                img,
                (max(64, int(w * scale_factor)), max(64, int(h * scale_factor))),
                interpolation=cv2.INTER_AREA,
            )
            dst_small = os.path.join(tmp_dir, f"{base}_small.jpg")
            cv2.imwrite(dst_small, small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            aug_paths.append(dst_small)
            aug_labels.append(label)
            tmp_paths.append(dst_small)

        except Exception:
            continue  # Skip any image that fails to augment

    return aug_paths, aug_labels, tmp_paths


def cleanup_augmented_files(tmp_paths: List[str]) -> None:
    """
    Deletes temporary augmented image files created by augment_dataset_with_jpeg.

    Args:
        tmp_paths: List of file paths to delete (as returned by augment_dataset_with_jpeg).
    """
    for p in tmp_paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    # Also try to remove the parent temp directory if empty
    if tmp_paths:
        try:
            parent = os.path.dirname(tmp_paths[0])
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        except Exception:
            pass

