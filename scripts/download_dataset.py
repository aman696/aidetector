#!/usr/bin/env python3
"""
Dataset Download Script for AI Image Detector
=============================================

Downloads real images from Lorem Picsum and AI-generated images from
HuggingFace DiffusionDB, then organizes them into train/test splits.

Usage:
    python scripts/download_dataset.py              # Download all images
    python scripts/download_dataset.py --verify      # Verify existing dataset
    python scripts/download_dataset.py --real-only    # Download only real images
    python scripts/download_dataset.py --ai-only      # Download only AI images
"""

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REAL_DIR = DATA_DIR / "real"
AI_DIR = DATA_DIR / "ai_generated"
TEST_DIR = DATA_DIR / "test"
TEST_REAL_DIR = TEST_DIR / "real"
TEST_AI_DIR = TEST_DIR / "ai_generated"

NUM_REAL_IMAGES = 60
NUM_AI_IMAGES = 60
NUM_TEST_PER_CLASS = 10  # moved from train → test

IMAGE_SIZE = 1024  # Picsum download resolution (width = height)
MIN_RESOLUTION = 512  # Minimum acceptable shortest side


# ---------------------------------------------------------------------------
# Real Image Downloads (Lorem Picsum)
# ---------------------------------------------------------------------------

def download_real_images(count: int = NUM_REAL_IMAGES) -> int:
    """Download real photographs from Lorem Picsum.

    Picsum serves real photographs from Unsplash contributors.
    Each request to https://picsum.photos/{w}/{h} returns a random photo.

    Args:
        count: Number of images to download.

    Returns:
        Number of images successfully downloaded.
    """
    REAL_DIR.mkdir(parents=True, exist_ok=True)

    # Check how many already exist
    existing = list(REAL_DIR.glob("real_*.jpg"))
    start_idx = len(existing) + 1

    if start_idx > count:
        print(f"  ✓ Already have {len(existing)} real images (requested {count})")
        return len(existing)

    remaining = count - len(existing)
    print(f"  Downloading {remaining} real images from Lorem Picsum ...")

    downloaded = 0
    failures = 0
    seen_ids = set()

    pbar = tqdm(total=remaining, desc="  Real images", unit="img")

    while downloaded < remaining and failures < remaining * 2:
        idx = start_idx + downloaded
        url = f"https://picsum.photos/{IMAGE_SIZE}/{IMAGE_SIZE}"

        try:
            resp = requests.get(url, timeout=30, allow_redirects=True)
            resp.raise_for_status()

            # Extract Picsum image ID from the redirect URL to avoid dupes
            picsum_id = None
            if resp.history:
                final_url = resp.url
                # URL looks like https://fastly.picsum.photos/id/237/1024/1024.jpg?...
                parts = final_url.split("/id/")
                if len(parts) > 1:
                    picsum_id = parts[1].split("/")[0]

            if picsum_id and picsum_id in seen_ids:
                failures += 1
                continue
            if picsum_id:
                seen_ids.add(picsum_id)

            filepath = REAL_DIR / f"real_{idx:03d}.jpg"
            filepath.write_bytes(resp.content)
            downloaded += 1
            pbar.update(1)

            # Small delay to be respectful to the free API
            time.sleep(0.3)

        except (requests.RequestException, IOError) as e:
            failures += 1
            tqdm.write(f"  ⚠ Failed to download real image: {e}")
            time.sleep(1)

    pbar.close()

    total = len(existing) + downloaded
    print(f"  ✓ {total} real images in {REAL_DIR}")
    return total


# ---------------------------------------------------------------------------
# AI-Generated Image Downloads (HuggingFace datasets-server REST API)
# ---------------------------------------------------------------------------

def download_ai_images(count: int = NUM_AI_IMAGES) -> int:
    """Download AI-generated images from HuggingFace DiffusionDB.

    Uses the HuggingFace datasets-server REST API to fetch image rows
    from the poloclub/diffusiondb dataset (2m_first_1k subset).
    Each image is a 512×512 Stable Diffusion generation.

    Args:
        count: Number of images to download.

    Returns:
        Number of images successfully downloaded.
    """
    AI_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(AI_DIR.glob("ai_*.png"))
    start_idx = len(existing) + 1

    if start_idx > count:
        print(f"  ✓ Already have {len(existing)} AI images (requested {count})")
        return len(existing)

    remaining = count - len(existing)

    print(f"  Downloading {remaining} AI images from DiffusionDB (REST API) ...")

    # HuggingFace datasets-server REST API endpoint
    # Returns rows from the dataset including image URLs
    base_url = "https://datasets-server.huggingface.co/rows"

    downloaded = 0
    pbar = tqdm(total=remaining, desc="  AI images", unit="img")
    batch_size = 20  # API returns up to 100 rows at a time
    offset = 0

    while downloaded < remaining:
        params = {
            "dataset": "poloclub/diffusiondb",
            "config": "2m_first_1k",
            "split": "train",
            "offset": offset,
            "length": min(batch_size, remaining - downloaded),
        }

        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            tqdm.write(f"  ⚠ API request failed at offset {offset}: {e}")
            offset += batch_size
            if offset > remaining * 3:  # give up after too many failures
                break
            time.sleep(2)
            continue

        rows = data.get("rows", [])
        if not rows:
            tqdm.write(f"  ⚠ No rows returned at offset {offset}, advancing ...")
            offset += batch_size
            if offset > remaining * 3:
                break
            continue

        for row in rows:
            if downloaded >= remaining:
                break

            img_data = row.get("row", {}).get("image", {})
            img_url = img_data.get("src") if isinstance(img_data, dict) else None

            if not img_url:
                continue

            try:
                img_resp = requests.get(img_url, timeout=30)
                img_resp.raise_for_status()

                idx = start_idx + downloaded
                filepath = AI_DIR / f"ai_{idx:03d}.png"
                filepath.write_bytes(img_resp.content)
                downloaded += 1
                pbar.update(1)
                time.sleep(0.15)

            except requests.RequestException as e:
                tqdm.write(f"  ⚠ Failed to download image: {e}")

        offset += batch_size

    pbar.close()

    total = len(existing) + downloaded
    print(f"  ✓ {total} AI images in {AI_DIR}")
    return total


# ---------------------------------------------------------------------------
# Test Split
# ---------------------------------------------------------------------------

def create_test_split(n_per_class: int = NUM_TEST_PER_CLASS) -> None:
    """Move a random subset of images into the test directory.

    Moves `n_per_class` images from data/real/ → data/test/real/
    and `n_per_class` images from data/ai_generated/ → data/test/ai_generated/
    """
    TEST_REAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_AI_DIR.mkdir(parents=True, exist_ok=True)

    # Skip if test set already populated
    existing_test_real = list(TEST_REAL_DIR.glob("*"))
    existing_test_ai = list(TEST_AI_DIR.glob("*"))
    if len(existing_test_real) >= n_per_class and len(existing_test_ai) >= n_per_class:
        print(f"  ✓ Test split already exists ({len(existing_test_real)} real, "
              f"{len(existing_test_ai)} AI)")
        return

    # Select random images for test set
    real_images = sorted(REAL_DIR.glob("real_*.jpg"))
    ai_images = sorted(AI_DIR.glob("ai_*.png"))

    if len(real_images) < n_per_class:
        print(f"  ⚠ Only {len(real_images)} real images available, "
              f"need {n_per_class} for test split")
        n_real = len(real_images) // 5  # take 20%
    else:
        n_real = n_per_class

    if len(ai_images) < n_per_class:
        print(f"  ⚠ Only {len(ai_images)} AI images available, "
              f"need {n_per_class} for test split")
        n_ai = len(ai_images) // 5
    else:
        n_ai = n_per_class

    random.seed(42)  # Reproducible splits
    test_real = random.sample(real_images, n_real)
    test_ai = random.sample(ai_images, n_ai)

    print(f"  Moving {n_real} real + {n_ai} AI images to test set ...")

    for img_path in test_real:
        shutil.move(str(img_path), str(TEST_REAL_DIR / img_path.name))

    for img_path in test_ai:
        shutil.move(str(img_path), str(TEST_AI_DIR / img_path.name))

    print(f"  ✓ Test split: {n_real} real, {n_ai} AI images in {TEST_DIR}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_dataset() -> bool:
    """Verify the downloaded dataset for completeness and quality.

    Checks:
        1. Image counts in each directory
        2. All images load successfully with OpenCV
        3. Minimum resolution ≥ 512px on shortest side

    Returns:
        True if verification passes.
    """
    try:
        import cv2
    except ImportError:
        print("  ⚠ OpenCV not installed — skipping image load verification")
        cv2 = None

    all_ok = True
    directories = {
        "data/real": REAL_DIR,
        "data/ai_generated": AI_DIR,
        "data/test/real": TEST_REAL_DIR,
        "data/test/ai_generated": TEST_AI_DIR,
    }

    print("\n=== Dataset Verification ===\n")

    for label, dirpath in directories.items():
        if not dirpath.exists():
            print(f"  ✗ {label}: directory does not exist")
            all_ok = False
            continue

        images = list(dirpath.glob("*.*"))
        images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

        print(f"  {label}: {len(images)} images")

        if len(images) == 0:
            print(f"    ✗ No images found!")
            all_ok = False
            continue

        # Check each image loads and meets resolution
        bad_files = []
        low_res = []
        for img_path in images:
            if img_path.stat().st_size == 0:
                bad_files.append(img_path.name)
                continue

            if cv2 is not None:
                img = cv2.imread(str(img_path))
                if img is None:
                    bad_files.append(img_path.name)
                    continue
                h, w = img.shape[:2]
                if min(h, w) < MIN_RESOLUTION:
                    low_res.append((img_path.name, f"{w}×{h}"))

        if bad_files:
            print(f"    ✗ {len(bad_files)} corrupt/unreadable: {bad_files[:5]}")
            all_ok = False
        else:
            print(f"    ✓ All images load OK")

        if low_res:
            print(f"    ⚠ {len(low_res)} below {MIN_RESOLUTION}px: {low_res[:5]}")
        else:
            print(f"    ✓ All images ≥ {MIN_RESOLUTION}px")

    # Summary
    train_real = len(list(REAL_DIR.glob("*.*"))) if REAL_DIR.exists() else 0
    train_ai = len(list(AI_DIR.glob("*.*"))) if AI_DIR.exists() else 0
    test_real = len(list(TEST_REAL_DIR.glob("*.*"))) if TEST_REAL_DIR.exists() else 0
    test_ai = len(list(TEST_AI_DIR.glob("*.*"))) if TEST_AI_DIR.exists() else 0

    print(f"\n  Summary:")
    print(f"    Train: {train_real} real + {train_ai} AI = {train_real + train_ai}")
    print(f"    Test:  {test_real} real + {test_ai} AI = {test_real + test_ai}")
    print(f"    Total: {train_real + train_ai + test_real + test_ai} images")

    if all_ok:
        print("\n  ✓ Dataset verification PASSED\n")
    else:
        print("\n  ✗ Dataset verification FAILED — see issues above\n")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and organize dataset for AI image detector"
    )
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing dataset instead of downloading")
    parser.add_argument("--real-only", action="store_true",
                        help="Download only real images")
    parser.add_argument("--ai-only", action="store_true",
                        help="Download only AI-generated images")
    parser.add_argument("--skip-split", action="store_true",
                        help="Skip creating train/test split")
    parser.add_argument("--num-real", type=int, default=NUM_REAL_IMAGES,
                        help=f"Number of real images (default: {NUM_REAL_IMAGES})")
    parser.add_argument("--num-ai", type=int, default=NUM_AI_IMAGES,
                        help=f"Number of AI images (default: {NUM_AI_IMAGES})")

    args = parser.parse_args()

    if args.verify:
        success = verify_dataset()
        sys.exit(0 if success else 1)

    print("=" * 50)
    print("  AI Image Detector — Dataset Downloader")
    print("=" * 50)

    # Ensure data directories exist
    for d in [REAL_DIR, AI_DIR, TEST_REAL_DIR, TEST_AI_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Download
    if not args.ai_only:
        print(f"\n[1/3] Real Images ({args.num_real})")
        download_real_images(args.num_real)

    if not args.real_only:
        print(f"\n[2/3] AI-Generated Images ({args.num_ai})")
        download_ai_images(args.num_ai)

    # Split
    if not args.skip_split:
        print(f"\n[3/3] Creating Test Split")
        create_test_split()

    # Verify
    print("\n[✓] Running verification ...")
    verify_dataset()

    print("Done! Dataset is ready in data/")


if __name__ == "__main__":
    main()
