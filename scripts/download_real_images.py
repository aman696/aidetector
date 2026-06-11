#!/usr/bin/env python3
"""
download_real_images.py — Download 200 diverse real photographs for training.

Sources used (all free, no key required by default):
  1. Lorem Picsum  — random real Unsplash photos, served by id (0-1083 known)
  2. Pexels curated CC0 direct URLs (hand-picked IDs across many categories)

The two sources give you category diversity:
  - Picsum:  landscapes, portraits, architecture, still-life, cityscapes
  - Pexels:  animals, food, technology, sports, nature, macro

Usage:
    # From the project root (venv activated):
    python scripts/download_real_images.py

    # Only download N images (useful for a quick test):
    python scripts/download_real_images.py --count 50

    # Verify what was already downloaded without fetching anything new:
    python scripts/download_real_images.py --verify

    # Force re-download even if files exist:
    python scripts/download_real_images.py --force

Output:
    data/real/real_001.jpg … real_200.jpg   (training set)
    data/test/real/real_t_001.jpg … real_t_020.jpg  (held-out test set, 10% of count)
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ──────────────────────── paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
REAL_TRAIN   = DATA_DIR / "real"
REAL_TEST    = DATA_DIR / "test" / "real"
LOG_FILE     = DATA_DIR / "real_download_log.json"   # tracks what was fetched

# ──────────────────────── config ──────────────────────────────────────────────
TARGET_TRAIN  = 200   # images to put in data/real/
TARGET_TEST   = 20    # images to put in data/test/real/
TARGET_TOTAL  = TARGET_TRAIN + TARGET_TEST  # 220 fetched, 20 moved to test
IMG_SIZE      = 1024  # square size for Picsum downloads
MIN_RES       = 400   # minimum shortest-side resolution to accept
DELAY_S       = 0.35  # polite pause between requests

# ──────────────────────── Picsum IDs ──────────────────────────────────────────
# Lorem Picsum serves real Unsplash photos by numeric ID (0-1083 publicly known).
# We spread IDs evenly to get category variety (people, nature, city, objects…).
# 150 IDs selected manually to span the full range and cover diverse subjects.
PICSUM_IDS = [
    10, 15, 20, 25, 30, 35, 40, 45, 50, 55,
    60, 65, 70, 75, 80, 85, 90, 95, 100, 106,
    110, 115, 120, 125, 130, 135, 140, 145, 150, 155,
    160, 165, 170, 175, 180, 185, 190, 195, 200, 206,
    210, 215, 220, 225, 230, 235, 240, 245, 250, 255,
    260, 265, 270, 275, 280, 285, 290, 295, 300, 306,
    310, 315, 320, 325, 330, 335, 340, 345, 350, 355,
    360, 365, 370, 375, 380, 385, 390, 395, 400, 406,
    410, 415, 420, 425, 430, 435, 440, 445, 450, 455,
    460, 465, 470, 475, 480, 485, 490, 495, 500, 506,
    510, 515, 520, 525, 530, 535, 540, 545, 550, 555,
    560, 565, 570, 575, 580, 585, 590, 595, 600, 614,
    620, 625, 630, 635, 640, 645, 650, 655, 660, 665,
    670, 675, 680, 685, 690, 695, 700, 706, 710, 715,
    720, 725, 730, 735, 740, 745, 750, 755, 760, 765,
]  # 150 IDs

# ──────────────────────── Pexels direct photo URLs ────────────────────────────
# These are direct Pexels photo IDs — resolved to their CDN download links.
# Covers categories Picsum is weak on: animals, food, macro, sport, tech.
# 70 IDs to bring the total to 220.
PEXELS_PHOTO_IDS = [
    2253275, 3573351, 1563356, 247431,  1170986,
    1166209, 1108099, 631477,  3621341, 3621344,
    1440727, 3861458, 3773025, 2325446, 2422915,
    1366919, 1366960, 2304168, 616833,  1007426,
    1181671, 1456706, 442580,  1126993, 2280551,
    1591447, 163036,  326055,  709552,  1269968,
    1648372, 1002141, 1536619, 2388569, 3184325,
    3225517, 3184291, 3401903, 3401904, 3401906,
    3622608, 3622619, 3622622, 3757607, 3766106,
    3766108, 4109743, 4109754, 4109758, 4116610,
    4116625, 4122621, 4122625, 4458441, 4458445,
    4458448, 4458451, 4553028, 4553030, 4553034,
    4553038, 4553072, 4553079, 4553085, 4553088,
    4553091, 4553094, 4553097, 4770292, 4770294,
]  # 70 IDs

PEXELS_CDN = "https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg?auto=compress&cs=tinysrgb&w=1024"


# ──────────────────────── helpers ─────────────────────────────────────────────

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_log() -> dict:
    """Load the download log (tracks seen content hashes to avoid duplicates)."""
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_log(log: dict) -> None:
    LOG_FILE.write_text(json.dumps(log, indent=2))


def _is_valid_image(data: bytes, min_res: int = MIN_RES) -> bool:
    """Return True if bytes decode to an image meeting the minimum resolution."""
    try:
        import cv2, numpy as np
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        return min(h, w) >= min_res
    except Exception:
        # cv2 not available — skip resolution check, at least ensure non-empty
        return len(data) > 5_000


def _fetch(url: str, timeout: int = 30) -> bytes | None:
    """GET url, return raw bytes or None on error."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "aidetector-dataset-builder/1.0"})
        r.raise_for_status()
        return r.content
    except requests.RequestException as e:
        tqdm.write(f"    ⚠ fetch failed ({url[:60]}…): {e}")
        return None


# ──────────────────────── downloaders ─────────────────────────────────────────

def download_picsum(ids: list[int], dest_dir: Path,
                    log: dict, start_idx: int,
                    pbar: tqdm) -> tuple[int, list[str]]:
    """
    Download Picsum photos by ID.

    Returns:
        (n_downloaded, list_of_saved_paths)
    """
    downloaded = 0
    saved_paths = []

    for pic_id in ids:
        url = f"https://picsum.photos/id/{pic_id}/{IMG_SIZE}/{IMG_SIZE}"

        # Skip if we already have this Picsum ID in the log
        log_key = f"picsum_{pic_id}"
        if log_key in log:
            continue

        data = _fetch(url)
        if data is None:
            log[log_key] = "failed"
            time.sleep(DELAY_S)
            continue

        h = _sha256(data)
        if h in log.values():
            log[log_key] = "dupe"
            continue

        if not _is_valid_image(data):
            log[log_key] = "low_res"
            time.sleep(DELAY_S)
            continue

        idx = start_idx + downloaded
        path = dest_dir / f"real_{idx:03d}.jpg"
        path.write_bytes(data)
        log[log_key] = h
        saved_paths.append(str(path))
        downloaded += 1
        pbar.update(1)
        time.sleep(DELAY_S)

    return downloaded, saved_paths


def download_pexels(ids: list[int], dest_dir: Path,
                    log: dict, start_idx: int,
                    pbar: tqdm) -> tuple[int, list[str]]:
    """
    Download Pexels photos by photo ID via the CDN pattern.

    Returns:
        (n_downloaded, list_of_saved_paths)
    """
    downloaded = 0
    saved_paths = []

    for pid in ids:
        log_key = f"pexels_{pid}"
        if log_key in log:
            continue

        url = PEXELS_CDN.format(pid=pid)
        data = _fetch(url)
        if data is None:
            # Try alternate resolution
            url2 = f"https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg"
            data = _fetch(url2)

        if data is None:
            log[log_key] = "failed"
            time.sleep(DELAY_S)
            continue

        h = _sha256(data)
        if h in log.values():
            log[log_key] = "dupe"
            continue

        if not _is_valid_image(data):
            log[log_key] = "low_res"
            time.sleep(DELAY_S)
            continue

        idx = start_idx + downloaded
        path = dest_dir / f"real_px_{idx:03d}.jpg"
        path.write_bytes(data)
        log[log_key] = h
        saved_paths.append(str(path))
        downloaded += 1
        pbar.update(1)
        time.sleep(DELAY_S)

    return downloaded, saved_paths


# ──────────────────────── test split ──────────────────────────────────────────

def create_test_split(train_dir: Path, test_dir: Path,
                      n_test: int = TARGET_TEST) -> None:
    """
    Move the last `n_test` images from train_dir into test_dir.
    Safe to call multiple times — won't move more than needed.
    """
    import shutil, random

    test_dir.mkdir(parents=True, exist_ok=True)
    existing_test = list(test_dir.glob("*.jpg"))

    if len(existing_test) >= n_test:
        print(f"  ✓ Test split already has {len(existing_test)} real images — skipping.")
        return

    need = n_test - len(existing_test)
    candidates = sorted(train_dir.glob("*.jpg"))

    if len(candidates) < need:
        print(f"  ⚠ Only {len(candidates)} train images available, need {need}.")
        need = len(candidates) // 5

    random.seed(42)
    to_move = random.sample(candidates, need)

    for p in to_move:
        shutil.move(str(p), str(test_dir / ("real_t_" + p.name)))

    print(f"  ✓ Moved {need} images → {test_dir}")


# ──────────────────────── verification ────────────────────────────────────────

def verify(train_dir: Path, test_dir: Path) -> None:
    """Print a quick count + resolution check."""
    try:
        import cv2, numpy as np
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    def check_dir(d: Path, label: str):
        imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
        if not imgs:
            print(f"  {label}: 0 images ⚠")
            return
        bad = []
        low_res = []
        for p in imgs:
            if p.stat().st_size < 2000:
                bad.append(p.name)
                continue
            if has_cv2:
                import cv2, numpy as np
                img = cv2.imread(str(p))
                if img is None:
                    bad.append(p.name)
                    continue
                h, w = img.shape[:2]
                if min(h, w) < MIN_RES:
                    low_res.append(f"{p.name}({w}×{h})")
        status = "✓" if not bad else "✗"
        print(f"  {status} {label}: {len(imgs)} images | {len(bad)} corrupt | {len(low_res)} low-res")
        if bad:
            print(f"      corrupt: {bad[:5]}")
        if low_res:
            print(f"      low-res: {low_res[:5]}")

    print("\n══════════ Verification ══════════")
    check_dir(train_dir, "data/real (train)")
    check_dir(test_dir,  "data/test/real")
    print("══════════════════════════════════\n")


# ──────────────────────── main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download 200 diverse real images for AI detector training."
    )
    parser.add_argument("--count", type=int, default=TARGET_TRAIN,
                        help=f"Training images to download (default: {TARGET_TRAIN})")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing images without downloading")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if target count already reached")
    parser.add_argument("--skip-test-split", action="store_true",
                        help="Do not move images into test set")
    args = parser.parse_args()

    REAL_TRAIN.mkdir(parents=True, exist_ok=True)
    REAL_TEST.mkdir(parents=True, exist_ok=True)

    if args.verify:
        verify(REAL_TRAIN, REAL_TEST)
        return

    # ── Count existing ────────────────────────────────────────────────────────
    existing_train = list(REAL_TRAIN.glob("*.jpg")) + list(REAL_TRAIN.glob("*.png"))
    existing_test  = list(REAL_TEST.glob("*.jpg"))  + list(REAL_TEST.glob("*.png"))
    print(f"\nExisting: {len(existing_train)} train  |  {len(existing_test)} test")

    if not args.force and len(existing_train) >= args.count:
        print(f"✓ Already have {len(existing_train)} ≥ {args.count} real training images.")
        if not args.skip_test_split:
            create_test_split(REAL_TRAIN, REAL_TEST)
        verify(REAL_TRAIN, REAL_TEST)
        return

    # ── Load dedup log ────────────────────────────────────────────────────────
    log = _load_log()
    total_needed = args.count + TARGET_TEST
    start_idx    = len(existing_train) + 1

    print(f"\n{'='*52}")
    print(f"  AI Detector — Real Image Downloader")
    print(f"  Target: {args.count} train  +  {TARGET_TEST} test  =  {total_needed} total")
    print(f"  Sources: Lorem Picsum ({len(PICSUM_IDS)} IDs)  +  Pexels ({len(PEXELS_PHOTO_IDS)} IDs)")
    print(f"{'='*52}\n")

    remaining = max(0, total_needed - len(existing_train) - len(existing_test))
    pbar = tqdm(total=remaining, desc="Downloading", unit="img")

    # ── Phase 1: Picsum (150 images) ──────────────────────────────────────────
    print(f"[1/2] Picsum photos ({len(PICSUM_IDS)} unique IDs)...")
    n1, paths1 = download_picsum(PICSUM_IDS, REAL_TRAIN, log, start_idx, pbar)
    start_idx += n1
    _save_log(log)
    print(f"      → {n1} downloaded from Picsum")

    # ── Phase 2: Pexels (70 images) ───────────────────────────────────────────
    print(f"\n[2/2] Pexels CDN photos ({len(PEXELS_PHOTO_IDS)} IDs)...")
    n2, paths2 = download_pexels(PEXELS_PHOTO_IDS, REAL_TRAIN, log, start_idx, pbar)
    _save_log(log)
    print(f"      → {n2} downloaded from Pexels")

    pbar.close()

    total_dl = n1 + n2
    total_now = len(list(REAL_TRAIN.glob("*.jpg")))
    print(f"\n✅ Downloaded {total_dl} new images  |  Total in data/real: {total_now}")

    # ── Test split ────────────────────────────────────────────────────────────
    if not args.skip_test_split:
        print(f"\n[3/3] Creating test split ({TARGET_TEST} images)...")
        create_test_split(REAL_TRAIN, REAL_TEST)

    # ── Final verify ─────────────────────────────────────────────────────────
    verify(REAL_TRAIN, REAL_TEST)

    if total_now < args.count:
        short = args.count - total_now
        print(f"⚠  Still {short} images short of the {args.count} target.")
        print("   Some Picsum/Pexels IDs may have failed. Re-run the script to retry,")
        print("   or add your own photos to data/real/ manually.")
    else:
        print("✅ Done! Run 'python main.py --train' to retrain the model.\n")


if __name__ == "__main__":
    main()
