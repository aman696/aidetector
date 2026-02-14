#!/usr/bin/env python3
"""
Process Local AI Data
=====================

Ingests local AI-generated images and videos from the `un/` directory.
- Copies images to `data/ai_generated/`.
- Extracts frames from videos to `data/ai_generated/`.
- Renames files for clarity (e.g., ai_kling_001.png).
- Regenerates test split.
"""

import os
import shutil
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "un"
DATA_DIR = PROJECT_ROOT / "data"
AI_DIR = DATA_DIR / "ai_generated"
TEST_DIR = DATA_DIR / "test"
TEST_AI_DIR = TEST_DIR / "ai_generated"

# Target number of frames to extract per video to reach ~60 total images
# We have 6 images + 2 videos. We need ~54 frames total, so ~27 per video.
TARGET_FRAMES_PER_VIDEO = 30 

# ---------------------------------------------------------------------------
# Processing Logic
# ---------------------------------------------------------------------------

def clean_ai_directories():
    """Clear existing AI data to start fresh from local source."""
    print("  Cleaning existing AI data directories...")
    if AI_DIR.exists():
        shutil.rmtree(AI_DIR)
    AI_DIR.mkdir(parents=True, exist_ok=True)
    
    if TEST_AI_DIR.exists():
        shutil.rmtree(TEST_AI_DIR)
    TEST_AI_DIR.mkdir(parents=True, exist_ok=True)


def process_images():
    """Copy and rename standalone images from source."""
    if not SOURCE_DIR.exists():
        print(f"  ✗ Source directory {SOURCE_DIR} not found!")
        return 0

    images = sorted([
        p for p in SOURCE_DIR.glob("*") 
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ])
    
    print(f"  Found {len(images)} standalone images.")
    
    count = 0
    for img_path in tqdm(images, desc="  Processing images"):
        # Determine source label from filename
        name = img_path.name.lower()
        if "kling" in name:
            label = "kling"
        elif "gemini" in name:
            label = "gemini"
        elif "banana" in name:
            label = "banana"
        else:
            label = "misc"
            
        # New filename
        count += 1
        new_name = f"ai_{label}_img_{count:03d}{img_path.suffix}"
        shutil.copy2(img_path, AI_DIR / new_name)
        
    return count


def process_videos():
    """Extract frames from videos."""
    videos = sorted([
        p for p in SOURCE_DIR.glob("*") 
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    ])
    
    print(f"  Found {len(videos)} videos.")
    
    total_frames_extracted = 0
    
    for vid_path in videos:
        print(f"  Processing video: {vid_path.name}")
        
        # Determine source label
        name = vid_path.name.lower()
        if "kling" in name:
            label = "kling"
        elif "gemini" in name:
            label = "gemini"
        else:
            label = "video"

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"    ⚠ Could not open video {vid_path.name}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"    ⚠ Could not determine frame count for {vid_path.name}")
            # Try to read anyway
            total_frames = 1000 # fallback estimate
            
        # Calculate sampling interval
        # Interval = total_frames / target
        interval = max(1, int(total_frames / TARGET_FRAMES_PER_VIDEO))
        
        frame_idx = 0
        saved_count = 0
        
        pbar = tqdm(total=TARGET_FRAMES_PER_VIDEO, desc="    Extracting frames", unit="fr")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % interval == 0:
                # Save frame
                frame_name = f"ai_{label}_vid_{total_frames_extracted:03d}.png"
                cv2.imwrite(str(AI_DIR / frame_name), frame)
                saved_count += 1
                total_frames_extracted += 1
                pbar.update(1)
                
                if saved_count >= TARGET_FRAMES_PER_VIDEO:
                    break
            
            frame_idx += 1
            
        cap.release()
        pbar.close()
        print(f"    Extracted {saved_count} frames.")
        
    return total_frames_extracted


def update_test_split(n_test=10):
    """Move 10 random AI images to test set."""
    print(f"  Updating test split (moving {n_test} images)...")
    
    ai_images = sorted(list(AI_DIR.glob("*.png")) + list(AI_DIR.glob("*.jpg")))
    
    if len(ai_images) < n_test:
        print(f"  ⚠ Not enough AI images ({len(ai_images)}) for test split!")
        return
        
    random.seed(42)
    test_samples = random.sample(ai_images, n_test)
    
    for img_path in test_samples:
        shutil.move(str(img_path), str(TEST_AI_DIR / img_path.name))
        
    print(f"  ✓ Moved {n_test} images into {TEST_AI_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  AI Image Detector — Local Data Processor")
    print("=" * 50)
    
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    clean_ai_directories()
    
    # 1. Process Images
    n_images = process_images()
    print(f"  ✓ Processed {n_images} standalone images.")
    
    # 2. Process Videos
    n_frames = process_videos()
    print(f"  ✓ Extracted {n_frames} frames from videos.")
    
    total = n_images + n_frames
    print(f"\n  Total AI images generated: {total}")
    
    # 3. Update Test Split
    update_test_split()
    
    print("\n  Done! AI Data ready in data/ai_generated/")

if __name__ == "__main__":
    main()
