"""
AI Image Detector - CLI Entry Point (v2 unified model).

Usage:
    python main.py --image <path>        Classify a single image
    python main.py --batch <directory>   Classify every image in a directory
    python main.py --train [--gpu]       Train the unified v2 detector
    python main.py --evaluate            Evaluate v2 on the test + holdout splits

The unified model classifies fully-AI images across conditions (clean,
social-media-compressed, screenshotted, chained). It uses 85 classical
features + a DINOv2 embedding; without torch it falls back to the 85-feature
classical model automatically. Training/evaluation details: see CLAUDE.md and
V2_PROGRESS.md.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    print("=" * 60)
    print("  AI Image Detector — unified v2 (classical + DINOv2)")
    print("=" * 60)
    print()


def _load_detector():
    from src.unified_detector import UnifiedDetector
    try:
        return UnifiedDetector()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Train a model first:  python main.py --train --gpu")
        sys.exit(1)


def analyze_single_image(image_path: str):
    print_banner()
    print(f"Analyzing: {image_path}")
    print("-" * 60)
    detector = _load_detector()
    start = time.time()
    result = detector.predict(image_path)
    elapsed = time.time() - start

    print(f"\n{'=' * 40}")
    print(f"  RESULT: {result['label']}")
    print(f"  Confidence: {result['confidence']:.1%}  (p(AI)={result['probability_ai']:.1%})")
    print(f"  Model: {result['method']}{'  [classical fallback]' if result['fallback'] else ''}")
    print(f"{'=' * 40}")
    print(f"\n{result['explanation']}")
    if result.get("screenshot_warning"):
        print(f"\n{result['screenshot_warning']}")
    print(f"\nAnalysis completed in {elapsed:.2f}s")


def batch_analyze(directory: str):
    from src.utils import get_all_image_paths
    print_banner()
    print(f"Batch Analysis: {directory}")
    print("-" * 60)
    detector = _load_detector()
    paths = get_all_image_paths(directory)
    print(f"Found {len(paths)} images\n")

    summary = {"Real": 0, "AI-Generated": 0}
    for path in paths:
        try:
            result = detector.predict(path)
            summary[result["label"]] = summary.get(result["label"], 0) + 1
            print(f"  {os.path.basename(path):40s} -> {result['label']:15s} "
                  f"({result['confidence']:.1%})")
        except Exception as exc:
            print(f"  {os.path.basename(path):40s} -> ERROR: {exc}")

    print(f"\n{'=' * 40}\nSummary:")
    for label, count in summary.items():
        print(f"  {label}: {count}")
    print(f"  Total: {sum(summary.values())}")


def train_model(gpu: bool, folds: int):
    from src.train_unified import train_unified, enable_gpu
    print_banner()
    print("Training unified v2 detector")
    print("-" * 60)
    use_gpu = gpu and enable_gpu()
    if use_gpu:
        print("[gpu] cuML GPU SVC active for the grid search")
    train_unified(n_splits=folds, n_jobs=-1, gpu=use_gpu)


def evaluate_model():
    from src.evaluate_unified import evaluate_unified
    print_banner()
    print("Evaluating unified v2 detector")
    print("-" * 60)
    evaluate_unified()


def main():
    parser = argparse.ArgumentParser(
        description="AI Image Detector — detect fully-AI images across conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --image photo.jpg
  python main.py --batch data/real/coco/
  python main.py --train --gpu
  python main.py --evaluate
        """,
    )
    parser.add_argument("--image", "-i", type=str, help="Classify a single image")
    parser.add_argument("--batch", "-b", type=str, help="Classify a directory of images")
    parser.add_argument("--train", "-t", action="store_true",
                        help="Train the unified v2 detector")
    parser.add_argument("--evaluate", "-e", action="store_true",
                        help="Evaluate v2 on the test + holdout splits")
    parser.add_argument("--gpu", action="store_true",
                        help="use cuML GPU acceleration during --train")
    parser.add_argument("--folds", type=int, default=5,
                        help="CV folds for --train (default 5)")
    # Deprecated v1 flags — the unified model handles screenshots directly.
    parser.add_argument("--train-screenshot", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--screenshot-mode", "-s", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.train_screenshot or args.screenshot_mode:
        print("Note: --train-screenshot / --screenshot-mode are retired. The "
              "unified v2 model handles screenshots directly.")

    if args.train:
        train_model(gpu=args.gpu, folds=args.folds)
    elif args.evaluate:
        evaluate_model()
    elif args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        analyze_single_image(args.image)
    elif args.batch:
        if not os.path.isdir(args.batch):
            print(f"Error: Directory not found: {args.batch}")
            sys.exit(1)
        batch_analyze(args.batch)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
