"""
Dataset manifest and leak-proof split management.

The manifest is the single source of truth for what images exist, their labels
and their generator family. Splits are assigned at BASE-IMAGE level and every
derived variant (platform-compressed, screenshotted, chained) inherits the
split of its base image — this is the core mechanism preventing the same
underlying picture from appearing in both train and test under different
conditions.

Layout produced:
    data/manifests/base_manifest.json    one record per source image
    data/manifests/splits.json           {meta: {...}, assignments: {id: split}}

Splits: 'train' | 'val' | 'test' | 'test_holdout'
'test_holdout' contains ENTIRE generator families never seen in training,
used to measure generalization to unseen generators.
"""

import argparse
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

from PIL import Image

from src.utils import SUPPORTED_EXTENSIONS


MANIFEST_DIR = os.path.join('data', 'manifests')
BASE_MANIFEST_PATH = os.path.join(MANIFEST_DIR, 'base_manifest.json')
SPLITS_PATH = os.path.join(MANIFEST_DIR, 'splits.json')
DERIVED_MANIFEST_PATH = os.path.join(MANIFEST_DIR, 'derived_manifest.json')

# Entire families routed to 'test_holdout' (newest model of each product line):
# the model must never train on these so their scores measure generalization
# to unseen generators.
DEFAULT_HOLDOUT_FAMILIES = (
    'midjourney_7',
    'ideogram_3.0',
    'imagen_4.0',
    'flux_1',
    'recraft_v3',
)

# (relative directory, label, source name) — family is the subdirectory name
# for data/mixed, otherwise equal to the source name.
_SCAN_ROOTS = (
    (os.path.join('real', 'coco'), 0, 'coco'),
    (os.path.join('real', 'openfake'), 0, 'openfake'),
    ('Gemini', 1, 'gemini'),
    ('GPT', 1, 'gpt'),
)


def _slugify(text: str) -> str:
    """Lowercase and reduce to [a-z0-9_] so ids are filesystem/cache safe."""
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def _image_size(path: str) -> Tuple[int, int]:
    """Returns (width, height) from the file header without decoding pixels."""
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


def _scan_dir(abs_dir: str) -> List[str]:
    """Sorted list of supported image filenames directly inside abs_dir."""
    if not os.path.isdir(abs_dir):
        return []
    names = []
    for name in sorted(os.listdir(abs_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            names.append(name)
    return names


def build_manifest(data_root: str = 'data', verbose: bool = False) -> List[Dict]:
    """
    Scans the on-disk dataset and returns one record per base image.

    Record schema:
        id      stable unique slug, safe for use as a cache filename
        path    path relative to the project root
        label   0 = real, 1 = AI-generated
        source  'coco' | 'openfake' | 'gemini' | 'gpt' | 'mixed'
        family  generator family ('coco', 'dalle_3', 'sd_1.5', ...)
        width, height   from the image header (0 if unreadable)
    """
    records: List[Dict] = []
    seen_ids = set()

    def add(rel_dir: str, label: int, source: str, family: str) -> None:
        abs_dir = os.path.join(data_root, rel_dir)
        for name in _scan_dir(abs_dir):
            stem = _slugify(os.path.splitext(name)[0])
            rec_id = f"{_slugify(source)}_{_slugify(family)}_{stem}" \
                if source == 'mixed' else f"{_slugify(family)}_{stem}"
            # Collisions should not happen with current naming; make them
            # impossible anyway since ids key every cache in the pipeline.
            base_id, n = rec_id, 1
            while rec_id in seen_ids:
                n += 1
                rec_id = f"{base_id}_{n}"
            seen_ids.add(rec_id)

            path = os.path.join(data_root, rel_dir, name)
            width, height = _image_size(path)
            records.append({
                'id': rec_id,
                'path': path,
                'label': label,
                'source': source,
                'family': family,
                'width': width,
                'height': height,
            })

    for rel_dir, label, source in _SCAN_ROOTS:
        add(rel_dir, label, source, family=source)

    mixed_root = os.path.join(data_root, 'mixed')
    if os.path.isdir(mixed_root):
        for family in sorted(os.listdir(mixed_root)):
            fam_dir = os.path.join('mixed', family)
            if os.path.isdir(os.path.join(data_root, fam_dir)):
                add(fam_dir, label=1, source='mixed', family=family)

    if verbose:
        n_real = sum(1 for r in records if r['label'] == 0)
        print(f"Manifest: {len(records)} images "
              f"({n_real} real, {len(records) - n_real} AI, "
              f"{len({r['family'] for r in records})} families)")
    return records


def build_splits(manifest: List[Dict],
                 holdout_families: Tuple[str, ...] = DEFAULT_HOLDOUT_FAMILIES,
                 train_frac: float = 0.7,
                 val_frac: float = 0.1,
                 seed: int = 42) -> Dict[str, str]:
    """
    Assigns each base-image id to 'train' / 'val' / 'test' / 'test_holdout'.

    Held-out families go entirely to 'test_holdout'. The remaining images are
    split per family (stratified) so each family appears in train/val/test in
    the same proportions. Deterministic for a given seed.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    missing = set(holdout_families) - {r['family'] for r in manifest}
    if missing:
        raise ValueError(f"Holdout families not present in manifest: {sorted(missing)}")

    rng = random.Random(seed)
    assignments: Dict[str, str] = {}

    by_family: Dict[str, List[str]] = {}
    for rec in manifest:
        by_family.setdefault(rec['family'], []).append(rec['id'])

    for family, ids in sorted(by_family.items()):
        if family in holdout_families:
            for rec_id in ids:
                assignments[rec_id] = 'test_holdout'
            continue
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        for i, rec_id in enumerate(ids):
            if i < n_train:
                assignments[rec_id] = 'train'
            elif i < n_train + n_val:
                assignments[rec_id] = 'val'
            else:
                assignments[rec_id] = 'test'

    return assignments


def save_json(obj, path: str) -> None:
    """Writes JSON with parent-directory creation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


def load_manifest(path: str = BASE_MANIFEST_PATH) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def load_splits(path: str = SPLITS_PATH) -> Dict[str, str]:
    with open(path) as f:
        return json.load(f)['assignments']


def load_derived_manifest(path: str = DERIVED_MANIFEST_PATH) -> List[Dict]:
    """Derived-variant records (Stage 1). Empty list if not built yet."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def load_split(split_name: str,
               conditions: Optional[List[str]] = None,
               include_derived: bool = True) -> List[Dict]:
    """
    Returns the records belonging to one split, joining base images
    (condition 'clean') with derived variants that inherit the base's split.

    Args:
        split_name: 'train' | 'val' | 'test' | 'test_holdout'.
        conditions: optional filter, e.g. ['clean', 'facebook', 'screenshot'].
        include_derived: set False for base images only.
    """
    splits = load_splits()
    records = []
    for rec in load_manifest():
        if splits.get(rec['id']) == split_name:
            rec = dict(rec)
            rec['condition'] = 'clean'
            rec['base_id'] = rec['id']
            records.append(rec)
    if include_derived:
        for rec in load_derived_manifest():
            if splits.get(rec['base_id']) == split_name:
                records.append(rec)
    if conditions is not None:
        wanted = set(conditions)
        records = [r for r in records if r['condition'] in wanted]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset manifest and splits")
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-frac', type=float, default=0.7)
    parser.add_argument('--val-frac', type=float, default=0.1)
    args = parser.parse_args()

    manifest = build_manifest(args.data_root, verbose=True)
    save_json(manifest, BASE_MANIFEST_PATH)

    assignments = build_splits(manifest,
                               train_frac=args.train_frac,
                               val_frac=args.val_frac,
                               seed=args.seed)
    save_json({
        'meta': {
            'seed': args.seed,
            'train_frac': args.train_frac,
            'val_frac': args.val_frac,
            'holdout_families': list(DEFAULT_HOLDOUT_FAMILIES),
        },
        'assignments': assignments,
    }, SPLITS_PATH)

    counts: Dict[str, int] = {}
    for split in assignments.values():
        counts[split] = counts.get(split, 0) + 1
    print(f"Splits written to {SPLITS_PATH}: {counts}")


if __name__ == '__main__':
    main()
