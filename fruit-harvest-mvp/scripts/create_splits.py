"""
Create a reproducible 75 / 15 / 10 train/val/test split.

Expects annotated frames and YOLO labels to be organised as:
  data/images/<video_name>/<frame>.jpg
  data/labels/<video_name>/<frame>.txt   (YOLO format)

Outputs manifest files (lists of image paths, relative to data/) to data/splits/:
  train.txt, val.txt, test.txt

Manifest paths are always written with forward slashes ("/"), regardless
of the OS this script runs on.

Usage
-----
  python scripts/create_splits.py
  python scripts/create_splits.py --seed 99 --train 0.75 --val 0.15 --test 0.10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path, PurePosixPath


def to_posix_relative(path: Path, base: Path) -> str:
    """
    Return `path` relative to `base`, as a forward-slash string.
    """
    rel = path.relative_to(base)
    return str(PurePosixPath(*rel.parts))


def create_splits(
    images_dir: Path,
    labels_dir: Path,
    splits_dir: Path,
    data_root: Path,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42
    ) -> dict[str, int]:
    """
    Scan for annotated frames and write split manifests.

    Only frames that have a corresponding label file are included —
    unannotated frames are silently skipped.

    Manifest lines are written relative to `data_root` (typically the
    `data/` folder), using forward slashes, e.g.:
        images/video_01/f0000120.jpg

    Returns a dict with counts: {"train": N, "val": N, "test": N}.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # Clean exported file name from Label Studio
    #for fname in sorted(annotations_dir.glob("*.txt")):
    #    print(fname)
    #    new_name = fname.name.split("%5C")[-1]
    #    new_fname = fname.with_name(new_name)
    #    fname.rename(new_fname)
        

    # Collect all image paths that have a matching label
    image_paths: list[Path] = []
    for img in sorted(images_dir.rglob("*.jpg")):
        relative = img.relative_to(images_dir)
        label = labels_dir / relative.with_suffix(".txt")
        if label.exists():
            image_paths.append(img)

    if not image_paths:
        raise FileNotFoundError(
            f"No annotated frames found.\n"
            f"  Images expected in : {images_dir}\n"
            f"  Labels expected in : {labels_dir}"
        )

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(image_paths)

    n = len(image_paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": image_paths[:n_train],
        "val":   image_paths[n_train : n_train + n_val],
        "test":  image_paths[n_train + n_val :]
    }

    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, paths in splits.items():
        manifest = splits_dir / f"{split_name}.txt"
        lines = [to_posix_relative(p, data_root) for p in paths]
        manifest.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"  {split_name:5s}: {len(paths):>4d} frames  →  {manifest}")

    print(f"\n  Total annotated frames: {n}")
    print(f"  Seed used: {seed}  (re-run with the same seed for identical splits)")
    return {k: len(v) for k, v in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reproducible dataset splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/images"),
        help="Folder of extracted/annotated frames. Must be named 'images'."
        )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/labels"),
        help="Folder of YOLO .txt label files. Must be named 'labels'."
        )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/splits")
        )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root folder that manifest paths are written relative to. "
             "Should be the common parent of --images and --labels.",
    )
    parser.add_argument("--train", type=float, default=0.75)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Creating dataset splits...")
    create_splits(
        images_dir=args.images,
        labels_dir=args.labels,
        splits_dir=args.splits,
        data_root=args.data_root,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
