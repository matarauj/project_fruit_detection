"""
Dataset utilities for YOLO training.

Used in the Kaggle training notebooks to validate splits and
inspect annotations before/after training.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def verify_dataset(splits_dir: Path, max_samples: int = 5) -> dict:
    """
    Sanity-check the dataset splits.

    Verifies that:
    - Each split manifest (.txt) exists and is non-empty.
    - Every listed image file exists on disk.
    - Every image has a corresponding label file.
    - Label files are valid YOLO format (class_id cx cy w h, normalised 0–1).

    Parameters
    ----------
    splits_dir : Path
        Directory containing train.txt, val.txt, test.txt.
    max_samples : int
        Number of samples to print details for per split.

    Returns
    -------
    dict
        Summary with counts and any errors found.
    """
    # TODO: Implement during Phase 1 (after annotation is complete)
    raise NotImplementedError


def draw_annotations(image_path: Path, label_path: Path) -> np.ndarray:
    """
    Draw YOLO bounding boxes on an image for visual inspection.

    Parameters
    ----------
    image_path : Path
        Path to the image file.
    label_path : Path
        Path to the corresponding YOLO label (.txt).

    Returns
    -------
    np.ndarray
        BGR image with bounding boxes drawn.
    """
    # TODO: Implement during Phase 1
    raise NotImplementedError


def compute_class_distribution(splits_dir: Path) -> dict:
    """
    Count total bounding box annotations per split.

    Useful for checking class balance and annotation density.

    Returns
    -------
    dict
        {"train": N, "val": N, "test": N}
    """
    # TODO: Implement during Phase 1
    raise NotImplementedError
