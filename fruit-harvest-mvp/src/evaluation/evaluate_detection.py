"""
Detection evaluation utilities.

Wraps Ultralytics' built-in validation to compute and log
the target metrics: Precision, Recall, F1, AP, mAP50, mAP50-95.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from ultralytics import YOLO


TARGET_METRICS = {
    "precision": 0.80,
    "recall":    0.80,
    "f1":        0.80,
    "AP":        0.75,
    "mAP50":     0.75,
    "mAP50-95":  0.40,
}


def evaluate_detection(
    model_path: str | Path,
    dataset_yaml: str | Path,
    split: str = "test",
    imgsz: int = 640,
    device: str = "",
    save_dir: Path | None = None,
) -> dict:
    """
    Run Ultralytics validation and return a dict of metric names → values.

    Parameters
    ----------
    model_path : str | Path
        Path to the trained .pt weights.
    dataset_yaml : str | Path
        Path to the YOLO dataset.yaml.
    split : str
        Which split to evaluate ("val" or "test").
    imgsz : int
        Inference image size.
    device : str
        Device string (""=auto, "cpu", "0", ...).
    save_dir : Path | None
        If provided, save a CSV summary of results here.

    Returns
    -------
    dict
        Keys: precision, recall, f1, mAP50, mAP50-95, AP (per-class).
    """
    # TODO: Implement during Phase 2
    # Hint:
    #   model = YOLO(model_path)
    #   metrics = model.val(data=dataset_yaml, split=split, imgsz=imgsz, device=device)
    #   metrics.box.map50  → mAP50
    #   metrics.box.map    → mAP50-95
    #   metrics.box.mp     → mean precision
    #   metrics.box.mr     → mean recall
    raise NotImplementedError


def print_metric_report(metrics: dict) -> None:
    """Print a formatted pass/fail report against the target thresholds."""
    print("\n── Detection Metric Report ──────────────────────────────")
    for name, target in TARGET_METRICS.items():
        value = metrics.get(name, float("nan"))
        status = "✓ PASS" if value >= target else "✗ FAIL"
        print(f"  {name:<12} {value:.4f}  (target ≥ {target})  {status}")
    print("─────────────────────────────────────────────────────────\n")
