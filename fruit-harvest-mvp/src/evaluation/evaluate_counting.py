"""
Counting evaluation — MAE and standard deviation.

Target: MAE <= 1.5, SD <= 1.5
"""
from __future__ import annotations

import numpy as np


TARGET_MAE = 1.5
TARGET_SD  = 1.5


def evaluate_counting(
    predicted_counts: list[int],
    ground_truth_counts: list[int],
) -> dict:
    """
    Compute MAE and standard deviation of counting errors over
    multiple video clips.

    Parameters
    ----------
    predicted_counts : list[int]
        Model count for each clip.
    ground_truth_counts : list[int]
        Manual ground-truth count for each clip.

    Returns
    -------
    dict
        Keys: mae, sd, errors (per-clip absolute errors).
    """
    assert len(predicted_counts) == len(ground_truth_counts), \
        "Lists must be the same length."

    errors = [abs(p - g) for p, g in zip(predicted_counts, ground_truth_counts)]
    mae = float(np.mean(errors))
    sd  = float(np.std(errors))

    return {"mae": round(mae, 4), "sd": round(sd, 4), "errors": errors}


def print_counting_report(metrics: dict) -> None:
    """Print a pass/fail report against the target thresholds."""
    print("\n── Counting Metric Report ───────────────────────────────")
    for name, target in [("mae", TARGET_MAE), ("sd", TARGET_SD)]:
        value = metrics[name]
        status = "✓ PASS" if value <= target else "✗ FAIL"
        print(f"  {name:<6} {value:.4f}  (target ≤ {target})  {status}")
    print("─────────────────────────────────────────────────────────\n")
