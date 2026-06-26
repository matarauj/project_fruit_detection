"""
Tracking evaluation — IDF1 and associated MOT metrics.

IDF1 target: >= 0.75
Counting error target: <= 20%

Uses the `trackeval` library (or `motmetrics`) to compute standard
MOT metrics from ground-truth and hypothesis track files.
"""
from __future__ import annotations

from pathlib import Path


TARGET_IDF1 = 0.75
TARGET_COUNTING_ERROR = 0.20   # 20%


def evaluate_tracking(
    gt_dir: Path,
    pred_dir: Path,
    seq_name: str,
) -> dict:
    """
    Compute IDF1 and standard MOT metrics for a tracked sequence.

    Parameters
    ----------
    gt_dir : Path
        Directory containing ground-truth track files in MOTChallenge format.
    pred_dir : Path
        Directory containing predicted track files in MOTChallenge format.
    seq_name : str
        Sequence name (subfolder inside gt_dir and pred_dir).

    Returns
    -------
    dict
        Keys include: IDF1, MOTA, MOTP, FP, FN, IDsw.
    """
    # TODO: Implement during Phase 3
    # Consider: pip install motmetrics
    # import motmetrics as mm
    raise NotImplementedError


def compute_counting_error(predicted: int, ground_truth: int) -> float:
    """
    Compute the relative counting error.

    Returns
    -------
    float
        Error as a fraction (0.0 = perfect, 0.20 = 20% error).
    """
    if ground_truth == 0:
        return 0.0 if predicted == 0 else float("inf")
    return abs(predicted - ground_truth) / ground_truth
