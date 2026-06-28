"""
Per-frame tracking logic.

Combines the detector (which also runs ByteTrack) with lightweight
track history management used by the counting module.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.tracking.bytetrack_wrapper import TrackedObject, parse_tracking_results


# Track history: track_id → list of (frame_idx, cx, cy)
TrackHistory = dict[int, list[tuple[int, float, float]]]


def update_track_history(
    tracked_objects: list[TrackedObject],
    history: TrackHistory
    ) -> TrackHistory:
    """
    Append the current-frame positions of all active tracks to the history.

    Parameters
    ----------
    tracked_objects : list[TrackedObject]
        Active tracks in the current frame (from bytetrack_wrapper).
    history : TrackHistory
        Mutable dict mapping track_id → [(frame_idx, cx, cy), ...].
        Pass an empty dict on the first frame.

    Returns
    -------
    TrackHistory
        Updated history (same object, mutated in-place).
    """
    for obj in tracked_objects:
        if obj.track_id not in history:
            history[obj.track_id] = []
        history[obj.track_id].append((obj.frame_idx, obj.cx, obj.cy))

    return history


def track_frame(
    detector,
    frame: np.ndarray,
    frame_idx: int,
    history: TrackHistory,
    ) -> tuple[list[TrackedObject], TrackHistory]:
    """
    Run detection + tracking on one frame and update the history.

    Parameters
    ----------
    detector : FruitDetector
        Initialised FruitDetector instance (with tracker state).
    frame : np.ndarray
        BGR frame from OpenCV.
    frame_idx : int
        Zero-based frame index.
    history : TrackHistory
        Running track history dict (mutated in-place).

    Returns
    -------
    tuple[list[TrackedObject], TrackHistory]
        Active tracked objects in this frame, and the updated history.
    """
    # 1. Run detection + ByteTrack via the FruitDetector
    results = detector.track_frame(frame)

    # 2. Normalise Ultralytics output into TrackedObject instances
    tracked_objects = parse_tracking_results(results, frame_idx)

    # 3. Append current positions to the running history
    update_track_history(tracked_objects, history)

    return tracked_objects, history