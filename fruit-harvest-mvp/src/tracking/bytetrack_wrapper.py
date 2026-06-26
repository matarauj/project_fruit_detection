"""
ByteTrack wrapper.

Ultralytics handles ByteTrack natively via model.track(), so this module
is a thin, testable adapter that normalises the output into a consistent
format used by the rest of the pipeline.

Tracking state is managed entirely inside Ultralytics — this wrapper
does not maintain its own state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ultralytics.engine.results import Results


@dataclass
class TrackedObject:
    """
    A single tracked detection in one frame.

    Attributes
    ----------
    track_id : int
        Unique ID assigned by ByteTrack. Stable across frames.
    bbox_xyxy : np.ndarray
        Bounding box as [x1, y1, x2, y2] in pixel coordinates.
    confidence : float
        Detection confidence score.
    class_id : int
        Detected class index (0 = apple).
    frame_idx : int
        Frame number this detection belongs to.
    """
    track_id: int
    bbox_xyxy: np.ndarray
    confidence: float
    class_id: int
    frame_idx: int

    @property
    def cx(self) -> float:
        """Centre x coordinate."""
        return float((self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2)

    @property
    def cy(self) -> float:
        """Centre y coordinate."""
        return float((self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2)

    @property
    def centre(self) -> tuple[float, float]:
        return self.cx, self.cy


def parse_tracking_results(results: Results, frame_idx: int) -> list[TrackedObject]:
    """
    Convert an Ultralytics Results object (with track IDs) into a list
    of TrackedObject instances.

    Parameters
    ----------
    results : Results
        Output of model.track() for a single frame. Boxes must include
        track IDs (results.boxes.id is not None).
    frame_idx : int
        Current frame number (0-indexed).

    Returns
    -------
    list[TrackedObject]
        One TrackedObject per active track in the frame.
        Returns an empty list if no tracked detections are present.
    """
    # TODO: Implement during Phase 3
    # Hints:
    #   results.boxes.xyxy  → tensor of [x1,y1,x2,y2]
    #   results.boxes.id    → tensor of track IDs (may be None if no detections)
    #   results.boxes.conf  → tensor of confidence scores
    #   results.boxes.cls   → tensor of class IDs
    raise NotImplementedError
