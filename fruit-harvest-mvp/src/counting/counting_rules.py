"""
Counting rules and the virtual line-zone definition.

Kept separate from counter.py so rules can be unit-tested independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.tracking.track_objects import TrackHistory


class LineOrientation(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass
class LineZone:
    """
    A virtual counting line across the frame.

    For a horizontal line the position is a fraction of the frame height.
    For a vertical line it is a fraction of the frame width.

    Parameters
    ----------
    position : float
        Fractional position of the line (0.0–1.0).
    orientation : LineOrientation
        Whether the line runs horizontally or vertically.
    frame_width : int
        Frame width in pixels (needed to compute pixel coordinates).
    frame_height : int
        Frame height in pixels.
    """
    position: float
    orientation: LineOrientation
    frame_width: int
    frame_height: int

    @classmethod
    def from_config(
        cls,
        config: dict,
        frame_width: int,
        frame_height: int,
    ) -> "LineZone":
        """Construct from the counting.method1 section of settings.yaml."""
        return cls(
            position=config["line_position"],
            orientation=LineOrientation(config["line_orientation"]),
            frame_width=frame_width,
            frame_height=frame_height,
        )

    @property
    def pixel_position(self) -> float:
        """Absolute pixel position of the line."""
        if self.orientation == LineOrientation.HORIZONTAL:
            return self.position * self.frame_height
        return self.position * self.frame_width

    def start_point(self) -> tuple[int, int]:
        """(x, y) start point for drawing the line."""
        if self.orientation == LineOrientation.HORIZONTAL:
            return (0, int(self.pixel_position))
        return (int(self.pixel_position), 0)

    def end_point(self) -> tuple[int, int]:
        """(x, y) end point for drawing the line."""
        if self.orientation == LineOrientation.HORIZONTAL:
            return (self.frame_width, int(self.pixel_position))
        return (int(self.pixel_position), self.frame_height)


def has_crossed_line(
    prev_pos: tuple[float, float],
    curr_pos: tuple[float, float],
    line_zone: LineZone,
) -> bool:
    """
    Return True if a track crossed the counting line between two consecutive frames.

    Uses a simple threshold comparison on the relevant coordinate.

    Parameters
    ----------
    prev_pos : tuple[float, float]
        (cx, cy) in the previous frame.
    curr_pos : tuple[float, float]
        (cx, cy) in the current frame.
    line_zone : LineZone
        The virtual counting line.
    """
    # TODO: Implement during Phase 4
    # Hint: for a horizontal line, check if cy crossed line_zone.pixel_position.
    # A crossing occurs when the sign of (pos - line) flips between frames.
    raise NotImplementedError


def is_track_stable(
    track_id: int,
    history: TrackHistory,
    min_frames: int,
) -> bool:
    """
    Return True if the track has been visible for at least `min_frames` frames.

    Parameters
    ----------
    track_id : int
        The track to check.
    history : TrackHistory
        Full track history.
    min_frames : int
        Minimum number of frames required.
    """
    # TODO: Implement during Phase 4
    raise NotImplementedError
