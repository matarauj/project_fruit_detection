"""
Video I/O and frame annotation.

Handles reading video frames, drawing detections/tracks/counting
overlays, and writing the processed output video.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.tracking.bytetrack_wrapper import TrackedObject
from src.counting.counting_rules import LineZone


# Colour palette for bounding boxes (BGR)
BBOX_COLOUR = (0, 200, 80)          # Green
TRACK_ID_COLOUR = (255, 255, 255)   # White text
LINE_COLOUR = (0, 120, 255)         # Orange line
TRAIL_COLOUR = (0, 200, 80)         # Green trail


class VideoProcessor:
    """
    Opens a video for reading and (optionally) writes an annotated output.

    Parameters
    ----------
    input_path : Path
        Source video file.
    output_path : Path | None
        If provided, the annotated video is written here.
    output_codec : str
        FourCC codec string (default "mp4v").
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Path | None = None,
        output_codec: str = "mp4v",
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path

        self._cap = cv2.VideoCapture(str(input_path))
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS)
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._writer: cv2.VideoWriter | None = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*output_codec)
            self._writer = cv2.VideoWriter(
                str(output_path), fourcc, self.fps, (self.width, self.height)
            )

    # ------------------------------------------------------------------
    # Frame iteration
    # ------------------------------------------------------------------

    def read_frames(self):
        """Generator that yields (frame_idx, BGR frame) until the video ends."""
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        track_history: dict,
        line_zone: LineZone | None = None,
        count_m1: int = 0,
        count_m2: int = 0,
        trail_length: int = 30,
    ) -> np.ndarray:
        """
        Draw detections, track trails, counting line, and stats overlay
        onto a copy of the frame.

        Parameters
        ----------
        frame : np.ndarray
            Original BGR frame (not modified in-place).
        tracked_objects : list[TrackedObject]
            Active tracks to draw.
        track_history : dict
            Full track history for drawing trails.
        line_zone : LineZone | None
            Virtual counting line (drawn if not None).
        count_m1, count_m2 : int
            Current counts to display in the stats overlay.
        trail_length : int
            How many past positions to draw per track.

        Returns
        -------
        np.ndarray
            Annotated copy of the frame.
        """
        # TODO: Implement during Phase 5
        # 1. Draw bounding boxes + track IDs for each tracked_object
        # 2. Draw trail (last `trail_length` positions from track_history)
        # 3. Draw line_zone if provided
        # 4. Draw stats overlay (top-left corner): frame, M1, M2 counts
        raise NotImplementedError

    def write_frame(self, frame: np.ndarray) -> None:
        """Write an annotated frame to the output video."""
        if self._writer is not None:
            self._writer.write(frame)

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._cap.release()
        if self._writer is not None:
            self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
