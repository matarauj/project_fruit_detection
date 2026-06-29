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


# Colour palette (BGR)
BBOX_COLOUR     = (0, 200, 80)      # Green — bounding boxes
TRACK_ID_COLOUR = (255, 255, 255)   # White — track ID text
LINE_COLOUR     = (0, 120, 255)     # Orange — counting line
TRAIL_COLOUR    = (0, 200, 80)      # Green — track trails
OVERLAY_BG      = (0, 0, 0)        # Black — stats overlay background


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
        output_codec: str = "mp4v"
        ) -> None:

        self.input_path = input_path
        self.output_path = output_path

        self._cap = cv2.VideoCapture(str(input_path))
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        self.fps: float          = self._cap.get(cv2.CAP_PROP_FPS)
        self.width: int          = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int         = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames: int   = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._writer: cv2.VideoWriter | None = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*output_codec)
            self._writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )

    # ------------------------------------------------------------------
    # Frame iteration
    # ------------------------------------------------------------------

    def read_frames(self):
        """
        Generator that yields (frame_idx, BGR frame) until the video ends.
        """
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
        trail_length: int = 30
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
        out = frame.copy()

        # 1. Draw trail for each active track
        for obj in tracked_objects:
            positions = track_history.get(obj.track_id, [])
            # Take only the last `trail_length` positions
            recent = positions[-trail_length:]

            # Extract (cx, cy) pairs and convert to integer pixel coords
            points = [(int(cx), int(cy)) for (_fi, cx, cy) in recent]
            
            # Draw line segments connecting successive trail points
            for i in range(1, len(points)):
                # Fade trail: older segments are thinner
                alpha = i / len(points)           # 0 → oldest, 1 → newest
                thickness = max(1, int(alpha * 3))
                cv2.line(out, points[i - 1], points[i], TRAIL_COLOUR, thickness)

        # 2. Draw bounding box and track ID for each active track
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox_xyxy.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), BBOX_COLOUR, thickness=2)

            label = f"#{obj.track_id}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness  = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Small filled rectangle behind the text for legibility
            cv2.rectangle(
                out,
                (x1, y1 - th - baseline - 4),
                (x1 + tw + 4, y1),
                BBOX_COLOUR,
                cv2.FILLED
            )
            cv2.putText(
                out, label,
                (x1 + 2, y1 - baseline - 2),
                font, font_scale, TRACK_ID_COLOUR, thickness,
                cv2.LINE_AA
            )

        # 3. Draw counting line
        if line_zone is not None:
            cv2.line(
                out,
                line_zone.start_point(),
                line_zone.end_point(),
                LINE_COLOUR,
                thickness=2
            )
            # Small label at the start of the line
            cv2.putText(
                out, "COUNT LINE",
                (line_zone.start_point()[0] + 4, line_zone.start_point()[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_COLOUR, 1, cv2.LINE_AA
            )

        # 4. Stats overlay — semi-transparent black panel in the top-left corner
        lines = [
            f"M1 (line):  {count_m1}",
            f"M2 (uid) :  {count_m2}"
        ]
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1
        pad        = 8
        line_h     = 22

        panel_w = 180
        panel_h = pad * 2 + line_h * len(lines)

        # Draw the background panel using addWeighted for transparency
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), OVERLAY_BG, cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        for i, text in enumerate(lines):
            y = pad + (i + 1) * line_h - 4
            cv2.putText(
                out, text,
                (pad, y),
                font, font_scale, TRACK_ID_COLOUR, thickness, cv2.LINE_AA
            )
        return out

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write an annotated frame to the output video.
        """
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
