"""
Fruit counter — implements Method 1 (line-crossing) and Method 2 (global ID).

Both methods are updated incrementally frame by frame.
The FruitCounter is stateful; instantiate once per video, then call
update() on each frame, and read .count_m1 / .count_m2 at any time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.tracking.bytetrack_wrapper import TrackedObject
from src.tracking.track_objects import TrackHistory
from src.counting.counting_rules import LineZone, has_crossed_line, is_track_stable


@dataclass
class FrameStats:
    """Per-frame snapshot returned by FruitCounter.update()."""
    frame_idx: int
    detections_in_frame: int
    count_m1: int       # Running total from line-crossing
    count_m2: int       # Running total from global ID
    active_tracks: int


class FruitCounter:
    """
    Stateful fruit counter for a single video.

    Method 1 — line crossing:
        Counts a track the first time its centre crosses the virtual
        counting line, provided it has been stable for at least
        `min_track_frames` frames.

    Method 2 — global unique ID:
        Counts a track the first time it reaches `min_track_frames`
        frames of consecutive visibility.

    Parameters
    ----------
    fps : float
        Video frame rate. Used to compute the seconds-equivalent of
        min_track_frames for reporting.
    min_track_frames : int
        Minimum frames a track must be present before counting.
    line_zone : LineZone | None
        Virtual counting line for Method 1. If None, M1 is disabled.
    """

    def __init__(
        self,
        fps: float,
        min_track_frames: int = 15,
        line_zone: LineZone | None = None
        ) -> None:
        self.fps = fps
        self.min_track_frames = min_track_frames
        self.line_zone = line_zone

        # Method 1 state
        self._m1_counted: set[int] = set()          # track IDs already counted by M1
        self._m1_prev_positions: dict[int, tuple[float, float]] = {}  # last frame position

        # Method 2 state
        self._m2_counted: set[int] = set()          # track IDs already counted by M2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        tracked_objects: list[TrackedObject],
        history: TrackHistory
        ) -> FrameStats:
        """
        Process one frame and update both counters.

        Parameters
        ----------
        frame_idx : int
            Current frame index.
        tracked_objects : list[TrackedObject]
            Active tracked objects in this frame.
        history : TrackHistory
            Full track history up to this frame.

        Returns
        -------
        FrameStats
            Snapshot of counts after processing this frame.
        """
        for obj in tracked_objects:
            tid = obj.track_id
            stable = is_track_stable(tid, history, self.min_track_frames)

            # ── Method 2: global unique ID ────────────────────────────────
            if stable and tid not in self._m2_counted:
                self._m2_counted.add(tid)

            # ── Method 1: line crossing ───────────────────────────────────
            if (
                self.line_zone is not None
                and stable
                and tid not in self._m1_counted
                and tid in self._m1_prev_positions
            ):
                prev = self._m1_prev_positions[tid]
                curr = obj.centre
                if has_crossed_line(prev, curr, self.line_zone):
                    self._m1_counted.add(tid)

            # Always update previous position for next frame's crossing check
            self._m1_prev_positions[tid] = obj.centre

        return FrameStats(
            frame_idx=frame_idx,
            detections_in_frame=len(tracked_objects),
            count_m1=self.count_m1,
            count_m2=self.count_m2,
            active_tracks=len(tracked_objects)
        )


    def reset(self) -> None:
        """
        Reset all state (call between videos).
        """
        self._m1_counted.clear()
        self._m1_prev_positions.clear()
        self._m2_counted.clear()


    @property
    def count_m1(self) -> int:
        """
        Total fruit counted by Method 1 (line-crossing).
        """
        return len(self._m1_counted)


    @property
    def count_m2(self) -> int:
        """
        Total fruit counted by Method 2 (global unique ID).
        """
        return len(self._m2_counted)