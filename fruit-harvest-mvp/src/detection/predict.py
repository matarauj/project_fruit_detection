"""
YOLO inference wrapper for apple detection.

Loads a trained YOLO11 model and runs detection on frames or video.
All tracking is delegated to the tracking module; this module is
detection-only.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Results


class FruitDetector:
    """
    Wraps an Ultralytics YOLO model for single-class apple detection.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "",
        track_high_thresh: float = 0.25,
        tracker_config_path: str | Path = "configs/bytetrack.yaml"
        ) -> None:
        """
        Parameters
        ----------
        model_path : str | Path
            Path to the trained .pt weights file.
        conf : float
            Confidence threshold; detections below this are discarded.
        iou : float
            NMS IoU threshold.
        imgsz : int
            Inference image size (resized internally by YOLO).
        device : str
            Device string ("cpu", "0", "cuda:0", or "" for auto).
        track_high_thresh : float
            Score threshold for high-confidence ByteTrack association.
            Overrides the base tracker_config_path's value for this
            detector instance only (does not mutate the file on disk).
        tracker_config_path : str | Path
            Base ByteTrack YAML to patch. Other tracker settings
            (track_low_thresh, track_buffer, match_thresh, etc.) are
            taken from this file unchanged.
        """
        self.model_path = Path(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.track_high_thresh = track_high_thresh
        self.base_tracker_config_path = Path(tracker_config_path)

        self.model = YOLO(str(self.model_path))

        # Patch track_high_thresh into a per-instance temp YAML so the
        # repo's configs/bytetrack.yaml is never mutated.
        self.tracker_yaml_path = self._build_patched_tracker_config()

    def _build_patched_tracker_config(self) -> Path:
        """
        Write a temp copy of the base tracker YAML with track_high_thresh
        overridden, and return its path.
        """
        with open(self.base_tracker_config_path, "r") as f:
            tracker_cfg = yaml.safe_load(f)

        tracker_cfg["track_high_thresh"] = self.track_high_thresh

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.safe_dump(tracker_cfg, tmp)
        tmp.close()
        return Path(tmp.name)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_frame(self, frame: np.ndarray) -> Results:
        """
        Run detection on a single BGR frame (OpenCV format).

        Parameters
        ----------
        frame : np.ndarray
            HxWx3 BGR image array.

        Returns
        -------
        Results
            Ultralytics Results object for the frame.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        return results[0]

    def track_frame(self, frame: np.ndarray, persist: bool = True) -> Results:
        """
        Run detection + ByteTrack tracking on a single frame.

        Uses Ultralytics' built-in tracker. Call with `persist=True` on
        every consecutive frame of the same video to maintain track state.

        Parameters
        ----------
        frame : np.ndarray
            HxWx3 BGR image array.
        persist : bool
            Whether to persist the tracker state across calls.

        Returns
        -------
        Results
            Ultralytics Results object; boxes include `.id` (track IDs).
        """
        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            tracker=str(self.tracker_yaml_path),
            persist=persist,
            verbose=False
        )
        return results[0]

    def reset_tracker(self) -> None:
        """
        Reset the tracker state (call between different video files).
        """
        # Ultralytics trackers are stateful; re-instantiating the model
        # is the safest reset. A lighter alternative exists in some versions.
        self.model = YOLO(str(self.model_path))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """
        Run a dummy inference to warm up the model (reduces first-frame latency).
        """
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(source=dummy, verbose=False)

    def __repr__(self) -> str:
        return (
            f"FruitDetector(model={self.model_path.name}, "
            f"conf={self.conf}, iou={self.iou}, imgsz={self.imgsz})"
        )
