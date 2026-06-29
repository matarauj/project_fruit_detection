"""
YOLO inference wrapper for apple detection.

Loads a trained YOLO11 model and runs detection on frames or video.
All tracking is delegated to the tracking module; this module is
detection-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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
        device: str = ""
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
        """
        self.model_path = Path(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

        self.model = YOLO(str(self.model_path))

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
            tracker="configs/bytetrack.yaml",
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
