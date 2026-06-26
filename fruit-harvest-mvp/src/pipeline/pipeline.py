"""
End-to-end pipeline: video → detect → track → count → results.

FruitPipeline wires together the detector, counter, and video processor.
It is called by the Streamlit app and can also be used headlessly in
evaluation scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np

from src.utils import load_config
from src.detection.predict import FruitDetector
from src.counting.counter import FruitCounter, FrameStats
from src.counting.counting_rules import LineZone
from src.counting.yield_estimator import YieldEstimator


@dataclass
class PipelineConfig:
    """Flat configuration for one pipeline run, populated from settings.yaml."""
    model_path: Path
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: str = ""
    min_track_frames: int = 15
    avg_apple_weight_g: float = 180.0
    line_position: float = 0.5
    line_orientation: str = "horizontal"
    enable_m1: bool = True
    enable_m2: bool = True

    @classmethod
    def from_yaml(cls, config_path: str | Path = "configs/settings.yaml") -> "PipelineConfig":
        cfg = load_config(config_path)
        return cls(
            model_path=Path(cfg["detection"]["model_path"]),
            conf=cfg["detection"]["confidence_threshold"],
            iou=cfg["detection"]["iou_threshold"],
            imgsz=cfg["detection"]["imgsz"],
            device=cfg["detection"]["device"],
            min_track_frames=cfg["counting"]["min_track_frames"],
            avg_apple_weight_g=cfg["yield"]["avg_apple_weight_g"],
            line_position=cfg["counting"]["method1"]["line_position"],
            line_orientation=cfg["counting"]["method1"]["line_orientation"],
            enable_m1=cfg["counting"]["method1"]["enabled"],
            enable_m2=cfg["counting"]["method2"]["enabled"],
        )


@dataclass
class PipelineResult:
    """Final result returned after processing a complete video."""
    total_frames: int
    fps: float
    count_m1: int
    count_m2: int
    estimated_kg_m1: float
    estimated_kg_m2: float
    per_frame_stats: list[FrameStats] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)


class FruitPipeline:
    """
    Orchestrates the full detection → tracking → counting pipeline.

    Usage
    -----
    >>> pipeline = FruitPipeline.from_yaml()
    >>> result = pipeline.run("data/raw/orchard_01.mp4", output_path="data/output/result.mp4")
    >>> print(result.count_m2, result.estimated_kg_m2)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.detector = FruitDetector(
            model_path=config.model_path,
            conf=config.conf,
            iou=config.iou,
            imgsz=config.imgsz,
            device=config.device,
        )
        self.yield_estimator = YieldEstimator(config.avg_apple_weight_g)

    @classmethod
    def from_yaml(cls, config_path: str | Path = "configs/settings.yaml") -> "FruitPipeline":
        return cls(PipelineConfig.from_yaml(config_path))

    def run(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
    ) -> PipelineResult:
        """
        Process a complete video file.

        Parameters
        ----------
        video_path : str | Path
            Input video.
        output_path : str | Path | None
            If provided, save the annotated output video here.

        Returns
        -------
        PipelineResult
            Final counts, yield estimates, and per-frame stats.
        """
        # TODO: Implement during Phase 5
        # 1. Open video with VideoProcessor
        # 2. Get frame dimensions and fps → build LineZone → build FruitCounter
        # 3. Loop over frames:
        #    a. track_frame(detector, frame, frame_idx, history)
        #    b. counter.update(frame_idx, tracked_objects, history)
        #    c. video_processor.annotate_frame(frame, tracked_objects, counter, line_zone)
        #    d. Collect confidence scores for distribution chart
        # 4. Assemble and return PipelineResult
        raise NotImplementedError

    def run_streaming(
        self,
        video_path: str | Path,
    ) -> Generator[tuple[np.ndarray, FrameStats], None, PipelineResult]:
        """
        Process a video frame-by-frame, yielding annotated frames for
        real-time display in the Streamlit app.

        Yields
        ------
        tuple[np.ndarray, FrameStats]
            Annotated BGR frame and the current frame statistics.

        Returns
        -------
        PipelineResult
            Final result after the generator is exhausted.
        """
        # TODO: Implement during Phase 5
        raise NotImplementedError
