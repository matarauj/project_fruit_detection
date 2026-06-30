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
from src.tracking.track_objects import track_frame, TrackHistory
from src.counting.counter import FruitCounter, FrameStats
from src.counting.counting_rules import LineZone, LineOrientation
from src.counting.yield_estimator import YieldEstimator
from src.pipeline.video_processor import VideoProcessor


@dataclass
class PipelineConfig:
    """
    Flat configuration for one pipeline run, populated from settings.yaml.
    """
    model_path: Path
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: str = ""
    min_track_frames: int = 15
    avg_apple_weight_g: float = 180.0
    line_position: float = 0.5
    line_orientation: str = "vertical"
    enable_m1: bool = True
    enable_m2: bool = True
    trail_length: int = 30
    output_codec: str = "mp4v"
    track_high_thresh: float = 0.25
    tracker_config_path: Path = Path("configs/bytetrack.yaml")

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
            trail_length=cfg["video"].get("trail_length", 30),
            output_codec=cfg["video"].get("output_codec", "mp4v"),
            track_high_thresh=cfg["tracking"].get("track_high_thresh", 0.25),
            tracker_config_path=Path(cfg["tracking"]["tracker_config"])
        )


@dataclass
class PipelineResult:
    """
    Final result returned after processing a complete video.
    """
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
            track_high_thresh=config.track_high_thresh,
            tracker_config_path=config.tracker_config_path
        )
        self.yield_estimator = YieldEstimator(config.avg_apple_weight_g)

    @classmethod
    def from_yaml(cls, config_path: str | Path = "configs/settings.yaml") -> "FruitPipeline":
        return cls(PipelineConfig.from_yaml(config_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None
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
        result = None
        # Exhaust the shared generator; discard intermediate frames.
        for _annotated_frame, _frame_stats, _partial_result in self._process_video(
            video_path, output_path
        ):
            result = _partial_result  # updated each frame; last value is final

        # _process_video always yields at least a sentinel — result is never None
        # for a non-empty video, but guard defensively.
        if result is None:
            raise RuntimeError(f"No frames could be read from {video_path}")

        return result


    def run_streaming(
        self,
        video_path: str | Path
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
        result = None
        for annotated_frame, frame_stats, partial_result in self._process_video(
            video_path, output_path=None
        ):
            result = partial_result
            yield annotated_frame, frame_stats

        if result is None:
            raise RuntimeError(f"No frames could be read from {video_path}")

        return result

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _build_line_zone(self, width: int, height: int) -> LineZone | None:
        """
        Construct the LineZone from config, or return None if M1 is disabled.
        """
        if not self.config.enable_m1:
            return None
        return LineZone(
            position=self.config.line_position,
            orientation=LineOrientation(self.config.line_orientation),
            frame_width=width,
            frame_height=height
        )

    def _process_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None
        ) -> Generator[tuple[np.ndarray, FrameStats, PipelineResult], None, None]:
        """
        Core frame-by-frame generator shared by run() and run_streaming().

        Yields (annotated_frame, frame_stats, partial_result) on every frame.
        The caller decides whether to collect frames (streaming) or discard
        them (batch).
        """
        cfg = self.config

        # Reset tracker state so this video starts fresh.
        self.detector.reset_tracker()

        with VideoProcessor(
            input_path=Path(video_path),
            output_path=Path(output_path) if output_path else None,
            output_codec=cfg.output_codec
        ) as vp:
            line_zone = self._build_line_zone(vp.width, vp.height)
            counter = FruitCounter(
                fps=vp.fps,
                min_track_frames=cfg.min_track_frames,
                line_zone=line_zone
            )
            history: TrackHistory = {}
            per_frame_stats: list[FrameStats] = []
            confidence_scores: list[float] = []

            for frame_idx, frame in vp.read_frames():
                # 1. Detect + track
                tracked_objects, history = track_frame(
                    self.detector, frame, frame_idx, history
                )

                # 2. Collect confidence scores for the distribution chart
                for obj in tracked_objects:
                    confidence_scores.append(obj.confidence)

                # 3. Update counters
                frame_stats = counter.update(frame_idx, tracked_objects, history)
                per_frame_stats.append(frame_stats)

                # 4. Annotate frame
                annotated = vp.annotate_frame(
                    frame=frame,
                    tracked_objects=tracked_objects,
                    track_history=history,
                    line_zone=line_zone,
                    count_m1=counter.count_m1,
                    count_m2=counter.count_m2,
                    trail_length=cfg.trail_length
                )

                # 5. Write to output video (no-op if output_path is None)
                vp.write_frame(annotated)

                # 6. Build partial result and yield
                partial_result = PipelineResult(
                    total_frames=frame_idx + 1,
                    fps=vp.fps,
                    count_m1=counter.count_m1,
                    count_m2=counter.count_m2,
                    estimated_kg_m1=self.yield_estimator.estimate_kg(counter.count_m1),
                    estimated_kg_m2=self.yield_estimator.estimate_kg(counter.count_m2),
                    per_frame_stats=per_frame_stats,
                    confidence_scores=confidence_scores
                )

                yield annotated, frame_stats, partial_result