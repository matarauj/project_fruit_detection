"""
Extract frames from orchard videos for annotation.

Samples every Nth frame from each video and saves as JPEG images.
Subfolders are created per video to keep sources traceable.

Usage
-----
  # All videos in data/raw/, every 15th frame, stop at 4000 total:
  python scripts/extract_frames.py

  # Custom paths and step:
  python scripts/extract_frames.py --input data/raw --output data/images --step 10

  # Single video file:
  python scripts/extract_frames.py --input data/raw/orchard_01.mp4 --step 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}


def extract_frames(
    video_path: Path,
    output_dir: Path,
    step: int = 15,
    jpeg_quality: int = 95
    ) -> int:
    """
    Extract every `step`-th frame from a single video file.

    Parameters
    ----------
    video_path : Path
        Path to the source video.
    output_dir : Path
        Directory where frames will be saved (created if absent).
    step : int
        Sample one frame every `step` frames.
    jpeg_quality : int
        JPEG compression quality (0-100). 95 gives near-lossless output.

    Returns
    -------
    int
        Number of frames saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}", file=sys.stderr)
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected = total_frames // step

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem

    print(f"\n  Video   : {video_path.name}")
    print(f"  Frames  : {total_frames}  |  FPS: {fps:.1f}  |  Duration: {total_frames / fps:.1f}s")
    print(f"  Sampling: every {step}th frame  (~{expected} frames expected)")
    print(f"  Output  : {output_dir}")

    frame_idx = 0
    saved = 0

    with tqdm(total=total_frames, desc="  Progress", unit="frame", ncols=70) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                filename = output_dir / f"{stem}_f{frame_idx:07d}.jpg"
                cv2.imwrite(
                    str(filename),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                )
                saved += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"  Saved   : {saved} frames")
    return saved


def collect_videos(input_path: Path) -> list[Path]:
    """
    Return a sorted list of video files from a file or directory.
    """
    if input_path.is_file():
        if input_path.suffix in SUPPORTED_EXTENSIONS:
            return [input_path]
        print(f"[ERROR] Not a recognised video format: {input_path}", file=sys.stderr)
        sys.exit(1)

    videos = sorted(
        f for f in input_path.iterdir() if f.suffix in SUPPORTED_EXTENSIONS
    )
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames from orchard videos for annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw"),
        help="Input video file or directory of videos."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/images"),
        help="Root output directory. One sub-folder is created per video."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=15,
        help="Sample one frame every N frames."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=4000,
        help="Stop after extracting this many frames in total."
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG output quality (0-100)."
    )
    args = parser.parse_args()

    videos = collect_videos(args.input)
    if not videos:
        print(f"[ERROR] No video files found in: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(videos)} video(s)")

    total_saved = 0
    for video_path in videos:
        remaining = args.max_frames - total_saved
        if remaining <= 0:
            print(f"\nReached --max-frames limit ({args.max_frames}). Stopping early.")
            break

        per_video_output = args.output / video_path.stem
        saved = extract_frames(
            video_path=video_path,
            output_dir=per_video_output,
            step=args.step,
            jpeg_quality=args.quality
        )
        total_saved += saved

    print(f"\n{'─' * 40}")
    print(f"Total frames extracted : {total_saved}")
    print(f"Output root            : {args.output}")
    print(f"\nNext step: open Label Studio and import frames from {args.output}")


if __name__ == "__main__":
    main()
