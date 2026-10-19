"""Video navigation: run the pipeline over a video with temporal smoothing and write a browser-playable clip."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.navigation import DirectionSmoother, resample_path
from src.pipeline import Models, NavOutput, NavParams, run_navigation

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass
class VideoInfo:
    fps: float
    frames: int
    width: int
    height: int


def video_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open the video (corrupt or unsupported codec).")
    info = VideoInfo(cap.get(cv2.CAP_PROP_FPS) or 10.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def iter_navigation(path: str | Path, models: Models, params: NavParams, stride: int = 1,
                    max_frames: int = 0) -> Iterator[tuple[int, np.ndarray, NavOutput]]:
    """Yield ``(source_frame_index, frame, output)`` for every ``stride``-th frame.

    The heading is smoothed with hysteresis (a missing route stops immediately and needs two valid frames to
    resume) and the route points are averaged with the previous frame's route to avoid flicker.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open the video (corrupt or unsupported codec).")
    smoother, prev, i, n = DirectionSmoother(config=params.nav, start_stopped=False), None, 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % max(1, stride) == 0:
                out = run_navigation(frame, models, params, smoother)
                if out.plan is not None and out.plan.success:
                    cur = resample_path(out.plan.points)
                    prev = cur if prev is None else 0.5 * cur + 0.5 * prev
                    out.plan.points = prev
                else:
                    prev = None
                yield i, frame, out
                n += 1
                if max_frames and n >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


def write_webm(frames: list[np.ndarray], path: str | Path, fps: float) -> bool:
    """Write VP8 WebM (plays in browsers).  Returns False if the codec is unavailable."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), max(1.0, fps), (w, h))
    if not vw.isOpened():
        return False
    for f in frames:
        vw.write(f)
    vw.release()
    return Path(path).exists() and Path(path).stat().st_size > 0
