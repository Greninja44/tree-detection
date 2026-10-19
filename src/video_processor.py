"""Frame-by-frame video and webcam processing.

The detector is created **once** by the caller and reused for every frame, so
weights are loaded a single time.  All capture/writer resources are released
in ``finally`` blocks, including on error, so a failed run never leaks file
handles.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .detector import DetectionResult, TreeDetector
from .preprocessing import (
    InputValidationError,
    describe_video,
    list_images_in_folder,
    load_image,
    open_video,
)
from .visualization import draw_detections

logger = logging.getLogger(__name__)

#: Codec tried (in order) when writing annotated videos.
OUTPUT_CODECS: tuple[str, ...] = ("mp4v", "avc1", "XVID", "MJPG")


@dataclasses.dataclass
class VideoSummary:
    """Statistics produced by processing one video."""

    source: str
    output_path: str | None
    frames_processed: int
    total_detections: int
    mean_detections_per_frame: float
    mean_inference_ms: float
    processing_fps: float
    wall_time_s: float
    errors: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _open_writer(
    output_path: Path,
    fps: float,
    size: tuple[int, int],
) -> tuple[cv2.VideoWriter | None, str | None]:
    """Try codecs in order; return ``(writer, codec_used)`` or ``(None, None)``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fps <= 0 or not np.isfinite(fps):
        fps = 25.0
    for codec in OUTPUT_CODECS:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
        if writer.isOpened():
            logger.info("Writing video with codec '%s' -> %s", codec, output_path)
            return writer, codec
        writer.release()
    return None, None


def process_video(
    detector: TreeDetector,
    source: str | Path | int,
    output_path: str | Path,
    *,
    max_frames: int | None = None,
    progress_callback: Callable[[int, int, DetectionResult], None] | None = None,
    save_every: int = 1,
) -> VideoSummary:
    """Process a video file or webcam index frame by frame.

    Parameters
    ----------
    detector:
        A ready-to-use :class:`~src.detector.TreeDetector` (not loaded per frame).
    source:
        Video path or webcam index (``int``).
    output_path:
        Destination for the annotated video.
    max_frames:
        Stop after this many frames (useful for webcam smoke tests).
    progress_callback:
        Optional ``(frame_index, total_frames, result)`` hook for UI progress.
    save_every:
        Write every *n*-th annotated frame (detection still runs on each frame).

    Raises
    ------
    InputValidationError
        If the video cannot be opened or produces no decodable frames.
    """
    cap = open_video(source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_label = f"webcam_{source}" if isinstance(source, int) else Path(str(source)).name

    writer: cv2.VideoWriter | None = None
    frames_processed = 0
    total_detections = 0
    inference_times: list[float] = []
    errors: list[str] = []
    start = time.perf_counter()
    frame_index = 0

    try:
        while True:
            if max_frames is not None and frames_processed >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            try:
                result = detector.detect(frame, source_name=source_label, frame_index=frame_index)
            except Exception as exc:
                errors.append(f"frame {frame_index}: {exc}")
                logger.error("Detection failed on frame %d: %s", frame_index, exc)
                frame_index += 1
                continue

            frames_processed += 1
            total_detections += result.count
            if result.inference_time_ms > 0:
                inference_times.append(result.inference_time_ms)

            if writer is None:
                writer, codec = _open_writer(Path(output_path), fps, (width, height))
                if writer is None:
                    raise InputValidationError(
                        f"Could not open a video writer for '{output_path}'. "
                        f"Tried codecs: {', '.join(OUTPUT_CODECS)}. "
                        "The container/codec combination may be unsupported on this system."
                    )
                if codec not in ("mp4v", "avc1"):
                    errors.append(f"writer codec fallback used: {codec}")

            if frames_processed % max(1, save_every) == 0:
                annotated = draw_detections(frame, result)
                if annotated.shape[1] != width or annotated.shape[0] != height:
                    annotated = cv2.resize(annotated, (width, height))
                writer.write(annotated)

            if progress_callback is not None:
                progress_callback(frame_index, total_frames, result)
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    wall = time.perf_counter() - start
    if frames_processed == 0:
        raise InputValidationError(
            f"Video '{source_label}' yielded no decodable frames. "
            "The file may be corrupted or use an unsupported codec."
        )
    mean_ms = float(np.mean(inference_times)) if inference_times else 0.0
    summary = VideoSummary(
        source=source_label,
        output_path=str(output_path),
        frames_processed=frames_processed,
        total_detections=total_detections,
        mean_detections_per_frame=round(total_detections / frames_processed, 3),
        mean_inference_ms=round(mean_ms, 3),
        processing_fps=round(frames_processed / wall, 2) if wall > 0 else 0.0,
        wall_time_s=round(wall, 3),
        errors=errors,
    )
    logger.info(
        "Processed %d frame(s) of %s -> %s (%.1f fps)",
        frames_processed,
        source_label,
        output_path,
        summary.processing_fps,
    )
    return summary


def process_folder(
    detector: TreeDetector,
    folder: str | Path,
    output_dir: str | Path,
    *,
    recursive: bool = False,
    prefix: str = "annotated_",
    image_hook: Callable[[Path, np.ndarray], None] | None = None,
) -> tuple[list[DetectionResult], list[Path]]:
    """Run detection over every image in *folder*.

    Returns ``(results, annotated_paths)``.  Corrupted files are logged and
    skipped rather than aborting the batch.

    ``image_hook``, when given, is called as ``hook(path, image)`` right before
    each detection — useful for computing lighting statistics on real pixels.
    """
    images = list_images_in_folder(folder, recursive=recursive)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[DetectionResult] = []
    annotated_paths: list[Path] = []
    for img_path in images:
        try:
            image = load_image(img_path)
        except InputValidationError as exc:
            logger.warning("Skipping %s: %s", img_path.name, exc)
            continue
        if image_hook is not None:
            try:
                image_hook(img_path, image)
            except Exception as exc:  # hooks must never abort the batch
                logger.warning("image_hook failed for %s: %s", img_path.name, exc)
        result = detector.detect(image, source_name=img_path.name)
        annotated = draw_detections(image, result)
        dest = out / f"{prefix}{img_path.name}"
        if dest.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            dest = dest.with_suffix(".png")
        cv2.imwrite(str(dest), annotated)
        results.append(result)
        annotated_paths.append(dest)
    logger.info("Batch processed %d/%d images into %s", len(results), len(images), out)
    return results, annotated_paths
