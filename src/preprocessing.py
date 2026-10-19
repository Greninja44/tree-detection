"""Input validation and loading helpers for images, folders and videos.

All functions raise ``InputValidationError`` with a human-readable message when
the input is unusable (wrong extension, missing file, corrupted content, empty
directory).  No function here depends on a model or on GPU availability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: File extensions accepted for still images.
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
)

#: File extensions accepted for video input.
SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
)


class InputValidationError(ValueError):
    """Raised when user supplied input cannot be processed."""


def is_image_path(path: str | Path) -> bool:
    """Return ``True`` when *path* has a supported image extension (case-insensitive)."""
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_video_path(path: str | Path) -> bool:
    """Return ``True`` when *path* has a supported video extension (case-insensitive)."""
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def validate_image_path(path: str | Path) -> Path:
    """Validate that *path* points to a readable image file.

    Checks: existence, extension support, non-empty file size.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise InputValidationError(f"Image file not found: {p}")
    if not p.is_file():
        raise InputValidationError(f"Not a file: {p}")
    if not is_image_path(p):
        raise InputValidationError(
            f"Unsupported image format '{p.suffix or '<none>'}' for {p.name}. "
            f"Supported: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    if p.stat().st_size == 0:
        raise InputValidationError(f"Image file is empty (0 bytes): {p}")
    return p


def validate_video_path(path: str | Path) -> Path:
    """Validate that *path* points to a readable video file."""
    p = Path(path).expanduser()
    if not p.exists():
        raise InputValidationError(f"Video file not found: {p}")
    if not p.is_file():
        raise InputValidationError(f"Not a file: {p}")
    if not is_video_path(p):
        raise InputValidationError(
            f"Unsupported video format '{p.suffix or '<none>'}' for {p.name}. "
            f"Supported: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"
        )
    if p.stat().st_size == 0:
        raise InputValidationError(f"Video file is empty (0 bytes): {p}")
    return p


def load_image(path: str | Path, color: str = "bgr") -> np.ndarray:
    """Load an image from *path*.

    Parameters
    ----------
    path:
        Image file to read.
    color:
        ``"bgr"`` (default, OpenCV convention), ``"rgb"`` or ``"gray"``.

    Raises
    ------
    InputValidationError
        If the file is missing, has an unsupported extension, or is corrupted.
    """
    p = validate_image_path(path)
    flags = {"bgr": cv2.IMREAD_COLOR, "rgb": cv2.IMREAD_COLOR, "gray": cv2.IMREAD_GRAYSCALE}
    if color not in flags:
        raise InputValidationError(f"Unknown color mode '{color}'. Use 'bgr', 'rgb' or 'gray'.")
    image = cv2.imread(str(p), flags[color])
    if image is None:
        raise InputValidationError(
            f"Failed to decode image '{p.name}'. The file is corrupted or not a valid image."
        )
    if image.size == 0 or min(image.shape[:2]) == 0:
        raise InputValidationError(f"Image has invalid dimensions: {p}")
    if color == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.debug("Loaded image %s shape=%s", p, image.shape)
    return image


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Return *image* as a 3-channel BGR ``uint8`` array (grayscale is converted)."""
    if not isinstance(image, np.ndarray):
        raise InputValidationError("Expected a numpy image array.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise InputValidationError(f"Unsupported image array shape: {image.shape}")


def list_images_in_folder(
    folder: str | Path,
    recursive: bool = False,
    extensions: Sequence[str] | None = None,
) -> list[Path]:
    """Return a sorted list of image files inside *folder*.

    Raises ``InputValidationError`` when the folder does not exist, is not a
    directory, or contains no supported images.
    """
    f = Path(folder).expanduser()
    if not f.exists():
        raise InputValidationError(f"Folder not found: {f}")
    if not f.is_dir():
        raise InputValidationError(f"Not a directory: {f}")
    exts = tuple(e.lower() for e in (extensions or SUPPORTED_IMAGE_EXTENSIONS))
    it: Iterable[Path] = f.rglob("*") if recursive else f.glob("*")
    images = sorted(p for p in it if p.is_file() and p.suffix.lower() in exts)
    if not images:
        raise InputValidationError(
            f"No supported images found in '{f}' "
            f"(extensions: {', '.join(exts)}). The directory is empty or only contains other files."
        )
    logger.info("Found %d image(s) in %s", len(images), f)
    return images


def open_video(path: str | Path | int) -> cv2.VideoCapture:
    """Open a video file or webcam index and validate that it can be read.

    Raises ``InputValidationError`` when the capture cannot be opened or reports
    no frames / no resolution (typical for missing codecs or absent cameras).
    """
    source: str | int = path
    label = f"camera index {path}" if isinstance(path, int) else str(path)
    if isinstance(path, str | Path):
        source = str(validate_video_path(path))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise InputValidationError(f"Could not open video source: {label}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise InputValidationError(
            f"Video source '{label}' reported an invalid resolution ({width}x{height}). "
            "The codec may be unsupported or the camera is unavailable."
        )
    return cap


def read_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Read the next frame; return ``None`` at end of stream."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def describe_video(path: str | Path) -> dict[str, float | int | str]:
    """Return basic metadata for a video file (fps, size, frame count)."""
    p = validate_video_path(path)
    cap = open_video(p)
    try:
        return {
            "path": str(p),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
    finally:
        cap.release()


def resize_max(image: np.ndarray, max_side: int | None) -> np.ndarray:
    """Downscale *image* so its longest side is at most *max_side* pixels.

    ``max_side=None`` or values >= the current size leave the image unchanged.
    """
    if max_side is None or max_side <= 0:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)
