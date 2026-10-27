"""Live webcam capture that always returns the newest frame (a background thread drains the driver buffer)."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class CameraError(RuntimeError):
    """The camera could not be opened or stopped delivering frames."""


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self.index = index
        self._cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraError(f"Cannot open camera {index}. It may be unplugged or in use by another process.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        deadline = time.time() + 5
        while self._frame is None and time.time() < deadline:
            time.sleep(0.05)
        if self._frame is None:
            self.release()
            raise CameraError(f"Camera {index} opened but delivered no frames.")

    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame, self._stamp = frame, time.time()
            else:
                time.sleep(0.02)

    def read(self) -> np.ndarray:
        """Newest BGR frame; raises if the camera has gone silent for >2 s."""
        with self._lock:
            frame, stamp = self._frame, self._stamp
        if frame is None or time.time() - stamp > 2.0:
            raise CameraError("The camera stopped delivering frames.")
        return frame.copy()

    def release(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
        self._cap.release()


def list_cameras(max_index: int = 4) -> list[int]:
    """Indices that open and deliver a frame (video1 is usually a metadata node and is skipped)."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened() and cap.read()[0]:
            found.append(i)
        cap.release()
    return found
