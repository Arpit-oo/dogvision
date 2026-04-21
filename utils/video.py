"""Video IO: local files via cv2.VideoCapture, webcam via integer index.

Includes a reconnecting wrapper for webcam/RTSP sources.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class FrameMeta:
    idx: int
    t_ns: int
    stream_id: int


def _open(source: str | int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open source: {source}")
    return cap


def iter_frames(
    source: str | int,
    stream_id: int = 0,
    reconnect: bool = True,
    max_backoff_s: float = 30.0,
) -> Iterator[tuple[np.ndarray, FrameMeta]]:
    """Yield (bgr_frame, meta). Reconnects on read-failure for live sources."""
    is_live = isinstance(source, int) or (
        isinstance(source, str) and source.lower().startswith(("rtsp://", "http://", "https://"))
    )
    cap = _open(source)
    idx = 0
    backoff = 1.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_live and reconnect:
                    cap.release()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff_s)
                    try:
                        cap = _open(source)
                        backoff = 1.0
                        continue
                    except RuntimeError:
                        continue
                else:
                    break
            yield frame, FrameMeta(idx=idx, t_ns=time.time_ns(), stream_id=stream_id)
            idx += 1
    finally:
        cap.release()


class VideoWriter:
    """Lazy VideoWriter — initialized on first frame so we know size/fps."""

    def __init__(self, path: str, fps: float = 30.0):
        self.path = path
        self.fps = fps
        self._w: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> None:
        if self._w is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._w = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
        self._w.write(frame)

    def release(self) -> None:
        if self._w is not None:
            self._w.release()
            self._w = None
