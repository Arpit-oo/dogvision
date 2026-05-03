"""Video I/O: local files via cv2.VideoCapture, webcam via integer index.

Includes a reconnecting wrapper for webcam/RTSP sources that automatically
retries with exponential backoff when the connection drops.
"""
from __future__ import annotations  # Allow modern type hints on Python 3.8+

import time  # For backoff sleep
from dataclasses import dataclass  # Clean data classes
from typing import Iterator  # Generator type hint

import cv2        # OpenCV video capture and writing
import numpy as np  # Frame arrays


@dataclass
class FrameMeta:
    """Metadata attached to each decoded frame."""
    idx: int         # Sequential frame index (0, 1, 2, ...)
    t_ns: int        # Nanosecond timestamp when frame was captured
    stream_id: int   # Camera/stream identifier (for multi-camera setups)


def _open(source: str | int) -> cv2.VideoCapture:
    """Open a video source. Raises RuntimeError if it fails to open."""
    cap = cv2.VideoCapture(source)  # Works with file paths, webcam indices, RTSP URLs
    if not cap.isOpened():
        raise RuntimeError(f"could not open source: {source}")
    return cap


def iter_frames(
    source: str | int,           # File path, webcam index (0), or RTSP URL
    stream_id: int = 0,          # Camera ID for metadata
    reconnect: bool = True,      # Whether to retry on connection loss
    max_backoff_s: float = 30.0, # Maximum retry delay in seconds
) -> Iterator[tuple[np.ndarray, FrameMeta]]:
    """Yield (bgr_frame, meta) pairs from a video source.

    For live sources (webcam, RTSP), automatically reconnects on read failure
    with exponential backoff (1s, 2s, 4s, 8s, ... up to max_backoff_s).
    For files, stops at end of video.
    """
    # Detect whether this is a live source (webcam/RTSP) or a file
    is_live = isinstance(source, int) or (
        isinstance(source, str) and source.lower().startswith(("rtsp://", "http://", "https://"))
    )
    cap = _open(source)       # Open the video source
    idx = 0                    # Frame counter
    backoff = 1.0              # Initial retry delay (seconds)
    try:
        while True:
            ok, frame = cap.read()  # Read next frame from source
            if not ok:
                # Read failed — either end of file or connection drop
                if is_live and reconnect:
                    # Live source: attempt reconnection with exponential backoff
                    cap.release()             # Release the broken connection
                    time.sleep(backoff)       # Wait before retrying
                    backoff = min(backoff * 2, max_backoff_s)  # Double delay, cap at max
                    try:
                        cap = _open(source)   # Try to reopen the source
                        backoff = 1.0         # Reset backoff on success
                        continue
                    except RuntimeError:
                        continue  # Reopen failed — try again after next backoff
                else:
                    break  # File source: end of video — stop iteration
            # Successfully read a frame — yield it with metadata
            yield frame, FrameMeta(idx=idx, t_ns=time.time_ns(), stream_id=stream_id)
            idx += 1
    finally:
        cap.release()  # Always release the video capture handle on exit


class VideoWriter:
    """Lazy VideoWriter — initialized on first frame so we know size/fps.

    Delays cv2.VideoWriter construction until the first frame arrives,
    because we need the frame dimensions to configure the writer.
    """

    def __init__(self, path: str, fps: float = 30.0):
        self.path = path        # Output file path
        self.fps = fps          # Target frames per second
        self._w: cv2.VideoWriter | None = None  # Lazy-initialized writer

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the output video. Initializes writer on first call."""
        if self._w is None:
            h, w = frame.shape[:2]  # Get frame dimensions from first frame
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
            self._w = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
        self._w.write(frame)  # Write frame to disk

    def release(self) -> None:
        """Finalize and close the output video file."""
        if self._w is not None:
            self._w.release()   # Flush buffers and close file
            self._w = None
