"""Preallocated CuPy ring buffer for per-detection rows.

Avoids per-frame cuDF concat overhead. Snapshot converts the active slice to
a cuDF DataFrame for rolling-window aggregates.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import cupy as cp
except ImportError as _e:  # pragma: no cover
    cp = None  # type: ignore[assignment]
    _cp_err = _e
else:
    _cp_err = None

try:
    import cudf
except ImportError:  # pragma: no cover
    cudf = None  # type: ignore[assignment]


COLS = ("frame", "stream", "track_id", "x1", "y1", "x2", "y2", "conf", "t_ns")


@dataclass
class RingSnapshot:
    df: "cudf.DataFrame"
    n_rows: int


class DetectionRing:
    """Column-oriented ring on GPU. O(1) append per row."""

    def __init__(self, capacity: int = 54_000):
        if cp is None:
            raise RuntimeError(f"CuPy required for DetectionRing: {_cp_err}")
        self.capacity = int(capacity)
        self._head = 0   # next write index
        self._size = 0   # total rows stored (capped at capacity)

        self.frame = cp.zeros(capacity, dtype=cp.int64)
        self.stream = cp.zeros(capacity, dtype=cp.int32)
        self.track_id = cp.full(capacity, -1, dtype=cp.int32)
        self.x1 = cp.zeros(capacity, dtype=cp.float32)
        self.y1 = cp.zeros(capacity, dtype=cp.float32)
        self.x2 = cp.zeros(capacity, dtype=cp.float32)
        self.y2 = cp.zeros(capacity, dtype=cp.float32)
        self.conf = cp.zeros(capacity, dtype=cp.float32)
        self.t_ns = cp.zeros(capacity, dtype=cp.int64)

    def append_batch(
        self,
        frame_idx: int,
        stream_id: int,
        detections,
        t_ns: int,
    ) -> None:
        for d in detections:
            if d.track_id is None:
                continue
            i = self._head
            self.frame[i] = frame_idx
            self.stream[i] = stream_id
            self.track_id[i] = d.track_id
            self.x1[i] = d.bbox[0]
            self.y1[i] = d.bbox[1]
            self.x2[i] = d.bbox[2]
            self.y2[i] = d.bbox[3]
            self.conf[i] = d.conf
            self.t_ns[i] = t_ns
            self._head = (self._head + 1) % self.capacity
            if self._size < self.capacity:
                self._size += 1

    def _ordered_slice(self):
        """Return the ring unrolled into chronological order."""
        if self._size < self.capacity:
            sl = slice(0, self._size)
            return {
                "frame": self.frame[sl], "stream": self.stream[sl],
                "track_id": self.track_id[sl],
                "x1": self.x1[sl], "y1": self.y1[sl],
                "x2": self.x2[sl], "y2": self.y2[sl],
                "conf": self.conf[sl], "t_ns": self.t_ns[sl],
            }
        # full ring — roll so head is at 0
        def roll(a):
            return cp.concatenate([a[self._head:], a[: self._head]])
        return {
            "frame": roll(self.frame), "stream": roll(self.stream),
            "track_id": roll(self.track_id),
            "x1": roll(self.x1), "y1": roll(self.y1),
            "x2": roll(self.x2), "y2": roll(self.y2),
            "conf": roll(self.conf), "t_ns": roll(self.t_ns),
        }

    def snapshot(self, last_n: int | None = None) -> RingSnapshot:
        if cudf is None:
            raise RuntimeError("cuDF not available — install RAPIDS.")
        data = self._ordered_slice()
        if last_n is not None and self._size > last_n:
            for k, v in data.items():
                data[k] = v[-last_n:]
        df = cudf.DataFrame(data)
        return RingSnapshot(df=df, n_rows=len(df))

    @property
    def size(self) -> int:
        return self._size
