"""Preallocated CuPy ring buffer for per-detection rows.

Avoids per-frame cuDF concat overhead by storing detection data in fixed-size
GPU arrays. Snapshot converts the active slice to a cuDF DataFrame for analytics.

GPU ACCELERATION USED HERE:
  - CuPy: all arrays preallocated on GPU memory (cp.zeros, cp.full)
  - O(1) append: writes directly to GPU array at head pointer, no allocation
  - cp.concatenate: GPU-side array unrolling when ring wraps around
  - cuDF: snapshot materializes GPU DataFrame for groupby/agg analytics
  - Zero CPU transfer: data enters on GPU (from detection) and stays on GPU
"""
from __future__ import annotations  # Allow modern type hints on Python 3.8+

from dataclasses import dataclass  # Clean data classes

# CuPy import — gracefully handles missing GPU library
try:
    import cupy as cp              # GPU array library (NumPy replacement for CUDA)
except ImportError as _e:          # CuPy not installed (CPU-only system)
    cp = None                      # type: ignore[assignment]
    _cp_err = _e                   # Save error message for later
else:
    _cp_err = None                 # CuPy loaded successfully

# cuDF import — RAPIDS GPU DataFrame library
try:
    import cudf                    # GPU DataFrame (pandas replacement for CUDA)
except ImportError:                # cuDF not installed
    cudf = None                    # type: ignore[assignment]


# Column names for the ring buffer (matches cuDF DataFrame schema)
COLS = ("frame", "stream", "track_id", "x1", "y1", "x2", "y2", "conf", "t_ns")


@dataclass
class RingSnapshot:
    """A frozen snapshot of the ring buffer as a cuDF DataFrame."""
    df: "cudf.DataFrame"           # GPU DataFrame containing detection rows
    n_rows: int                    # Number of rows in this snapshot


class DetectionRing:
    """Column-oriented ring buffer on GPU. O(1) append per detection row.

    Instead of appending to a cuDF DataFrame every frame (expensive concat),
    we preallocate fixed-size CuPy arrays on GPU and write directly to them.
    When analytics need data, snapshot() materializes a cuDF DataFrame from
    the active portion of the ring.

    Capacity default: 54,000 rows = ~30 minutes at 30 FPS with 1 detection/frame.
    """

    def __init__(self, capacity: int = 54_000):
        if cp is None:
            raise RuntimeError(f"CuPy required for DetectionRing: {_cp_err}")
        self.capacity = int(capacity)  # Maximum rows before ring wraps
        self._head = 0                 # Next write position (circular)
        self._size = 0                 # Total rows stored (capped at capacity)

        # Preallocate GPU arrays — one per column, all same length
        # These stay on GPU memory for the lifetime of the pipeline
        self.frame = cp.zeros(capacity, dtype=cp.int64)       # Frame index
        self.stream = cp.zeros(capacity, dtype=cp.int32)      # Stream/camera ID
        self.track_id = cp.full(capacity, -1, dtype=cp.int32) # ByteTrack ID (-1 = unset)
        self.x1 = cp.zeros(capacity, dtype=cp.float32)        # Bbox left
        self.y1 = cp.zeros(capacity, dtype=cp.float32)        # Bbox top
        self.x2 = cp.zeros(capacity, dtype=cp.float32)        # Bbox right
        self.y2 = cp.zeros(capacity, dtype=cp.float32)        # Bbox bottom
        self.conf = cp.zeros(capacity, dtype=cp.float32)      # Detection confidence
        self.t_ns = cp.zeros(capacity, dtype=cp.int64)        # Nanosecond timestamp

    def append_batch(
        self,
        frame_idx: int,        # Current frame number
        stream_id: int,        # Camera/stream ID
        detections,            # List of Detection objects
        t_ns: int,             # Frame timestamp in nanoseconds
    ) -> None:
        """Append detections to the ring buffer. O(1) per detection.

        Writes directly to preallocated GPU arrays at the head pointer.
        When head reaches capacity, it wraps to 0 (circular buffer).
        """
        for d in detections:
            if d.track_id is None:
                continue           # Skip untracked detections
            i = self._head         # Current write position in GPU arrays
            # Write each field directly to GPU memory (no allocation)
            self.frame[i] = frame_idx
            self.stream[i] = stream_id
            self.track_id[i] = d.track_id
            self.x1[i] = d.bbox[0]
            self.y1[i] = d.bbox[1]
            self.x2[i] = d.bbox[2]
            self.y2[i] = d.bbox[3]
            self.conf[i] = d.conf
            self.t_ns[i] = t_ns
            # Advance head pointer with wraparound
            self._head = (self._head + 1) % self.capacity
            # Track total rows (capped at capacity)
            if self._size < self.capacity:
                self._size += 1

    def _ordered_slice(self):
        """Return the ring unrolled into chronological order as GPU arrays.

        If the ring hasn't wrapped, just slice [0:size].
        If wrapped, concatenate [head:end] + [0:head] to restore time order.
        """
        if self._size < self.capacity:
            # Ring hasn't wrapped — simple slice
            sl = slice(0, self._size)
            return {
                "frame": self.frame[sl], "stream": self.stream[sl],
                "track_id": self.track_id[sl],
                "x1": self.x1[sl], "y1": self.y1[sl],
                "x2": self.x2[sl], "y2": self.y2[sl],
                "conf": self.conf[sl], "t_ns": self.t_ns[sl],
            }
        # Ring has wrapped — unroll via GPU-side concatenate
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
        """Materialize the active ring data as a cuDF GPU DataFrame.

        This is where CuPy arrays become cuDF columns — zero-copy when possible.
        Called every 30 frames (not every frame) to amortize cuDF overhead.
        """
        if cudf is None:
            raise RuntimeError("cuDF not available — install RAPIDS.")
        data = self._ordered_slice()  # Get chronologically ordered GPU arrays
        # Optionally trim to last N rows (for windowed analytics)
        if last_n is not None and self._size > last_n:
            for k, v in data.items():
                data[k] = v[-last_n:]
        # Construct cuDF DataFrame from GPU arrays (stays on GPU)
        df = cudf.DataFrame(data)
        return RingSnapshot(df=df, n_rows=len(df))

    @property
    def size(self) -> int:
        """Number of rows currently stored in the ring buffer."""
        return self._size
