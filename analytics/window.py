"""cuDF rolling-window aggregates over the DetectionRing snapshot.

Computes real-time analytics entirely on GPU using cuDF (RAPIDS):
  - Dogs per frame, per stream (groupby + nunique)
  - Unique dog count (nunique)
  - Dogs per minute (time-based aggregation)
  - Normalized movement speed per track (trajectory distance / time / bbox size)
  - Peak activity periods

GPU ACCELERATION USED HERE:
  - cudf.DataFrame: GPU-resident DataFrame (pandas replacement)
  - groupby().nunique(): parallel group-by counting on GPU
  - groupby().agg(): multi-column aggregation on GPU
  - Vectorized arithmetic: (dx² + dy²)^0.5 computed on GPU
  - .where(), .fillna(): conditional operations on GPU
  - ALL operations stay on GPU — zero CPU transfer for analytics
"""
from __future__ import annotations  # Allow modern type hints on Python 3.8+

from dataclasses import asdict, dataclass  # Clean data classes

# cuDF import — RAPIDS GPU DataFrame library
try:
    import cudf                    # GPU DataFrame (pandas replacement for CUDA)
except ImportError:                # cuDF not installed (CPU-only system)
    cudf = None                    # type: ignore[assignment]


@dataclass
class WindowStats:
    """Aggregated statistics from one analytics window (e.g., last 30 frames)."""
    n_rows: int                    # Total detection rows in the window
    unique_dogs: int               # Distinct track_ids in the window
    dogs_now: int                  # Unique track_ids in the LATEST frame only
    avg_dogs_per_frame: float      # Mean dogs per frame across the window
    peak_dogs_per_frame: int       # Maximum dogs seen in a single frame
    dogs_per_minute: float         # Unique dogs / window duration in minutes
    avg_norm_speed: float          # Mean normalized speed (pixels/sec / bbox diagonal)

    def to_dict(self) -> dict:
        """Convert to plain dict for JSON serialization."""
        return asdict(self)


class AnalyticsWindow:
    """Computes rolling-window statistics from a DetectionRing snapshot.

    Called every 30 frames (~1 second at 30 FPS). Takes a cuDF DataFrame
    snapshot from the ring buffer and computes aggregates entirely on GPU.
    """

    def compute(self, snapshot) -> WindowStats:
        """Run all analytics on the GPU DataFrame snapshot.

        Returns a WindowStats dataclass with the computed metrics.
        """
        if cudf is None:
            raise RuntimeError("cuDF not available — install RAPIDS.")
        df = snapshot.df               # cuDF GPU DataFrame from ring buffer
        if len(df) == 0:               # Empty snapshot — return zeros
            return WindowStats(0, 0, 0, 0.0, 0, 0.0, 0.0)

        # ---- Dogs per frame per stream (GPU groupby + nunique) ----
        # Groups detections by (stream, frame), counts unique track_ids per group
        per_frame = (
            df.groupby(["stream", "frame"])["track_id"].nunique().reset_index()
        )
        avg_per_frame = float(per_frame["track_id"].mean())  # Mean across all frames
        peak = int(per_frame["track_id"].max())               # Max in any single frame

        # ---- Dogs in latest frame (GPU filter + nunique) ----
        # How many distinct dogs are visible RIGHT NOW
        latest_frame = df["frame"].max()                       # Most recent frame number
        dogs_now = int(
            df[df["frame"] == latest_frame]["track_id"].nunique()  # Filter + count
        )

        # ---- Total unique dogs in window (GPU nunique) ----
        unique = int(df["track_id"].nunique())

        # ---- Dogs per minute (GPU min/max + arithmetic) ----
        t_min = float(df["t_ns"].min())                        # Earliest timestamp
        t_max = float(df["t_ns"].max())                        # Latest timestamp
        dt_min = max((t_max - t_min) / 1e9 / 60.0, 1e-9)     # Duration in minutes
        dogs_per_minute = unique / dt_min                      # Rate

        # ---- Normalized speed per track (GPU vectorized math) ----
        # Add computed columns: center (cx, cy) and bbox diagonal
        df = df.assign(
            cx=(df["x1"] + df["x2"]) / 2.0,                   # Center X (GPU arithmetic)
            cy=(df["y1"] + df["y2"]) / 2.0,                   # Center Y (GPU arithmetic)
            diag=((df["x2"] - df["x1"]) ** 2 + (df["y2"] - df["y1"]) ** 2) ** 0.5,  # Diagonal
        )
        # Aggregate per track_id: spatial extent + time extent + avg bbox size
        agg = (
            df.groupby("track_id")                             # Group by dog/person ID
            .agg(
                cx_first=("cx", "min"),                        # Leftmost center X (proxy for start)
                cx_last=("cx", "max"),                         # Rightmost center X (proxy for end)
                cy_first=("cy", "min"),                        # Topmost center Y
                cy_last=("cy", "max"),                         # Bottommost center Y
                t_first=("t_ns", "min"),                       # First seen timestamp
                t_last=("t_ns", "max"),                        # Last seen timestamp
                diag_mean=("diag", "mean"),                    # Average bbox diagonal (for normalization)
            )
            .reset_index()                                     # Flatten back to columns
        )
        # Compute displacement distance on GPU (vectorized)
        dx = agg["cx_last"] - agg["cx_first"]                 # X displacement
        dy = agg["cy_last"] - agg["cy_first"]                 # Y displacement
        dt_s = (agg["t_last"] - agg["t_first"]).astype("float64") / 1e9  # Duration in seconds
        dist = (dx * dx + dy * dy) ** 0.5                      # Euclidean distance (GPU)
        # Speed = distance / time, normalized by bbox diagonal for scale-invariance
        speed = dist / dt_s.where(dt_s > 0, 1e9)              # Avoid div-by-zero
        norm_speed = speed / agg["diag_mean"].where(agg["diag_mean"] > 0, 1.0)  # Normalize
        avg_norm_speed = float(norm_speed.fillna(0).mean())    # Transfer single float to CPU

        return WindowStats(
            n_rows=int(len(df)),                               # Total rows processed
            unique_dogs=unique,                                # Distinct dogs
            dogs_now=dogs_now,                                 # Dogs in latest frame
            avg_dogs_per_frame=avg_per_frame,                  # Average per frame
            peak_dogs_per_frame=peak,                          # Peak per frame
            dogs_per_minute=dogs_per_minute,                   # Rate
            avg_norm_speed=avg_norm_speed,                     # Movement metric
        )
