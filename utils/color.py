"""Deterministic track-ID to BGR color mapping.

Uses the golden ratio to spread hue values evenly across the color wheel.
Same track ID always produces the same color, across runs and frames.
This lets viewers visually track "which dog is which" by color alone.
"""
from __future__ import annotations  # Allow modern type hints on Python 3.8+

import colorsys  # Python standard library for color space conversion


def id_to_bgr(track_id: int) -> tuple[int, int, int]:
    """Convert a tracker ID to a deterministic BGR color.

    Uses the golden ratio (0.618...) to distribute hues evenly. This ensures
    that consecutive IDs (1, 2, 3...) produce visually distinct colors rather
    than similar shades.

    Returns BGR tuple for OpenCV (not RGB).
    """
    # Multiply by golden ratio and take modulo 1 to get a hue in [0, 1)
    # This spreads IDs across the entire color wheel evenly
    h = (track_id * 0.61803398875) % 1.0
    # Convert HSV to RGB (s=0.75 for saturation, v=0.95 for brightness)
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    # Convert to 0-255 range and swap R↔B for OpenCV's BGR format
    return int(b * 255), int(g * 255), int(r * 255)
