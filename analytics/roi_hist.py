"""GPU color-histogram per ROI using CuPy.

Crops each detection from the frame on-GPU and computes an HSV histogram.
Used as a re-ID hint in future versions; stored alongside detections for QA.
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None  # type: ignore[assignment]


def bgr_to_hsv_gpu(frame_bgr_u8):
    """Minimal BGR→HSV on GPU. Matches cv2.COLOR_BGR2HSV semantics approximately."""
    f = frame_bgr_u8.astype(cp.float32) / 255.0
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    cmax = cp.maximum(cp.maximum(r, g), b)
    cmin = cp.minimum(cp.minimum(r, g), b)
    delta = cmax - cmin
    h = cp.zeros_like(cmax)
    mask = delta > 0
    rc = (g - b) / cp.where(delta == 0, 1, delta)
    gc = (b - r) / cp.where(delta == 0, 1, delta) + 2
    bc = (r - g) / cp.where(delta == 0, 1, delta) + 4
    h = cp.where((cmax == r) & mask, rc % 6, h)
    h = cp.where((cmax == g) & mask, gc, h)
    h = cp.where((cmax == b) & mask, bc, h)
    h = (h / 6.0) % 1.0
    s = cp.where(cmax == 0, 0, delta / cp.where(cmax == 0, 1, cmax))
    v = cmax
    return cp.stack([h, s, v], axis=-1)


def roi_histograms(
    frame_bgr_u8_gpu,
    bboxes: list[tuple[float, float, float, float]],
    bins: int = 16,
) -> np.ndarray:
    """Return an (N, bins**3) float32 array of normalized HSV histograms on CPU.

    Kept small because the hist is only for logging / future re-ID.
    """
    if cp is None:
        raise RuntimeError("CuPy not available.")
    if not bboxes:
        return np.zeros((0, bins ** 3), dtype=np.float32)
    H, W = frame_bgr_u8_gpu.shape[:2]
    hsv = bgr_to_hsv_gpu(frame_bgr_u8_gpu)
    out = cp.zeros((len(bboxes), bins ** 3), dtype=cp.float32)
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        x1i = max(0, int(x1)); y1i = max(0, int(y1))
        x2i = min(W, int(x2)); y2i = min(H, int(y2))
        if x2i <= x1i or y2i <= y1i:
            continue
        crop = hsv[y1i:y2i, x1i:x2i].reshape(-1, 3)
        h_idx = cp.minimum((crop[:, 0] * bins).astype(cp.int32), bins - 1)
        s_idx = cp.minimum((crop[:, 1] * bins).astype(cp.int32), bins - 1)
        v_idx = cp.minimum((crop[:, 2] * bins).astype(cp.int32), bins - 1)
        flat = h_idx * bins * bins + s_idx * bins + v_idx
        hist = cp.bincount(flat, minlength=bins ** 3).astype(cp.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        out[i] = hist
    return cp.asnumpy(out)
