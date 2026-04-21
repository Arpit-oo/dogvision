import colorsys


def id_to_bgr(track_id: int) -> tuple[int, int, int]:
    """Deterministic color for a tracker ID. Same ID → same color every run."""
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)
