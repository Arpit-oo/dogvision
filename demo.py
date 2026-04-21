"""Entry point.

Examples:
    python demo.py --source video.mp4
    python demo.py --source 0                 # webcam
    python demo.py --source rtsp://cam/stream
    python demo.py --source video.mp4 --no-display --out out/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pipeline.orchestrator import Pipeline


def _parse_source(s: str):
    return int(s) if s.isdigit() else s


def main() -> None:
    p = argparse.ArgumentParser("dogvision")
    p.add_argument("--source", required=True, help="path | webcam index | rtsp url")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--weights", default=None, help="override config weights")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--no-trt", action="store_true")
    p.add_argument("--out", default="out")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.weights:
        cfg["model"]["weights"] = args.weights
    if args.no_trt:
        cfg["model"]["trt"] = False
    if args.no_display:
        cfg["output"]["display"] = False

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg["output"]["video_path"] = str(out / "annotated.mp4")
    cfg["output"]["detections_path"] = str(out / "detections.parquet")
    cfg["output"]["analytics_path"] = str(out / "analytics_window.json")

    pipe = Pipeline(cfg, source=_parse_source(args.source))
    pipe.run()


if __name__ == "__main__":
    main()
