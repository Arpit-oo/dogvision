"""Optional: fine-tune YOLOv8 on dog-only datasets.

Not required for the default pipeline (COCO-pretrained weights already detect
dogs well). Use this when you want improved recall on your specific scenes.

Expected dataset layout (Ultralytics YOLO):
    datasets/dogs/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt    # class 0 = dog
        dogs.yaml                   # train/val paths + names: ['dog']

Run:
    python train.py --data datasets/dogs/dogs.yaml --epochs 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to dataset yaml")
    p.add_argument("--weights", default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="dogvision")
    args = p.parse_args()

    Path(args.project).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        amp=True,
        pretrained=True,
        exist_ok=True,
    )
    # After training, the best.pt will live at runs/train/dogvision/weights/best.pt
    print(f"\nTraining done. Use the new weights:\n"
          f"  python demo.py --source video.mp4 --weights "
          f"{args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
