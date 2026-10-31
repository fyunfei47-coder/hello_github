#!/usr/bin/env python3
"""Train a nano YOLO model on NEU-DET, with CPU-friendly defaults.

Example (CPU, first working run):
    python scripts/train.py --model yolov8n.pt --epochs 20 --imgsz 320 --batch 16

The pretrained weight (yolov8n.pt / yolo11n.pt) is downloaded automatically
by ultralytics on first use. Best weights end up in
runs/detect/neu-det-n/weights/best.pt
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Pretrained model, e.g. yolov8n.pt or yolo11n.pt")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=320,
                        help="NEU-DET images are 200x200, so 320 is enough for a first run")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu",
                        help="Use 'cpu' on an ordinary computer, '0' for a GPU")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="neu-det-n")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=10,
        save_period=-1,
    )
    print(f"Training done. Best weights: {results.save_dir}/weights/best.pt")
    print(f"Metrics: mAP50={results.results_dict.get('metrics/mAP50(B)')}, "
          f"mAP50-95={results.results_dict.get('metrics/mAP50-95(B)')}")


if __name__ == "__main__":
    main()
