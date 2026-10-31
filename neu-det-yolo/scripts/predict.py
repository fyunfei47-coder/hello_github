#!/usr/bin/env python3
"""Run a trained YOLO model on an image or a folder of images.

Example:
    python scripts/predict.py --model runs/detect/neu-det-n/weights/best.pt \
        --source dataset/images/val --save-csv runs/predictions.csv

Annotated images go to runs/predict/ by default (ultralytics behaviour).
With --save-csv, one row is written per detected box:
image, class, confidence, x_min, y_min, x_max, y_max (pixel coordinates).
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Image file or folder")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--save-csv", default=None, help="Optional CSV output path")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save annotated images (default: on)")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.predict(source=args.source, conf=args.conf,
                            imgsz=args.imgsz, save=args.save, verbose=False)

    rows = []
    for result in results:
        image_name = Path(result.path).name
        boxes = result.boxes
        if boxes is None:
            continue
        names = result.names
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            rows.append({
                "image": image_name,
                "class": names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "x_min": round(xyxy[0], 1),
                "y_min": round(xyxy[1], 1),
                "x_max": round(xyxy[2], 1),
                "y_max": round(xyxy[3], 1),
            })
            print(f"{image_name}: {rows[-1]['class']} "
                  f"conf={rows[-1]['confidence']} box={xyxy}")

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "image", "class", "confidence",
                "x_min", "y_min", "x_max", "y_max"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved: {csv_path} ({len(rows)} boxes)")
    else:
        print(f"Total boxes: {len(rows)}")


if __name__ == "__main__":
    main()
