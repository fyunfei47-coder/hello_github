#!/usr/bin/env python3
"""Convert NEU-DET (VOC XML) annotations to YOLO txt format.

Supported source layouts (under --src):
  1. Kaggle-style split:
       train/images/*.jpg, train/annotations/*.xml
       validation/images/*.jpg, validation/annotations/*.xml
  2. Flat original layout:
       IMAGES/*.jpg, ANNOTATIONS/*.xml
     In this case a random split is made with --val-ratio.

Output layout (under --out):
  dataset/images/train, dataset/images/val
  dataset/labels/train, dataset/labels/val

Class order (must match data.yaml):
  crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches

NEU-DET class names in XML may use slightly different spellings, so common
variants are normalised here. Unknown classes are reported and skipped,
never silently renamed.
"""

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

# NEU-DET XML class names seen in the wild -> canonical name.
ALIASES = {
    "crazing": "crazing",
    "cr": "crazing",
    "inclusion": "inclusion",
    "in": "inclusion",
    "patches": "patches",
    "pa": "patches",
    "pitted_surface": "pitted_surface",
    "ps": "pitted_surface",
    "pitted surface": "pitted_surface",
    "rolled-in_scale": "rolled-in_scale",
    "rs": "rolled-in_scale",
    "rolled-in scale": "rolled-in_scale",
    "scratches": "scratches",
    "sc": "scratches",
}


def parse_xml(xml_path):
    """Return a list of (class_name, xmin, ymin, xmax, ymax)."""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.iter("object"):
        name_el = obj.find("name")
        box_el = obj.find("bndbox")
        if name_el is None or box_el is None:
            continue
        raw_name = (name_el.text or "").strip()
        name = ALIASES.get(raw_name, ALIASES.get(raw_name.lower()))
        if name is None:
            print(f"WARNING: unknown class {raw_name!r} in {xml_path}, skipped")
            continue
        try:
            xmin = float(box_el.findtext("xmin"))
            ymin = float(box_el.findtext("ymin"))
            xmax = float(box_el.findtext("xmax"))
            ymax = float(box_el.findtext("ymax"))
        except (TypeError, ValueError):
            print(f"WARNING: bad box in {xml_path}, skipped")
            continue
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes


def convert_one(image_path, xml_path, out_image_path, out_label_path):
    with Image.open(image_path) as im:
        width, height = im.size
    boxes = parse_xml(xml_path)
    lines = []
    for name, xmin, ymin, xmax, ymax in boxes:
        # Clip to the image, then drop degenerate boxes.
        xmin, ymin = max(0.0, xmin), max(0.0, ymin)
        xmax, ymax = min(float(width), xmax), min(float(height), ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"{CLASS_INDEX[name]} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_path, out_image_path)
    out_label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def find_pairs(src: Path):
    """Return (split_pairs, flat_pairs) as lists of (image, xml)."""
    split_pairs = {}
    for split, img_dir_name, ann_dir_name in [
        ("train", "train/images", "train/annotations"),
        ("val", "validation/images", "validation/annotations"),
        ("val", "valid/images", "valid/annotations"),
        ("train", "IMAGES", "ANNOTATIONS"),
        ("val", "Validation_Images", "Validation_Annotations"),
    ]:
        img_dir = src / img_dir_name
        ann_dir = src / ann_dir_name
        if not img_dir.is_dir() or not ann_dir.is_dir():
            continue
        pairs = []
        for image_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            xml_path = ann_dir / f"{image_path.stem}.xml"
            if xml_path.is_file():
                pairs.append((image_path, xml_path))
        if pairs:
            split_pairs.setdefault(split, []).extend(pairs)

    flat_pairs = []
    img_dir = src / "IMAGES"
    ann_dir = src / "ANNOTATIONS"
    if img_dir.is_dir() and ann_dir.is_dir():
        for image_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            xml_path = ann_dir / f"{image_path.stem}.xml"
            if xml_path.is_file():
                flat_pairs.append((image_path, xml_path))
    return split_pairs, flat_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Downloaded NEU-DET folder")
    parser.add_argument("--out", required=True, help="Output YOLO dataset folder")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation share when source has no split (default 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        print(f"ERROR: source folder not found: {src}", file=sys.stderr)
        return 1

    split_pairs, flat_pairs = find_pairs(src)
    jobs = []  # (split, image, xml)
    if split_pairs:
        for split, pairs in split_pairs.items():
            jobs.extend((split, img, xml) for img, xml in pairs)
    elif flat_pairs:
        rng = random.Random(args.seed)
        shuffled = flat_pairs[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_ratio)) if len(shuffled) > 1 else 0
        for img, xml in shuffled[n_val:]:
            jobs.append(("train", img, xml))
        for img, xml in shuffled[:n_val]:
            jobs.append(("val", img, xml))
    else:
        print(f"ERROR: no image/annotation pairs found under {src}", file=sys.stderr)
        print("Expected train/images + train/annotations, or IMAGES + ANNOTATIONS.", file=sys.stderr)
        return 1

    total_boxes = 0
    counts = {"train": 0, "val": 0}
    for split, image_path, xml_path in jobs:
        out_image = out / "images" / split / image_path.name
        out_label = out / "labels" / split / f"{image_path.stem}.txt"
        total_boxes += convert_one(image_path, xml_path, out_image, out_label)
        counts[split] += 1

    print(f"Done. train={counts['train']} images, val={counts['val']} images, "
          f"total boxes={total_boxes}")
    print(f"Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
