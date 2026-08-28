#!/usr/bin/env python3
"""Prepare DIOR-R-RSVG as LocateAnything rotated visual-grounding data.

The train and val ID lists are merged into a training JSONL.  The test list is
written separately.  Each valid XML object becomes one single-instance sample
using its natural-language ``description`` and an ``le90`` rotated box.
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


VERTEX_FIELDS = (
    ("x_left_top", "y_left_top"),
    ("x_right_top", "y_right_top"),
    ("x_right_bottom", "y_right_bottom"),
    ("x_left_bottom", "y_left_bottom"),
)
PROMPT_TEMPLATE = "Locate a single instance that matches the following description: {description}"


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def quantize(value, lower=0, upper=1000):
    return max(lower, min(upper, int(round(value))))


def regularize_le90(cx, cy, width, height, angle_degrees):
    """Convert OpenCV output to a unique long-edge box in [-90, 90)."""
    width, height = float(width), float(height)
    angle = math.radians(float(angle_degrees))
    if height >= width:
        width, height = height, width
        angle += math.pi / 2.0
    angle = ((angle + math.pi / 2.0) % math.pi) - math.pi / 2.0
    return float(cx), float(cy), width, height, angle


def polygon_to_cxcywha(points, image_width, image_height):
    """Convert a four-point pixel polygon into normalized ``le90`` tokens."""
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError("invalid robndbox polygon")

    (cx, cy), (width, height), angle_degrees = cv2.minAreaRect(polygon)
    cx, cy, width, height, angle = regularize_le90(cx, cy, width, height, angle_degrees)
    if min(width, height) <= 1e-6:
        raise ValueError("degenerate robndbox polygon")

    angle_bin = max(0, min(999, int(round((angle + math.pi / 2.0) / math.pi * 1000.0))))
    return (
        quantize(cx / image_width * 1000.0),
        quantize(cy / image_height * 1000.0),
        quantize(width / image_width * 1000.0, lower=1),
        quantize(height / image_height * 1000.0, lower=1),
        angle_bin,
    )


def read_ids(list_path):
    if not list_path.is_file():
        raise FileNotFoundError(f"Missing split file: {list_path}")
    with list_path.open("r", encoding="utf-8") as source:
        return [line.strip() for line in source if line.strip()]


def resolve_xml(annotation_root, image_id):
    """Resolve numeric IDs whether their list is padded or unpadded."""
    candidates = [annotation_root / f"{image_id}.xml"]
    try:
        candidates.append(annotation_root / f"{int(image_id):05d}.xml")
    except ValueError:
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def parse_annotation(annotation_path):
    root = ET.parse(annotation_path).getroot()
    image_name = normalize_text(root.findtext("filename"))
    if not image_name:
        raise ValueError("missing XML filename")

    records = []
    for object_index, obj in enumerate(root.findall("object")):
        description = normalize_text(obj.findtext("description"))
        robndbox = obj.find("robndbox")
        if not description or robndbox is None:
            continue
        try:
            points = []
            for x_key, y_key in VERTEX_FIELDS:
                x = float(robndbox.findtext(x_key))
                y = float(robndbox.findtext(y_key))
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("non-finite vertex")
                points.append((x, y))
        except (TypeError, ValueError):
            continue
        records.append((object_index, description, points))
    return image_name, records


def sample_key(image_name, description, box):
    return image_name, description, tuple(box)


def convert_ids(ids, annotation_root, image_root, output_path):
    stats = Counter()
    seen = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as target:
        for image_id in ids:
            annotation_path = resolve_xml(annotation_root, image_id)
            if annotation_path is None:
                print(f"skip missing XML for ID {image_id}")
                stats["missing_xml"] += 1
                continue
            try:
                image_name, objects = parse_annotation(annotation_path)
                image_path = image_root / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
                if image_width <= 0 or image_height <= 0:
                    raise ValueError("non-positive image dimensions")
            except (ET.ParseError, OSError, ValueError) as error:
                print(f"skip invalid annotation {annotation_path}: {error}")
                stats["invalid_annotation"] += 1
                continue

            stats["images"] += 1
            for object_index, description, points in objects:
                stats["input_objects"] += 1
                try:
                    box = polygon_to_cxcywha(points, image_width, image_height)
                except ValueError as error:
                    print(f"skip object {annotation_path}:{object_index}: {error}")
                    stats["invalid_object"] += 1
                    continue

                key = sample_key(image_name, description, box)
                if key in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(key)

                cx, cy, width, height, angle = box
                sample = {
                    "conversations": [
                        {"from": "human", "value": PROMPT_TEMPLATE.format(description=description)},
                        {
                            "from": "gpt",
                            "value": (
                                f"<ref>{description}</ref>"
                                f"<box><{cx}><{cy}><{width}><{height}><{angle}></box>"
                            ),
                        },
                    ],
                    "image": image_name,
                }
                target.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["written"] += 1
    return stats


def write_recipe(output_dir, image_root):
    recipe = {
        "dior_r_rsvg_rotated_train": {
            "annotation": str((output_dir / "dior_r_rsvg_rotated_train.jsonl").resolve()),
            "root": str(image_root.resolve()),
            "repeat_time": 1.0,
            "data_augment": False,
        },
        "dior_r_rsvg_rotated_test": {
            "annotation": str((output_dir / "dior_r_rsvg_rotated_test.jsonl").resolve()),
            "root": str(image_root.resolve()),
            "repeat_time": 1.0,
            "data_augment": False,
        },
    }
    recipe_path = output_dir / "dior_r_rsvg_rotated_locany_recipe.json"
    with recipe_path.open("w", encoding="utf-8") as target:
        json.dump(recipe, target, ensure_ascii=False, indent=2)
        target.write("\n")
    return recipe_path


def main():
    parser = argparse.ArgumentParser(description="Prepare DIOR-R-RSVG for LocateAnything rotated grounding.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/dior_r_rsvg"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/locany_recipe/dior_r_rsvg_rotated"))
    args = parser.parse_args()

    train_ids = read_ids(args.dataset_root / "train.txt") + read_ids(args.dataset_root / "val.txt")
    test_ids = read_ids(args.dataset_root / "test.txt")
    annotation_root = args.dataset_root / "Annotations_obb"
    image_root = args.dataset_root / "JPEGImages"

    train_stats = convert_ids(
        train_ids, annotation_root, image_root, args.output_dir / "dior_r_rsvg_rotated_train.jsonl"
    )
    test_stats = convert_ids(
        test_ids, annotation_root, image_root, args.output_dir / "dior_r_rsvg_rotated_test.jsonl"
    )
    print(f"train (train + val): {dict(train_stats)}")
    print(f"test: {dict(test_stats)}")
    print(f"recipe: {write_recipe(args.output_dir, image_root)}")
    print("Fine-tune with --box_coord_dim 5 --block_size 7 (or larger).")


if __name__ == "__main__":
    main()
