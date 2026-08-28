#!/usr/bin/env python3
"""Convert VRSBench referring expressions to LocateAnything rotated grounding.

Each VRSBench object contributes one sample using its ``referring_sentence``:

    <ref>description</ref><box><cx><cy><w><h><angle></box>

The five coordinates follow the project's canonical ``le90`` convention:
``w >= h`` and ``angle`` is a [0, 1000) bin for [-90, 90) degrees.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROMPT_TEMPLATE = "Locate a single instance that matches the following description: {description}"


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def quantize(value, lower=0, upper=1000):
    return max(lower, min(upper, int(round(value))))


def regularize_le90(cx, cy, width, height, angle_degrees):
    """Canonicalize OpenCV's box to long-edge ``le90`` representation."""
    width, height = float(width), float(height)
    angle = math.radians(float(angle_degrees))
    if height >= width:
        width, height = height, width
        angle += math.pi / 2.0
    angle = ((angle + math.pi / 2.0) % math.pi) - math.pi / 2.0
    return float(cx), float(cy), width, height, angle


def polygon_to_cxcywha(points, image_width, image_height):
    """Convert four pixel-coordinate vertices to normalized cxcywha tokens."""
    if len(points) != 4:
        raise ValueError("a VRSBench object must have exactly four polygon vertices")
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError("invalid polygon coordinates")

    (cx, cy), (width, height), angle_degrees = cv2.minAreaRect(polygon)
    cx, cy, width, height, angle = regularize_le90(cx, cy, width, height, angle_degrees)
    if min(width, height) <= 1e-6:
        raise ValueError("degenerate polygon")

    angle_bin = max(0, min(999, int(round((angle + math.pi / 2.0) / math.pi * 1000.0))))
    return (
        quantize(cx / image_width * 1000.0),
        quantize(cy / image_height * 1000.0),
        quantize(width / image_width * 1000.0, lower=1),
        quantize(height / image_height * 1000.0, lower=1),
        angle_bin,
    )


def extract_pixel_polygon(obj, image_width, image_height):
    """Read VRSBench's normalized ``obj_corner`` into pixel coordinates.

    Values outside [0, 1] are intentionally retained: VRSBench can annotate
    objects partially outside the image, and clipping before minAreaRect would
    change the target rectangle.
    """
    corners = obj.get("obj_corner")
    if isinstance(corners, list) and len(corners) == 8:
        values = [float(value) for value in corners]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("obj_corner contains a non-finite coordinate")
        return [
            (values[index] * image_width, values[index + 1] * image_height)
            for index in range(0, 8, 2)
        ]

    # Fallback for an incomplete export: obj_coord is normalized xyxy.
    coord = obj.get("obj_coord")
    if isinstance(coord, list) and len(coord) == 4:
        x1, y1, x2, y2 = [float(value) for value in coord]
        return [
            (x1 * image_width, y1 * image_height),
            (x2 * image_width, y1 * image_height),
            (x2 * image_width, y2 * image_height),
            (x1 * image_width, y2 * image_height),
        ]
    raise ValueError("missing valid obj_corner (or fallback obj_coord)")


def sample_key(image_name, description, box):
    """Remove exact duplicate referring annotations within one split."""
    return image_name, normalize_text(description), tuple(box)


def convert_split(annotation_root, image_root, output_path):
    stats = Counter()
    seen = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(annotation_root.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No JSON annotations found in {annotation_root}")

    with output_path.open("w", encoding="utf-8") as target:
        for annotation_path in annotation_files:
            try:
                with annotation_path.open("r", encoding="utf-8") as source:
                    annotation = json.load(source)
                image_name = str(annotation["image"])
                image_path = image_root / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
                if image_width <= 0 or image_height <= 0:
                    raise ValueError("non-positive image dimensions")
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                print(f"skip annotation {annotation_path}: {error}")
                stats["invalid_annotation"] += 1
                continue

            objects = annotation.get("objects")
            if not isinstance(objects, list):
                print(f"skip annotation with non-list objects: {annotation_path}")
                stats["invalid_annotation"] += 1
                continue

            for object_index, obj in enumerate(objects):
                stats["input_objects"] += 1
                try:
                    description = normalize_text(obj["referring_sentence"])
                    if not description:
                        raise ValueError("empty referring_sentence")
                    polygon = extract_pixel_polygon(obj, image_width, image_height)
                    box = polygon_to_cxcywha(polygon, image_width, image_height)
                except (KeyError, TypeError, ValueError) as error:
                    print(f"skip object {annotation_path}:{object_index}: {error}")
                    stats["invalid_object"] += 1
                    continue

                key = sample_key(image_name, description, box)
                if key in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(key)

                cx, cy, width, height, angle = box
                answer = (
                    f"<ref>{description}</ref>"
                    f"<box><{cx}><{cy}><{width}><{height}><{angle}></box>"
                )
                sample = {
                    "conversations": [
                        {"from": "human", "value": PROMPT_TEMPLATE.format(description=description)},
                        {"from": "gpt", "value": answer},
                    ],
                    "image": image_name,
                }
                target.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["written"] += 1

    return stats


def write_recipe(output_dir, dataset_root, splits):
    recipe = {}
    for split in splits:
        recipe[f"vrsbench_rotated_{split}"] = {
            "annotation": str((output_dir / f"vrsbench_rotated_{split}.jsonl").resolve()),
            "root": str((dataset_root / f"Images_{split}").resolve()),
            "repeat_time": 1.0,
            "data_augment": False,
        }
    recipe_path = output_dir / "vrsbench_rotated_locany_recipe.json"
    with recipe_path.open("w", encoding="utf-8") as target:
        json.dump(recipe, target, ensure_ascii=False, indent=2)
        target.write("\n")
    return recipe_path


def main():
    parser = argparse.ArgumentParser(description="Prepare VRSBench for LocateAnything cxcywha grounding.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/VRSBench"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/locany_recipe/vrsbench_rotated"))
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    args = parser.parse_args()

    for split in args.splits:
        stats = convert_split(
            args.dataset_root / f"Annotations_{split}",
            args.dataset_root / f"Images_{split}",
            args.output_dir / f"vrsbench_rotated_{split}.jsonl",
        )
        print(f"{split}: {dict(stats)}")

    recipe_path = write_recipe(args.output_dir, args.dataset_root, args.splits)
    print(f"recipe: {recipe_path}")
    print("Fine-tune with --box_coord_dim 5 --block_size 7 (or larger).")


if __name__ == "__main__":
    main()
