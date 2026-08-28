#!/usr/bin/env python3
"""Convert AVVG annotations to LocateAnything rotated-box grounding JSONL.

AVVG stores a quadrilateral in pixel coordinates.  This script converts it to
the five-coordinate LocateAnything protocol::

    <ref>query</ref><box><cx><cy><w><h><angle></box>

``cx``/``w`` are normalized by image width, ``cy``/``h`` by image height, and
``angle`` maps the canonical long-edge angle in [-90, 90) degrees to [0, 1000).
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROMPT_TEMPLATE = "Locate a single instance that matches the following description: {question}."


def quantize(value, lower=0, upper=1000):
    """Round a normalized value and keep it representable by coordinate tokens."""
    return max(lower, min(upper, int(round(value))))


def normalize_question(question):
    return " ".join(str(question).strip().split())


def regularize_le90(cx, cy, width, height, angle_degrees):
    """Canonicalize a ``cv2.minAreaRect`` box to MMRotate's ``le90`` form."""
    angle = math.radians(float(angle_degrees))
    width, height = float(width), float(height)
    if height >= width:
        width, height = height, width
        angle += math.pi / 2.0
    # Equivalent to regularize_boxes(pattern="le90"): w >= h and
    # angle in [-pi/2, pi/2).
    angle = ((angle + math.pi / 2.0) % math.pi) - math.pi / 2.0
    return float(cx), float(cy), width, height, angle


def quantize_le90_angle(angle_radians):
    """Map a canonical [-pi/2, pi/2) angle to coordinate-token bins."""
    value = (angle_radians + math.pi / 2.0) / math.pi * 1000.0
    return max(0, min(999, int(round(value))))


def polygon_to_cxcywha(poly, image_width, image_height):
    """Return normalized integer ``(cx, cy, w, h, angle)`` for a rectangle.

    ``poly`` can start at any vertex and can be clockwise or anticlockwise.
    ``cv2.minAreaRect`` estimates its minimum-area rectangle, then ``le90``
    regularization creates a unique long-edge representation. Coordinates
    outside the image are retained while computing geometry (AVVG has such
    cases), then serialized values are clipped to the token vocabulary.
    """
    if not isinstance(poly, list) or len(poly) != 4:
        raise ValueError("poly must contain exactly four vertices")

    points = []
    for point in poly:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each polygon vertex must be [x, y]")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("polygon contains a non-finite coordinate")
        points.append((x, y))

    (cx_px, cy_px), (width_px, height_px), angle_degrees = cv2.minAreaRect(
        np.asarray(points, dtype=np.float32)
    )
    cx_px, cy_px, width_px, height_px, angle_radians = regularize_le90(
        cx_px, cy_px, width_px, height_px, angle_degrees
    )
    if min(width_px, height_px) <= 1e-6:
        raise ValueError("degenerate polygon")
    return (
        quantize(cx_px / image_width * 1000.0),
        quantize(cy_px / image_height * 1000.0),
        quantize(width_px / image_width * 1000.0, lower=1),
        quantize(height_px / image_height * 1000.0, lower=1),
        quantize_le90_angle(angle_radians),
    )


def record_key(record):
    """Key for exact duplicated AVVG annotations within one split."""
    poly = tuple(tuple(float(value) for value in point) for point in record["poly"])
    return str(record["image_id"]), normalize_question(record["question"]), poly


def locate_image(image_root, image_id):
    image_path = image_root / image_id
    if image_path.is_file():
        return image_path
    # Some AVVG exports differ only in the extension case.
    matches = list(image_root.glob(f"{Path(image_id).stem}.*"))
    return matches[0] if len(matches) == 1 else None


def convert_split(annotation_path, image_root, output_path):
    stats = Counter()
    seen = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with annotation_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            stats["input"] += 1
            try:
                record = json.loads(line)
                key = record_key(record)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                print(f"skip malformed record {annotation_path}:{line_number}: {error}")
                stats["malformed"] += 1
                continue

            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)

            image_path = locate_image(image_root, str(record["image_id"]))
            if image_path is None:
                print(f"skip missing image {record['image_id']} ({annotation_path}:{line_number})")
                stats["missing_image"] += 1
                continue

            try:
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
                if image_width <= 0 or image_height <= 0:
                    raise ValueError("non-positive image size")
                cx, cy, width, height, angle = polygon_to_cxcywha(
                    record["poly"], image_width, image_height
                )
            except (OSError, KeyError, TypeError, ValueError) as error:
                print(f"skip invalid geometry {annotation_path}:{line_number}: {error}")
                stats["invalid"] += 1
                continue

            question = normalize_question(record["question"])
            answer = f"<ref>{question}</ref><box><{cx}><{cy}><{width}><{height}><{angle}></box>"
            sample = {
                "conversations": [
                    {"from": "human", "value": PROMPT_TEMPLATE.format(question=question)},
                    {"from": "gpt", "value": answer},
                ],
                "image": image_path.name,
            }
            target.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stats["written"] += 1

    return stats


def write_recipe(output_dir, image_root, splits):
    recipe = {
        f"avvg_rotated_{split}": {
            "annotation": str((output_dir / f"avvg_rotated_{split}.jsonl").resolve()),
            "root": str(image_root.resolve()),
            "repeat_time": 1.0,
            "data_augment": False,
        }
        for split in splits
    }
    recipe_path = output_dir / "avvg_rotated_locany_recipe.json"
    with recipe_path.open("w", encoding="utf-8") as handle:
        json.dump(recipe, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return recipe_path


def main():
    parser = argparse.ArgumentParser(description="Prepare AVVG for LocateAnything cxcywha grounding.")
    # parser.add_argument("--metainfo-root", type=Path, default=Path("/root/refGeo/metainfo"))
    parser.add_argument("--metainfo-root", type=Path, default=Path("/root/autodl-tmp/avvg_resized_1024x576/metainfo"))
    # parser.add_argument("--image-root", type=Path, default=Path("/root/refGeo/images/avvg"))
    parser.add_argument("--image-root", type=Path, default=Path("/root/autodl-tmp/avvg_resized_1024x576/images"))
    # parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/locany_recipe/avvg_rotated"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/locany_recipe/avvg_1024x576_rotated"))
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    args = parser.parse_args()

    for split in args.splits:
        annotation_path = args.metainfo_root / f"avvg_{split}.jsonl"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"AVVG annotation file does not exist: {annotation_path}")
        output_path = args.output_dir / f"avvg_rotated_{split}.jsonl"
        stats = convert_split(annotation_path, args.image_root, output_path)
        print(f"{split}: {dict(stats)} -> {output_path}")

    recipe_path = write_recipe(args.output_dir, args.image_root, args.splits)
    print(f"recipe: {recipe_path}")
    print("Fine-tune with --box_coord_dim 5 --block_size 7 (or larger).")


if __name__ == "__main__":
    main()
