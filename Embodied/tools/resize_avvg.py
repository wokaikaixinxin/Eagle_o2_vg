#!/usr/bin/env python3
"""Resize the raw AVVG dataset while keeping polygon annotations aligned.

The output dataset is self-contained::

    <output-root>/
      images/
      metainfo/avvg_train.jsonl
      metainfo/avvg_test.jsonl

Images are resized directly to the requested dimensions (the default is
1024x576), so polygons are scaled independently along the x and y axes.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def locate_image(image_root, image_id):
    """Resolve an AVVG image ID, tolerating extension-case differences."""
    requested = image_root / image_id
    if requested.is_file():
        return requested

    parent = requested.parent
    if not parent.is_dir():
        return None
    matches = [
        path
        for path in parent.iterdir()
        if path.is_file() and path.stem == requested.stem and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return matches[0] if len(matches) == 1 else None


def scale_polygon(poly, scale_x, scale_y):
    """Scale an AVVG quadrilateral, rejecting malformed coordinates."""
    if not isinstance(poly, list) or len(poly) != 4:
        raise ValueError("poly must contain exactly four vertices")

    scaled = []
    for point in poly:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each polygon vertex must be [x, y]")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("polygon contains a non-finite coordinate")
        scaled.append([x * scale_x, y * scale_y])
    return scaled


def scale_bbox(bbox, scale_x, scale_y):
    """Scale an AVVG ``[x1, y1, x2, y2]`` pixel bounding box."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")
    values = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox contains a non-finite coordinate")
    x1, y1, x2, y2 = values
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def resize_image(source_path, destination_path, target_size):
    """Resize one image and return its original ``(width, height)``."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        original_size = image.size
        if min(original_size) <= 0:
            raise ValueError("image has a non-positive dimension")
        resized = image.resize(target_size, Image.Resampling.LANCZOS)
        # PIL selects the encoder from the destination extension. AVVG image
        # IDs normally use the same extension as their source files.
        if destination_path.suffix.lower() in {".jpg", ".jpeg"} and resized.mode not in {
            "RGB",
            "L",
            "CMYK",
        }:
            resized = resized.convert("RGB")
        resized.save(destination_path)
    return original_size


def resize_images(image_root, output_image_root, target_size):
    """Resize every supported image under ``image_root`` preserving layout."""
    stats = Counter()
    sizes = {}
    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative_path = image_path.relative_to(image_root)
        try:
            sizes[relative_path.as_posix()] = resize_image(
                image_path, output_image_root / relative_path, target_size
            )
            stats["written"] += 1
        except (OSError, UnidentifiedImageError, ValueError) as error:
            print(f"skip unreadable image {image_path}: {error}")
            stats["invalid"] += 1
    return sizes, stats


def transform_annotations(annotation_path, image_root, output_path, image_sizes, target_size):
    """Write a transformed AVVG JSONL file and return processing statistics."""
    stats = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_width, target_height = target_size

    with annotation_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            stats["input"] += 1
            try:
                record = json.loads(line)
                image_path = locate_image(image_root, str(record["image_id"]))
                if image_path is None:
                    raise FileNotFoundError(f"image does not exist: {record['image_id']}")
                relative_image = image_path.relative_to(image_root).as_posix()
                image_width, image_height = image_sizes[relative_image]
                transformed = dict(record)
                transformed["image_id"] = relative_image
                transformed["bbox"] = scale_bbox(
                    record["bbox"], target_width / image_width, target_height / image_height
                )
                transformed["poly"] = scale_polygon(
                    record["poly"], target_width / image_width, target_height / image_height
                )
            except (
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                print(f"skip invalid record {annotation_path}:{line_number}: {error}")
                stats["invalid"] += 1
                continue

            target.write(json.dumps(transformed, ensure_ascii=False) + "\n")
            stats["written"] += 1
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Resize raw AVVG images and scale their quadrilateral annotations."
    )
    parser.add_argument("--metainfo-root", type=Path, default=Path("/root/autodl-tmp/refGeo/metainfo"))
    parser.add_argument("--image-root", type=Path, default=Path("/root/autodl-tmp/refGeo/images/avvg"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/avvg_resized_1024x576"),
        help="New dataset directory; input files are never modified.",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"AVVG image directory does not exist: {args.image_root}")

    output_root = args.output_root.resolve()
    if output_root == args.image_root.resolve() or output_root == args.metainfo_root.resolve():
        parser.error("--output-root must be a new directory, not an input directory")

    target_size = (args.width, args.height)
    output_image_root = output_root / "images"
    image_sizes, image_stats = resize_images(args.image_root, output_image_root, target_size)
    print(f"images: {dict(image_stats)} -> {output_image_root}")

    output_metainfo_root = output_root / "metainfo"
    for split in args.splits:
        annotation_path = args.metainfo_root / f"avvg_{split}.jsonl"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"AVVG annotation file does not exist: {annotation_path}")
        output_path = output_metainfo_root / annotation_path.name
        stats = transform_annotations(
            annotation_path, args.image_root, output_path, image_sizes, target_size
        )
        print(f"{split}: {dict(stats)} -> {output_path}")


if __name__ == "__main__":
    main()
