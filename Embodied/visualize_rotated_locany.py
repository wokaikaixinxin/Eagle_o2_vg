#!/usr/bin/env python3
"""Run LocateAnything cxcywha inference and render rotated boxes."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from locateanything_worker import LocateAnythingWorker


def rotated_corners(box):
    """Return four corners for clockwise image-space angle in degrees."""
    import math

    cx, cy, width, height, angle = (box[key] for key in ("cx", "cy", "w", "h", "angle"))
    radians = math.radians(angle)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    corners = []
    for dx, dy in ((-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)):
        corners.append((cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a))
    return corners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--categories", required=True, help='Separate multiple categories with "</c>".')
    parser.add_argument("--output", default=None)
    parser.add_argument("--generation-mode", default="hybrid", choices=["fast", "hybrid", "slow"])
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    worker = LocateAnythingWorker(args.model)
    result = worker.detect_rotated(image, args.categories, generation_mode=args.generation_mode)
    print(result["answer"])

    boxes = worker.parse_rotated_boxes(result["answer"], image.width, image.height)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        corners = rotated_corners(box)
        draw.line(corners + [corners[0]], fill="red", width=3)
        draw.text(corners[0], f'{box["label"]} {box["angle"]:.1f}°', fill="red")

    output = args.output or str(Path(args.image).with_name(Path(args.image).stem + "_rotated_vis.jpg"))
    canvas.save(output)
    print("saved:", output)


if __name__ == "__main__":
    main()
