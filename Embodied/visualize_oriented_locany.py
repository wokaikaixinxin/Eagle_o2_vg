#!/usr/bin/env python3
"""Run LocateAnything oriented-box inference and draw four-point polygons."""

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw

from locateanything_worker import LocateAnythingWorker


def parse_oriented_boxes(answer, width, height):
    pattern = (
        r"<ref>(.*?)</ref><box>"
        r"<(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)>"
        r"</box>"
    )
    boxes = []
    for match in re.finditer(pattern, answer):
        label = match.group(1)
        coords = [int(v) for v in match.groups()[1:]]
        points = [
            (coords[i] / 1000 * width, coords[i + 1] / 1000 * height)
            for i in range(0, 8, 2)
        ]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if max(xs) <= min(xs) or max(ys) <= min(ys):
            print("skip invalid oriented box:", label, coords)
            continue
        boxes.append((label, points))
    return boxes


def draw_oriented_boxes(image, boxes, output):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for label, points in boxes:
        draw.line(points + [points[0]], fill="red", width=3)
        x, y = points[0]
        draw.text((x, max(0, y - 14)), label, fill="red")
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--categories", required=True, help='Use "</c>" to separate multiple categories.')
    parser.add_argument("--output", default=None)
    parser.add_argument("--generation-mode", default="hybrid", choices=["fast", "hybrid", "slow"])
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    worker = LocateAnythingWorker(args.model)

    result = worker.detect_oriented(image, args.categories, generation_mode=args.generation_mode)
    answer = result["answer"]
    print(answer)

    boxes = parse_oriented_boxes(answer, image.width, image.height)
    output = args.output or str(Path(args.image).with_name(Path(args.image).stem + "_oriented_vis.jpg"))
    draw_oriented_boxes(image, boxes, output)
    print("saved:", output)


if __name__ == "__main__":
    main()
