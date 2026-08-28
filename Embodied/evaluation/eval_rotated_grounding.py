#!/usr/bin/env python3
"""Evaluate LocateAnything single-instance rotated visual grounding.

Input annotations must use the converted LocateAnything format with one target:
``<ref>description</ref><box><cx><cy><w><h><angle></box>``.  The angle bin
uses the project's ``le90`` convention: ``angle / 1000 * 180 - 90`` degrees.
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# The worker lives one level above this evaluation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from locateanything_worker import LocateAnythingWorker  # noqa: E402


GT_PATTERN = re.compile(
    r"<ref>(.*?)</ref><box>"
    r"<(\d+)><(\d+)><(\d+)><(\d+)><(\d+)></box>"
)


def parse_target(answer, image_width, image_height):
    """Parse the single ground-truth box from a LocateAnything answer."""
    matches = list(GT_PATTERN.finditer(answer))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one cxcywha target, found {len(matches)}")
    match = matches[0]
    cx, cy, width, height, angle = [int(value) for value in match.groups()[1:]]
    if width <= 0 or height <= 0:
        raise ValueError("ground-truth width and height must be positive")
    return {
        "label": match.group(1),
        "cx": cx / 1000.0 * image_width,
        "cy": cy / 1000.0 * image_height,
        "w": width / 1000.0 * image_width,
        "h": height / 1000.0 * image_height,
        "angle": angle / 1000.0 * 180.0 - 90.0,
    }


def rotated_corners(box):
    """Return clockwise image-coordinate vertices for a cxcywha box."""
    angle = math.radians(box["angle"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners = []
    for dx, dy in (
        (-box["w"] / 2.0, -box["h"] / 2.0),
        (box["w"] / 2.0, -box["h"] / 2.0),
        (box["w"] / 2.0, box["h"] / 2.0),
        (-box["w"] / 2.0, box["h"] / 2.0),
    ):
        corners.append((box["cx"] + dx * cos_a - dy * sin_a,
                        box["cy"] + dx * sin_a + dy * cos_a))
    return np.asarray(corners, dtype=np.float32)


def hbb_iou(first, second):
    """Axis-aligned IoU of the enclosing horizontal boxes, for ablation only."""
    first_points, second_points = rotated_corners(first), rotated_corners(second)
    ax1, ay1 = first_points.min(axis=0)
    ax2, ay2 = first_points.max(axis=0)
    bx1, by1 = second_points.min(axis=0)
    bx2, by2 = second_points.max(axis=0)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return (intersection / union if union > 0 else 0.0), intersection, union


def rotated_iou(first, second):
    """Exact quadrilateral IoU using OpenCV convex-polygon intersection."""
    first_points, second_points = rotated_corners(first), rotated_corners(second)
    first_area = float(first["w"] * first["h"])
    second_area = float(second["w"] * second["h"])
    try:
        intersection, _ = cv2.intersectConvexConvex(first_points, second_points)
    except cv2.error:
        return 0.0, 0.0, first_area + second_area
    intersection = max(0.0, float(intersection))
    union = max(0.0, first_area + second_area - intersection)
    return (intersection / union if union > 0 else 0.0), intersection, union


def extract_conversations(sample):
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("missing conversations list")
    human = next((entry.get("value") for entry in conversations if entry.get("from") == "human"), None)
    gpt = next((entry.get("value") for entry in conversations if entry.get("from") == "gpt"), None)
    if not isinstance(human, str) or not isinstance(gpt, str):
        raise ValueError("conversations must contain human and gpt strings")
    return human, gpt


def load_samples(annotation_path):
    with annotation_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                print(f"skip malformed JSON {annotation_path}:{line_number}: {error}")


def summarize(records, thresholds):
    valid = [record for record in records if record["evaluated"]]
    if not valid:
        return {"num_samples": len(records), "num_evaluated": 0}
    ious = np.asarray([record["iou"] for record in valid], dtype=np.float64)
    intersections = sum(record["intersection"] for record in valid)
    unions = sum(record["union"] for record in valid)
    metrics = {
        "num_samples": len(records),
        "num_evaluated": len(valid),
        "num_invalid_or_missing_prediction": len(records) - len(valid),
        "meanIoU": float(ious.mean() * 100.0),
        "cumIoU": float(intersections / unions * 100.0) if unions > 0 else 0.0,
    }
    for threshold in thresholds:
        metrics[f"Acc@{threshold:g}"] = float((ious >= threshold).mean() * 100.0)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate LocateAnything rotated visual grounding.")
    parser.add_argument("--model", required=True, help="Fine-tuned checkpoint directory, e.g. checkpoint-150.")
    parser.add_argument("--annotation", required=True, type=Path, help="Converted rotated test JSONL.")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Predictions and per-sample IoUs as JSONL.")
    parser.add_argument("--metrics-output", default=None, type=Path, help="Summary JSON; defaults beside --output.")
    parser.add_argument("--iou-type", choices=["rotated", "hbb"], default="rotated")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-mode", choices=["fast", "hybrid", "slow"], default="hybrid")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.iou_type == "hbb":
        print("WARNING: HBB IoU is an ablation. Use rotated IoU for the primary result.")
    if not args.annotation.is_file():
        raise FileNotFoundError(f"Missing annotation JSONL: {args.annotation}")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"Missing image root: {args.image_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output = args.metrics_output or args.output.with_suffix(".metrics.json")
    worker = LocateAnythingWorker(args.model, device=args.device)
    calculate_iou = rotated_iou if args.iou_type == "rotated" else hbb_iou
    records = []
    start_time = time.perf_counter()

    with args.output.open("w", encoding="utf-8") as target:
        for sample_index, (line_number, sample) in enumerate(load_samples(args.annotation), start=1):
            if args.max_samples is not None and len(records) >= args.max_samples:
                break
            record = {"line": line_number, "evaluated": False}
            try:
                prompt, target_answer = extract_conversations(sample)
                image_name = sample["image"]
                image_path = args.image_root / image_name
                with Image.open(image_path) as image_file:
                    image = image_file.convert("RGB")
                gt = parse_target(target_answer, image.width, image.height)
                result = worker.predict(
                    image, prompt, generation_mode=args.generation_mode,
                    max_new_tokens=args.max_new_tokens, temperature=0.0,
                    top_p=1.0, top_k=0, repetition_penalty=1.0, verbose=False,
                )
                answer = result["answer"]
                predictions = worker.parse_rotated_boxes(answer, image.width, image.height)
                record.update({"image": image_name, "prompt": prompt, "ground_truth": gt, "answer": answer})
                if predictions:
                    # Generative LocateAnything has no detection score. The first serialized box is Top-1.
                    prediction = predictions[0]
                    iou, intersection, union = calculate_iou(prediction, gt)
                    record.update({
                        "evaluated": True,
                        "prediction": prediction,
                        "iou": iou,
                        "intersection": intersection,
                        "union": union,
                    })
                else:
                    record["error"] = "no valid cxcywha box in model output"
            except (OSError, KeyError, TypeError, ValueError) as error:
                record["error"] = str(error)

            records.append(record)
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            if sample_index % 25 == 0:
                print(f"processed {sample_index} samples")

    elapsed_seconds = time.perf_counter() - start_time
    metrics = summarize(records, args.thresholds)
    metrics.update({
        "iou_type": args.iou_type,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_sample": elapsed_seconds / len(records) if records else 0.0,
        "model": args.model,
        "annotation": str(args.annotation),
    })
    with metrics_output.open("w", encoding="utf-8") as target:
        json.dump(metrics, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"per-sample results: {args.output}")
    print(f"metrics: {metrics_output}")


if __name__ == "__main__":
    main()
