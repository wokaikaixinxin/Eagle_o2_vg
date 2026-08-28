# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def is_point_in_bbox(point: List[float], bbox: List[float]) -> bool:
    """
    Check if a point is inside a bounding box

    Args:
        point: [x, y] point coordinates
        bbox: [x1, y1, x2, y2] bounding box coordinates (xyxy format)

    Returns:
        bool: Whether the point is inside the bounding box
    """
    if len(point) != 2 or len(bbox) != 4:
        return False

    x, y = point
    x1, y1, x2, y2 = bbox

    # Ensure bbox coordinates are in correct order
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    return x1 <= x <= x2 and y1 <= y <= y2


def get_box_center(box: List[float]) -> List[float]:
    """
    Calculate the center point of a bounding box

    Args:
        box: [x1, y1, x2, y2] bounding box (xyxy format)

    Returns:
        List[float]: [center_x, center_y] center point coordinates
    """
    if len(box) != 4:
        return []

    x1, y1, x2, y2 = box

    # Ensure coordinates are in correct order
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return [center_x, center_y]


def extract_predictions(
    extracted_predictions: Dict,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Extract all point and box coordinates from the extracted_predictions dict

    Args:
        extracted_predictions: e.g. {"street sign": [[209.45945945945945, 269.2692692692693]]} or
                              {"button": [[x1, y1, x2, y2]]}

    Returns:
        Tuple[List[List[float]], List[List[float]]]: (list of point coordinates, list of box coordinates)
    """
    points = []
    boxes = []

    if not isinstance(extracted_predictions, dict):
        return points, boxes

    for key, value in extracted_predictions.items():
        if isinstance(value, list):
            for coord in value:
                if isinstance(coord, list):
                    try:
                        if len(coord) == 2:
                            # Point format [x, y]
                            x, y = float(coord[0]), float(coord[1])
                            points.append([x, y])
                        elif len(coord) == 4:
                            # Box format [x1, y1, x2, y2]
                            x1, y1, x2, y2 = (
                                float(coord[0]),
                                float(coord[1]),
                                float(coord[2]),
                                float(coord[3]),
                            )
                            boxes.append([x1, y1, x2, y2])
                    except (ValueError, TypeError):
                        continue

    return points, boxes


def get_data_source_from_image_name(image_name: str) -> str:
    """
    Infer data source from image filename.

    Args:
        image_name: Image filename

    Returns:
        str: Data source type (Desktop, Mobile, Web, Unknown)
    """
    if not isinstance(image_name, str):
        return "Unknown"

    if image_name.startswith("pc_"):
        return "Desktop"
    elif image_name.startswith("mobile_"):
        return "Mobile"
    elif image_name.startswith("web_"):
        return "Web"
    else:
        return "Unknown"


def build_filename_mapping(type_match_file: str) -> Dict[str, Dict[str, str]]:
    """
    Build a mapping from filename to source and type from type_match_file.

    Args:
        type_match_file: Path to a JSONL file with img_filename and type fields.
                        Supported field pairs:
                        1. data_type, data_source (preferred)
                        2. ui_type, group (fallback)

    Returns:
        Dict[str, Dict[str, str]]: Filename -> {"source": source, "type": type}
    """
    filename_mapping = {}

    if not type_match_file or not os.path.exists(type_match_file):
        print(f"Warning: type_match_file {type_match_file} not found or not provided")
        return filename_mapping

    try:
        with open(type_match_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    img_filename = data.get("img_filename", "")

                    if not img_filename:
                        continue

                    # Prefer data_type and data_source when both are present and non-empty
                    data_type_val = data.get("data_type")
                    data_source_val = data.get("data_source")
                    ui_type_val = data.get("ui_type")
                    group_val = data.get("group")

                    if (
                        data_type_val
                        and data_source_val
                        and data_type_val != ""
                        and data_source_val != ""
                    ):
                        # Both data_type and data_source present and non-empty; use them
                        filename_mapping[img_filename] = {
                            "source": data_source_val,
                            "type": data_type_val,
                        }
                    elif (
                        ui_type_val
                        and group_val
                        and ui_type_val != ""
                        and group_val != ""
                    ):
                        # Both ui_type and group present and non-empty; use as fallback
                        filename_mapping[img_filename] = {
                            "source": group_val,
                            "type": ui_type_val,
                        }
                    else:
                        # Mixed fields or defaults
                        final_type = (
                            data_type_val
                            if data_type_val and data_type_val != ""
                            else (
                                ui_type_val
                                if ui_type_val and ui_type_val != ""
                                else "Unknown"
                            )
                        )
                        final_source = (
                            data_source_val
                            if data_source_val and data_source_val != ""
                            else (
                                group_val
                                if group_val and group_val != ""
                                else "Unknown"
                            )
                        )

                        filename_mapping[img_filename] = {
                            "source": final_source,
                            "type": final_type,
                        }

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Failed to parse line {line_num} in type_match_file: {e}"
                    )
                    continue

        print(
            f"Loaded {len(filename_mapping)} filename mappings from {type_match_file}"
        )

    except Exception as e:
        print(f"Error reading type_match_file {type_match_file}: {e}")

    return filename_mapping


def get_filename_from_path(image_path: str) -> str:
    """
    Extract the filename from an image path.

    Args:
        image_path: Image path

    Returns:
        str: Filename
    """
    if not image_path:
        return ""
    return os.path.basename(image_path)


def evaluate_single_prediction(
    prediction: Dict, filename_mapping: Optional[Dict] = None
) -> Tuple[bool, str, str]:
    """
    Evaluate a single prediction.

    Args:
        prediction: Single prediction dict
        filename_mapping: Filename -> source/type mapping

    Returns:
        Tuple[bool, str, str]: (is_correct, data_type, data_source)
    """
    try:
        # Extract required fields
        extracted_predictions = prediction.get("extracted_predictions", {})
        gt = prediction.get("gt", [])
        data_type = prediction.get("data_type", "")
        image_name = prediction.get("image_name", "")
        image_path = prediction.get("image_path", "")

        # Resolve filename
        if not image_name and image_path:
            image_name = get_filename_from_path(image_path)

        # Infer data_source from image_name (legacy logic)
        data_source = get_data_source_from_image_name(image_name)

        # If data_type or data_source is empty or Unknown, try filename_mapping
        if filename_mapping and image_name:
            mapping_info = filename_mapping.get(image_name, {})

            # If data_type is empty or Unknown, use type from mapping
            if not data_type or data_type == "Unknown":
                data_type = mapping_info.get("type", data_type or "Unknown")

            # If data_source is Unknown, use source from mapping
            if data_source == "Unknown":
                data_source = mapping_info.get("source", data_source)

        # Extract predicted points and boxes
        predicted_points, predicted_boxes = extract_predictions(extracted_predictions)

        # No predicted points or boxes -> treat as failure
        if not predicted_points and not predicted_boxes:
            return False, data_type, data_source

        # Empty or invalid GT -> treat as failure
        if not gt or len(gt) != 4:
            return False, data_type, data_source

        # Check if any predicted point lies inside the GT box
        for point in predicted_points:
            if is_point_in_bbox(point, gt):
                return True, data_type, data_source

        # Check if any predicted box center lies inside the GT box
        for box in predicted_boxes:
            center_point = get_box_center(box)
            if center_point and is_point_in_bbox(center_point, gt):
                return True, data_type, data_source

        return False, data_type, data_source

    except Exception as e:
        print(f"Error evaluating prediction: {e}")
        return False, "Unknown", "Unknown"


def calculate_metrics(jsonl_file: str, filename_mapping: Optional[Dict] = None) -> Dict:
    """
    Compute GUI grounding metrics.

    Args:
        jsonl_file: Path to JSONL file
        filename_mapping: Filename -> source/type mapping

    Returns:
        Dict: Metrics dictionary
    """
    # Counters
    total_correct = 0
    total_count = 0
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    # Read JSONL file
    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    prediction = json.loads(line)
                    is_correct, data_type, data_source = evaluate_single_prediction(
                        prediction, filename_mapping
                    )

                    # Update overall stats
                    total_count += 1
                    if is_correct:
                        total_correct += 1

                    # Update per-category stats
                    category_key = f"{data_source}-{data_type}"
                    category_stats[category_key]["total"] += 1
                    if is_correct:
                        category_stats[category_key]["correct"] += 1

                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: File {jsonl_file} not found")
        return {}
    except Exception as e:
        print(f"Error reading file {jsonl_file}: {e}")
        return {}

    # Compute accuracies
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    category_accuracies = {}
    for category, stats in category_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        category_accuracies[category] = {
            "accuracy": accuracy,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    return {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_count": total_count,
        "category_accuracies": category_accuracies,
    }


def print_results(results: Dict):
    """
    Print evaluation results.

    Args:
        results: Results dictionary
    """
    if not results:
        print("No results to display")
        return

    print("=" * 60)
    print("GUI Grounding Evaluation Results")
    print("=" * 60)

    # Overall accuracy
    overall_acc = results["overall_accuracy"]
    total_correct = results["total_correct"]
    total_count = results["total_count"]

    print(f"\nOverall Results:")
    print(f"  Average Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"  Correct Predictions: {total_correct}")
    print(f"  Total Predictions: {total_count}")

    # Per-category accuracy
    print(f"\nResults by Data Source and Type:")
    print("-" * 60)
    print(f"{'Category':<20} {'Accuracy':<12} {'Correct/Total':<15}")
    print("-" * 60)

    category_accuracies = results["category_accuracies"]
    for category in sorted(category_accuracies.keys()):
        stats = category_accuracies[category]
        accuracy = stats["accuracy"]
        correct = stats["correct"]
        total = stats["total"]

        print(f"{category:<20} {accuracy:.4f} ({accuracy*100:.2f}%) {correct}/{total}")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Calculate GUI grounding metrics")
    parser.add_argument(
        "--jsonl_file",
        default="path/to/EvalData/_locate_anything_eval_results/sspro/benchmark_screenspotpro_box.jsonl",
        help="Path to the JSONL file containing predictions",
    )
    parser.add_argument(
        "--output",
        default="path/to/EvalData/_locate_anything_eval_results/sspro/metric_point.jsonl",
        help="Output file to save results (optional)",
    )
    parser.add_argument(
        "--type_match_file",
        default="path/to/EvalData/ScreenSpot-Pro/converted_box.jsonl",
        help="JSONL file containing img_filename, ui_type, group mappings",
    )

    args = parser.parse_args()

    # Build filename mapping
    print(f"Loading filename mappings from: {args.type_match_file}")
    filename_mapping = build_filename_mapping(args.type_match_file)

    # Compute metrics
    print(f"Evaluating predictions from: {args.jsonl_file}")
    results = calculate_metrics(args.jsonl_file, filename_mapping)

    # Print results
    print_results(results)

    # Save results to file if output path is set
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()
