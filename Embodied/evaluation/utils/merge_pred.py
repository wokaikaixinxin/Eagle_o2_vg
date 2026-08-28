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
import re

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Merge prediction results")
    parser.add_argument(
        "--root_path",
        type=str,
        default="path/to/EvalData/_locate_anything_eval_results/box_eval/LVIS",
        help="path to coco json file",
    )
    parser.add_argument(
        "--dump_anno_path",
        type=str,
        default="path/to/EvalData/_locate_anything_eval_results/box_eval/LVIS/answer.jsonl",
        help="path to coco image file",
    )
    args = parser.parse_args()
    return args


def extract_numbers(file_path):
    match = re.search(r"(\d+)_(\d+)", file_path)
    if match:
        return int(match.group(1)), int(match.group(2))
    return float("inf"), float("inf")


if __name__ == "__main__":
    args = get_args()
    root_path = args.root_path
    dump_anno_path = args.dump_anno_path
    files = os.listdir(root_path)
    tsv_files = [f for f in files if f.endswith(".jsonl")]
    # Sort by leading numeric values
    json_files = sorted(tsv_files, key=extract_numbers)

    json_files = [os.path.join(root_path, f) for f in json_files]

    total_anno = []
    for json_file in tqdm(json_files):
        with open(json_file, "r") as f:
            lines = [json.loads(line) for line in f.readlines()]
        total_anno.extend(lines)
    with open(args.dump_anno_path, "w") as f:
        for anno in total_anno:
            f.write(json.dumps(anno, ensure_ascii=False) + "\n")
