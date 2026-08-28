# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_log_line(line: str, data_store: Dict[str, List[float]]) -> bool:
    """Parse a single log line and update the data store. Returns True if parsed successfully."""
    if "Statistic Info" not in line:
        return False

    matches = re.findall(r"([\w\(\)]+)=([^;]+)", line)
    if not matches:
        return False

    for key, value in matches:
        key = key.strip()
        value = value.strip()

        try:
            num_val = float(value)
            data_store[key].append(num_val)
        except ValueError:
            if value.lower() == "true":
                data_store[key].append(1.0)
            elif value.lower() == "false":
                data_store[key].append(0.0)
    
    return True


def print_statistics(data_store: Dict[str, List[float]], count: int) -> None:
    """Print the calculated statistics in a formatted table."""
    print(f"\n{'='*15} Speed Statistics ({count} samples found) {'='*15}")

    if count == 0:
        print("No log lines containing 'Statistic Info' were found.")
        return

    header_format = "{:<25} | {:<15} | {:<10} | {:<10} | {:<10}"
    row_format = "{:<25} | {:<15.4f} | {:<10.4f} | {:<10.4f} | {:<10.2f}"
    
    print(header_format.format("Key", "Average", "Min", "Max", "Total"))
    print("-" * 80)

    for key, values in sorted(data_store.items()):
        if not values:
            continue
            
        avg_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        sum_val = sum(values)

        print(row_format.format(key, avg_val, min_val, max_val, sum_val))

    print("-" * 80 + "\n")


def analyze_log(file_path: str) -> None:
    """Analyze the log file and print statistics."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: '{file_path}'")
        return

    data_store: Dict[str, List[float]] = defaultdict(list)
    count = 0

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if parse_log_line(line, data_store):
                    count += 1
                    
        print_statistics(data_store, count)
        
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze speed statistics from log files.")
    parser.add_argument(
        "--log_file", 
        type=str, 
        required=True,
        help="Path to the log file to analyze"
    )
    args = parser.parse_args()

    analyze_log(args.log_file)


if __name__ == "__main__":
    main()
