# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Multi-GPU DDP inference script for LocateAnything ScreenSpot Pro evaluation.
Supports multi-node multi-GPU distributed inference.
"""
import argparse
import json
import os
import re
import datetime

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from inference_compat import (
    apply_chat_template,
    build_generate_kwargs,
    decode_generation_output,
    prepare_generation_inputs,
    process_vision_info,
)

# Set longer NCCL timeout (2 hours)
os.environ["NCCL_TIMEOUT"] = "7200"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="path/to/LocateAnything",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--test_jsonl_path",
        type=str,
        default="path/to/EvalData/ScreenSpot-Pro/converted_box.jsonl",
        help="Path to the test JSONL file containing benchmark data",
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        default="path/to/EvalData/ScreenSpot-Pro/images",
        help="Root directory to prepend to image paths. If empty, use image_path as is.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="path/to/EvalData/eval_results/benchmark_screenspotpro_box.jsonl",
        help="Path to save the evaluation results",
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--short_side_size",
        type=int,
        default=None,
        help="Resize image short side to this value before inference. Coordinates will be mapped back to original size.",
    )
    parser.add_argument(
        "--generation_mode",
        type=str,
        default="hybrid",
        choices=["fast", "slow", "hybrid"],
        help="Generation mode: 'fast' (MTP only), 'slow' (AR only), 'hybrid' (MTP + AR fallback).",
    )

    # DDP parameters
    parser.add_argument("--world_size", type=int, default=1, help="Total number of GPUs")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="Master address")
    parser.add_argument("--master_port", type=str, default="29500", help="Master port")
    parser.add_argument("--use_tcp", action="store_true", help="Use TCP for communication")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank (set by torchrun)")

    return parser.parse_args()


def setup_distributed():
    """Initialize distributed environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        print("Not using distributed mode")
        return 0, 1, 0

    torch.cuda.set_device(local_rank)

    # Initialize process group with longer timeout (2 hours)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(hours=2),
    )

    dist.barrier()

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank():
    """Get the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get the total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


class LocateAnythingWorker:
    def __init__(self, model_path, device='cuda', generation_mode: str = 'hybrid'):
        self.model_id = model_path
        self.device = device
        self.generation_mode = generation_mode
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        if hasattr(self.processor, 'tokenizer'):
            try:
                self.processor.tokenizer.padding_side = "left"
            except Exception:
                pass
        self.model = self.model.to(device)
        self.model.eval()

    def build_messages(self, image, question):
        """Build messages using the original question as input."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

    @torch.inference_mode()
    def generate(self, image, question, max_new_tokens=4096):
        """Generate predictions for a given image and question.
        Returns: output_text (str)
        """
        messages = self.build_messages(image, question)
        text_list = [apply_chat_template(self.processor, messages)]
        image_inputs, video_inputs = process_vision_info(
            self.processor,
            messages,
        )
        processor_inputs = self.processor(
            text=text_list,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )

        prepared_inputs = prepare_generation_inputs(processor_inputs, self.device)
        generate_kwargs = build_generate_kwargs(
            prepared_inputs,
            self.processor,
            generation_mode=self.generation_mode,
            max_new_tokens=max_new_tokens,
        )

        raw_output = self.model.generate(**generate_kwargs)
        output_text = decode_generation_output(
            raw_output,
            prepared_inputs["input_ids"],
            self.processor,
        )

        return output_text


class ScreenSpotProDataset(Dataset):
    """ScreenSpot Pro dataset class for distributed inference."""
    def __init__(self, test_data, image_root_dir):
        self.test_data = test_data
        self.image_root_dir = image_root_dir

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        entry = self.test_data[idx]
        image_name = entry["file_name"]
        instruction = entry["instruction"]
        # Build the question format
        question = f"Point to: {instruction}."
        # question = f"Locate the region that matches the following description: {instruction}."
        gt = entry["answer"]
        data_type = entry.get("data_type", "Unknown")
        data_source = entry.get("data_source", "Unknown")

        # Construct full image path
        full_image_path = os.path.join(self.image_root_dir, image_name)

        return {
            "image_name": image_name,
            "full_image_path": full_image_path,
            "instruction": instruction,
            "question": question,
            "gt": gt,
            "data_type": data_type,
            "data_source": data_source,
            "idx": idx,
        }


def resize_image_short_side(image, short_side_size):
    """Resize image short side to the specified value, preserving aspect ratio.

    Args:
        image: PIL.Image object
        short_side_size: target short side size

    Returns:
        resized_image: resized image
        scale_factor: scale factor (resized / original)
    """
    w, h = image.size
    if w <= h:
        # Width is the short side
        new_w = short_side_size
        scale_factor = new_w / w
        new_h = int(h * scale_factor)
    else:
        # Height is the short side
        new_h = short_side_size
        scale_factor = new_h / h
        new_w = int(w * scale_factor)

    resized_image = image.resize((new_w, new_h), Image.BILINEAR)
    return resized_image, scale_factor


def parse_bbox_with_labels(text):
    """Parse <ref>category</ref><box>...</box> format.
    Supports two formats:
    1. Point coordinates: <box><x><y></box> (2 values)
    2. Box coordinates: <box><x1><y1><x2><y2></box> (4 values)
    Returns [(category, coords, is_point), ...]
    where is_point=True for point coordinates, is_point=False for box coordinates.
    """
    results = []
    ref_pattern = r'<ref>([^<]+)</ref>((?:<box>.*?</box>)+)'
    
    point_pattern = (
        r'<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*'
        r'<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>'
    )
    
    box_pattern = (
        r'<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*'
        r'<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*'
        r'<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*'
        r'<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>'
    )

    ref_matches = re.findall(ref_pattern, text)
    for category, boxes_str in ref_matches:
        point_matches = re.findall(point_pattern, boxes_str)
        for match in point_matches:
            try:
                x, y = map(float, match)
                if 0 <= x <= 10000 and 0 <= y <= 10000:
                    results.append((category, [x, y], True))
            except Exception:
                continue
        
        box_matches = re.findall(box_pattern, boxes_str)
        for match in box_matches:
            try:
                x1, y1, x2, y2 = map(float, match)
                if 0 <= x1 <= 10000 and 0 <= y1 <= 10000 and 0 <= x2 <= 10000 and 0 <= y2 <= 10000:
                    results.append((category, [x1, y1, x2, y2], False))
            except Exception:
                continue
    return results


def convert_normalized_point_to_absolute(nor_point, img_size):
    """Convert normalized point coordinates (0-1000) to absolute coordinates.

    Args:
        nor_point: [x, y] in range 0-1000
        img_size: (w, h)

    Returns:
        [x, y] in absolute coordinates
    """
    w, h = img_size
    x, y = nor_point
    x = x * w / 1000
    y = y * h / 1000
    # Clamp coordinates to valid range
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    return [x, y]


def convert_normalized_bbox_to_absolute(nor_bbox, img_size):
    """Convert normalized coordinates (0-1000) to absolute coordinates.

    Args:
        nor_bbox: [x1, y1, x2, y2] in range 0-1000
        img_size: (w, h)

    Returns:
        [x1, y1, x2, y2] in absolute coordinates (xyxy format)
    """
    w, h = img_size
    x1, y1, x2, y2 = nor_bbox
    x1 = x1 * w / 1000
    y1 = y1 * h / 1000
    x2 = x2 * w / 1000
    y2 = y2 * h / 1000
    # Clamp coordinates to valid range
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    # Ensure x1 < x2, y1 < y2
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1, x2)
    y_max = max(y1, y2)
    return [x_min, y_min, x_max, y_max]


def parse_prediction(text, w, h):
    """Parse model output and extract points/boxes grouped by category.

    Example input format:
    Point coordinates: <ref>person</ref><box><100><200></box>
    Box coordinates: <ref>car</ref><box><150><250><350><450></box>

    Returns:
    {
        'person': [[x, y], ...],  # point coordinates
        'car': [[x1,y1,x2,y2], ...],  # box coordinates
        ...
    }
    Note: point coordinates have length 2, box coordinates have length 4.
    """
    result = {}

    parsed_items = parse_bbox_with_labels(text)

    for category, coords, is_point in parsed_items:
        if category not in result:
            result[category] = []
        
        if is_point:
            abs_point = convert_normalized_point_to_absolute(coords, (w, h))
            result[category].append(abs_point)
        else:
            abs_bbox = convert_normalized_bbox_to_absolute(coords, (w, h))
            result[category].append(abs_bbox)

    return result


def load_test_data(test_jsonl_path):
    """
    Load test data from JSONL file.

    Returns:
    List of test entries, each containing:
    {
        'file_name': str,
        'question': str,
        'answer': dict/list,
        'data_type': str,
        'data_source': str,
    }
    """
    test_data = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    test_data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")

    return test_data


def main():
    args = get_args()

    # Set up distributed environment
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if local_rank >= 0 else "cuda"

    if is_main_process():
        print(f"=== DDP Inference Configuration ===")
        print(f"World Size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local Rank: {local_rank}")
        print(f"Device: {device}")
        print(f"Model Path: {args.model_path}")
        print(f"Generation Mode: {args.generation_mode}")
        print(f"Test JSONL Path: {args.test_jsonl_path}")
        print(f"Save Path: {args.save_path}")
        print(f"=====================")

    # Create save directory
    if is_main_process():
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    # Wait for main process to create directory
    if dist.is_initialized():
        dist.barrier()

    # Initialize model
    if is_main_process():
        print(f"Loading model from: {args.model_path}")
    worker = LocateAnythingWorker(args.model_path, device=device, generation_mode=args.generation_mode)

    # Load test data
    if is_main_process():
        print(f"Loading test data from: {args.test_jsonl_path}")
    test_data = load_test_data(args.test_jsonl_path)

    if is_main_process():
        print(f"Loaded {len(test_data)} test entries")

    # Create dataset and distributed sampler
    dataset = ScreenSpotProDataset(test_data, args.image_root_dir)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    # DataLoader with batch_size=1 for single-sample inference
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )

    if is_main_process():
        print(f"Rank {rank}: Processing {len(dataloader)} samples")

    local_predictions = []

    # Progress bar (only on main process)
    iterator = tqdm(dataloader, desc=f"Rank {rank} processing", disable=not is_main_process())

    for sample in iterator:
        image_name = sample["image_name"]
        full_image_path = sample["full_image_path"]
        instruction = sample["instruction"]
        question = sample["question"]
        gt = sample["gt"]
        data_type = sample["data_type"]
        data_source = sample["data_source"]

        if not os.path.exists(full_image_path):
            print(f"[Rank {rank}] Warning: Image not found: {full_image_path}")
            continue

        try:
            image = Image.open(full_image_path).convert("RGB")
            original_w, original_h = image.size
        except Exception as e:
            print(f"[Rank {rank}] Error loading image {full_image_path}: {e}")
            continue

        # Resize image if short_side_size is set
        scale_factor = 1.0
        if args.short_side_size is not None:
            image, scale_factor = resize_image_short_side(image, args.short_side_size)
            resized_w, resized_h = image.size
            if is_main_process():
                print(f"[Resize] Original: {original_w}x{original_h} -> Resized: {resized_w}x{resized_h} (scale: {scale_factor:.4f})")
        
        # Run inference
        output = worker.generate(
            image,
            question,
            max_new_tokens=args.max_new_tokens,
        )

        if is_main_process():
            print(f"[{image_name}]")
            print(f"Question: {question}")
            print(f"Output: {output[:200]}...")
            print("-" * 50)

        try:
            # Parse with resized dimensions, then convert back
            resized_w, resized_h = image.size
            extracted_predictions = parse_prediction(
                output,
                resized_w,
                resized_h,
            )

            # Convert coordinates back to original image size if resized
            if args.short_side_size is not None and scale_factor != 1.0:
                for category in extracted_predictions:
                    for i, bbox in enumerate(extracted_predictions[category]):
                        extracted_predictions[category][i] = [
                            bbox[0] / scale_factor,
                            bbox[1] / scale_factor,
                            bbox[2] / scale_factor,
                            bbox[3] / scale_factor,
                        ]

            prediction = {
                "image_name": image_name,
                "data_type": data_type,
                "data_source": data_source,
                "instruction": instruction,
                "extracted_predictions": extracted_predictions,
                "gt": gt,
                "question": question,
                "raw_response": output,
            }

        except Exception as e:
            print(f"[Rank {rank}] Parse failed, error is {e}")
            prediction = {
                "image_name": image_name,
                "data_type": data_type,
                "data_source": data_source,
                "instruction": instruction,
                "extracted_predictions": {},
                "gt": gt,
                "question": question,
                "raw_response": output,
            }

        local_predictions.append(prediction)

    print(f"[Rank {rank}] Finished processing {len(local_predictions)} samples")

    # Each rank saves its own results to avoid all_gather timeout
    save_dir = os.path.dirname(args.save_path)
    if not save_dir:
        save_dir = "."
    base_name = os.path.basename(args.save_path)
    name_without_ext = os.path.splitext(base_name)[0]
    ext = os.path.splitext(base_name)[1] or ".jsonl"

    rank_save_path = os.path.join(save_dir, f"{name_without_ext}_rank{rank}{ext}")
    with open(rank_save_path, "w") as f:
        for prediction in local_predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(f"[Rank {rank}] Saved {len(local_predictions)} predictions to: {rank_save_path}")

    # Synchronize all processes
    if dist.is_initialized():
        dist.barrier()

    # Main process merges all results
    if is_main_process():
        print(f"\n🔄 Merging predictions from all ranks...")
        all_predictions = []
        for r in range(world_size):
            r_save_path = os.path.join(save_dir, f"{name_without_ext}_rank{r}{ext}")
            if os.path.exists(r_save_path):
                with open(r_save_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_predictions.append(json.loads(line))

        print(f"💾 Saving merged predictions to: {args.save_path}")
        with open(args.save_path, "w") as f:
            for prediction in all_predictions:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        print(f"✅ Saved {len(all_predictions)} predictions!")

        for r in range(world_size):
            r_save_path = os.path.join(save_dir, f"{name_without_ext}_rank{r}{ext}")
            if os.path.exists(r_save_path):
                os.remove(r_save_path)
                print(f"🗑️ Cleaned up temporary file: {r_save_path}")

    # Clean up distributed environment
    cleanup_distributed()


if __name__ == "__main__":
    main()
