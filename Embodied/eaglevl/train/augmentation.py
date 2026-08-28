# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# -*- coding: utf-8 -*-
"""
Image augmentation module for training-time data augmentation.
"""

import random
from PIL import Image
from typing import Union, List


def resize_image_keep_aspect_ratio(
    image: Image.Image, 
    target_long_edge: int
) -> Image.Image:
    """
    Resize the image to the specified long edge length while preserving aspect ratio.
    
    Args:
        image: PIL Image
        target_long_edge: Target long edge length
    
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    long_edge = max(width, height)
    
    if long_edge == target_long_edge:
        return image
    
    scale = target_long_edge / long_edge
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return image.resize((new_width, new_height), Image.LANCZOS)


def apply_resize_augmentation(
    image: Image.Image,
    data_augment: bool = True,
    min_long_edge: int = 640,
    max_long_edge: int = 2048,
    augment_prob: float = 0.5
) -> Image.Image:
    """
    Apply resize augmentation.
    
    Rules:
    - If data_augment=True, no processing is done and the original image is returned.
    - If data_augment=False, with augment_prob probability the original image is kept,
      and with (1-augment_prob) probability the long edge is resized to a random value
      in [min_long_edge, max_long_edge].
    
    Args:
        image: PIL Image
        data_augment: The data_augment value from dataset config
        min_long_edge: Minimum long edge length
        max_long_edge: Maximum long edge length
        augment_prob: Probability of keeping the original image (default 50%)
    
    Returns:
        Processed PIL Image
    """
    if not data_augment:
        return image
    
    if random.random() < augment_prob:
        return image
    
    width, height = image.size
    current_long_edge = max(width, height)
    
    target_long_edge = random.randint(min_long_edge, max_long_edge)
    
    if target_long_edge == current_long_edge:
        return image
    
    return resize_image_keep_aspect_ratio(image, target_long_edge)


def apply_resize_augmentation_to_list(
    images: List[Image.Image],
    data_augment: bool = True,
    min_long_edge: int = 640,
    max_long_edge: int = 2048,
    augment_prob: float = 0.5
) -> List[Image.Image]:
    """
    Apply resize augmentation to a list of images.
    
    Note: Each image independently decides whether to augment and the target size.
    
    Args:
        images: List of PIL Images
        data_augment: The data_augment value from dataset config
        min_long_edge: Minimum long edge length
        max_long_edge: Maximum long edge length
        augment_prob: Probability of keeping the original image
    
    Returns:
        List of processed PIL Images
    """
    return [
        apply_resize_augmentation(
            img, 
            data_augment=data_augment,
            min_long_edge=min_long_edge,
            max_long_edge=max_long_edge,
            augment_prob=augment_prob
        )
        for img in images
    ]
