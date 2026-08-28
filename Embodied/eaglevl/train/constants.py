# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'
INTERVAL_START_TOKEN = '<interval>'
INTERVAL_END_TOKEN = '</interval>'
TEXT_MASK_TOKEN = '<text_mask>'
NULL_TOKEN = '<null>'
SEP_TOKEN = '</c>'

special_tokens_list = [
    IMG_CONTEXT_TOKEN,
    IMG_START_TOKEN, IMG_END_TOKEN,
    BOX_START_TOKEN, BOX_END_TOKEN,
    QUAD_START_TOKEN, QUAD_END_TOKEN,
    REF_START_TOKEN, REF_END_TOKEN,
    INTERVAL_START_TOKEN, INTERVAL_END_TOKEN,
    TEXT_MASK_TOKEN,
    NULL_TOKEN,
    SEP_TOKEN,
] 

number_tokens_list = [f'<{i}>' for i in range(1001)]
