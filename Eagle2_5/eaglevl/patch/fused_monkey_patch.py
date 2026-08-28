# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from .fused_ops.fused_rms_norm import LigerRMSNorm
from .fused_ops.fused_rotary_pos_emb import liger_rotary_pos_emb
from .fused_ops.fused_swiglu import LigerSwiGLUMLP


def replace_liger_fused_ops():
    from transformers.models.qwen2 import modeling_qwen2
    modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP
    modeling_qwen2.Qwen2RMSNorm = LigerRMSNorm
    modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb
    
    from transformers.models.llama import modeling_llama
    modeling_llama.LlamaMLP = LigerSwiGLUMLP
    modeling_llama.LlamaRMSNorm = LigerRMSNorm

    from transformers.models.qwen3 import modeling_qwen3
    modeling_qwen3.Qwen3MLP = LigerSwiGLUMLP
    modeling_qwen3.Qwen3RMSNorm = LigerRMSNorm
    modeling_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb
    
    



