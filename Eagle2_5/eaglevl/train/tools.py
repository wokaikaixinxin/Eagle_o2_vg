# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum, unique
from typing import  Dict, List, Sequence, Set, Union
from datasets import concatenate_datasets, interleave_datasets
import bisect
import numpy as np


import time
from transformers import TrainerCallback, Trainer, TrainingArguments
import glob
import os
from transformers.trainer_utils import get_last_checkpoint
import torch.distributed as dist
import shutil

def get_last_checkpoint_guard(folder):
    while True:
        last_checkpoint = get_last_checkpoint(folder)
        if last_checkpoint is None:
            break
        
        world_size = dist.get_world_size()
        if len(glob.glob(os.path.join(last_checkpoint, "*.pth"))) != world_size:
            # incomplete xxx.pth
            shutil.rmtree(last_checkpoint)
        else:
            break

    return last_checkpoint

class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, initial_interval_hours, save_interval_minutes):
        super().__init__()
        self.initial_interval_seconds = initial_interval_hours * 3600 - 15 * 60
        self.save_interval_seconds = save_interval_minutes * 60
        self.start_time = None
        self.first_save_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return control

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Check if the initial interval has passed
        if self.first_save_time is None and elapsed_time >= self.initial_interval_seconds:
            self.first_save_time = current_time
            control.should_save = True
        # Check if the subsequent save interval has passed
        elif self.first_save_time is not None and (current_time - self.first_save_time) >= self.save_interval_seconds:
            self.first_save_time = current_time
            control.should_save = True

        return control


import os
import torch
from transformers import TrainerCallback
import torch.distributed as dist
from pynvml import (
    nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetComputeRunningProcesses_v2, nvmlShutdown,
    NVML_TEMPERATURE_GPU, nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature
)

MB = 1024 ** 2

class MemoryLoggerCallback(TrainerCallback):
    def __init__(self):
        nvmlInit()
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.device_id = torch.cuda.current_device()
        self.pid = os.getpid()

        # 建议：训练开始时清零峰值统计
        torch.cuda.reset_peak_memory_stats(self.device_id)

    def log_gpu_info(self, step):
        # 先同步，避免异步内核导致读数偏小
        torch.cuda.synchronize(self.device_id)

        # -------- PyTorch 口径（本进程）--------
        alloc = torch.cuda.memory_allocated(self.device_id)            # 张量实际占用
        reserv = torch.cuda.memory_reserved(self.device_id)            # 预留（缓存）
        max_alloc = torch.cuda.max_memory_allocated(self.device_id)    # 运行以来峰值
        max_reserv = torch.cuda.max_memory_reserved(self.device_id)

        # -------- CUDA 驱动口径（当前设备可用/总量）--------
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device_id)
        used_cuda = total_bytes - free_bytes

        # -------- NVML 口径（设备整卡 + 本进程）--------
        handle = nvmlDeviceGetHandleByIndex(self.device_id)
        mem_info = nvmlDeviceGetMemoryInfo(handle)  # 整卡 used/free/total（所有进程合计）

        # 本进程在 NVML 里的用量（可能因权限/MIG返回不可用）
        proc_used = None
        try:
            procs = nvmlDeviceGetComputeRunningProcesses_v2(handle)
            for p in procs:
                if getattr(p, "pid", None) == self.pid:
                    proc_used = getattr(p, "usedGpuMemory", None)  # bytes
                    break
        except Exception:
            pass

        temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        power_w = nvmlDeviceGetPowerUsage(handle) / 1000.0

        print(
            f"[Step {step} | Rank {self.rank} / GPU {self.device_id}] "
            f"PT alloc={alloc/MB:.2f}MB (peak {max_alloc/MB:.2f}MB), "
            f"PT reserved={reserv/MB:.2f}MB (peak {max_reserv/MB:.2f}MB), "
            f"CUDA used={used_cuda/MB:.2f}MB / total={total_bytes/MB:.0f}MB, "
            f"NVML device used={mem_info.used/MB:.2f}MB"
            + (f", NVML proc used={proc_used/MB:.2f}MB" if proc_used else "")
            + f", Temp={temperature}°C, Power={power_w:.1f}W"
        )

    def on_step_end(self, args, state, control, **kwargs):
        # 只在少数 rank 打印，避免刷屏
        if self.rank % 32 == 0:
            self.log_gpu_info(state.global_step)

    def __del__(self):
        try:
            nvmlShutdown()
        except Exception:
            pass



