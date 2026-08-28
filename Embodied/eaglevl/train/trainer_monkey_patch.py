# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from functools import partial
import json
import os

import torch
import torch.nn as nn
import transformers
from transformers import Trainer, logging
from transformers.trainer import is_sagemaker_mp_enabled
from transformers.trainer_callback import TrainerCallback  # noqa: E402

# Fix: import torch.distributed as dist
import torch.distributed as dist

logger = logging.get_logger(__name__)


def create_optimizer_various_lr(self):
    """
    Setup the optimizer with customizable learning rate scales for different model components.
    `lr_scale` is a string that defines the scale of learning rates for different components 
    in the form of "vision_model: 0.1, mlp: 1.0, llm: 1.0".
    """
    opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
    # if self.optimizer is None:
    if True:
        base_lr = self.args.learning_rate
        
        # Parse lr_scale and apply the scaling factors
        lr_scale = dict()
        if hasattr(self.args, 'lr_scale') and self.args.lr_scale:
            for pair in self.args.lr_scale.split(','):
                k, v = pair.split(':')
                lr_scale[k.strip()] = float(v.strip())
        else:
            raise ValueError(f"Please provide lr_scale in the form of 'vision_model: 0.1, mlp: 1.0, llm: 1.0' {self.args.lr_scale}")
        
        # Set default scaling to 1.0 if not provided in the lr_scale
        llm_scale = lr_scale.get('llm', 1.0)
        vit_scale = lr_scale.get('vision_model', 1.0)
        mlp_scale = lr_scale.get('mlp', 1.0)

        # Iterate through named parameters and group them based on the keyword in the name
        optimizer_grouped_parameters = []
        param_to_name = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_to_name[param] = name
                if 'vision_model' in name:
                    lr = base_lr * vit_scale
                    optimizer_grouped_parameters.append({"params": [param], "lr": lr})
                elif 'language_model' in name:
                    lr = base_lr * llm_scale
                    optimizer_grouped_parameters.append({"params": [param], "lr": lr})
                else:
                    lr = base_lr * mlp_scale
                    optimizer_grouped_parameters.append({"params": [param], "lr": lr})
                
                print(f"Parameter: {name}, Learning Rate: {lr}")
        # Add weight decay for parameters based on decay conditions if necessary
        decay_parameters = self.get_decay_parameter_names(self.model)
        for group in optimizer_grouped_parameters:
            group['weight_decay'] = self.args.weight_decay if any(p.requires_grad and param_to_name.get(p, '') in decay_parameters for p in group['params']) else 0.0
        
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        if optimizer_cls.__name__ == "Adam8bit":
            import bitsandbytes
            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    logger.info(f"skipped {module}: {skipped/2**20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            logger.info(f"skipped: {skipped/2**20}M params")
    
    if is_sagemaker_mp_enabled():
        self.optimizer = smp.DistributedOptimizer(self.optimizer)
    
    # Track learning rate for each parameter
    param_to_lr = {}
    for group in optimizer_grouped_parameters:
        lr = group.get('lr', None)
        for param in group['params']:
            param_to_lr[param] = lr

    # Print trainable parameters and their learning rates
    # Fix: dist was undefined, need to import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("=" * 40)
        logger.info("{:<40} {:<20} {:<10}".format("Parameter Name", "Size", "Learning Rate"))
        logger.info("=" * 40)

        # Print parameter names, sizes, and associated learning rates
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                lr = param_to_lr.get(param, "N/A")
                logger.info(f"{name:<40} {str(param.shape):<20} {lr:<10}")
        
        logger.info("=" * 40)

    return self.optimizer

def replace_create_optimizer_with_various_lr():
    print('Replace original create_optimizer with custom create_optimizer')
    transformers.Trainer.create_optimizer = create_optimizer_various_lr
