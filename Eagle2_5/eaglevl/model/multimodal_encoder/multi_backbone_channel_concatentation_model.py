# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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

import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Optional, Tuple, Union

from .multi_backbone_channel_concatenation_encoder import MultiBackboneChannelConcatenationVisionTower
from .configuration_multi_backbone_channel_concatentation_model import MultiBackboneChannelConcatenationVisionModelConfig


class MultiBackboneChannelConcatenationVisionModel(nn.Module):

    """
    A vision model wrapper that concatenates channels from multiple backbones.

    Args:
        config (MultiBackboneChannelConcatenationVisionModelConfig): The configuration for the model.

    Attributes:
        vision_model (MultiBackboneChannelConcatenationVisionTower): The vision tower that performs the channel concatenation.

    Notes:
        **The class is not inherited from the PreTrainedModel in transformers**

    """

    config_class = MultiBackboneChannelConcatenationVisionModelConfig
    main_input_name = "pixel_values"
    
    def __init__(self, config: MultiBackboneChannelConcatenationVisionModelConfig, raw_config=None):
        super().__init__()
        self.vision_model = MultiBackboneChannelConcatenationVisionTower(
            vision_tower=config.vision_tower,
            args=config,
            grid_size=config.grid_size,
            convnext_img_size=config.convnext_img_size, 
            radio_img_size=config.radio_img_size,
            siglip_img_size=config.siglip_img_size,
            normalize_type=config.normalize_type,
            raw_config=raw_config
        )


    def get_input_embeddings(self):
        # You might need to adjust this depending on how you want to handle input embeddings
        return self.vision_model.vision_towers[0].get_input_embeddings()

    def forward(
        self,
        pixel_values,
        return_dict: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert return_dict is True, "We only support return_dict"
        assert output_hidden_states is False, "We do not support output_hidden_states"
        features = self.vision_model(pixel_values)

        # We only supports features as model outputs
        return BaseModelOutputWithPooling(
            last_hidden_state=features,
            pooler_output=None,
            hidden_states=None,
            attentions=None,
        )

    @property
    def dummy_feature(self):
        return self.vision_model.dummy_feature

    @property
    def dtype(self):
        return self.vision_model.dtype

    @property
    def device(self):
        return self.vision_model.device

    @property
    def config(self):
        return self.vision_model.config

    @property
    def hidden_size(self):
        return self.vision_model.hidden_size

    @property
    def num_patches(self):
        return self.vision_model.num_patches
