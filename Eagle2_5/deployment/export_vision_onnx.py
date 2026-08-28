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

import argparse
from transformers import AutoModel, AutoConfig
from eaglevl.model.eagle2_5.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
import torch
import time
import numpy as np
import onnx
import tensorrt as trt
from torch import nn

def parse_arguments():
    parser = argparse.ArgumentParser(description='Export ONNX file and TensorRT Engine for vision part of Eagle2')
    parser.add_argument("--config_path", type=str, default="/workspace/Eagle2.5-8B")
    parser.add_argument("--onnx_file", type=str, default="test.onnx")
    parser.add_argument("--model_path", type=str,default="/workspace/Eagle2.5-8B")
    parser.add_argument("--trt_engine", type=str, default=None)
    parser.add_argument("--minBS", type=int, default=1)
    parser.add_argument("--optBS", type=int, default=16)
    parser.add_argument("--maxBS", type=int, default=32)
    args = parser.parse_args()
    return args

def generate_trt_engine(onnxFile,
                            planFile,
                            minBS=1,
                            optBS=7,
                            maxBS=32,
                            input_shape=[448, 448]):
        print("Start converting TRT engine!")
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        parser = trt.OnnxParser(network, logger)

        with open(onnxFile, "rb") as model:
            if not parser.parse(model.read(), "/".join(onnxFile.split("/"))):
                print("Failed parsing %s" % onnxFile)
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print("Succeeded parsing %s" % onnxFile)

        nBS = -1
        nMinBS = minBS
        nOptBS = optBS
        nMaxBS = maxBS
        inputT = network.get_input(0)
        inputT.shape = [nBS, 3, input_shape[0], input_shape[1]]
        profile.set_shape(
            inputT.name,
            [nMinBS, 3, input_shape[0], input_shape[1]],
            [nOptBS, 3, input_shape[0], input_shape[1]],
            [nMaxBS, 3, input_shape[0], input_shape[1]],
        )

        config.add_optimization_profile(profile)

        t0 = time.time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time.time()
        if engineString == None:
            print("Failed building %s" % planFile)
        else:
            print("Succeeded building %s in %d s" % (planFile, t1 - t0))
            with open(planFile, "wb") as f:
                f.write(engineString)

class FeatureExtractorWrapper(nn.Module):
    def __init__(self, eagle2_5_model, mlp_checkpoint=False):
        super().__init__()
        self.vision_model = eagle2_5_model.vision_model.eval()
        self.mlp1 = eagle2_5_model.mlp1.eval()
        self.select_layer = eagle2_5_model.select_layer
        self.downsample_ratio = eagle2_5_model.downsample_ratio
        self.use_pixel_shuffle = getattr(eagle2_5_model, "use_pixel_shuffle", True)


    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))

        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
            if hasattr(vit_embeds, 'last_hidden_state'):
                vit_embeds = vit_embeds.last_hidden_state
        else:
            vit_embeds = self.vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
            vit_embeds = vit_embeds.hidden_states[self.select_layer]
        if self.use_pixel_shuffle:
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)  # Skip checkpoint for ONNX

        return vit_embeds

def main():
    args = parse_arguments()
    # Load pretrained model/config from huggingface checkpoints
    device = torch.cuda.current_device()
    model = Eagle2_5_VLForConditionalGeneration.from_pretrained(
                                            args.model_path, 
                                            trust_remote_code=True, 
                                            torch_dtype=torch.bfloat16
                                        ).eval().to(device)
    config = AutoConfig.from_pretrained(args.config_path, 
                                        trust_remote_code=True, 
                                        torch_dtype=torch.bfloat16
                                        )

    vit_input_shape = config.vision_config.image_size
    model.vision_model.config._attn_implementation = 'eager'
    model._supports_flash_attn_2 = False
    
    # Define inputs, input_names and output_names for ONNX file
    inputs = torch.randn(1, 3, vit_input_shape, vit_input_shape).to(device)
    input_args = [
            inputs
        ]
    input_names = [
            "pixel_values", 
        ]
    output_names = [
            "vit_embeds",
        ]

    # Vision model for Eagle 2.5
    model_for_onnx = FeatureExtractorWrapper(model)

    # Export ONNX file for vision part
    torch.onnx.export(
        model_for_onnx.float(), tuple(input_args),
        args.onnx_file,
        input_names=input_names, output_names=output_names,
        do_constant_folding=True,
        opset_version=17,
        verbose=False,
        dynamic_axes={"pixel_values": {
                0: "batch"
            }},
        )

    # Build TensorRT engine for vision part
    if args.trt_engine is not None:
        generate_trt_engine(
                        args.onnx_file, args.trt_engine, 
                        args.minBS, args.optBS, args.maxBS, 
                        input_shape=[vit_input_shape, vit_input_shape]
                    )


if __name__ == '__main__':
    main()

