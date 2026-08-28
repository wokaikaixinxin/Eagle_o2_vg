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
#
# This file is adopted from the InternVL project
# (https://github.com/OpenGVLab/InternVL), licensed under the MIT License.
# 
# Modifications © 2025 NVIDIA CORPORATION & AFFILIATES, licensed under
# the Apache License, Version 2.0.
#
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License
# --------------------------------------------------------

import base64
import json
from io import BytesIO

import requests
from PIL import Image


def get_model_list(controller_url):
    """Query controller to get available model list."""
    ret = requests.post(controller_url + '/refresh_all_workers')
    assert ret.status_code == 200
    ret = requests.post(controller_url + '/list_models')
    models = ret.json()['models']
    return models


def get_selected_worker_ip(controller_url, selected_model):
    """Ask controller for the target worker address by model name."""
    ret = requests.post(controller_url + '/get_worker_address',
            json={'model': selected_model})
    worker_addr = ret.json()['address']
    return worker_addr


def pil_image_to_base64(image):
    """Convert PIL image to base64-encoded PNG string."""
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


controller_url = 'http://127.0.0.1:10075'
model_list = get_model_list(controller_url)
print(f'Model list: {model_list}')

# Pick a model (first available or a default Eagle2.5 model name)
selected_model = model_list[0] if len(model_list) > 0 else 'Eagle2.5-8B'
worker_addr = get_selected_worker_ip(controller_url, selected_model)
print(f'model_name: {selected_model}, worker_addr: {worker_addr}')


# For multi-turn and multi-image conversations, organize data as follows:
# send_messages = [{'role': 'system', 'content': system_message}]
# send_messages.append({'role': 'user', 'content': 'question1 to image1', 'image': [pil_image_to_base64(image)]})
# send_messages.append({'role': 'assistant', 'content': 'answer1'})
# send_messages.append({'role': 'user', 'content': 'question2 to image2', 'image': [pil_image_to_base64(image)]})
# send_messages.append({'role': 'assistant', 'content': 'answer2'})
# send_messages.append({'role': 'user', 'content': 'question3 to image1 & 2', 'image': []})

image = Image.open('image1.jpg')
print(f'Loading image, size: {image.size}')
system_message = "You are a helpful assistant."
send_messages = [{'role': 'system', 'content': system_message}]
send_messages.append({'role': 'user', 'content': 'Describe this image in detail.', 'image': [pil_image_to_base64(image)]})

pload = {
    'model': selected_model,
    'prompt': send_messages,
    'temperature': 0.8,
    'top_p': 0.7,
    'max_new_tokens': 2048,
    'max_input_tiles': 12,
    'repetition_penalty': 1.0,
}
headers = {'User-Agent': 'Eagle2.5-Chat Client'}
response = requests.post(worker_addr + '/worker_generate_stream',
                         headers=headers, json=pload, stream=True, timeout=10)
for chunk in response.iter_lines(decode_unicode=True, delimiter=b'\0'):
    if chunk:
        data = json.loads(chunk)
        if data['error_code'] == 0:
            output = data['text']  # streaming output
        else:
            output = data['text'] + f" (error_code: {data['error_code']})"
# final output
print(output)
