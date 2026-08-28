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

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import json
import os
import threading
import time
import uuid
from functools import partial
from io import BytesIO
from threading import Thread
import math
import requests
import torch
import torchvision.transforms as T
import uvicorn
from constants import  WORKER_HEART_BEAT_INTERVAL
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import StreamingResponse
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer, AutoConfig, AutoProcessor
from utils import build_logger, pretty_print_semaphore, server_error_msg

worker_id = str(uuid.uuid4())[:6]
logger = build_logger('model_worker', f'model_worker_{worker_id}.log')
global_counter = 0
model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()

def split_model(model_path, device):

    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers

    print('world_size', world_size)
    num_layers_per_gpu_ = math.floor(num_layers / (world_size - 1))
    num_layers_per_gpu = [num_layers_per_gpu_] * world_size
    num_layers_per_gpu[device] = num_layers - num_layers_per_gpu_ * (world_size-1)
    print(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = device
    device_map['mlp1'] = device
    device_map['language_model.model.tok_embeddings'] = device
    device_map['language_model.model.embed_tokens'] = device
    device_map['language_model.output'] = device
    device_map['language_model.model.norm'] = device
    device_map['language_model.lm_head'] = device
    device_map['language_model.model.rotary_emb'] = device
    device_map[f'language_model.model.layers.{num_layers - 1}'] = device
    return device_map

class ModelWorker:
    def __init__(self, controller_addr, worker_addr, worker_id, model_path, model_name,
                 load_8bit, device):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith('/'):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split('/')
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + '_' + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        logger.info(f'Loading the model {self.model_name} on worker {worker_id} ...')

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        tokens_to_keep = ['<box>', '</box>', '<ref>', '</ref>']
        tokenizer.additional_special_tokens = [item for item in tokenizer.additional_special_tokens if item not in tokens_to_keep]
        self.tokenizer = tokenizer
        self.device = torch.cuda.current_device()

        if any(x in model_path.lower() for x in ['70b', '72b', '40b', '34b', '30b']):
            device_map = split_model(model_path, self.device)
        else:
            device_map = None
            device_map = None
        
        if device_map is not None:    
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True,
                                               device_map=device_map, 
                                               trust_remote_code=True,
                                               attn_implementation='flash_attention_2',
                                               load_in_8bit=load_8bit).eval()
        else:
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True,
                                               attn_implementation='flash_attention_2',
                                               load_in_8bit=load_8bit).eval()  

        print(device_map, load_8bit)
        if not load_8bit and device_map is None:
            self.model = self.model.to(device)
        self.load_8bit = load_8bit
        
        self.model_path = model_path
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker, args=(self,))
        self.heart_beat_thread.start()

    def reload_model(self):
        del self.model
        torch.cuda.empty_cache()
        if self.device == 'auto':
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            # This can make distributed deployment work properly
            self.model = AutoModel.from_pretrained(
                self.model_path,
                load_in_8bit=self.load_8bit,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                trust_remote_code=True).eval()
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                load_in_8bit=self.load_8bit,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True).eval()
        if not self.load_8bit and not self.device == 'auto':
            self.model = self.model.cuda()

    def register_to_controller(self):
        logger.info('Register to controller')

        url = self.controller_addr + '/register_worker'
        data = {
            'worker_name': self.worker_addr,
            'check_heart_beat': True,
            'worker_status': self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f'Send heart beat. Models: {[self.model_name]}. '
                    f'Semaphore: {pretty_print_semaphore(model_semaphore)}. '
                    f'global_counter: {global_counter}')

        url = self.controller_addr + '/receive_heart_beat'

        while True:
            try:
                ret = requests.post(url, json={
                    'worker_name': self.worker_addr,
                    'queue_length': self.get_queue_length()}, timeout=5)
                exist = ret.json()['exist']
                break
            except requests.exceptions.RequestException as e:
                logger.error(f'heart beat error: {e}')
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            'model_names': [self.model_name],
            'speed': 1,
            'queue_length': self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        # system_message = params['prompt']
        send_messages = params['prompt']
        max_input_tiles = params['max_input_tiles']
        temperature = params['temperature']
        top_p = params['top_p']
        max_new_tokens = params['max_new_tokens']
        repetition_penalty = params['repetition_penalty']
        do_sample = True if temperature > 0.0 else False
        # print('send_messages:', send_messages)

        for i in range(len(send_messages)):
            if send_messages[i]['role'] == 'user':
                images = send_messages[i]['image']
                if len(images) > 0:
                    content = []
                    txt = {'type': 'text', 'text': send_messages[i]['content']}
                    for image in images:
                        print('image:', type(image))
                        content.append({'type': 'image', 'image': 'data:image/jpeg;base64,' + image})
                    content.append(txt)
                    send_messages[i]['content'] = content
        if send_messages[-1]['role'] == 'assistant':
            send_messages = send_messages[:-1]
        text_list = [self.processor.apply_chat_template(
            send_messages, tokenize=False, add_generation_prompt=True
        )]
        print('text_list:', text_list)
        image_inputs, video_inputs = self.processor.process_vision_info(send_messages)
        inputs = self.processor(text = text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True, images_kwargs={'max_dynamic_tiles': max_input_tiles})
        inputs = inputs.to("cuda")
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ''
        for new_text in streamer:
            generated_text += new_text
            yield json.dumps({'text': generated_text, 'error_code': 0}).encode() + b'\0'

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print('Caught ValueError:', e)
            ret = {
                'text': server_error_msg,
                'error_code': 1,
            }
            yield json.dumps(ret).encode() + b'\0'
        except torch.cuda.CudaError as e:
            print('Caught torch.cuda.CudaError:', e)
            ret = {
                'text': server_error_msg,
                'error_code': 1,
            }
            yield json.dumps(ret).encode() + b'\0'
        except Exception as e:
            print('Caught Unknown Error', e)
            ret = {
                'text': server_error_msg,
                'error_code': 1,
            }
            yield json.dumps(ret).encode() + b'\0'


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post('/worker_generate_stream')
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post('/worker_get_status')
async def get_status(request: Request):
    return worker.get_status()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=6210)
    parser.add_argument('--worker-address', type=str, default='http://localhost:6210')
    parser.add_argument('--controller-address', type=str, default='http://localhost:10075')
    parser.add_argument('--model-path', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--limit-model-concurrency', type=int, default=5)
    parser.add_argument('--stream-interval', type=int, default=1)
    parser.add_argument('--load-8bit', action='store_true')
    args = parser.parse_args()
    logger.info(f'args: {args}')

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.model_path,
                         args.model_name,
                         args.load_8bit,
                         args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
   

