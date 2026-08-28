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

import argparse
import base64
import datetime
import hashlib
import json
import os
import random
import re
import sys
import traceback
# from streamlit_js_eval import streamlit_js_eval
from functools import partial
from io import BytesIO

import cv2
import numpy as np
import requests
import streamlit as st
import decord
from constants import LOGDIR, server_error_msg
from library import Library
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_select import image_select

custom_args = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--controller_url', type=str, default='http://127.0.0.1:10075', help='url of the controller')
parser.add_argument('--sd_worker_url', type=str, default='http://0.0.0.0:40006', help='url of the stable diffusion worker')
parser.add_argument('--max_image_limit', type=int, default=128, help='maximum number of images')
args = parser.parse_args(custom_args)
controller_url = args.controller_url
sd_worker_url = args.sd_worker_url
max_image_limit = args.max_image_limit
print('args:', args)


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f'{t.year}-{t.month:02d}-{t.day:02d}-conv.json')
    return name



def get_model_list():
    ret = requests.post(controller_url + '/refresh_all_workers')
    assert ret.status_code == 200
    ret = requests.post(controller_url + '/list_models')
    models = ret.json()['models']
    print('models:', models)
    return models


def extract_frames_by_number(video_path, num_frames):
    """Extract frames by specifying number of frames"""
    video = decord.VideoReader(video_path)
    total_frames = len(video)
    num_frames = min(num_frames, total_frames)
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frames = video.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames]
    # return [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

def extract_frames_by_interval(video_path, fps):
    """Extract frames by specifying interval/fps"""
    video = decord.VideoReader(video_path)
    video_fps = video.get_avg_fps()
    sample_fps = min(fps, video_fps)
    interval = int(video_fps / sample_fps)
    frame_indices = range(0, len(video), interval)
    frames = video.get_batch(list(frame_indices)).asnumpy()
    return [Image.fromarray(frame) for frame in frames]
    # return [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

def load_upload_file_and_show():
    video_count = 0
    if uploaded_files is not None:
        images, filenames = [], []
        for file in uploaded_files:
            if file.name.endswith('.mp4'):
                # Handle MP4 video files
                video_bytes = file.read()
                temp_file = f"temp_{file.name}"
                with open(temp_file, "wb") as f:
                    f.write(video_bytes)
                
                # Default strategy: sample 8 frames
                if 'frame_number' in st.session_state:
                    frames = extract_frames_by_number(temp_file, st.session_state.frame_number)
                elif 'frame_interval' in st.session_state:
                    frames = extract_frames_by_interval(temp_file, st.session_state.frame_interval)
                else:
                    frames = extract_frames_by_number(temp_file, 8)
                
                images.extend(frames)
                os.remove(temp_file)
                video_count += 1
            else:
                # Handle image files
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                images.append(img)
                
        with upload_image_preview.container():
            Library(images)

        image_hashes = [hashlib.md5(image.tobytes()).hexdigest() for image in images]
        for image, hash in zip(images, image_hashes):
            t = datetime.datetime.now()
            filename = os.path.join(LOGDIR, 'serve_images', f'{t.year}-{t.month:02d}-{t.day:02d}', f'{hash}.jpg')
            filenames.append(filename)
            if not os.path.isfile(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                image.save(filename)
    return images, filenames, video_count


def get_selected_worker_ip():
    ret = requests.post(controller_url + '/get_worker_address',
            json={'model': selected_model})
    worker_addr = ret.json()['address']
    return worker_addr


def save_chat_history():
    messages = st.session_state.messages
    new_messages = []
    for message in messages:
        new_message = {'role': message['role'], 'content': message['content']}
        if 'filenames' in message:
            new_message['filenames'] = message['filenames']
        new_messages.append(new_message)
    if len(new_messages) > 0:
        fout = open(get_conv_log_filename(), 'a')
        data = {
            'type': 'chat',
            'model': selected_model,
            'messages': new_messages,
        }
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')
        fout.close()


def generate_response(messages, default_response=None):
    send_messages = [{'role': 'system', 'content': persona_rec}]
    for message in messages:
        if message['role'] == 'user':
            user_message = {'role': 'user', 'content': message['content']}
            if 'image' in message and len('image') > 0:
                user_message['image'] = []
                for image in message['image']:
                    user_message['image'].append(pil_image_to_base64(image))
            send_messages.append(user_message)
        else:
            send_messages.append({'role': 'assistant', 'content': message['content']})
    pload = {
        'model': selected_model,
        'prompt': send_messages,
        'temperature': float(temperature),
        'top_p': float(top_p),
        'max_new_tokens': max_length,
        'max_input_tiles': max_input_tiles,
        'repetition_penalty': float(repetition_penalty),
    }
    worker_addr = get_selected_worker_ip()
    headers = {'User-Agent': 'InternVL-Chat Client'}
    placeholder, output = st.empty(), ''
    try:
        if default_response is None:    
            print('worker_addr:', worker_addr)
            response = requests.post(worker_addr + '/worker_generate_stream',
                                 headers=headers, json=pload, stream=True, timeout=10)
            # print('response', response)
        else:
            response = default_response
            placeholder.markdown(response)
            return response
        for chunk in response.iter_lines(decode_unicode=True, delimiter=b'\0'):
            if chunk:
                data = json.loads(chunk.decode())
                if data['error_code'] == 0:
                    output = data['text']
                    # Phi3-3.8B will produce abnormal `�` output
                    if '4B' in selected_model and '�' in output[-2:]:
                        output = output.replace('�', '')
                        break
                    placeholder.markdown(output + '▌')
                else:
                    output = data['text'] + f" (error_code: {data['error_code']})"
                    placeholder.markdown(output)
        placeholder.markdown(output)
    except requests.exceptions.RequestException as e:
        traceback.print_exc()
        placeholder.markdown(server_error_msg)
    return output


def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def clear_chat_history():
    st.session_state.messages = []
    st.session_state['image_select'] = -1


def clear_file_uploader():
    st.session_state.uploader_key += 1
    st.rerun()


def combined_func(func_list):
    for func in func_list:
        func()


def show_one_or_multiple_images(message, total_image_num, is_input=True):
    if 'image' in message:
        if is_input:
            total_image_num = total_image_num + len(message['image'])
            if len(message['image']) == 1 and total_image_num == 1:
                label = f"(In this conversation, {len(message['image'])} image was uploaded, {total_image_num} image in total)"
            elif len(message['image']) == 1 and total_image_num > 1:
                label = f"(In this conversation, {len(message['image'])} image was uploaded, {total_image_num} images in total)"
            else:
                label = f"(In this conversation, {len(message['image'])} images were uploaded, {total_image_num} images in total)"
            
        upload_image_preview = st.empty()
        with upload_image_preview.container():
            Library(message['image'])
        if is_input and len(message['image']) > 0:
            st.markdown(label)


def is_valid_bounding_box(input_string):
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, input_string, re.DOTALL)
    print('matches:', matches)
    res_list = []
    try:
        matches = [each.strip(' \n').strip('```json\n').strip('\n```') for each in matches]
        for each in matches:
            print('each:', each)

        matches = [json.loads(each) for each in matches]
        for m in matches:
            print('m:', m)
            for item in m:
                print('item:', item)
                if 'bbox_2d' in item or 'point_2d' in item:
                    res_list.append(item)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []
    return res_list


def draw_bounding_boxes(res_list):
    """
    [
        {
            'bbox_2d': [x1, y1, x2, y2],
            'label': 'label'
        },
        {
            'point_2d': [x, y],
            'label': 'label'
        }
    ]
    """
    returned_image = None
    for message in st.session_state.messages:
        if message['role'] == 'user' and 'image' in message and len(message['image']) > 0:
            last_image = message['image'][-1]
            width, height = last_image.size
            returned_image = last_image.copy()
            draw = ImageDraw.Draw(returned_image)
            break

    print('results:', res_list)

    if returned_image is None:
        return None

    for result in res_list:
        # 可调节的尺寸因子
        scale = (width * height) ** 0.5 / 1000  # 比例缩放因子
        line_width = max(1, int(scale * 2))
        point_radius = max(2, int(scale * 3))
        font_size = max(10, int(scale * 12))

        random_color = (random.randint(0, 128), random.randint(0, 128), random.randint(0, 128))
        x1, y1 = 0, 0  # 默认初始化

        if 'bbox_2d' in result:
            x1, y1, x2, y2 = [float(v) / 1000 for v in result['bbox_2d']]
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
            draw.rectangle((x1, y1, x2, y2), outline=random_color, width=line_width)

        if 'point_2d' in result:
            x1, y1 = [float(v) / 1000 for v in result['point_2d']]
            x1, y1 = int(x1 * width), int(y1 * height)
            draw.ellipse((x1 - point_radius, y1 - point_radius, x1 + point_radius, y1 + point_radius),
                         fill=random_color)

        if 'label' in result:
            label = result['label']
            try:
                font = ImageFont.truetype('static/SimHei.ttf', font_size)
            except:
                font = ImageFont.load_default()
            text_size = font.getbbox(label)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]
            text_position = (x1, max(0, y1 - text_height))  # 防止文字超出顶部
            draw.rectangle(
                [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
                fill=random_color
            )
            draw.text(text_position, label, fill='white', font=font)

    return returned_image if res_list else None

def regenerate():
    st.session_state.messages = st.session_state.messages[:-1]
    st.rerun()


logo_code = """
<svg width="1700" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color: #76B900; stop-opacity: 1" />
      <stop offset="100%" style="stop-color: #76B900; stop-opacity: 1" />
    </linearGradient>
  </defs>
  <text x="000" y="160" font-size="180" font-weight="bold" fill="url(#gradient1)" style="font-family: Arial, sans-serif;">
    Eagle2.5 Demo
  </text>
</svg>
"""

# App title
st.set_page_config(page_title='Eagle2.5')

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0


system_message_long = """You are Eagle 2.5, a cutting-edge large language model developed by NVIDIA. You are highly capable, efficient, and aligned, specialized in understanding complex multimodal inputs and providing expert-level responses across domains.

Always be concise, accurate, and helpful. You respond like a reliable co-pilot to researchers, developers, and engineers, offering deep technical insight, step-by-step reasoning, and practical suggestions.

You can interpret long contexts, follow nuanced instructions, and dynamically adjust your tone to match the user's intent. If the user does not specify a tone, default to a professional, technical, yet friendly style.

You understand you are Eagle 2.5, and may refer to yourself as such when asked.."""

system_message_short = "You are a helpful assistant."

# Replicate Credentials
with st.sidebar:
    model_list = get_model_list()
    # "[![Open in GitHub](https://github.com/codespaces/badge.svg)](https://github.com/OpenGVLab/InternVL)"
    lan='English'
    st.logo(logo_code, link='', icon_image=logo_code)
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Eagle chat model', model_list, key='selected_model',
                                          on_change=clear_chat_history,
                                          help='We depolyed these models: 1B, 2B, 9B, 34B and Video-8B. 1/2/9/34B are only trained on image data, while Video-8B (ongoing project) is trained on image/video data.')
    with st.expander('🤖 System Prompt'):
        persona_rec = st.text_area('System Prompt', value=system_message_short,
                                   help='System prompt is a pre-defined message used to instruct the assistant at the beginning of a conversation.',
                                   height=200)
    with st.expander('🔥 Advanced Options'):
        temperature = st.slider('temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        top_p = st.slider('top_p', min_value=0.0, max_value=1.0, value=0.95, step=0.05)
        repetition_penalty = st.slider('repetition_penalty', min_value=1.0, max_value=1.5, value=1.1, step=0.02)
        max_length = st.slider('max_new_token', min_value=0, max_value=4096, value=1024, step=128)
        max_input_tiles = st.slider('max_input_tiles (control image resolution)', min_value=1, max_value=12,
                                    value=12,step=1)
    with st.expander('🎥 Video Frame Options'):
        frame_selection = st.radio('Frame Selection Method', ['Default (8 frames)', 'Frame Number', 'Frame Interval'])
        if frame_selection == 'Frame Number':
            st.session_state.frame_number = st.number_input('Number of frames to extract', min_value=1, max_value=128, value=8)
            if 'frame_interval' in st.session_state:
                del st.session_state.frame_interval
        elif frame_selection == 'Frame Interval':
            st.session_state.frame_interval = st.number_input('Frames per second (fps)', min_value=1, max_value=24, value=2)
            if 'frame_number' in st.session_state:
                del st.session_state.frame_number
        else:
            if 'frame_number' in st.session_state:
                del st.session_state.frame_number
            if 'frame_interval' in st.session_state:
                del st.session_state.frame_interval
                
    upload_image_preview = st.empty()
    uploaded_files = st.file_uploader('Upload files', accept_multiple_files=True,
                                      type=['png', 'jpg', 'jpeg', 'webp', 'mp4'],
                                      help='You can upload multiple images or a single video.',
                                      key=f'uploader_{st.session_state.uploader_key}',
                                      on_change=st.rerun)
    uploaded_pil_images, save_filenames, video_count = load_upload_file_and_show()
    todo_list = st.sidebar.selectbox('Our to-do list', ['👏This is our to-do list',
                                                        '1. Support for PDF and more diverse data formats',
                                                        '2. Write a usage document'], key='todo_list',
                                     help='Here are some features we plan to support in the future.')
    
    # TODO: add a button to jump to the internal user portal
    # st.markdown("""
    #     <a href="https://www.nvidia.com" target="_blank">
    #         <button style="background-color:#76B900; color:white; border:none; padding:10px 15px; border-radius:5px;">
    #             NVIDIA Internal User Portal
    #     </button>
    #     </a>
    # """, unsafe_allow_html=True)


gradient_text_html = """
<style>
.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, #76B900, #76B900);
    background: linear-gradient(to right, #76B900, #76B900);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
}
</style>
<div class="gradient-text">Eagle2.5</div>
"""

st.markdown(gradient_text_html, unsafe_allow_html=True)
st.caption('NVIDIA\'s cutting-edge VLM')


# Store LLM generated responses
if 'messages' not in st.session_state.keys():
    clear_chat_history()

gallery_placeholder = st.empty()
with gallery_placeholder.container():
    examples = ['gallery/prod_1.jpeg']
    images = [Image.open(image).convert('RGB') for image in examples]
    captions = [ "Please give me the bounding bbox of the description \"polar bear\", providing their labels and bounding box coordinates in JSON."]
    img_idx = image_select(
        label='',
        images=images,
        captions=captions,
        use_container_width=True,
        index=-1,
        return_value='index',
        key='image_select'
    )
    st.caption(
        'Note: For non-commercial research use only. AI responses may contain errors. Users should not spread or allow others to spread hate speech, violence, pornography, or fraud-related harmful information.')
    if img_idx != -1 and len(st.session_state.messages) == 0 and selected_model is not None:
        gallery_placeholder.empty()
        st.session_state.messages.append({'role': 'user', 'content': captions[img_idx], 'image': [images[img_idx]],
                                          'filenames': [examples[img_idx]]})
        st.rerun()  # Fixed an issue where examples were not emptied

if len(st.session_state.messages) > 0:
    gallery_placeholder.empty()

# Display or clear chat messages
total_image_num = 0
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        show_one_or_multiple_images(message, total_image_num, is_input=message['role'] == 'user')
        if 'image' in message and message['role'] == 'user':
            total_image_num += len(message['image'])

print('total_image_num:', total_image_num, 'len(uploaded_files):', len(uploaded_files), 'max_image_limit:', max_image_limit)
input_disable_flag = (len(model_list) == 0) or total_image_num + len(uploaded_files) > max_image_limit
input_disable_flag = input_disable_flag or video_count > 1
print('input_disable_flag:', input_disable_flag)

if input_disable_flag:
    st.sidebar.button('Clear Chat History',
                      on_click=partial(combined_func, func_list=[clear_chat_history, clear_file_uploader]))
    st.error('Too many images have been uploaded. Please clear the history.')
    prompt = st.chat_input('Input is disabled.', disabled=True)
else:
    prompt = st.chat_input('Send messages to Eagle', disabled=input_disable_flag)


if prompt:
    # prompt = alias_instructions[prompt] if prompt in alias_instructions else prompt
    gallery_placeholder.empty()
    image_list = uploaded_pil_images
    st.session_state.messages.append(
        {'role': 'user', 'content': prompt, 'image': image_list, 'filenames': save_filenames})
    with st.chat_message('user'):
        st.write(prompt)
        show_one_or_multiple_images(st.session_state.messages[-1], total_image_num, is_input=True)
    if image_list:
        clear_file_uploader()

# Generate a new response if last message is not from assistant
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]['role'] != 'assistant':
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            if not prompt:
                prompt = st.session_state.messages[-1]['content']
            default_response = 'Input is disabled.' if input_disable_flag else None
            response = generate_response(st.session_state.messages, default_response=default_response)
            message = {'role': 'assistant', 'content': response}
        with st.spinner('Drawing...'):
            try:
                print('response:', response)
                res_list = is_valid_bounding_box(response)
                print('res_list:', res_list)
                if len(res_list) > 0:
                    has_returned_image = draw_bounding_boxes(res_list)
                    message['image'] = [has_returned_image] if has_returned_image else []
                st.session_state.messages.append(message)
                show_one_or_multiple_images(message, total_image_num, is_input=False)
            except Exception as e:
                st.error(f"Error: {e}")
                st.error(traceback.format_exc())

if len(st.session_state.messages) > 0:
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1.3])
    text1 = 'Clear Chat History'
    text2 = 'Regenerate' 
    text3 = 'Copy' 
    with col1:
        st.button(text1, on_click=partial(combined_func, func_list=[clear_chat_history, clear_file_uploader]),
                  key='clear_chat_history_button')
    with col2:
        st.button(text2, on_click=regenerate, key='regenerate_button')

print(st.session_state.messages)
save_chat_history()