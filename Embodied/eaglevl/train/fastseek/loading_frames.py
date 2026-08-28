from .fastseek import Fastseek
import av
from fractions import Fraction
from typing import List
import numpy as np
from PIL import Image
from pathlib import Path
import collections
from decord import VideoReader
import lmdb
import io
import cv2
import os
import pickle
import signal
import functools

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _resolve_repo_relative_path(path, anchor="Eagle2"):
    normalized = path.replace("\\", "/")
    marker = f"/{anchor}/"
    if marker not in normalized:
        return None

    rel_path = normalized.split(marker, 1)[1]
    return os.path.join(_REPO_ROOT, anchor, *[part for part in rel_path.split("/") if part])


def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set up signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)  # Restore the original handler
            return result
        return wrapper
    return decorator

@timeout(10)  # 10-second timeout
def load_image_from_lmdb(lmdb_file, lmdb_key):
    
    if not os.path.exists(lmdb_file):
        resolved_lmdb_file = _resolve_repo_relative_path(lmdb_file)
        if resolved_lmdb_file is not None and os.path.exists(resolved_lmdb_file):
            lmdb_file = resolved_lmdb_file
        else:
            raise ValueError(f"LMDB file {lmdb_file} does not exist")

    # special case for AgiBotWorld
    if 'AgiBotWorld' in lmdb_file:
        return read_img_from_lmdb_v2(lmdb_file, lmdb_key)

    env = None
    try:
        env = lmdb.open(lmdb_file, readonly=True, lock=False, max_readers=10240)
        with env.begin(write=False) as txn:
            try:
                image_bin = txn.get(lmdb_key.encode('ascii'))
                if image_bin is None:
                    raise ValueError(f"Key {lmdb_key} not found in LMDB file {lmdb_file}")
                buf = io.BytesIO(image_bin)
            except Exception as e:
                raise ValueError(f"Fail to get image from LMDB file {lmdb_file}: {str(e)}")

        try:
            pil_image = Image.open(buf)
        except Exception as e:
            image_np = np.frombuffer(image_bin, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        return pil_image
    finally:
        if env is not None:
            env.close()


def read_img_from_lmdb_v2(lmdb_file, lmdb_key):
    # special case for AgiBotWorld
    key = lmdb_key.encode('ascii')
    env = lmdb.open(lmdb_file, max_readers=10240, readonly=True, lock=False, readahead=False, meminit=False)
    try:
        txn = env.begin()
        value = txn.get(key)
        if value is None:
            print(f"Warning: Key {key} not found.")
            return None
        record = pickle.loads(value)
        image_bgr = cv2.imdecode(np.frombuffer(record['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        return image
    finally:
        env.close()

def frame_to_pts(frame: int, frame_duration: int) -> int:
    return int(frame * frame_duration)

def pts_to_frame(pts: int, frame_duration) -> int:
    return pts / frame_duration

def frame_to_time(frame: int, rate: Fraction, time_base: Fraction) -> int:
    return int(frame / rate / time_base)


def decord_read(video_path: str, frame_indices: List[int]=None, timestamps: List[float]=None):
    # ftype = filetype.guess(video_path)
    try:
        video_reader = VideoReader(video_path, num_threads=2)
        # get total frames
        total_frames = len(video_reader)
        # get real fps
        input_fps = video_reader.get_avg_fps()
        
        # if timestamps is not None, we need to update the frame_indices
        if timestamps is not None:
            frame_indices = [int(timestamp * input_fps) for timestamp in timestamps]

        # reset frame out of range
        frame_indices = [max(0, min(total_frames-1, frame_idx)) for frame_idx in frame_indices] 
        # remap_frame_indices, because some indices might be repeated
        original_frame_indices = frame_indices
        frame_indices = list(dict.fromkeys(frame_indices))
        frame_idx_to_image_dict = {}
        for frame_idx in frame_indices:
            try:
                frame_img = video_reader[frame_idx].asnumpy()
                frame_idx_to_image_dict[frame_idx] = Image.fromarray(frame_img)
            except Exception as e:
                print(f"Warning: Failed to read frame {frame_idx} from {video_path}: {str(e)}")
                # Use a black image as fallback
                frame_idx_to_image_dict[frame_idx] = Image.new('RGB', (224, 224), (0, 0, 0))
        frames = [frame_idx_to_image_dict[frame_idx] for frame_idx in original_frame_indices]
        return frames
    except Exception as e:
        print(f"Error reading video {video_path}: {str(e)}")
        # Return black images as fallback
        return [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(len(frame_indices) if frame_indices else 1)]


def pyav_read(video_path: str, frame_indices=None, timestamps=None):
    with av.open(video_path) as container:
        container.streams.video[0].thread_type = 3
        video_stream = container.streams.video[0]
        input_fps: Fraction = video_stream.average_rate
        input_tb: Fraction = video_stream.time_base
        frame_duration = round(1 / (input_fps * input_tb))
        duration_seconds = video_stream.duration * input_tb
        total_frames = round(duration_seconds * input_fps)
        if timestamps is not None:
            frame_indices = [round(timestamp * input_fps) for timestamp in timestamps]
        frame_indices = [max(0, min(total_frames - 1, frame_idx)) for frame_idx in frame_indices]
        original_frame_indices = frame_indices
        frame_idx_to_image_dict = {}
        for frame_idx in frame_indices:
            timestamp = round(frame_idx * frame_duration)
            container.seek(timestamp, stream=video_stream)
            for frame in container.decode(video=0):
                if frame.pts >= timestamp:
                    frame_idx_to_image_dict[frame_idx] = Image.fromarray(frame.to_ndarray(format='rgb24'))
                    break
        frames = [frame_idx_to_image_dict[frame_idx] for frame_idx in original_frame_indices]
        return frames



@timeout(30)  # 30-second timeout
def read_frames_sequential(video_path: str, frame_indices: List[int]=None, timestamps: List[float]=None):
    assert frame_indices is not None or timestamps is not None
    return decord_read(video_path, frame_indices, timestamps)



def get_frames_for_multiple_videos(video_frame_tuples):
    # [("video1.mp4", 1), ("video1.mp4", 2), ("video2.mp4", 1), ("video2.mp4", 2)]
    videos_frame_dict = collections.defaultdict(list)
    
    for i, (video_path, frame_index) in enumerate(video_frame_tuples):
        videos_frame_dict[video_path].append((i, frame_index))

    output_images = [None] * len(video_frame_tuples)
    for video_path in videos_frame_dict:
        videos_frame_dict[video_path].sort(key=lambda x: x[1])
        frame_indices = [x[1] for x in videos_frame_dict[video_path]]
        images = read_frames_sequential(video_path, frame_indices)
        for video_frame_idx, image in zip(videos_frame_dict[video_path], images):
            output_images[video_frame_idx[0]] = image

    return output_images


def get_frames_for_multiple_videos_and_images(video_frame_or_image_tuples):
    # [("video1.mp4", 1), ("video1.mp4", 2), ("video2.mp4", 1), ("video2.mp4", 2), Image1, Image2]
    videos_frame_dict = collections.defaultdict(list)
    image_list = []
    
    for i, image_target in enumerate(video_frame_or_image_tuples):
        if isinstance(image_target, list):
            # video case
            video_path, frame_index = image_target
            videos_frame_dict[video_path].append((i, frame_index))
        else:
            # str and dict case
            # image case
            image_list.append((i, image_target))

    output_images = [None] * len(video_frame_or_image_tuples)
    for video_path in videos_frame_dict:
        videos_frame_dict[video_path].sort(key=lambda x: x[1])
        frame_indices = [x[1] for x in videos_frame_dict[video_path]]
        images = read_frames_sequential(video_path, frame_indices)
        for video_frame_idx, image in zip(videos_frame_dict[video_path], images):
            output_images[video_frame_idx[0]] = image

    for image in image_list:
        if isinstance(image[1], dict):
            output_images[image[0]] = load_image_from_lmdb(image[1]['lmdb_file'], image[1]['lmdb_key'])
        else:
            output_images[image[0]] = Image.open(image[1])
    
    assert None not in output_images
    return output_images



# TODO
def get_frames_for_multiple_videos_images_and_audio(video_frame_or_image_or_audio_tuples):
    pass