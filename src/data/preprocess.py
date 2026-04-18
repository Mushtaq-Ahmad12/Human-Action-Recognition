import os
import cv2
import numpy as np
import yaml
import torch
from tqdm import tqdm

def extract_frames(video_path, frames_per_video=20, image_size=128):
    """
    Given a video path, extract evenly spaced frames, resize, normalize,
    and return as a pyTorch-compatible formatted numpy array (C, Sequence_Length, H, W)
    Or actually return PyTorch shape (Sequence_Length, C, H, W).
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    if frame_count == 0:
        # Handle empty/corrupt videos
        return np.zeros((frames_per_video, 3, image_size, image_size), dtype=np.float32)
        
    indices = np.linspace(0, frame_count - 1, frames_per_video, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size))
            
        frames.append(frame)
        
    cap.release()
    
    # Normalize to [0, 1]
    frames = np.array(frames, dtype=np.float32) / 255.0
    
    # Shape: (Frames, H, W, C) -> PyTorch Shape: (Frames, C, H, W)
    frames = np.transpose(frames, (0, 3, 1, 2))
    
    return frames

def preprocess_dataset(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Could not read config.yaml: {e}")
        return
        
    raw_path = config['data']['raw_path']
    processed_path = config['data']['processed_path']
    classes = config['data']['subset_classes']
    frames_per_video = config['data']['frames_per_video']
    image_size = config['data']['image_size']
    
    print(f"Starting Preprocessing from {raw_path} to {processed_path}")
    
    os.makedirs(processed_path, exist_ok=True)
    
    for cls_name in classes:
        cls_dir = os.path.join(raw_path, cls_name)
        out_cls_dir = os.path.join(processed_path, cls_name)
        
        if not os.path.isdir(cls_dir):
            print(f"Directory missing: {cls_dir}")
            continue
            
        os.makedirs(out_cls_dir, exist_ok=True)
        
        video_files = [f for f in os.listdir(cls_dir) if f.endswith('.avi') or f.endswith('.mp4')]
        
        print(f"Processing class: {cls_name}")
        for video_name in tqdm(video_files):
            v_path = os.path.join(cls_dir, video_name)
            out_path = os.path.join(out_cls_dir, video_name.split('.')[0] + ".npy")
            
            if not os.path.exists(out_path):
                frames = extract_frames(v_path, frames_per_video, image_size)
                np.save(out_path, frames)

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    preprocess_dataset(config_file)
