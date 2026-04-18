import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

class VideoDataset(Dataset):
    def __init__(self, data_root, classes, frames_per_video=20, image_size=128):
        self.data_root = data_root
        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        
        self.video_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_root, cls_name) # This is now the processed_path
            if not os.path.isdir(cls_dir):
                continue
            for npy_name in os.listdir(cls_dir):
                if npy_name.endswith('.npy'):
                    self.video_paths.append(os.path.join(cls_dir, npy_name))
                    self.labels.append(self.class_to_idx[cls_name])
                    
    def __len__(self):
        return len(self.video_paths)
        
    def __getitem__(self, idx):
        npy_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load preprocessed arrays directly from disk.
        # Shape is assumed to be (Sequence_Length, C, H, W)
        frames = np.load(npy_path)
        
        # Convert to tensor safely
        frames = torch.tensor(frames)
        
        return frames, label
