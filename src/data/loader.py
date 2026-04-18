import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml
import numpy as np

from src.data.dataset import VideoDataset

def get_dataloaders(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    processed_path = config['data']['processed_path']
    classes = config['data']['subset_classes']
    frames_per_video = config['data']['frames_per_video']
    image_size = config['data']['image_size']
    batch_size = config['training']['batch_size']
    
    dataset = VideoDataset(processed_path, classes, frames_per_video, image_size)
    
    # Train / Val / Test split (70% - 15% - 15%)
    total_len = len(dataset)
    if total_len == 0:
        raise ValueError(f"No .npy files found in {processed_path}. Have you run download.py and then preprocess.py?")
        
    train_size = int(0.7 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    # ==========================================
    # Handle Class Imbalance via Oversampling
    # ==========================================
    train_labels = [dataset.labels[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels, minlength=len(classes))
    
    # Prevent divide by zero if a class is entirely missing from the random split
    class_counts_safe = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts_safe
    
    # Generate weight for every single training sample
    sample_weights = [class_weights[label] for label in train_labels]
    
    # The Sampler takes care of drawing miniority classes more frequently
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Shuffle is manually disabled because the Sampler algorithm inherently shuffles randomly
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset.classes, class_weights

if __name__ == "__main__":
    print("Testing data loader with Class Balance constraints...")
    try:
        config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
        train_loader, val_loader, test_loader, classes, weights = get_dataloaders(config_file)
        print(f"Classes: {classes}")
        print(f"Computed Class Weights: {weights}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        for x, y in train_loader:
            print(f"Sample batch input shape [Batch, Channels, Frames, Height, Width]: {x.shape}")
            print(f"Sample batch labels: {y}")
            break
        print("DataLoader test passed!")
    except Exception as e:
        print(f"Error: {e}")
