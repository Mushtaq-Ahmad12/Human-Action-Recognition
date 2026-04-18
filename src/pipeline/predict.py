import os
import yaml
import torch
import cv2
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model.model import CNN_RNN_Model

def predict_video(video_path, config_path="config.yaml"):
    """
    Given a raw video file, preprocess it, feed it to the trained CNN-RNN model, and return prediction.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cpu')
    classes = config['data']['subset_classes']
    frames_per_video = config['data']['frames_per_video']
    image_size = config['data']['image_size']
    model_path = config['training']['model_save_path']
    
    if not os.path.exists(model_path):
        return {"error": "Model file not found. Train model first."}
        
    # Preprocess Video
    from src.data.preprocess import extract_frames
    
    frames = extract_frames(video_path, frames_per_video, image_size)
    
    # To Tensor shape (1, Sequence_length, C, H, W)
    input_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Load Model
    model = CNN_RNN_Model(
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().numpy()
        
    pred_idx = np.argmax(probabilities)
    confidence = probabilities[pred_idx]
    
    return {
        "class": classes[pred_idx],
        "confidence": float(confidence),
        "probabilities": {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    }

if __name__ == "__main__":
    # Test quickly if __main__
    # res = predict_video("data/raw/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi")
    # print(res)
    pass
