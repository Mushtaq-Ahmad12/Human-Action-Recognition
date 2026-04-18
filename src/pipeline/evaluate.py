import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model.model import CNN_RNN_Model
from src.data.loader import get_dataloaders

def evaluate_model(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cpu')
    
    model_path = config['training']['model_save_path']
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Train the model first.")
        return
        
    _, _, test_loader, classes, _ = get_dataloaders(config_path)
    
    model = CNN_RNN_Model(
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating model on test dataset...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    os.makedirs('data/processed', exist_ok=True)
    plt.savefig('data/processed/confusion_matrix.png')
    print("Saved confusion matrix to data/processed/confusion_matrix.png")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    evaluate_model(config_file)
