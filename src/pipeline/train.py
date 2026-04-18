import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model.model import CNN_RNN_Model
from src.data.loader import get_dataloaders

def train_model(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    save_path = config['training']['model_save_path']
    patience = config['training']['patience']
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print("Loading class-balanced data...")
    train_loader, val_loader, _, classes, class_weights = get_dataloaders(config_path)
    
    # Convert weights to PyTorch Tensor for Weighted Cross Entropy
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    model = CNN_RNN_Model(
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers']
    ).to(device)
    
    # 1. Weighted Loss (Crucial for class imbalance)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    
    # 2. Optimizer with regularized constraints
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 3. Learning Rate Scheduler (verbose removed for compatibility with PyTorch 2.2+)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Advanced Data Augmentations
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15), # Gentle rotation fixes tilted camera imbalances
        T.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0))
    ])
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"Starting Highly Optimized Training on {device}...")
    for epoch in range(epochs):
        # ---------------- TRAINING PHASE ----------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Dynamically Inject Spatial Augmentations natively on GPU
            b, s, c, h, w = inputs.shape
            inputs_flat = inputs.view(b * s, c, h, w)
            inputs_aug = train_transforms(inputs_flat)
            inputs = inputs_aug.view(b, s, c, h, w)
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct / total
        
        # ---------------- VALIDATION PHASE ----------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * correct / total
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # Step the Learning Rate Scheduler
        scheduler.step(epoch_val_loss)
        
        # ---------------- EARLY STOPPING & SAVING ----------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            print(f"--> Valid loss decreased. Saving highly generalized model to {save_path}")
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"--> Validation loss plateaued for {patience} epochs. Triggering Early Stopping!")
                break
            
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    
def plot_metrics(t_loss, v_loss, t_acc, v_acc):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t_loss, label='Train Loss')
    plt.plot(v_loss, label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(t_acc, label='Train Acc')
    plt.plot(v_acc, label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/processed/training_history.png')
    print("Saved training history plot to data/processed/training_history.png")

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    train_model(config_file)
