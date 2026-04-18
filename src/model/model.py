import torch
import torch.nn as nn
import torchvision.models as models

class CNN_RNN_Model(nn.Module):
    def __init__(self, num_classes=5, hidden_size=256, num_layers=2):
        super(CNN_RNN_Model, self).__init__()
        
        # 1. Spatial Feature Extractor (CNN)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # FREEZE CNN Backbone to prevent massive overfitting on small datasets
        for param in resnet.parameters():
            param.requires_grad = False
            
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_output_size = 512
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.4)
        
        # 2. Temporal Sequence Learner (BiLSTM)
        # Replaced standard LSTM with fully Bidirectional LSTM for deep contextual mapping
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=0.4 if num_layers > 1 else 0.0 # Standard structural dropout
        )
        
        # 3. Classifier
        # Input features doubled because BiLSTM pushes forward and backward contexts simultaneously
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch, frames, channels, height, width)
        batch_size, num_frames, c, h, w = x.shape
        
        # Treat each frame as an independent image for the CNN
        x_reshaped = x.view(batch_size * num_frames, c, h, w)
        
        # CNN Forward Pass
        spatial_features = self.cnn(x_reshaped) 
        spatial_features = spatial_features.view(batch_size * num_frames, -1) 
        
        # Apply first dropout
        spatial_features = self.dropout1(spatial_features)
        
        # Reshape back to sequence for BiLSTM
        lstm_input = spatial_features.view(batch_size, num_frames, -1) 
        
        # BiLSTM Forward Pass
        lstm_out, _ = self.lstm(lstm_input)
        
        # Average pooling across all temporal sequences gives a significantly smoother and
        # more stable classification gradient than violently snipping off the final frame
        pooled_out = torch.mean(lstm_out, dim=1) 
        
        # Apply second dropout
        pooled_out = self.dropout2(pooled_out)
        
        # FC Forward Pass
        out = self.fc(pooled_out)
        return out
