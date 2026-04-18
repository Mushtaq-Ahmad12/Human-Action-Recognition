# Human Action Recognition using CNN-BiLSTM
This repository contains a fully automated deep learning pipeline constructed with PyTorch for complex Human Action Recognition leveraging a dual-layer Bidirectional LSTM backbone over an extracted ResNet18 sequence.

## 🚀 Kaggle GPU Training Workaround
If your local machine takes too long to crunch the 30-frame temporal layers, you can seamlessly deploy this to Kaggle for free robust T4 / P100 GPUs!

1. Push all your latest code to your GitHub Repository.
2. Go to Kaggle -> **New Notebook**.
3. Import the `notebooks/kaggle_training.ipynb` file directly.
4. Turn **Accelerator -> 'GPU P100' or 'GPU T4x2'** ON.
5. Run all cells! The notebook will automatically fetch your repo, sequence the video files natively, train the model dynamically, and provide one-click download links for your newly generated `<best_model.pth>`.

## 💻 Local Testing & Usage
