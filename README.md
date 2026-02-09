# Convolutional-Neural-Network-CNN

This GitHub repository demonstrates Convolutional Neural Networks (CNNs) using CIFAR-10 and MNIST datasets, focusing on model comparisons and sustainable AI practices.

**Key Highlights:**
- Achieves up to 90% accuracy on CIFAR-10 and 99% on MNIST.
- Explores basic to advanced architectures, including transfer learning.
- Emphasizes green computing with model pruning and CO2 tracking for efficiency.

## Overview
The repo includes two Jupyter notebooks:
- **Basic vs Advance CNN.ipynb**: Builds and compares simple CNNs, deeper models with regularization, and VGG16 transfer learning in TensorFlow on CIFAR-10.
- **Green-Computing-Custom-CNN.ipynb**: Implements lightweight custom CNNs and pruned models (MobileNetV2, ResNet18) in PyTorch on MNIST/CIFAR-10, measuring performance and environmental impact.

## Setup
1. Clone: `git clone https://github.com/Zahin2470/Convolutional-Neural-Network-CNN.git`
2. Install: `pip install tensorflow torch torchvision codecarbon ptflops torchsummary numpy pandas matplotlib scikit-learn`
3. Run notebooks in Jupyter.

## Results
- Basic CNN: ~75% accuracy (CIFAR-10).
- Advanced: ~82%.
- Transfer: ~90%.
- Custom/Pruned: Balances accuracy (65-81%) with low CO2 emissions.

For details, explore the notebooks.

---

This repository provides hands-on examples of Convolutional Neural Networks (CNNs) for image classification, leveraging standard datasets like CIFAR-10 (60,000 color images in 10 classes) and MNIST (70,000 grayscale digit images). It serves as an educational tool for understanding CNN fundamentals, architectural advancements, and the integration of sustainability in AI model development. Built with TensorFlow/Keras and PyTorch, the project is suitable for beginners and intermediate learners in computer vision.

### Project Structure
- **Notebooks**:
  - Basic vs Advance CNN.ipynb (TensorFlow focus).
  - Green-Computing-Custom-CNN.ipynb (PyTorch focus, efficiency emphasis).
- No data files needed; datasets load via APIs.

### Detailed Notebook Insights

#### Basic vs Advance CNN.ipynb
Introduces CNN concepts on CIFAR-10:
- **Data Prep**: Normalization, one-hot encoding, optional augmentation.
- **Models**:
  - Basic: 3 conv layers, ~319K params.
  - Advanced: Added BatchNorm/Dropout, ~553K params.
  - Transfer: VGG16 base, ~14.8M params (mostly frozen).
- **Training**: Adam optimizer, early stopping, up to 50 epochs.
- **Metrics**: Accuracy plots, confusion matrices.
- Sample Results: Basic (70-75%), Advanced (78-82%), Transfer (85-90%).

#### Green-Computing-Custom-CNN.ipynb
Focuses on efficient models for MNIST/CIFAR-10:
- **Data Prep**: Custom loaders, normalization.
- **Models**:
  - Custom CNN: Depth-wise conv, ~102K-133K params.
  - Pruned: MobileNetV2/ResNet18 with 30% pruning, quantization.
- **Training**: Adam/SGD, emissions tracking with CodeCarbon.
- **Metrics**: Accuracy, GFLOPs, size, inference time, CO2.
- Visuals: Summaries, plots.

These demonstrate trade-offs between performance and efficiency, ideal for edge devices.
