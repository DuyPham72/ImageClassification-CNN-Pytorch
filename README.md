# Animal Image Classification with Custom CNN

A deep learning project that classifies animal images using a custom-built Convolutional Neural Network (CNN) implemented in **PyTorch**.  
The model was trained with advanced techniques such as **data augmentation**, **AdamW optimizer**, and **learning rate scheduling (ReduceLROnPlateau)** to achieve high accuracy.

---

## Project Overview

This project demonstrates the full deep learning workflow — from dataset preparation to deployment:

1. **Data loading & preprocessing** using `torchvision.transforms`
2. **Model architecture** designed from scratch (5 convolutional blocks + Adaptive Average Pooling + Linear)
3. **Training loop** with real-time TensorBoard logging
4. **Validation & checkpointing** to save best and latest models
5. **Inference script** for single-image prediction

---

## Project Structure
Animal-CNN-Classifier
├── dataset.py   # Custom dataset loader for training & validation
├── model.py         # CNN architecture definition
├── train.py         # Training loop with checkpoint saving & tensorboard logging
├── inference.py     # Script for running inference on single image
├── curr_model.pt    # Latest model checkpoint
├── best_model.pt    # Best model weights (highest validation accuracy)
├── my_tensorboard/  # TensorBoard logs (loss & accuracy curves)
└── README.md        # Project description (this file)
