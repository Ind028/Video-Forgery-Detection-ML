# 🎥 Video Forgery Detection using ResNet50

A deep learning system for detecting copy-move forgeries in digital videos using a ResNet50 convolutional neural network implemented from scratch.

## 📋 Overview

This project presents a deep learning-based approach for detecting copy-move forgeries in digital videos. The system performs per-frame binary classification to identify forged and authentic video frames using a ResNet50 architecture trained from scratch on the REWIND video copy-move forgeries dataset.

**Key Achievements:**
- ✅ Training Accuracy: **79.20%**
- ✅ Validation Accuracy: **78.33%**
- ✅ Precision: **0.76** | Recall: **0.80** | F1-Score: **0.78**
- ✅ ROC-AUC: **0.84**

## 🏗️ Architecture
Input Frame (224×224×3)
↓
Conv1 (7×7, 64)
↓
Conv2_x (3 bottleneck blocks)
↓
Conv3_x (4 bottleneck blocks)
↓
Conv4_x (6 bottleneck blocks)
↓
Conv5_x (3 bottleneck blocks)
↓
Global Average Pooling
↓
Dense(512) + ReLU + Dropout(0.5)
↓
Dense(1) + Sigmoid
↓
Forged / Authentic
