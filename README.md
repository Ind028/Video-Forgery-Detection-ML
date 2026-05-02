# 🎥 Video Forgery Detection using ResNet50

A deep learning system for detecting **copy-move forgeries in digital videos** using a **ResNet50 architecture implemented from scratch**.

---

## 📌 Overview

This project focuses on identifying **tampered video frames** where regions are copied and pasted within the same frame (copy-move forgery). The system processes videos frame-by-frame and performs **binary classification** to determine whether a frame is:

- ✅ Authentic  
- ❌ Forged  

The model is trained on the **REWIND video copy-move forgery dataset**, making it robust for real-world tampering scenarios.

---

## 🚀 Key Results

| Metric              | Score |
|--------------------|------|
| Training Accuracy  | **79.20%** |
| Validation Accuracy| **78.33%** |
| Precision          | **0.76** |
| Recall             | **0.80** |
| F1-Score           | **0.78** |
| ROC-AUC            | **0.84** |

---


---

## 📂 Dataset

- **Dataset Used:** REWIND (Video Copy-Move Forgery Dataset)
- Contains:
  - Forged video frames
  - Authentic video frames

### Preprocessing
- Frame extraction from videos  
- Resize to **224×224**  
- Normalization  

---

## ⚙️ Features

- 🔍 Detects **copy-move forgeries in videos**
- 🧱 ResNet50 implemented **from scratch**
- 🎯 Frame-level binary classification
- 📊 Evaluation using **Precision, Recall, F1, ROC-AUC**
- ⚡ Dropout for regularization

---

## 🛠️ Tech Stack

- Python 🐍
- TensorFlow / PyTorch
- NumPy
- OpenCV
- Matplotlib

---

## 📈 Training Details

- Input Size: **224 × 224 × 3**
- Loss Function: **Binary Crossentropy**
- Optimizer: **Adam**
- Regularization: **Dropout (0.5)**
- Activation:
  - ReLU (hidden layers)
  - Sigmoid (output layer)

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision & Recall  
- F1 Score  
- ROC-AUC  

---

## 🧪 How It Works

1. 🎞️ Extract frames from input video  
2. 🖼️ Preprocess frames (resize + normalize)  
3. 🧠 Pass frames through ResNet50  
4. 📊 Predict probability of forgery  
5. ✅ Classify as **Forged** or **Authentic**

---

## ▶️ Usage

```bash
# Clone the repository
git clone https://github.com/your-username/video-forgery-detection.git

# Navigate to project directory
cd video-forgery-detection

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Run inference
python predict.py
