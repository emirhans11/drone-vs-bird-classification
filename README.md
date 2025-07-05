# drone-vs-bird-classification
MobileNetV2 based model for drones and birdwatching
# 🕊️ Drone vs Bird Image Classification with MobileNetV2

This project is a deep learning-based image classifier that distinguishes between **drones** and **birds** using transfer learning with **MobileNetV2**.

---

## 📦 Dataset

- Source: [Kaggle - Drone vs Bird Dataset](https://www.kaggle.com/datasets/muhammadsaoodsarwar/drone-vs-bird)
- Structure:
  - `train/` → images labeled as `bird/` or `drone/`
  - `val/` → 20% validation split
  - `custom_test/` → manually collected real-world images for testing

---

## 📌 Project Overview

- 🔍 **Goal:** Classify an image as either `bird` or `drone`
- 🧠 **Model:** MobileNetV2 (pretrained on ImageNet, fine-tuned)
- 🧪 **Accuracy:**
  - Training Accuracy: **99.88%**
  - Validation Accuracy: **99.88%**
  - Real-world test: **10/10 correct predictions**

---

## 📁 Files

| File | Description |
|------|-------------|
| `drone_vs_bird_classifier.ipynb` | Jupyter Notebook with training and evaluation code |
| `best_model.pt` | Trained model weights |
| `README.md` | Main project documentation (English) |
| `README_TR.md` | Turkish translation of the documentation |

---

## 🔧 Usage

### Load the model:
```python
import torch
from torchvision import models
import torch.nn as nn

model = models.mobilenet_v2(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 2)
)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
