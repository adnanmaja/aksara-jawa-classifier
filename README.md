# 🧠 Aksara Jawa Classifier

A simple deep learning model to classify characters in the Javanese script (Aksara Jawa) using PyTorch and a ResNet18 backbone.

This project aims to preserve and promote the Javanese script by enabling machine-based recognition of its characters, including:
- **Aksara Nglegena**
- **Sandhangan**
- **Pasangan**
- **Angka**

> 🧪 Currently in beta — more features and improvements are on the way!

---

## 🚀 Demo (Coming Soon)
The model will soon be deployed on a web app using **Flask**, allowing users to upload or draw Aksara characters for instant classification.

---

## 📦 Features
- Image classification of segmented Aksara Jawa characters
- Trained on custom dataset with augmentation
- ResNet18 architecture fine-tuned with PyTorch
- JSON label mapping
- Evaluation logs after each epoch

### Planned:
- ✅ Model retrained with segmented dataset
- ⏳ Real-time drawing input via canvas
- ⏳ Web UI with confidence scores
- ⏳ Dataset expansion & noise filtering
- ⏳ Support for compound syllables

---

## 🧰 Tech Stack
- Python, PyTorch
- Torchvision
- NumPy, Matplotlib
- Flask (for upcoming deployment)

---

## 🗂️ Dataset
Custom-built dataset of Aksara Jawa characters, including segmentation and augmentation. Will be released publicly once cleaned and documented.

---

## 🧪 Training Details
- **Architecture:** ResNet18 (transfer learning)
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 30
- **Accuracy:** ~74.4% (baseline on initial dataset)

---
