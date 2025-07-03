# 🧠 Aksara Jawa Classifier

A simple deep learning model to classify characters in the Javanese script (Aksara Jawa) using PyTorch and a ResNet18 backbone.

This project aims to preserve and promote the Javanese script by enabling machine-based recognition of its characters, including:
- **Aksara Nglegena**
- **Sandhangan**
- **Pasangan**
- **Angka**

> 🧪 Currently in beta — more features and improvements are on the way!
Still very barebone, updates coming soon
---

## 📦 Features
- Image classification of segmented Aksara Jawa characters
- Trained on custom dataset with augmentation
- ResNet18 architecture fine-tuned with PyTorch
- JSON label mapping
- Evaluation logs after each epoch

### Planned:
- ⏳ Model retrained with segmented dataset
- ⏳ Dataset expansion & noise filtering
- ⏳ Numerical characters (Aksara angka)
- ⏳ Murda characters (Aksara murda)
- ⏳ Support for compound syllables (More pasangan, sandhangan, and their combinations)

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
- **Accuracy:** ~93.40% (baseline on initial dataset)

---

## 🙏 Acknowledgements
- PyTorch team
- Javanese script community
- Wordpress Romonadha (https://romonadha.wordpress.com/2019/12/21/font-jawa/)
- OpenCV, Matplotlib, and other amazing open-source tools
- My code supervisors ChatGPT, Claude, Gemini

---

## 🧩 Contributing
Have suggestions or want to contribute Aksara samples? Feel free to open an issue or pull request!

---

## ⚠️ Licensing of Assets
This repository uses public assets (such as fonts and character samples) that are **not redistributed** and remain under their respective licenses. You must independently obtain those resources if you wish to reproduce or train the model.

The code in this project is released under the [MIT License](LICENSE).