# 🧠 Aksara Jawa Classifier

A lightweight Flask-based web API that classifies Javanese script (Aksara Jawa) using ONNX-optimized models.  
Designed for speed, low memory usage, and easy integration into web frontends.

This project aims to preserve and promote the Javanese script by enabling machine-based recognition of its characters, including:
- **Aksara Nglegena**
- **Sandhangan**
- **Pasangan**
- **Angka** (Coming soon)

---

## 🚀 Live Demo

Frontend: [https://www.nulisjawa.my.id](https://www.nulisjawa.my.id)  
API: `https://aksara-container.delightfulcliff-10a792b4.southeastasia.azurecontainerapps.io/`  
*(Note: You can try uploading an image of Aksara Jawa characters to see JSON output.)*

---

> 🧪 Currently in beta — more features and improvements are on the way!

---

## 📦 Features
- Image classification of segmented Aksara Jawa characters
- Trained on custom dataset with augmentation
- ResNet18 architecture fine-tuned with PyTorch and powered by ONNX for fast, CPU-friendly inference
- Dockerized backend deployed via **Azure Container Apps**
- Frontend hosted with GitHub Pages + custom domain
- HTTPS enabled via DNS and SSL setup
- JSON label mapping

---

### Planned:
- ⏳ Model retrained with more datasets
- ⏳ Dataset expansion & noise filtering
- ⏳ Numerical characters (Aksara angka)
- ⏳ Murda characters (Aksara murda)
- ⏳ Support for compound syllables (More pasangan, sandhangan, and their combinations)

---

## 🧰 Tech Stack
| Layer        | Tool                        |
|--------------|-----------------------------|
| Backend      | Python + Flask              |
| ML Inference | ONNX Runtime                |
| Development  | Pytorch                     |
| Container    | Docker                      |
| Hosting      | Azure Container App (API)   |
| Frontend     | HTML/CSS/JS via GitHub Pages|
| Domain       | `.my.id` domain with DNS/API setup |

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