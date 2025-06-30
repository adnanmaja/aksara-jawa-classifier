from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import torch
import io
import torch.nn as nn
from torchvision.models import resnet18
from segment_characters import segment_characters
from torchvision import transforms
import numpy as np
import cv2

app = FastAPI()

num_classes = 20
model = resnet18(weights=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("aksara_model.pth", map_location="cpu"))
model.eval()

label_map = [
    "ba", "ca", "da", "dha", "ga",
    "ha", "ja", "ka", "la", "ma",
    "na", "nga", "nya", "pa", "ra",
    "sa", "ta", "tha", "wa", "ya"
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=(-20, 20)),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1), 
        scale=(0.8, 1.2),     
        shear=(-15, 15)       
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


char_imgs = segment_characters("char_1.png")

for img in char_imgs:
    tensor = transform(img).unsqueeze(0)
    outputs = model(tensor)
    pred_idx = torch.argmax(outputs, dim=1).item()
    pred_label = label_map[pred_idx]
    print(pred_label)



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0) 
    
    with torch.no_grad():
        outputs = model(tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()
        pred_label = label_map[pred_idx]

    return { "prediction": pred_idx,
        "label": pred_label}