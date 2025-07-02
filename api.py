from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import torch
import io
import torch.nn as nn
from torchvision.models import resnet18
from segment_characters import segment_characters
from aksara_parser import group_sandhangan
from torchvision import transforms
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

num_classes = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(weights=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("SAVED_MODELS/aksaraUpdate_model.pth", map_location="cpu"))
model.to(device)
model.eval()

label_map = [
    "ba", "ba_suku", "ca", "ca_suku", "da", "da_suku", "dha", "dha_suku", "ga", "ga_suku",
    "ha", "ha_suku", "ja", "ja_suku", "ka", "ka_suku", "la", "la_suku", "ma", "ma_suku",
    "na", "na_suku", "nga", "nga_suku", "nya", "nya_suku", "pa", "pa_suku", "ra", "ra_suku",
    "sa", "sa_suku", "ta", "ta_suku", "tha", "tha_suku", "wa", "wa_suku", "ya", "ya_suku"
]

sandhanganSound_map = {
    'suku': 'u',
    'taling': 'e',
    'taling_tarung': 'o',
    'pepet': 'ê',
    'cakra': 'r',
    'wignyan': 'h',
    'keret': 'ng',  # optional
    'cecak': 'ng',  # optional
}


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    char_segments = segment_characters(pil_image)  # Now works with updated function

    results = []
    images = [item['image'] for item in char_segments]
    for img in images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()
            pred_label = label_map[pred_idx]
            results.append(pred_label)
    resultsGrouped = group_sandhangan(results)

    def transliterate_grouped(resultsGrouped):
        result = []
        for base, sandhangan in resultsGrouped:
            if sandhangan and sandhangan in sandhanganSound_map:
                vowel = sandhanganSound_map[sandhangan]
                result.append(base[0] + vowel)  # crude base-to-syllable mapping
            else:
                result.append(base)
        return ''.join(result)

    final_text = transliterate_grouped(resultsGrouped)
    return {"prediction": final_text}