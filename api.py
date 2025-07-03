from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import torch
import io
import torch.nn as nn
from torchvision.models import resnet18
from segment_characters import segment_characters, segment_by_projection
from aksara_parser import group_sandhangan, join_base_and_sandhangan
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model = resnet18(weights=False)
base_model.fc = nn.Linear(base_model.fc.in_features, 40)
base_model.load_state_dict(torch.load("SAVED_MODELS/aksaraUpdate_model.pth", map_location="cpu"))
base_model.to(device)
base_model.eval()

sandhangan_model = resnet18(weights=False)
sandhangan_model.fc = nn.Linear(sandhangan_model.fc.in_features, 20)
sandhangan_model.load_state_dict(torch.load("SAVED_MODELS/sandhangan_model.pth", map_location="cpu"))
sandhangan_model.to(device)
sandhangan_model.eval()

pasangan_model = resnet18(weights=False)
pasangan_model.fc = nn.Linear(pasangan_model.fc.in_features, 20)
pasangan_model.load_state_dict(torch.load("SAVED_MODELS/pasangan_model.pth", map_location="cpu"))
pasangan_model.to(device)
pasangan_model.eval()

label_map = [
    "ba", "ba_suku", "ca", "ca_suku", "da", "da_suku", "dha", "dha_suku", "ga", "ga_suku",
    "ha", "ha_suku", "ja", "ja_suku", "ka", "ka_suku", "la", "la_suku", "ma", "ma_suku",
    "na", "na_suku", "nga", "nga_suku", "nya", "nya_suku", "pa", "pa_suku", "ra", "ra_suku",
    "sa", "sa_suku", "ta", "ta_suku", "tha", "tha_suku", "wa", "wa_suku", "ya", "ya_suku"
]

sandhangan_map = ['cakra', 'cakra2', 'keret',
'mbuhai', 'mbuhau', 'mbuhii', 'mbuhuu', 'pangkal',
'pepet', 'rongga', 'suku', 'taling', 'talingtarung',
'wignyan', 'wulu'
]

sandhanganSound_map = {
    'suku': 'u',
    'wulu': 'i',
    'taling': 'e',
    'taling_tarung': 'o',
    'pepet': 'ê',
    'cakra': 'r',
    'wignyan': 'h',
    'keret': 'ng',  
    'cecak': 'ng',  
}

pasangan_map = [
'b', 'c', 'd', 'dh', 'g',
'h', 'j', 'k', 'l', 'm',
'n', 'ng', 'ny', 'p', 'r',
's', 't', 'th', 'w', 'y'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def basePredict(images):
    base_results = []
    if images.mode != 'RGB':
        images = images.convert('RGB')
    tensor = transform(images).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = base_model(tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()
        base_results.append(label_map[pred_idx])
    return base_results

def sandhanganPredict(images):
    sandhangan_results = []
    if images.mode != 'RGB':
        images = images.convert('RGB')
    tensor = transform(images).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = sandhangan_model(tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()
    return sandhangan_map[pred_idx]

def pasanganPredict(images):
    pasangan_results = []
    if images.mode != 'RGB':
        images = images.convert('RGB')
    tensor = transform(images).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = pasangan_model(tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()
    return pasangan_map[pred_idx]

def classify_region(bbox, avg_height, avg_y, top_thresh=0.65, bottom_thresh=1.4):
    x, y, w, h = bbox
    cy = y + h / 2

    # Very short
    if h < 0.35 * avg_height or w < 0.3 * avg_height :
        return 'pasangan'
    # Base in the center, large enough
    elif h >= 0.7 * avg_height and (avg_y - 0.3 * avg_height) < cy < (avg_y + 0.3 * avg_height):
        return 'sandhangan' # Originially base
    # Significantly below baseline
    elif cy > avg_y + 0.35 * avg_height:
        return 'base' # Originally pasangan
    else:
        return 'sandhangan'
        
def transliterate_grouped(resultsGrouped):
    result = []
    for base, sandhangan in resultsGrouped:
        if sandhangan and sandhangan in sandhanganSound_map:
            vowel = sandhanganSound_map[sandhangan]
            result.append(base[0] + vowel)  # crude base-to-syllable mapping
        else:
            result.append(base)
    return ''.join(result)

def integrate_pasangan(base_stream, pasangan_stream):
    final = []
    pasangan_buffer = []

    for i, base in enumerate(base_stream):
        final.append(base)

        if i < len(pasangan_stream) and pasangan_stream[i] != '_':
            pasangan_buffer.append(pasangan_stream[i])

    # Now interleave pasangan after base if needed
    # Or you can attach to previous base if context says no vowel between
    final_with_pasangan = []

    for char in final:
        final_with_pasangan.append(char)
        if pasangan_buffer:
            final_with_pasangan.append(pasangan_buffer.pop(0))

    return final_with_pasangan


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    char_segments = segment_by_projection(pil_image)  # Now works with updated function

    bboxes = [seg['bbox'] for seg in char_segments]
    avg_h = np.mean([h for _, _, _, h in bboxes])
    avg_y = np.mean([y for _, y, _, _ in bboxes])

    for seg in char_segments: # DEBUG
        print(f"Segment: bbox={seg['bbox']}, h={seg['bbox'][3]}, cy={seg['bbox'][1] + seg['bbox'][3]/2}, role={classify_region(seg['bbox'], avg_h, avg_y)}")

    base_preds = []
    sandhangan_preds = []
    pasangan_preds = []
    
    for seg in char_segments:
        role = classify_region(seg['bbox'], avg_h, avg_y)

        if role == 'base':
            base_preds.append(basePredict(seg['image'])[0])
            sandhangan_preds.append('_')
            pasangan_preds.append('_')

        elif role == 'sandhangan':
            sandhangan_preds.append(sandhanganPredict(seg['image'])[0])
            base_preds.append('_')
            pasangan_preds.append('_')

        elif role == 'pasangan':
            pasangan_preds.append(pasanganPredict(seg['image'])[0])
            base_preds.append('_')
            sandhangan_preds.append('_')


    integrated_result = integrate_pasangan(base_preds, pasangan_preds)
    grouped_result = join_base_and_sandhangan(base_preds, sandhangan_preds)
    grouped = group_sandhangan(grouped_result)
    final_text = transliterate_grouped(grouped)

    return {
    "base": base_preds,
    "sandhangan": sandhangan_preds,
    "pasangan": pasangan_preds, 
    "prediction": final_text}

