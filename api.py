from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import torch
import io
import torch.nn as nn
from torchvision.models import resnet18
from segment_characters import segment_characters
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
    'keret': 'ng',  # optional
    'cecak': 'ng',  # optional
}


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
        pred_label = label_map[pred_idx]
        base_results.append(pred_label)
    return base_results

def sandhanganPredict(images):
    sandhangan_results = []
    if images.mode != 'RGB':
        images = images.convert('RGB')
    tensor = transform(images).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = sandhangan_model(tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()
        pred_label = sandhangan_map[pred_idx]
        sandhangan_results.append(pred_label)
    return sandhangan_results

def classify_region(bbox, avg_height, avg_y, top_thresh=0.65, bottom_thresh=1.4):
        """
        Classifies a segment as base or sandhangan based on bbox position and size.
        
        Args:
            bbox: (x, y, w, h)
            avg_height: mean height of all segments
            avg_y: mean y position of all segments
            top_thresh: anything significantly above average is atas sandhangan
            bottom_thresh: anything significantly below average is pasangan
            
        Returns:
            'base' or 'sandhangan'
        """
        x, y, w, h = bbox
        cy = y + h / 2

        if h >= 0.7 * avg_height and abs(cy - avg_y) < 0.4 * avg_height:
            return 'base'
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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    char_segments = segment_characters(pil_image)  # Now works with updated function

    bboxes = [seg['bbox'] for seg in char_segments]
    avg_h = np.mean([h for _, _, _, h in bboxes])
    avg_y = np.mean([y for _, y, _, _ in bboxes])

    base_preds = []
    sandhangan_preds = []
    
    for seg in char_segments:
        role = classify_region(seg['bbox'], avg_h, avg_y)
        if role == 'base':
            base_preds.append('_')
            sandhangan_preds.append(sandhanganPredict(seg['image'])[0])
        else:
            base_preds.append(basePredict(seg['image'])[0])
            sandhangan_preds.append('_')
    grouped_result = join_base_and_sandhangan(base_preds, sandhangan_preds)
    grouped = group_sandhangan(grouped_result)
    final_text = transliterate_grouped(grouped)

    return {
    "base": base_preds,
    "sandhangan": sandhangan_preds, 
    "prediction": final_text}


# async def predict(file: UploadFile = File(...)):
#     img_bytes = await file.read()
#     pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#     char_segments = segment_characters(pil_image)  # Now works with updated function

#     results = []
#     images = [item['image'] for item in char_segments]
#     for img in images:
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
#         tensor = transform(img).unsqueeze(0).to(device)
#         with torch.no_grad():
#             outputs = base_model(tensor)
#             pred_idx = torch.argmax(outputs, dim=1).item()
#             pred_label = label_map[pred_idx]
#             results.append(pred_label)
#     resultsGrouped = group_sandhangan(results)


#     def transliterate_grouped(resultsGrouped):
#         result = []
#         for base, sandhangan in resultsGrouped:
#             if sandhangan and sandhangan in sandhanganSound_map:
#                 vowel = sandhanganSound_map[sandhangan]
#                 result.append(base[0] + vowel)  # crude base-to-syllable mapping
#             else:
#                 result.append(base)
#         return ''.join(result)

#     final_text = transliterate_grouped(resultsGrouped)
#     return {"prediction": final_text}

