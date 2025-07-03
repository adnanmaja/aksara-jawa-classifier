from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import torch
import io
import torch.nn as nn
from torchvision.models import resnet18
from segment_characters import segment_characters, segment_by_projection
from aksara_parser import basePredict, sandhanganPredict, pasanganPredict
from aksara_parser import classify_region, group_sandhangan, join_base_and_sandhangan, transliterate_grouped, integrate_pasangan
from aksara_parser import baseDebug, sandhanganDebug, pasanganDebug
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

@app.post("/predict")
async def predict(file : UploadFile = File(...)):
    img_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    char_segments = segment_by_projection(pil_image)  

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
            base_preds.append(basePredict(seg['image']))
            sandhangan_preds.append('_')
            pasangan_preds.append('_')

        elif role == 'sandhangan':
            pred = sandhanganPredict(seg['image'])
            print("🟢 Sandhangan detected:", pred)  # debug line
            sandhangan_preds.append(pred)
            base_preds.append('_')
            pasangan_preds.append('_')

        elif role == 'pasangan':
            pasangan_preds.append(pasanganPredict(seg['image']))
            base_preds.append('_')
            sandhangan_preds.append('_')
            

    print(f"[BEFORE GROUPING AND INTEGRATING] Base: {len(base_preds)}, Sandhangan: {len(sandhangan_preds)}, Pasangan: {len(pasangan_preds)}")
    integrated_result = integrate_pasangan(base_preds, pasangan_preds)
    grouped_result = join_base_and_sandhangan(base_preds, sandhangan_preds)
    print(f"[GROUPED_RESULT] Type: {type(grouped_result)}, {grouped_result}")
    grouped = group_sandhangan(grouped_result)
    print(f"[GROUPED] Type: {type(grouped)}, {grouped}")
    final_text = transliterate_grouped(grouped_result)

    base_debug = baseDebug(seg['image'])
    sandhangan_debug = sandhanganDebug(seg['image'])
    pasangan_debug = pasanganDebug(seg['image'])

    return {
    "debug": {
        "base": base_debug,
        "sandhangan": sandhangan_debug,
        "pasangan": pasangan_debug, 
        "joined_base_and_sandhangan()": grouped_result,
        "group_sandhangan()": grouped,}, 
    "prediction": final_text}

if __name__ == "__main__":
    testimg = Image.open("TESTS/test_4.png")
    predict(testimg)

