import torch
import numpy as np
from torchvision import transforms
from segment_characters import segment_characters
from PIL import Image
import cv2
from torchvision.models import resnet18
import torchvision.models as models
import torch.nn as nn

# === LOAD MODELS ===
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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === LABELS MAPPING ===
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
    # Vowels
    'suku': 'u',
    'wulu': 'i',
    'taling': 'e',
    'taling_tarung': 'o',
    'pepet': 'ê',

    # Consonants
    'cakra': 'r',
    'wignyan': 'h',
    'keret': 'ng',  
    'cecak': 'ng',  
}

pasangan_map = [
'b', 'c', 'd', 'dh', 'g',
'm', 'j', 'k', 'l', 'h',
'n', 'ng', 'ny', 'p', 'r',
's', 't', 'th', 'w', 'y'
]


# === PREDICTION LOGICS ===
def basePredict(images):
    base_results = []
    if images.mode != 'RGB':
        images = images.convert('RGB')
    tensor = transform(images).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = base_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        print(f"[DEBUG] Predicted {label_map[pred_idx.item()]} with confidence {conf.item():.2f}")
        base_debug = f"[DEBUG] Predicted {label_map[pred_idx.item()]} with confidence {conf.item():.2f}"
        pred_idx = torch.argmax(outputs, dim=1).item()
        base_results.append(label_map[pred_idx])
    return label_map[pred_idx]

def sandhanganPredict(images):
    sandhangan_results = []
    if images.mode != 'RGB':
        images = images.convert('RGB')
    tensor = transform(images).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = sandhangan_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        print(f"[DEBUG] Predicted {sandhangan_map[pred_idx.item()]} with confidence {conf.item():.2f}")
        sandhangan_debug = f"[DEBUG] Predicted {sandhangan_map[pred_idx.item()]} with confidence {conf.item():.2f}"
    return sandhangan_map[pred_idx]

def pasanganPredict(images):
    pasangan_results = []
    if images.mode != 'RGB':
        images = images.convert('RGB')
    tensor = transform(images).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = pasangan_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        print(f"[DEBUG] Predicted {pasangan_map[pred_idx.item()]} with confidence {conf.item():.2f}")
        pasangan_debug = f"[DEBUG] Predicted {pasangan_map[pred_idx.item()]} with confidence {conf.item():.2f}"
        pred_idx = torch.argmax(outputs, dim=1).item()
    return pasangan_map[pred_idx]


# === DEBUG MESSAGES (looks cool) ===
def baseDebug(images):
    if images.mode != 'RGB':
        images = images.convert('RGB')
    tensor = transform(images).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = base_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        base_debug = f"[DEBUG] Predicted {label_map[pred_idx.item()]} with confidence {conf.item():.2f}"
    return base_debug

def sandhanganDebug(images):
    if images.mode != 'RGB':
        images = images.convert('RGB')
    tensor = transform(images).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = sandhangan_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        sandhangan_debug = f"[DEBUG] Predicted {sandhangan_map[pred_idx.item()]} with confidence {conf.item():.2f}"
    return sandhangan_debug

def pasanganDebug(images):
    if images.mode != 'RGB':
        images = images.convert('RGB')
    tensor = transform(images).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = pasangan_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        pasangan_debug = f"[DEBUG] Predicted {pasangan_map[pred_idx.item()]} with confidence {conf.item():.2f}"
    return pasangan_debug

# === DEFINING BASE, SANDHANGAN, AND PASANGAN ===
# Theres role mismatch here i have no idea why but it works
def classify_region(bbox, avg_height, avg_y, top_thresh=0.65, bottom_thresh=1.4):
    x, y, w, h = bbox
    cy = y + h / 2

    # Very short
    if h < 0.5 * avg_height or w < 0.3 * avg_height :
        return 'sandhangan' # Originally sandhangan
    # Base in the center, large enough
    elif h >= 0.7 * avg_height and (avg_y - 0.3 * avg_height) < cy < (avg_y + 0.3 * avg_height):
        return 'pasangan' # Originially base
    # Significantly below baseline
    elif cy > avg_y + 0.35 * avg_height:
        return 'base' # Originally pasangan
    else:
        return 'sandhangan' # Originally sandhangan
        

# === COMBINING BASE AND SANDHANGAN ===
def join_base_and_sandhangan(base_preds, sandhangan_preds):
    joined = []
    last_base_idx = -1

    for i in range(len(base_preds)):
        base = base_preds[i]
        sandh = sandhangan_preds[i]

        if base != "_" and sandh == "_":
            joined.append(base)
            last_base_idx = len(joined) - 1

        elif base != "_" and sandh != "_":
            joined.append(f"{base}_{sandh}")
            last_base_idx = len(joined) - 1

        elif base == "_" and sandh != "_":
            if last_base_idx >= 0:
                # Attach sandhangan to previous base
                prev = joined[last_base_idx]
                if '_' in prev:
                    parts = prev.split('_')
                    joined[last_base_idx] = f"{parts[0]}_{sandh}"
                else:
                    joined[last_base_idx] = f"{prev}_{sandh}"
                print(f"[DEBUG] Attached sandhangan '{sandh}' to previous base '{joined[last_base_idx]}'")
            else:
                # orphan sandhangan
                joined.append(f"_{sandh}")
        else:
            joined.append("_")

    return joined

def transliterate_grouped(joined_labels):
    result = []

    for label in joined_labels:
        if '_' in label:
            parts = label.split('_')
            base = parts[0]
            modifiers = parts[1:]

            base_char = base
            vowel = ''
            final_consonants = ''

            for mod in modifiers:
                if mod in sandhanganSound_map:
                    sound = sandhanganSound_map[mod]
                    if sound in ['u', 'e', 'o', 'ê']:  # For vowels
                        vowel = sound
                        base_char = base[0]
                        print(f"IM A VOWEL → {mod} → {sound}")
                    else:
                        final_consonants += sound  # For consonants
                        print(f"NOT a vowel → {mod} → {sound}")
                else:
                    base += mod  # pasangan

            result.append(base_char + vowel + final_consonants)
        else:
            result.append(label)

    return ''.join(result)

def group_sandhangan(predictions):
    grouped = []

    for label in predictions:
        if '_' in label:
            base, mark = label.split('_', 1)
            grouped.append((base, mark))
        else:
            grouped.append((label, None))
    
    return grouped


# === COMBINING PASANGAN ===
def integrate_pasangan(base_stream, pasangan_stream):
    result = []

    result = []
    for i, base in enumerate(base_stream):
        if base != '_':
            result.append(base)
            
            if i + 1 < len(pasangan_stream) and pasangan_stream[i + 1] != '_':
                pasangan = pasangan_stream[i + 1]
                result[-1] += f"_{pasangan}"
        else:
            continue  

        if pasangan_stream[i] != '_':
            if result:
                result[-1] += f"_{pasangan_stream[i]}"
    return result


# == TEST AND DEBUG PURPOSES==
if __name__ == "__main__":
    openIMG = Image.open("TESTS/test_4.png")
    result = group_sandhangan(openIMG)
    for r in result:
        print(r)
