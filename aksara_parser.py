import torch
import numpy as np
from torchvision import transforms
from segment_characters import segment_characters  # your segmenter
from PIL import Image
import cv2
from torchvision.models import resnet18
import torchvision.models as models

# === LOAD MODELS ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'


base_model = models.resnet18(num_classes=40)
base_model.load_state_dict(torch.load("SAVED_MODELS/aksaraUpdate_model.pth", map_location=device))
base_model = base_model.to(device)
base_model.eval()

sandhangan_model = models.resnet18(num_classes=20)
sandhangan_model.load_state_dict(torch.load("SAVED_MODELS/sandhangan_model.pth", map_location=device))
sandhangan_model = base_model.to(device)
sandhangan_model.eval()

pasangan_model = models.resnet18(num_classes=20)
pasangan_model.load_state_dict(torch.load("SAVED_MODELS/pasangan_model.pth", map_location=device))
pasangan_model = base_model.to(device)
pasangan_model.eval()

# === TRANSFORM (same as training) ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

base_classes = [
    "ba", "ba_suku", "ca", "ca_suku", "da", "da_suku", "dha", "dha_suku", "ga", "ga_suku",
    "ha", "ha_suku", "ja", "ja_suku", "ka", "ka_suku", "la", "la_suku", "ma", "ma_suku",
    "na", "na_suku", "nga", "nga_suku", "nya", "nya_suku", "pa", "pa_suku", "ra", "ra_suku",
    "sa", "sa_suku", "ta", "ta_suku", "tha", "tha_suku", "wa", "wa_suku", "ya", "ya_suku"
]

sandhangan_classes = ['cakra', 'cakra2', 'keret',
'mbuhai', 'mbuhau', 'mbuhii', 'mbuhuu', 'pangkal',
'pepet', 'rongga', 'suku', 'taling', 'talingtarung',
'wigyan', 'wulu'
]

pasangan_classes = [
'b', 'c', 'd', 'dh', 'g',
'h', 'j', 'k', 'l', 'm',
'n', 'ng', 'ny', 'p', 'r',
's', 't', 'th', 'w', 'y'
]

# === PAD AND RESIZE ===
def pad_and_resize(img, size=224, pad_color=255):
    img = np.array(img.convert("L"))
    h, w = img.shape
    scale = min((size - 20) / h, (size - 20) / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), pad_color, dtype=np.uint8)
    x_off, y_off = (size - new_w)//2, (size - new_h)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return Image.fromarray(canvas).convert("RGB")

# === CLASSIFY SEGMENT ===
def classify(image_tensor, model, class_names, label_type="base"):
    with torch.no_grad():
        output = model(image_tensor.to(device))
        pred_idx = output.argmax(dim=1).item()

        if pred_idx >= len(class_names):
            print(f"[⚠️] Invalid prediction index {pred_idx} for {label_type} (len={len(class_names)})")
            return "UNKNOWN"

        return class_names[pred_idx]

# === GROUP BY POSITION ===
def assign_roles(segments):
    grouped = []
    base_boxes = []

    # Identify base characters
    avg_h = np.mean([s['bbox'][3] for s in segments])
    for i, s in enumerate(segments):
        if s['bbox'][3] >= 0.7 * avg_h:
            s['role'] = 'base'
            base_boxes.append((i, s))
    
    # Assign sandhangan/pasangan to base characters
    for i, seg in enumerate(segments):
        if 'role' in seg: continue  # already assigned as base

        x, y, w, h = seg['bbox']
        cx, cy = x + w//2, y + h//2
        assigned = False

        for base_i, base in base_boxes:
            bx, by, bw, bh = base['bbox']
            bcx, bcy = bx + bw//2, by + bh//2

            if abs(cx - bcx) < bw * 0.6:
                if cy < by:
                    seg['role'] = 'sandhangan_atas'
                    seg['base_idx'] = base_i
                    assigned = True
                    break
                elif cy > by + bh:
                    seg['role'] = 'pasangan'
                    seg['base_idx'] = base_i
                    assigned = True
                    break
                elif abs(cy - bcy) < bh * 0.3:
                    seg['role'] = 'sandhangan_samping'
                    seg['base_idx'] = base_i
                    assigned = True
                    break

        if not assigned:
            seg['role'] = 'unknown'

    # Group by base char
    output = []
    for base_i, base in base_boxes:
        entry = {'base': base['pred'], 'sandhangan': [], 'pasangan': []}
        for s in segments:
            if s.get('base_idx') == base_i:
                if 'sandhangan' in s['role']:
                    entry['sandhangan'].append(s['pred'])
                elif s['role'] == 'pasangan':
                    entry['pasangan'].append(s['pred'])
        output.append(entry)

    return output

# === MAIN PARSER ===
def parse_aksara_sentence(image_path):
    segments = segment_characters(image_path)
    parsed = []

    for i, img in enumerate(segments):
        for seg in segments:
            padded = pad_and_resize(seg["image"])
        tensor = transform(padded).unsqueeze(0)

        # Predict all 3, pick best
        base_pred = classify(tensor, base_model, base_classes)
        sandhangan_pred = classify(tensor, sandhangan_model, sandhangan_classes)
        pasangan_pred = classify(tensor, pasangan_model, pasangan_classes)

        # Store with bbox (assumes segment_characters returns that too!)
        parsed.append({
            'img': padded,
            'tensor': tensor,
            'pred_base': base_pred,
            'pred_sandhangan': sandhangan_pred,
            'pred_pasangan': pasangan_pred,
            'bbox': segments[i]["bbox"]  # you'll need to return this from segment_characters!
        })

    # Assign main prediction & role
    avg_h = np.mean([seg['bbox'][3] for seg in parsed])
    avg_y = np.mean([seg['bbox'][1] for seg in parsed])

    for p in parsed:
        x, y, w, h = p['bbox']
        cx = x + w // 2
        cy = y + h // 2

        if h >= 0.7 * avg_h and abs(cy - avg_y) < 0.3 * avg_h:
            p['pred'] = p['pred_base']
        elif cy < avg_y:
            p['pred'] = p['pred_sandhangan']
        elif cy > avg_y + 0.5 * avg_h:
            p['pred'] = p['pred_pasangan']
        else:
            p['pred'] = p['pred_sandhangan']


    return assign_roles(parsed)

def group_sandhangan(predictions):
    grouped = []

    for label in predictions:
        if '_' in label:
            base, mark = label.split('_', 1)
            grouped.append((base, mark))
        else:
            grouped.append((label, None))
    
    return grouped

def join_base_and_sandhangan(base_preds, sandhangan_preds):
   
    joined = []
    last_base_idx = -1

    for i in range(len(base_preds)):
        base = base_preds[i]
        sandh = sandhangan_preds[i]

        if base != '_' and sandh == '_':
            joined.append(base)
            last_base_idx = len(joined) - 1

        elif base != '_' and sandh != '_':
            joined.append(f"{base}_{sandh}")
            last_base_idx = len(joined) - 1

        elif base == '_' and sandh != '_':
            if last_base_idx >= 0:
                # Append sandhangan to previous base
                if '_' in joined[last_base_idx]:
                    base_part, prev_sandh = joined[last_base_idx].split('_', 1)
                    joined[last_base_idx] = f"{base_part}_{sandh}"  # overwrite previous sandhangan
                else:
                    joined[last_base_idx] = f"{joined[last_base_idx]}_{sandh}"
            else:
                # no base to attach to
                joined.append(f"_{sandh}")
        else:
            joined.append('_')  # blank segment
    
    return joined

# openIMG = Image.open("TESTS/test_4.png")
# result = parse_aksara_sentence(openIMG)
# for r in result:
#     print(r)
