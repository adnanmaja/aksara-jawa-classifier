import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import os
import threading


# === THREAD COUNT LIMIT ===
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["ONNX_NUM_THREADS"] = "1"



# === PREPROCESSING ===
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))
    
    img_array = np.array(image).astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return img_array

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


# === LOAD MODELS & PREDICTION LOGICS ===
def basePredict(image):
    base_session = ort.InferenceSession("ONNX_MODELS/aksaraUpdate.onnx", providers=["CPUExecutionProvider"])
    input_data = preprocess_image(image)
    
    # Run inference
    outputs = base_session.run(None, {"input": input_data})
    logits = outputs[0]
    
    # Apply softmax and get prediction
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    pred_idx = np.argmax(probs)
    confidence = probs[0][pred_idx]
    
    print(f"[BASE] Predicted {label_map[pred_idx]} with confidence {confidence:.2f}")
    return label_map[pred_idx]

def sandhanganPredict(image):
    sandhangan_session = ort.InferenceSession("ONNX_MODELS/sandhangan.onnx", providers=["CPUExecutionProvider"])
    input_data = preprocess_image(image)
    
    # Run inference
    outputs = sandhangan_session.run(None, {"input": input_data})
    logits = outputs[0]
    
    # Apply softmax and get prediction
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    pred_idx = np.argmax(probs)
    confidence = probs[0][pred_idx]
    
    return sandhangan_map[pred_idx]

def pasanganPredict(image):
    pasangan_session = ort.InferenceSession("ONNX_MODELS/pasangan.onnx", providers=["CPUExecutionProvider"])
    input_data = preprocess_image(image)
    
    # Run inference
    outputs = pasangan_session.run(None, {"input": input_data})
    logits = outputs[0]
    
    # Apply softmax and get prediction
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    pred_idx = np.argmax(probs)
    confidence = probs[0][pred_idx]
    
    return pasangan_map[pred_idx]


# === DEBUG MESSAGES (looks cool) ===
def baseDebug(image):
    base_session = ort.InferenceSession("ONNX_MODELS/aksaraUpdate.onnx", providers=["CPUExecutionProvider"])
    input_data = preprocess_image(image)
    outputs = base_session.run(None, {"input": input_data})
    logits = outputs[0]
    
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    pred_idx = np.argmax(probs)
    confidence = probs[0][pred_idx]
    
    return f"[BASE] Predicted {label_map[pred_idx]} with confidence {confidence:.2f}"

def sandhanganDebug(image):
    sandhangan_session = ort.InferenceSession("ONNX_MODELS/sandhangan.onnx", providers=["CPUExecutionProvider"])
    input_data = preprocess_image(image)
    outputs = sandhangan_session.run(None, {"input": input_data})
    logits = outputs[0]
    
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    pred_idx = np.argmax(probs)
    confidence = probs[0][pred_idx]
    
    return f"[SANDHANG] Predicted {sandhangan_map[pred_idx]} with confidence {confidence:.2f}"

def pasanganDebug(image):
    pasangan_session = ort.InferenceSession("ONNX_MODELS/pasangan.onnx", providers=["CPUExecutionProvider"])
    input_data = preprocess_image(image)
    outputs = pasangan_session.run(None, {"input": input_data})
    logits = outputs[0]
    
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    pred_idx = np.argmax(probs)
    confidence = probs[0][pred_idx]
    
    return f"[PASANG] Predicted {pasangan_map[pred_idx]} with confidence {confidence:.2f}"

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
    basePredict(openIMG)
    result = baseDebug(openIMG)
    print(result)
    print("Active threads:", threading.active_count())