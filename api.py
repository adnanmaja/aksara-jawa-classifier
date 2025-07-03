from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Read and process the image
        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        char_segments = segment_by_projection(pil_image)

        bboxes = [seg['bbox'] for seg in char_segments]
        avg_h = np.mean([h for _, _, _, h in bboxes])
        avg_y = np.mean([y for _, y, _, _ in bboxes])

        for seg in char_segments:  # DEBUG
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

        # Get debug info from the last segment (you might want to modify this logic)
        if char_segments:
            last_seg = char_segments[-1]
            base_debug = baseDebug(last_seg['image'])
            sandhangan_debug = sandhanganDebug(last_seg['image'])
            pasangan_debug = pasanganDebug(last_seg['image'])
        else:
            base_debug = sandhangan_debug = pasangan_debug = None

        return jsonify({
            "debug": {
                "base": base_debug,
                "sandhangan": sandhangan_debug,
                "pasangan": pasangan_debug,
                "joined_base_and_sandhangan()": grouped_result,
                "group_sandhangan()": grouped,
            },
            "prediction": final_text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

if __name__ == "__main__":
    # For development
    app.run(debug=True, host='0.0.0.0', port=5000)
    
    # For testing with local image (commented out for production)
    # testimg = Image.open("TESTS/test_4.png")
    # with app.test_client() as client:
    #     # This would need to be adapted for actual testing