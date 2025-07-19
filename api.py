from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from segment_characters import segment_by_projection, draw_bounding_boxes_pil
from aksara_parser import basePredict, sandhanganPredict, pasanganPredict
from aksara_parser import classify_region, group_sandhangan, join_base_and_sandhangan, transliterate_grouped, integrate_pasangan
from aksara_parser import baseDebug, sandhanganDebug, pasanganDebug
import numpy as np
import base64

app = Flask(__name__)
CORS(app, origins=['https://www.nulisjawa.my.id', 'https://www.nulisjawa.my.id/'], methods=['POST', 'GET', 'OPTIONS'])
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === MAIN ENDPOINT FUNCTIONALITY ===
@app.route('/', methods=['POST'])
def predict():
    try:
        # Check if the file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        print("File received")
        file = request.files['file']
        
        # Check if the file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if the file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Read and process the image
        img_bytes = file.read()
        print(f"Image read, size: {len(img_bytes)} bytes")
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        char_segments = segment_by_projection(pil_image)
        bbox_visualization = draw_bounding_boxes_pil(pil_image, char_segments)

        base_preds = []
        sandhangan_preds = []
        pasangan_preds = []

        base_debug = []
        sandhangan_debug = []
        pasangan_debug = []
        
        # Calculating glyph sizes
        bboxes = [seg['bbox'] for seg in char_segments]
        avg_h = np.mean([h for _, _, _, h in bboxes])
        avg_y = np.mean([y for _, y, _, _ in bboxes])
            
        for seg in char_segments: 
            # Determining the roles of each segmented images
            role = classify_region(seg['bbox'], avg_h, avg_y) 
            print(f"Segment: bbox={seg['bbox']}, h={seg['bbox'][3]}, cy={seg['bbox'][1] + seg['bbox'][3]/2}, role={role}") # Debug stuff

            # Passing the segmented images to their respective models based on their role
            if role == 'base':
                base_preds.append(basePredict(seg['image']))
                base_debug.append(baseDebug(seg['image']))
                sandhangan_preds.append('_')
                pasangan_preds.append('_')

            elif role == 'sandhangan':
                sandhangan_preds.append(sandhanganPredict(seg['image']))
                sandhangan_debug.append(sandhanganDebug(seg['image']))
                base_preds.append('_')
                pasangan_preds.append('_')

            elif role == 'pasangan':
                pasangan_preds.append(pasanganPredict(seg['image']))
                pasangan_debug.append(pasanganDebug(seg['image']))
                base_preds.append('_')
                sandhangan_preds.append('_')

        # Combining the results from all 3 models
        print(f"[BEFORE GROUPING AND INTEGRATING] Base: {len(base_preds)}, Sandhangan: {len(sandhangan_preds)}, Pasangan: {len(pasangan_preds)}")
        grouped_result = join_base_and_sandhangan(base_preds, sandhangan_preds)
        print(f"[GROUPED_RESULT] Type: {type(grouped_result)}, {grouped_result}")
        grouped = group_sandhangan(grouped_result)
        print(f"[GROUPED] Type: {type(grouped)}, {grouped}")
        final_text = transliterate_grouped(grouped_result)

        # Turning a PIL.Image (bbox_visualization) to a base64 for the frontend to display it
        print(f"About to do base64 conversion for {bbox_visualization}")
        buffer = io.BytesIO()
        bbox_visualization.save(buffer, format='PNG')
        bbox_img_str = base64.b64encode(buffer.getvalue()).decode()
        print(f"Base64 conversion complete: {bbox_img_str[:100]}")

        return jsonify({
            "details": {
                "bbox": bbox_img_str,
                "base": base_debug,
                "sandhangan": sandhangan_debug,
                "pasangan": pasangan_debug,
                "joined_base_and_sandhangan()": grouped_result,
                "group_sandhangan()": grouped
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
    # app.run()
    app.run(debug=True, host='0.0.0.0', port=8000)   