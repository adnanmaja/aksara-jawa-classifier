import cv2
import numpy as np
from PIL import Image


def segment_characters(pil_image, min_width=10, min_height=10):
    import numpy as np
    import cv2
    from PIL import Image

    def pad_and_resize(img_array, size=224, pad_color=255, padding=100):
        h, w = img_array.shape
        new_img = np.full((h + 2 * padding, w + 2 * padding), pad_color, dtype=np.uint8)
        new_img[padding:padding+h, padding:padding+w] = img_array
        pil = Image.fromarray(new_img)
        return pil.resize((size, size), Image.Resampling.LANCZOS)

    def vertical_split(gray_char_img, min_height_split=20):
        """Split a vertically long image using horizontal projection profile."""
        h, w = gray_char_img.shape
        projection = np.sum(255 - gray_char_img, axis=1)
        peaks = []
        threshold = np.max(projection) * 0.1

        # Find low valleys
        for i in range(1, h - 1):
            if projection[i-1] > threshold and projection[i] < threshold and projection[i+1] > threshold:
                peaks.append(i)

        # Split based on valleys
        slices = []
        last = 0
        for p in peaks:
            if p - last >= min_height_split:
                slices.append(gray_char_img[last:p, :])
                last = p
        if h - last >= min_height_split:
            slices.append(gray_char_img[last:, :])
        return slices if len(slices) > 0 else [gray_char_img]

    if isinstance(pil_image, str):
        from PIL import Image
        pil_image = Image.open(pil_image)

    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    gray = np.array(pil_image).astype(np.uint8)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    char_segments = []
    for (x, y, w, h) in bounding_boxes:
        if w >= min_width and h >= min_height:
            char_crop = gray[y:y+h, x:x+w]
            splits = vertical_split(char_crop)

            for split_img in splits:
                pil_img = pad_and_resize(split_img)
                char_segments.append({
                    "image": pil_img,
                    "bbox": (x, y, w, h)
                })

    return char_segments

import os
segments = segment_characters("test_5.png") 
output_dir = "segmented_chars"
os.makedirs(output_dir, exist_ok=True)

for i, seg in enumerate(segments):
    img = seg["image"]
    x, y, w, h = seg["bbox"]
    img.save(os.path.join(output_dir, f"char_{i}_x{x}_y{y}.png"))