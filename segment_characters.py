import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks


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

    def split_vertically_or_horizontally_if_needed(gray_char_img, aspect_thresh=2.0):
        h, w = gray_char_img.shape

        if h / w > aspect_thresh:
            # Try horizontal projection (for stacked glyphs)
            projection = np.sum(255 - gray_char_img, axis=1)
            min_val = np.min(projection)
            is_blank = projection < (min_val + 10)

            from scipy.signal import find_peaks
            peaks, _ = find_peaks(is_blank.astype(np.uint8), distance=10)

            if len(peaks) > 0:
                parts = []
                last = 0
                for y in peaks:
                    if y - last > 10:
                        part = gray_char_img[last:y, :]
                        if part.shape[0] > 5:
                            parts.append(part)
                    last = y
                part = gray_char_img[last:, :]
                if part.shape[0] > 5:
                    parts.append(part)
                return parts

        # Otherwise don't split
        return [gray_char_img]

    # Convert to grayscale
    if isinstance(pil_image, str):
        pil_image = Image.open(pil_image)
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    gray = np.array(pil_image).astype(np.uint8)

    # Threshold to binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (external only)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left to right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    char_segments = []

    for (x, y, w, h) in bounding_boxes:
        if w >= min_width and h >= min_height:
            char_crop = gray[y:y+h, x:x+w]
            parts = split_vertically_or_horizontally_if_needed(char_crop)

            for part in parts:
                pil_img = pad_and_resize(part, size=224, pad_color=255)
                char_segments.append({
                    "image": pil_img,
                    "bbox": (x, y, w, h)
                })

    return char_segments


# import os
# segments = segment_characters("./TESTS/test_5.png") 
# output_dir = "./DATA/segmented_chars"
# os.makedirs(output_dir, exist_ok=True)

# for i, seg in enumerate(segments):
#     img = seg["image"]
#     x, y, w, h = seg["bbox"]
#     img.save(os.path.join(output_dir, f"char_{i}_x{x}_y{y}.png"))