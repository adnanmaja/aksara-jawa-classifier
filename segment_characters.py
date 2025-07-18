import cv2
import numpy as np
from PIL import Image, ImageDraw

def pad_and_resize(img, size=224, pad_color=255, padding=100):
    if isinstance(img, Image.Image):
        img = np.array(img.convert("L"))

    h, w = img.shape
    padded = np.full((h + 2 * padding, w + 2 * padding), pad_color, dtype=np.uint8)
    padded[padding:padding + h, padding:padding + w] = img

    scale = min((size - 20) / padded.shape[1], (size - 20) / padded.shape[0])
    new_w, new_h = int(padded.shape[1] * scale), int(padded.shape[0] * scale)
    resized = cv2.resize(padded, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size), pad_color, dtype=np.uint8)
    x_off, y_off = (size - new_w)//2, (size - new_h)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    return Image.fromarray(canvas).convert("RGB")

def split_vertically_or_horizontally_if_needed(gray_char_img, aspect_thresh=2.0):
        h, w = gray_char_img.shape

        if h / w > aspect_thresh:
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

        return [gray_char_img]

def segment_characters(pil_image, min_width=10, min_height=10):    
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

def segment_by_projection(pil_image, min_char_width=5, min_char_height=5):
    """
    Segment image into character glyphs using projection profiles.
    Returns: List of {image: PIL.Image, bbox: (x, y, w, h)}
    """
    img = pil_image.convert('L')
    gray = np.array(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Line segmentation
    horizontal_sum = np.sum(binary, axis=1)
    line_boundaries = []
    in_line = False
    for i, val in enumerate(horizontal_sum):
        if val > 0 and not in_line:
            start = i
            in_line = True
        elif val == 0 and in_line:
            end = i
            in_line = False
            if end - start > min_char_height:
                line_boundaries.append((start, end))

    segments = []

    # Character segmentation within each line 
    for (y1, y2) in line_boundaries:
        line_img = binary[y1:y2, :]
        vertical_sum = np.sum(line_img, axis=0)
        in_char = False
        char_start = 0
        for x in range(len(vertical_sum)):
            if vertical_sum[x] > 0 and not in_char:
                char_start = x
                in_char = True
            elif vertical_sum[x] == 0 and in_char:
                char_end = x
                in_char = False
                w = char_end - char_start
                h = y2 - y1
                if w > min_char_width and h > min_char_height:
                    char_crop = gray[y1:y2, char_start:char_end]
                    padded_img = pad_and_resize(char_crop, size=224, pad_color=255, padding=90)
                    segments.append({
                        'image': padded_img,
                        'bbox': (char_start, y1, w, h)
                    })
    return segments

def draw_bounding_boxes_pil(original_image, segmentation_results):
    """
    Draw bounding boxes on original image using PIL
    
    Args:
        original_image: PIL.Image - the original input image
        segmentation_results: list of dict - [{image: PIL.Image, bbox: (x, y, w, h)}, ...]
    
    Returns:
        PIL.Image with bounding boxes drawn
    """
    # Create a copy to avoid modifying original
    img_with_boxes = original_image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Default colors if not provided
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    for i, result in enumerate(segmentation_results):
        bbox = result['bbox']  # (x, y, w, h)
        x, y, w, h = bbox
        
        # Convert to corner coordinates for PIL
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        
        # Get color for this box
        color = colors[i % len(colors)]
        
        # Draw rectangle
        draw.rectangle([top_left, bottom_right], outline=color, width=2)
        
        # Add index label
        draw.text((x, y - 15), str(i), fill=color)
    
    return img_with_boxes

if __name__ == "__main__":
    image = Image.open("./TESTS/test_4.png")
    segments = segment_by_projection(image) 
    print(type(draw_bounding_boxes_pil(image, segments)))