import cv2
import numpy as np
from PIL import Image


def pad_and_resize(image, size=224, pad_color=255, padding=50):
    """
    Resize image while maintaining aspect ratio and pad to square.
    
    Args:
        image: Input image (numpy array or PIL Image)
        size: Target size (will create size x size image)
        pad_color: Color for padding (255 for white, 0 for black)
        padding: Amount of padding around the character (higher = more zoomed out)
    
    Returns:
        PIL Image: Resized and padded image
    """
    img = np.array(image)
    h, w = img.shape[:2]
    
    scale = min((size - padding) / h, (size - padding) / w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), pad_color, dtype=np.uint8)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    return Image.fromarray(canvas).convert("RGB")


def segment_characters(pil_image, min_width=10, min_height=10):

    # Load image and convert to grayscale
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (external only)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left to right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])  # sort by x

    char_images = []
    
    for (x, y, w, h) in bounding_boxes:
        if w >= min_width and h >= min_height:
            char_crop = gray[y:y+h, x:x+w]
            
            # Use pad_and_resize with custom padding for more "zoomed out" look
            pil_img = pad_and_resize(char_crop, size=224, pad_color=255, padding=100)
            char_images.append(pil_img)

    return char_images