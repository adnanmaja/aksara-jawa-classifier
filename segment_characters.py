import cv2
from PIL import Image


def segment_characters(image_path, min_width=10, min_height=10):
    """
    Segments individual base characters from an image of a full Aksara Jawa sentence.
    
    Args:
        image_path (str): Path to sentence image.
        min_width (int): Minimum width of character to be considered valid.
        min_height (int): Minimum height of character to be considered valid.

    Returns:
        List[PIL.Image]: List of cropped/resized PIL images (224x224) for each character.
    """
    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (external only)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []

    # Sort contours left to right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])  # sort by x

    for (x, y, w, h) in bounding_boxes:
        if w >= min_width and h >= min_height:
            char_crop = gray[y:y+h, x:x+w]
            char_crop = cv2.resize(char_crop, (224, 224), interpolation=cv2.INTER_AREA)

            # Convert to 3-channel PIL image
            pil_img = Image.fromarray(char_crop).convert("RGB")
            char_images.append(pil_img)

    
    return char_images

