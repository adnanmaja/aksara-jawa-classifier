import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2
from pathlib import Path
import argparse

class JavaneseCharacterGenerator:
    def __init__(self, font_paths, output_dir, image_size=(64, 64)):
        self.font_paths = font_paths
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_base_image(self, character, font_path, font_size):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            # Fallback to default font if custom font fails
            font = ImageFont.load_default()
            
        # Create image with white background
        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)
        
        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), character, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        x = (self.image_size[0] - text_width) // 2
        y = (self.image_size[1] - text_height) // 2
        
        # Draw the character in black
        draw.text((x, y), character, fill='black', font=font)
        
        return img
    
    def apply_rotation(self, img, max_angle=5):
        angle = random.uniform(-max_angle, max_angle)
        return img.rotate(angle, fillcolor='white', expand=False)
    
    def apply_position_jitter(self, img, max_shift=3):
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        
        # Create new image and paste with offset
        new_img = Image.new('RGB', self.image_size, 'white')
        new_img.paste(img, (shift_x, shift_y))
        return new_img
    
    def apply_blur(self, img, blur_probability=0.3):
        if random.random() < blur_probability:
            blur_radius = random.uniform(0.2, 0.8)
            return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return img
    
    def apply_noise(self, img, noise_probability=0.3, noise_intensity=10):
        if random.random() < noise_probability:
            # Convert to numpy array
            img_array = np.array(img)
            
            # Generate noise
            noise = np.random.normal(0, noise_intensity, img_array.shape)
            
            # Add noise and clip values
            noisy_img = img_array + noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            
            return Image.fromarray(noisy_img)
        return img
    
    def apply_brightness_contrast(self, img, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
        # Brightness adjustment
        brightness_factor = random.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        
        # Contrast adjustment
        contrast_factor = random.uniform(*contrast_range)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        
        return img
    
    def apply_background_variations(self, img, bg_variation_prob=0.2):
        if random.random() < bg_variation_prob:
            # Convert to RGBA to work with transparency
            img_rgba = img.convert('RGBA')
            
            # Create background with slight color variation
            bg_color = random.randint(240, 255)  # Light gray to white
            bg = Image.new('RGB', self.image_size, (bg_color, bg_color, bg_color))
            
            # Composite the character onto the new background
            bg.paste(img_rgba, mask=img_rgba)
            return bg
        return img
    
    def apply_morphological_operations(self, img, morph_prob=0.2):
        if random.random() < morph_prob:
            # Convert to opencv format
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Convert to binary
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Random morphological operation
            kernel_size = random.randint(1, 2)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if random.random() < 0.5:
                # Erosion (thinner strokes)
                binary = cv2.erode(binary, kernel, iterations=1)
            else:
                # Dilation (thicker strokes)
                binary = cv2.dilate(binary, kernel, iterations=1)
            
            # Convert back to PIL
            binary = cv2.bitwise_not(binary)  # Invert back
            img_cv = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return img
    
    def generate_augmented_character(self, character, font_path, font_name, variations=100):
        """Generate multiple augmented versions of a character"""
        # Create safe directory name using character's Unicode representation
        char_name = character.encode('unicode_escape').decode('ascii').replace('\\', '_')
        char_dir = self.output_dir / f"char_{char_name}"
        char_dir.mkdir(exist_ok=True)
        
        for i in range(variations):
            # Random font size
            font_size = random.randint(20, 50)
            
            # Generate base image
            img = self.generate_base_image(character, font_path, font_size)
            
            # Apply augmentations in random order
            augmentations = [
                self.apply_rotation,
                self.apply_position_jitter,
                self.apply_blur,
                self.apply_noise,
                self.apply_brightness_contrast,
                self.apply_background_variations,
                self.apply_morphological_operations
            ]
            
            # Randomly shuffle and apply augmentations
            random.shuffle(augmentations)
            for aug_func in augmentations:
                img = aug_func(img)
            
            # Save image
            filename = f"{font_name}_{i:04d}.png"
            img.save(char_dir / filename)
            
        print(f"Generated {variations} variations for character '{character}' with font {font_name}")
    
    def generate_dataset(self, characters, variations_per_font=100):
        """Generate complete dataset for all characters and fonts"""
        total_images = len(characters) * len(self.font_paths) * variations_per_font
        print(f"Generating {total_images} images total...")
        
        generated_count = 0
        for character in characters:
            for font_name, font_path in self.font_paths.items():
                if os.path.exists(font_path):
                    self.generate_augmented_character(character, font_path, font_name, variations_per_font)
                    generated_count += variations_per_font
                else:
                    print(f"Warning: Font file not found: {font_path}")
            
            print(f"Progress: {generated_count}/{total_images} images generated")


if __name__ == "__main__":
    # Define your Javanese characters (add more as needed)
    javanese_characters = [
        'ꦲ', 'ꦤ', 'ꦕ', 'ꦫ', 'ꦏ',
        'ꦢ', 'ꦠ', 'ꦱ', 'ꦮ', 'ꦭ', 
        'ꦥ', 'ꦣ', 'ꦗ', 'ꦪ', 'ꦚ',
        'ꦩ', 'ꦒ', 'ꦧ', 'ꦛ', 'ꦔ',
    ]

    sandhangan_characters = [
        'ꦶ', 'ꦸ', 'ꦺ', 'ꦺꦴ', 'ꦼ',
        'ꦂ', 'ꦁ', 'ꦃ',
        'ꦿ', 'ꦽ', 'ꦾ', 
        'ꦷ', 'ꦹ', '', 'ꦻ', 'ꦾ꦳'
    ]
    
    pasangan_characters = [
        '', '', '꧀ꦕ', '꧀ꦫ', '꧀ꦏ',
        '꧀ꦢ', '꧀ꦠ', '꧀ꦱ', '꧀ꦮ','꧀ꦭ', 
        '꧀ꦥ', '꧀ꦝ', '꧀ꦗ', '꧀ꦪ', '꧀ꦚ',
        '꧀ꦩ', '꧀ꦒ', '꧀ꦧ', '꧀ꦛ', '꧀ꦔ'
    ]


    print(f"Number of characters: {len(pasangan_characters)}")
    print("Characters:", pasangan_characters)
    
    # Define font paths (update these paths to your actual font locations)
    font_paths = {
        'javatext': 'D:/Python/Aksara/fonts/javatext.ttf',
        'indonesian_texts': 'D:/Python/Aksara/fonts/IndonesianTexts-Regular.ttf',
        'noto_sans_javanese_bold': 'D:/Python/Aksara/fonts/static/NotoSansJavanese-Bold.ttf',
        'noto_sans_javanese_medium': 'D:/Python/Aksara/fonts/static/NotoSansJavanese-Medium.ttf',
        'noto_sans_javanese_regular': 'D:/Python/Aksara/fonts/static/NotoSansJavanese-Regular.ttf',
        'noto_sans_javanese_semibold': 'D:/Python/Aksara/fonts/static/NotoSansJavanese-SemiBold.ttf'
    }
    
    # Create generator
    generator = JavaneseCharacterGenerator(
        font_paths=font_paths,
        output_dir='./pasangan_dataset',
        image_size=(224, 224)
    )
    
    # Generate dataset
    generator.generate_dataset(
        characters=pasangan_characters,
        variations_per_font=50 
    )



