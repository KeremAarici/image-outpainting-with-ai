import numpy as np
import cv2
from PIL import Image

class ImageProcessor:
    """
    Utility class for image preprocessing, mask generation, 
    and ControlNet condition extraction.
    """
    
    @staticmethod
    def create_outpainting_mask(image: Image.Image, padding: dict) -> tuple[Image.Image, Image.Image]:
        """
        Expands the image canvas and generates a corresponding inpainting mask.
        
        padding format: {'top': int, 'bottom': int, 'left': int, 'right': int}
        Returns: (expanded_image, generated_mask)
        """
        width, height = image.size
        
        new_width = width + padding.get('left', 0) + padding.get('right', 0)
        new_height = height + padding.get('top', 0) + padding.get('bottom', 0)
        
        # Create padded background image (filled with black/neutral pixels)
        padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        padded_image.paste(image, (padding.get('left', 0), padding.get('top', 0)))
        
        # Create mask: Black (0) for preserved original areas, White (255) for expansion zones
        mask = Image.new("L", (new_width, new_height), 255)
        original_area = Image.new("L", (width, height), 0)
        mask.paste(original_area, (padding.get('left', 0), padding.get('top', 0)))
        
        return padded_image, mask

    @staticmethod
    def extract_canny_edges(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """
        Generates Canny edge feature map required for ControlNet conditioning.
        """
        image_np = np.array(image.convert("RGB"))
        canny_edges = cv2.Canny(image_np, low_threshold, high_threshold)
        
        # Convert 1-channel grayscale edge map to 3-channel RGB image
        canny_3channel = np.stack([canny_edges] * 3, axis=-1)
        return Image.fromarray(canny_3channel)

if __name__ == "__main__":
    print("[INFO] ImageProcessor module verified successfully.")