import numpy as np
import cv2
from PIL import Image, ImageFilter

class ImageProcessor:
    """
    Utility class for image preprocessing, mask generation, 
    and boundary-safe ControlNet edge extraction.
    """
    
    @staticmethod
    def create_outpainting_mask(image: Image.Image, padding: dict, blur_radius: int = 12) -> tuple[Image.Image, Image.Image]:
        """
        Expands canvas and creates a blurred mask for seamless boundary blending.
        """
        width, height = image.size
        
        new_width = width + padding.get('left', 0) + padding.get('right', 0)
        new_height = height + padding.get('top', 0) + padding.get('bottom', 0)
        
        padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        padded_image.paste(image, (padding.get('left', 0), padding.get('top', 0)))
        
        mask = Image.new("L", (new_width, new_height), 255)
        
        # Inner preserved area slightly shrinked for feathering overlap
        overlap = 12
        inner_width = max(1, width - (overlap * 2))
        inner_height = max(1, height - (overlap * 2))
        
        original_area = Image.new("L", (inner_width, inner_height), 0)
        mask.paste(original_area, (padding.get('left', 0) + overlap, padding.get('top', 0) + overlap))
        
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
        return padded_image, mask

    @staticmethod
    def extract_canny_edges(image: Image.Image, padding: dict, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """
        Extracts Canny edges ONLY from the original image to avoid detecting 
        the artificial canvas frame border as a real edge.
        """
        width, height = image.size
        new_width = width + padding.get('left', 0) + padding.get('right', 0)
        new_height = height + padding.get('top', 0) + padding.get('bottom', 0)

        # Run the corner detection on only original image 
        image_np = np.array(image.convert("RGB"))
        canny_edges = cv2.Canny(image_np, low_threshold, high_threshold)
        canny_3channel = np.stack([canny_edges] * 3, axis=-1)
        canny_pil = Image.fromarray(canny_3channel)

        padded_canny = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        padded_canny.paste(canny_pil, (padding.get('left', 0), padding.get('top', 0)))

        return padded_canny