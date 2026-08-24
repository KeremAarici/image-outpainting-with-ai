import os
from PIL import Image
from sd_inpainter import SDControlNetInpainter
from image_processor import ImageProcessor

def main():
    print("[INFO] Starting outpainting pipeline execution...")
    
    # Load sample input image (ensure you have a test image in your workspace)
    input_image_path = "input.jpg"
    if not os.path.exists(input_image_path):
        print(f"[WARNING] '{input_image_path}' not found. Please place a sample image in the directory.")
        return

    base_image = Image.open(input_image_path).convert("RGB")
    
    # Define padding configuration for outpainting (e.g., expand canvas by 128 pixels on each side)
    padding_config = {
        'top': 128,
        'bottom': 128,
        'left': 128,
        'right': 128
    }
    
    print("[INFO] Expanding canvas and generating mask...")
    expanded_image, mask_image = ImageProcessor.create_outpainting_mask(base_image, padding_config)
    
    # Initialize the inpainting pipeline
    inpainter = SDControlNetInpainter()
    
    # Define text prompt for the outpainted regions
    prompt = "a photorealistic continuation of the scene, highly detailed, 4k resolution"
    negative_prompt = "blurry, low quality, distortion, artifacts"
    
    print("[INFO] Running diffusion inference...")
    result_image = inpainter.predict(
        image=expanded_image,
        mask_image=mask_image,
        prompt=prompt,
        negative_prompt=negative_prompt
    )
    
    # Save output result
    output_path = "output_outpainted.png"
    result_image.save(output_path)
    print(f"[INFO] Outpainting successful! Saved result to {output_path}")

if __name__ == "__main__":
    main()