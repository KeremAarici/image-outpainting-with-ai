import os
from PIL import Image
from sd_inpainter import SDControlNetInpainter
from image_processor import ImageProcessor

def main():
    
    
    input_image_path = "input.jpg" 
    output_image_path = "output_outpainted.png"
    device = "cuda" # For NVIDIA GPU Usage
    
    
    if not os.path.exists(input_image_path):
        print(f"[Error] Input image not found: {input_image_path}")
        print("Please place an image named 'input.jpg' in the project's root directory.")
        return

    print("[INFO] Models starting...")
    processor = ImageProcessor()
    inpainter = SDControlNetInpainter(device=device)

    print("[INFO] images downloading")
    init_image = Image.open(input_image_path).convert("RGB")

    # You can adjust where and how far the image extends right here
    padding = {
        "top": 50,
        "bottom": 500,
        "left": 50,
        "right": 50
    }

    expanded_image, mask_image = processor.create_outpainting_mask(init_image, padding)

    
    print("[INFO] Creating the Canny corner mapping (boundary-safe)...")
    canny_image = processor.extract_canny_edges(init_image, padding)

    mask_image.save("debug_mask.png")
    canny_image.save("debug_canny.png")

    # Production Param.
    # Spesific Promt For Your Image. You have to cahnge here if you want to use a different input image
    prompt = "tropical island beach, seamless clear turquoise ocean, bright blue sky with light clouds, high resolution, photorealistic"
    negative_prompt = "blurry, dark, rocks, brown soil, frame, borders, sharp lines, low quality, text, watermark, canvas border"
    
    # Inference (Production)
    print("[INFO] Diffusion is starting.")
    
    # strength=1.0, causes the masked area to be completely redrawn.
    result_image = inpainter.predict(
        image=expanded_image,
        mask_image=mask_image,
        control_image=canny_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=0.42,
        strength=1.0,
        guidance_scale=7,
        num_inference_steps=50
    )

    # Save
    print(f"[INFO] Result saving: {output_image_path}")
    result_image.save(output_image_path)
    print("[INFO] Operation is successfull.")

if __name__ == "__main__":
    main()