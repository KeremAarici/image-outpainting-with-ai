import os
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel

class SDControlNetInpainter:
    """
    Inpainting pipeline leveraging Stable Diffusion and ControlNet (Canny edge guidance)
    for seamless image completion and semantic object generation.
    """
    def __init__(self, device: str = None):
        # Automatically detect CUDA GPU availability if not specified
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"[INFO] Initializing SDControlNetInpainter on device: {self.device}")
        
        # Load pre-trained Canny ControlNet checkpoint
        print("[INFO] Loading ControlNet model (Canny)...")
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=self.torch_dtype
        )
        
        # Initialize the Stable Diffusion Inpainting pipeline integrated with ControlNet
        print("[INFO] Loading Stable Diffusion Inpainting pipeline...")
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=self.controlnet,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        # Enable VRAM optimizations when running on GPU
        if self.device == "cuda":
            print("[INFO] Enabling attention slicing for VRAM optimization.")
            self.pipe.enable_attention_slicing()

    def generate_canny_map(self, image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """
        Extract Canny edges from an input PIL image to serve as ControlNet structural condition.
        """
        image_np = np.array(image.convert("RGB"))
        canny_edges = cv2.Canny(image_np, low_threshold, high_threshold)
        canny_3channel = np.stack([canny_edges] * 3, axis=-1)
        return Image.fromarray(canny_3channel)

    def predict(
        self, 
        image: Image.Image, 
        mask_image: Image.Image, 
        prompt: str, 
        negative_prompt: str = "blurry, low quality, distorted, extra limbs, bad anatomy",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0
    ) -> Image.Image:
        """
        Executes the inpainting pipeline with spatial edge guidance.

        Parameters:
            image (PIL.Image): Original base image.
            mask_image (PIL.Image): Inpainting mask (White = regenerate, Black = preserve).
            prompt (str): Text describing the intended completion/generation.
            negative_prompt (str): Text describing unwanted artifacts or features.
            num_inference_steps (int): Total diffusion denoising steps.
            guidance_scale (float): Classifier-free guidance scale.
            controlnet_conditioning_scale (float): Weight of the ControlNet edge influence.
        """
        print(f"[INFO] Processing image generation with prompt: '{prompt}'")
        
        # Extract edge map from input image for structural guidance
        control_image = self.generate_canny_map(image)

        # Run Latent Diffusion inference
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            control_image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale
        ).images[0]

        print("[INFO] Inference complete successfully.")
        return output

if __name__ == "__main__":
    # Self-test block for pipeline verification
    print("[INFO] Running pipeline verification test...")
    inpainter = SDControlNetInpainter()
    print("[INFO] Pipeline initialized ready for commits.")