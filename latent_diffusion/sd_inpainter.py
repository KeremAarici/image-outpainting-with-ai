import torch
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image

class SDControlNetInpainter:
    """
    Manages the Stable Diffusion Inpainting + ControlNet Canny pipeline for high-quality outpainting.
    """
    def __init__(self, device="cuda", model_id="runwayml/stable-diffusion-inpainting", controlnet_id="lllyasviel/sd-controlnet-canny"):
        self.device = device
        
        # 1. Load ControlNet model
        print(f"[INFO] Loading ControlNet model: {controlnet_id}...")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id, torch_dtype=torch.float16
        )
        
        # 2. Load Inpainting Pipeline with ControlNet attached
        print(f"[INFO] Loading Inpainting Pipeline: {model_id}...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, controlnet=self.controlnet, torch_dtype=torch.float16
        )
        
        # 3. Optimize scheduler for faster, high-quality generation
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        # 4. Move to GPU (or CPU offload if VRAM is tight)
        self.pipe.to(device)
        
        # Optional VRAM optimization (uncomment if you get CUDA OOM)
        # self.pipe.enable_model_cpu_offload()

    def predict(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        control_image: Image.Image, # For Canny Corner Mapping Algorithm
        prompt: str,
        negative_prompt: str = "",
        controlnet_conditioning_scale: float = 0.5,
        strength: float = 1.0,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: int = -1
    ) -> Image.Image:
        """
        Executes the diffusion inpainting process with ControlNet guidance.
        """
        print("[INFO] Generating image...")

        # Setup generator for reproducibility if needed
        if seed != -1:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Execute pipeline
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image, # Padded/expanded base image
            mask_image=mask_image, # Gaussian blurred mask
            control_image=control_image, # ControlNet input
            controlnet_conditioning_scale=controlnet_conditioning_scale, # ControlNet influence
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )

        return output.images[0]