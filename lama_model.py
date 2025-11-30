from diffusers import RePaintPipeline
import torch
from PIL import Image
import numpy as np

class LamaModel:
    def __init__(self, device="cpu"):
        """
        Initialize the RePaint inpainting model.
        device: "cpu" or "cuda"
        """
        print("Loading RePaint model (this may take a few minutes)...")
        self.pipe = RePaintPipeline.from_pretrained("google/remat-lama-office")
        self.pipe.to(device)
        print("RePaint model loaded!")

    def inpaint_pil(self, pil_img: Image.Image, mask_img: Image.Image):
        """
        Arguments:
            pil_img: PIL RGB image
            mask_img: PIL grayscale mask (white = area to fill, black = keep)
        Returns:
            PIL.Image (RGB) with inpainted result
        """
        # Ensure correct formats
        pil_img = pil_img.convert("RGB")
        mask_img = mask_img.convert("L")

        result = self.pipe(image=pil_img, mask_image=mask_img, num_inference_steps=50)
        return result.images[0]
