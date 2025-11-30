from PIL import Image
import numpy as np

def resize_image(pil_img, size=(512,512)):
    return pil_img.convert("RGB").resize(size)

def auto_generate_mask(pil_img, white_threshold=250):
    """Generate a mask from white pixels (white = hole)."""
    img_np = np.array(pil_img.convert("RGB"))
    mask = np.all(img_np >= white_threshold, axis=2).astype(np.uint8) * 255
    return Image.fromarray(mask, mode='L')
