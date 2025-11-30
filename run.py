# run_inpaint.py
from lama_model import LamaModel
from PIL import Image

# Load images
image = Image.open("half_face.png")
mask = Image.open("mask.png")

# Initialize model
model = LamaModel(device="cpu")  # use "cuda" if you have GPU

# Run inpainting
result = model.inpaint_pil(image, mask)

# Save result
result.save("completed_face.png")
print("✅ Completed face saved as completed_face.png")
