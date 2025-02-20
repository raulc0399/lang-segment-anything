import numpy as np
from PIL import Image
import os

from lang_sam import LangSAM
from lang_sam.utils import draw_image

# Create output directory if it doesn't exist
os.makedirs("imgs/output", exist_ok=True)

# Initialize model
model = LangSAM(sam_type="sam2.1_hq_hiera_large", device="cuda")

# Process both images
for img_name in ["l.png", "r.png"]:
    # Load image
    image_pil = Image.open(f"imgs/{img_name}").convert("RGB")
    
    # Get predictions
    results = model.predict(
        images_pil=[image_pil],
        texts_prompt=["arms. hands."],
        box_threshold=0.3,
        text_threshold=0.25,
    )[0]
    
    if len(results["masks"]):
        # Draw results on the image
        image_array = np.asarray(image_pil)
        output_image = draw_image(
            image_array,
            results["masks"],
            results["boxes"],
            results["scores"],
            results["labels"],
        )
        output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")
    else:
        output_image = image_pil
        
    # Save output
    output_image.save(f"imgs/output/{img_name}")
    print(f"Processed {img_name}")
