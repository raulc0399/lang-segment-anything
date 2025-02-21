import os
from PIL import Image
import numpy as np
from models import LangSAMModel, ZoeDepthModel, DepthAnythingV2Model

os.makedirs("imgs/output", exist_ok=True)

generate_depth = False

if generate_depth:
    # zoe needs another version of timm, use pip install timm==0.6.7
    # zoe_model = ZoeDepthModel()
    depth_anything_model = DepthAnythingV2Model()

    for img_name in ["l.png", "r.png"]:
        # zoe_model.process_image(img_name)
        depth_anything_model.process_image(img_name)
else:
    langsam_model = LangSAMModel()
    
    for img_name in ["l.png", "r.png"]:
        arm_mask, _ = langsam_model.process_image(img_name, return_mask=True)
        
        if arm_mask is not None:
            # Load depth image
            depth_img = Image.open(f"imgs/output/depth_depth_anything_2_{img_name}")
            depth_array = np.array(depth_img)
            
            # Apply arm mask to depth
            masked_depth = depth_array.copy()
            masked_depth[~arm_mask] = 0
            
            # Save masked depth image
            masked_depth_img = Image.fromarray(masked_depth)
            masked_depth_img.save(f"imgs/output/depth_arms_masked_{img_name}")
            print(f"Saved masked depth for {img_name}")
        
