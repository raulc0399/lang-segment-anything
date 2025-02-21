import os
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
        langsam_model.process_image(img_name, return_mask=True)
        
        
