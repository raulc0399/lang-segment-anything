import os
from models import LangSAMModel, ZoeDepthModel, DepthAnythingV2Model

# Create output directory if it doesn't exist
os.makedirs("imgs/output", exist_ok=True)

# Initialize models
langsam_model = LangSAMModel()
# zoe needs another version of timm, use pip install timm==0.6.7
# zoe_model = ZoeDepthModel()
depth_anything_model = DepthAnythingV2Model()

# Process images with each model
for img_name in ["l.png", "r.png"]:
    langsam_model.process_image(img_name)
    # zoe_model.process_image(img_name)
    depth_anything_model.process_image(img_name)
