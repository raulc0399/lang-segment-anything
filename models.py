import torch
import cv2
import numpy as np
from PIL import Image
from lang_sam import LangSAM
from depth_anything_v2.dpt import DepthAnythingV2
from lang_sam.utils import draw_image
from colorize import colorize

class LangSAMModel:
    def __init__(self, device="cuda"):
        self.model = LangSAM(sam_type="sam2.1_hq_hiera_large", device=device)
        
    def process_image(self, img_name):
        # Load image
        image_pil = Image.open(f"imgs/{img_name}").convert("RGB")
        
        # Get predictions
        results = self.model.predict(
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

class ZoeDepthModel:
    def __init__(self, device="cuda"):
        self.model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(device).eval()
        
    def process_image(self, img_name):
        image_pil = Image.open(f"imgs/{img_name}").convert("RGB")
        depth = self.model.infer_pil(image_pil)
        
        colored_depth = colorize(depth, cmap='gray_r')
        raw_depth = Image.fromarray(colored_depth)

        # Save depth image
        raw_depth.save(f"imgs/output/depth_zoe_{img_name}")
        print(f"Processed depth for {img_name}")

class DepthAnythingV2Model:
    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    def __init__(self, encoder='vitl', device="cuda"):
        self.model = DepthAnythingV2(**self.MODEL_CONFIGS[encoder])
        self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model = self.model.to(device).eval()
        
    def process_image(self, img_name):
        raw_img = cv2.imread(f"imgs/{img_name}")
        depth = self.model.infer_image(raw_img)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        gray_depth = Image.fromarray(depth)
        
        # Save depth image
        gray_depth.save(f"imgs/output/depth_depth_anything_2_{img_name}")
        print(f"Processed depth for {img_name}")
