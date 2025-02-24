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
        

    def process_image(self, img_name, return_mask=False):
        """Process image and optionally return separated arm/hand masks"""
        image_pil = Image.open(f"imgs/{img_name}").convert("RGB")
        results = self.model.predict(
            images_pil=[image_pil],
            texts_prompt=["arms. hands."],
            box_threshold=0.3,
            text_threshold=0.25,
        )[0]
        
        if len(results["masks"]):
            image_array = np.asarray(image_pil)
            output_image = draw_image(
                image_array,
                results["masks"],
                results["boxes"],
                results["scores"],
                results["labels"],
            )
            output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")
            
            if return_mask:
                # Initialize masks
                mask_shape = results["masks"][0].shape
                arm_mask = np.zeros(mask_shape, dtype=bool)
                hand_mask = np.zeros(mask_shape, dtype=bool)
                
                # Separate masks based on labels
                for mask, label in zip(results["masks"], results["labels"]):
                    if label == 'arms':
                        arm_mask = np.logical_or(arm_mask, mask)
                    elif label == 'hands':
                        hand_mask = np.logical_or(hand_mask, mask)
                
                # Remove hand regions from arm mask
                arm_mask = np.logical_and(arm_mask, ~hand_mask)

                # Save arm mask
                # arm_mask_img = Image.fromarray((arm_mask * 255).astype(np.uint8))
                # arm_mask_img.save(f"imgs/output/arm_mask_{img_name}")

                # Save hand mask
                # hand_mask_img = Image.fromarray((hand_mask * 255).astype(np.uint8))
                # hand_mask_img.save(f"imgs/output/hand_mask_{img_name}")

                return arm_mask, hand_mask
        else:
            output_image = image_pil
            if return_mask:
                return None, None

        # Save visualization
        output_image.save(f"imgs/output/hands_{img_name}")
                    
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
