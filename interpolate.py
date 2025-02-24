import numpy as np
from PIL import Image
import cv2

def interpolate_between_masks(arm_image_path, hand_image_path):
    # Load images
    arm_img = np.array(Image.open(arm_image_path))
    hand_img = np.array(Image.open(hand_image_path))
    
    # Convert hand image to grayscale if it's RGB
    if len(hand_img.shape) == 3:
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_RGB2GRAY)
    
    # Create masks for non-zero regions
    arm_mask = arm_img > 0
    hand_mask = hand_img > 0
    
    # Create target mask for interpolation (area between arm and hand)
    # Find the bounding box that contains both masks
    y_indices, x_indices = np.where(arm_mask | hand_mask)
    top, bottom = y_indices.min(), y_indices.max()
    left, right = x_indices.min(), x_indices.max()
    
    # Create a mask for the region between arm and hand
    region_mask = np.zeros_like(arm_mask)
    region_mask[top:bottom+1, left:right+1] = True
    target_mask = region_mask & ~(arm_mask | hand_mask)
    
    # Initialize result image with original values
    result = np.zeros_like(arm_img, dtype=np.float32)
    result[arm_mask] = arm_img[arm_mask]
    result[hand_mask] = hand_img[hand_mask]
    
    # Create distance maps to both masks
    dist_to_arm = cv2.distanceTransform((~arm_mask).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_hand = cv2.distanceTransform((~hand_mask).astype(np.uint8), cv2.DIST_L2, 5)
    
    # Calculate weights for interpolation
    total_dist = dist_to_arm + dist_to_hand
    epsilon = 1e-10  # Small value to prevent division by zero
    arm_weight = dist_to_hand / (total_dist + epsilon)
    hand_weight = dist_to_arm / (total_dist + epsilon)
    
    # Interpolate in the target region
    target_pixels = target_mask & (total_dist > 0)
    result[target_pixels] = (
        arm_weight[target_pixels] * arm_img[arm_mask].mean() +
        hand_weight[target_pixels] * hand_img[hand_mask].mean()
    )
    
    # Save result
    result = result.astype(np.uint8)
    result_img = Image.fromarray(result)
    result_img.save('./imgs/output/interpolated_depth.png')

if __name__ == "__main__":
    arm_image_path = './imgs/output/depth_arms_masked_r.png'  # Image 1
    hand_image_path = './imgs/r_controlnet_img.png'  # Image 2
    
    interpolate_between_masks(arm_image_path, hand_image_path)
    print("Created interpolated depth map")
