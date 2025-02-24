import cv2
import numpy as np

def merge_depth_images(arm_path, hand_path):
    # Read the images
    arm = cv2.imread(arm_path, cv2.IMREAD_GRAYSCALE)
    hand = cv2.imread(hand_path, cv2.IMREAD_GRAYSCALE)
    
    # Create masks for each part
    # For the arm image - we want the white/gray part
    _, arm_mask = cv2.threshold(arm, 1, 255, cv2.THRESH_BINARY)
    
    # For the hand image - we want the grayscale part
    _, hand_mask = cv2.threshold(hand, 1, 255, cv2.THRESH_BINARY)
    
    # Create the final image
    result = np.zeros_like(arm)
    
    # Copy the arm part
    result = cv2.bitwise_and(arm, arm_mask)
    
    # Add the hand part
    hand_region = cv2.bitwise_and(hand, hand_mask)
    
    # Combine them
    # Where hand_region has values, use those; otherwise keep arm values
    result = np.where(hand_region > 0, hand_region, result)
    
    # Create a transition mask for the wrist area
    kernel = np.ones((5,5), np.uint8)
    transition_mask = cv2.dilate(hand_mask, kernel) & cv2.dilate(arm_mask, kernel)

    # Apply stronger blending in the transition area
    transition_area = cv2.GaussianBlur(result, (7,7), 0)

    # Blend the transition smoothly
    alpha = cv2.GaussianBlur(transition_mask.astype(float), (15,15), 0) / 255
    result = cv2.convertScaleAbs(result * (1 - alpha) + transition_area * alpha)
    
    return result

def interpolate_gap(arm_path, hand_path):
    # Read the images
    arm = cv2.imread(arm_path, cv2.IMREAD_GRAYSCALE)
    hand = cv2.imread(hand_path, cv2.IMREAD_GRAYSCALE)

    # Create masks for non-black pixels
    arm_mask = arm > 0
    hand_mask = hand > 0
    
    # Find the gap (black pixels between arm and hand)
    # Dilate both masks slightly to ensure we catch the gap
    kernel = np.ones((5,5), np.uint8)
    dilated_arm = cv2.dilate(arm_mask.astype(np.uint8), kernel)
    dilated_hand = cv2.dilate(hand_mask.astype(np.uint8), kernel)
    
    # The gap area is where both dilated masks overlap but original pixels are black
    gap_mask = (dilated_arm & dilated_hand) & ~(arm_mask | hand_mask)
    
    # Create the result image
    result = np.copy(arm)
    result = np.where(hand > 0, hand, result)
    
    # For each black pixel in the gap, interpolate between nearest non-black pixels
    y_indices, x_indices = np.where(gap_mask)
    
    for y, x in zip(y_indices, x_indices):
        # Look for nearest non-black pixels horizontally
        left_x = x
        while left_x >= 0 and result[y, left_x] == 0:
            left_x -= 1
            
        right_x = x
        while right_x < result.shape[1] and result[y, right_x] == 0:
            right_x += 1
            
        # If we found valid pixels on both sides
        if left_x >= 0 and right_x < result.shape[1]:
            left_val = result[y, left_x]
            right_val = result[y, right_x]
            # Linear interpolation
            alpha = (x - left_x) / (right_x - left_x)
            result[y, x] = int((1 - alpha) * left_val + alpha * right_val)
    
    return result

def main():
    # Paths to your images
    arm_image_path = './imgs/output/depth_arms_masked_r.png'  # Image 1
    hand_image_path = './imgs/r_controlnet_img.png'  # Image 2
    
    # Merge the images
    result = merge_depth_images(arm_image_path, hand_image_path)

    # result = interpolate_gap(arm_image_path, hand_image_path)
    
    # Save the result
    cv2.imwrite('./imgs/output/merged_depth.png', result)

if __name__ == "__main__":
    main()