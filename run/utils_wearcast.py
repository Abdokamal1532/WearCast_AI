import numpy as np
import cv2
from PIL import Image, ImageDraw

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

def extend_arm_mask(wrist, elbow, scale):
    wrist = elbow + scale * (wrist - elbow)
    return wrist

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)
    return refine_mask

def analyze_sleeve_length(garm_mask_np):
    """
    Classify garment as short_sleeve or long_sleeve based on mask.
    Robustly handles wide t-shirts by checking how far DOWN the outer edges go.
    """
    import numpy as np
    
    if len(garm_mask_np.shape) == 3:
        # Keep only the largest connected component if the background removal was messy.
        garm_mask_np = garm_mask_np[:, :, 0]
        
    y_indices, x_indices = np.where(garm_mask_np > 0)
    if len(y_indices) == 0:
        return False

    y_min, y_max = np.min(y_indices), np.max(y_indices)
    height = y_max - y_min
    
    # [FIX] Boxy T-shirts were falsely triggering the "Long Sleeve" detector because their 
    # torsos are wide enough to fill the outer boundaries of the mask.
    # Instead of checking the outer 25% of the X-axis, we calculate the width profile of the garment.
    # A short-sleeve shirt is widest at the top (shoulders+sleeves) and much narrower at the bottom (torso).
    # A long-sleeve shirt has arms extending down, so the bottom is equal to or wider than the top.
    
    cropped_mask = garm_mask_np[y_min:y_max+1, :]
    widths = []
    for row in cropped_mask:
        xs = np.where(row > 0)[0]
        if len(xs) > 0:
            widths.append(xs[-1] - xs[0])
        else:
            widths.append(0)
    widths = np.array(widths)
    
    # Compare the top 30% width to the bottom 30% width
    idx_30 = int(0.3 * height)
    idx_70 = int(0.7 * height)
    
    if idx_30 == 0 or idx_70 >= len(widths):
        return False
        
    max_top_width = np.max(widths[:idx_30])
    max_bottom_width = np.max(widths[idx_70:])
    
    ratio = max_bottom_width / max_top_width if max_top_width > 0 else 1.0
    
    # If the bottom is at least 85% as wide as the top, it's a long sleeve.
    return ratio > 0.85

def get_mask_location(model_type, category, model_parse: Image.Image, keypoint: dict, width=384, height=512, is_long_sleeve=False):
    import cv2
    import numpy as np
    
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    print(" -> Constructing High-Precision Mask (Strict Clothing Boundaries)...")

    # 1. Base Mask: ONLY Upper clothes (4) and Dress (7)
    inpaint_mask = ((parse_array == 4) | (parse_array == 7)).astype(np.uint8) * 255

    pose_data = np.array(keypoint["pose_keypoints_2d"]).reshape((-1, 2))
    scale_factor = height / 512.0
    pt = lambda idx: np.multiply(tuple(pose_data[idx][:2]), scale_factor)

    # 3. Protect Identity (Face, Hair, Hat)
    # We protect the neck (18) to ensure smooth collar seams.
    skin_protect_base = ((parse_array == 11) | (parse_array == 2) | (parse_array == 1) | (parse_array == 18)).astype(np.uint8) * 255

    if category == 'upperbody':
        print(" -> [MASK] Covering full arms in generation mask to ensure all old sleeves are hidden.")
        arms = ((parse_array == 14) | (parse_array == 15)).astype(np.uint8) * 255
        inpaint_mask = cv2.bitwise_or(inpaint_mask, arms)
        skin_protect = skin_protect_base
    else:
        arms_protect = ((parse_array == 14) | (parse_array == 15)).astype(np.uint8) * 255
        skin_protect = cv2.bitwise_or(skin_protect_base, arms_protect)
    
    # 4. Protect Bottoms (Pants, Skirts)
    bottoms = ((parse_array == 5) | (parse_array == 6) | (parse_array == 9) | (parse_array == 10) | (parse_array == 12) | (parse_array == 13)).astype(np.uint8) * 255
    
    # 5. Adaptive Dilation
    # [FIX B] Kernel size adapts to how much of the image is already labeled as garment.
    # Large people (Gemini-style product models) often have UpperClothes = 25–35% of frame,
    # Normal real-world subjects (like Abdo) have 15-22%.
    # If the threshold is too low, we under-mask normal people, causing black boundary seams.
    garment_pixel_pct = np.sum(parse_array == 4) / max(1, parse_array.size)
    if garment_pixel_pct > 0.28:
        dilation_kernel_size = 15   # Massive parsed region (e.g. tight crop) — conservative
    elif garment_pixel_pct > 0.22:
        dilation_kernel_size = 20   # Medium-Large
    else:
        dilation_kernel_size = 25   # Standard real human — expand aggressively to prevent seams
    
    # [DEBUG MASK] Precise print
    print(f"   [DEBUG MASK] garment_pct={garment_pixel_pct:.2%} → selected dilation kernel={dilation_kernel_size}x{dilation_kernel_size}")
    mask_expanded = cv2.dilate(inpaint_mask, np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8), iterations=1)
    
    # [FIX C] Armpit Bridging
    arms = ((parse_array == 14) | (parse_array == 15)).astype(np.uint8) * 255
    junction_region = cv2.dilate(arms, np.ones((5,5), np.uint8), iterations=2)
    silhouette = (parse_array > 0).astype(np.uint8) * 255
    junction_region = cv2.bitwise_and(junction_region, silhouette)
    mask_expanded = cv2.bitwise_or(mask_expanded, junction_region)
    
    # SILHOUETTE LOCK: 
    mask_expanded = cv2.bitwise_and(mask_expanded, silhouette)
    
    # [FIX D] Cap the mask's lower boundary using the hip keypoints
    hip_y = max(pose_data[8][1], pose_data[11][1])  # RHip, LHip keypoints
    hip_y_scaled = int(hip_y * scale_factor)
    if 0 < hip_y_scaled < height:
        mask_expanded[hip_y_scaled:, :] = 0
        
    # [FIX E] Enforce aspect-ratio constraint:
    shoulder_width = abs(pose_data[2][0] - pose_data[5][0]) * scale_factor
    mask_cols = np.where(np.any(mask_expanded > 0, axis=0))[0]
    if len(mask_cols) > 0 and shoulder_width > 0:
        mask_actual_width = mask_cols[-1] - mask_cols[0]
        if mask_actual_width > shoulder_width * 1.15:
            trim = int((mask_actual_width - shoulder_width * 1.15) / 2)
            if mask_cols[0] + trim < mask_cols[-1] - trim:
                mask_expanded[:, :mask_cols[0] + trim] = 0
                mask_expanded[:, mask_cols[-1] - trim:] = 0

    # [FIX F] Explicit bottom-protection step
    pants_protect = ((parse_array == 5) | (parse_array == 6)).astype(np.uint8) * 255
    pants_protect_dilated = cv2.dilate(pants_protect, np.ones((10,10), np.uint8))
    mask_expanded[pants_protect_dilated == 255] = 0

    # Apply protections
    mask_expanded[skin_protect == 255] = 0
    mask_expanded[bottoms == 255] = 0

    # 6. Fill internal holes
    def hole_fill_local(img):
        img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
        img_copy = img.copy()
        mask_fill = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(img, mask_fill, (0, 0), 255)
        img_inverse = cv2.bitwise_not(img)
        return cv2.bitwise_or(img_copy, img_inverse)

    mask_hard = hole_fill_local(mask_expanded)

    # Organic Smoothing
    mask_hard = cv2.GaussianBlur(mask_hard, (11, 11), 0)
    mask_hard = (mask_hard > 127).astype(np.uint8) * 255

    # Feathered Soft Mask
    mask_soft = cv2.GaussianBlur(mask_hard.astype(np.float32), (9, 9), 0)
    inpaint_mask_soft = np.clip(mask_soft / 255.0, 0, 1)

    percentage = 100 * np.sum(mask_hard > 0) / (width * height)
    print(f" -> Optimized Mask: {percentage:.1f}% coverage.")

    return Image.fromarray(mask_hard), Image.fromarray((inpaint_mask_soft * 255).astype(np.uint8))


def smart_resize(img: Image.Image, target_size=(768, 1024), fill_color=(255, 255, 255)):
    """
    Resizes image while maintaining aspect ratio, adding padding to reach target_size.
    """
    img_w, img_h = img.size
    target_w, target_h = target_size

    aspect = img_w / img_h
    target_aspect = target_w / target_h

    if aspect > target_aspect:
        new_w = target_w
        new_h = int(target_w / aspect)
    else:
        new_h = target_h
        new_w = int(target_h * aspect)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    new_img = Image.new("RGB", target_size, fill_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))

    return new_img