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
    """
    import numpy as np
    
    if len(garm_mask_np.shape) == 3:
        garm_mask_np = garm_mask_np[:, :, 0]
        
    y_indices, x_indices = np.where(garm_mask_np > 0)
    if len(y_indices) == 0:
        return False

    height = np.max(y_indices) - np.min(y_indices)
    width = np.max(x_indices) - np.min(x_indices)
    
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    
    # Check bottom 25% of the image height
    bottom_y_start = y_max - int(0.25 * height)
    # Check outer 20% of the image width
    left_x_end = x_min + int(0.20 * width)
    right_x_start = x_max - int(0.20 * width)
    
    bottom_left_region = garm_mask_np[bottom_y_start:y_max, x_min:left_x_end]
    bottom_right_region = garm_mask_np[bottom_y_start:y_max, right_x_start:x_max]
    
    left_density = np.sum(bottom_left_region > 0) / max(1, bottom_left_region.size)
    right_density = np.sum(bottom_right_region > 0) / max(1, bottom_right_region.size)
    
    # TEMPORARY FIX: Disable long sleeve detection to prevent false positives for wide short-sleeve shirts.
    return False
        
    return False

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

    # 3. Protect Identity & Skin (Face, Hair, Hat, Neck)
    # Neck is explicitly protected to avoid glitches around collar
    skin_protect_base = ((parse_array == 11) | (parse_array == 2) | (parse_array == 1) | (parse_array == 18)).astype(np.uint8) * 255

    if category == 'upperbody':
        if is_long_sleeve:
            print(" -> [MASK] Long sleeve detected. Including full arms in generation mask.")
            arms = ((parse_array == 14) | (parse_array == 15)).astype(np.uint8) * 255
            inpaint_mask = cv2.bitwise_or(inpaint_mask, arms)
            skin_protect = skin_protect_base
        else:
            # FIX #3: Short-sleeve — include only the UPPER half of each arm (shoulder→elbow).
            # Previously the full arm to the wrist was sometimes included, creating phantom
            # sleeve strips when the person wore a sleeveless garment originally.
            print(" -> [MASK] Short sleeve detected. Including upper arm only (shoulder→elbow).")
            # Draw thick line from shoulder keypoint to a point 50% down towards the elbow
            for s_idx, e_idx in [(2, 3), (5, 6)]:
                if pt(s_idx)[0] > 1 and pt(e_idx)[0] > 1:
                    s_pt = (int(pt(s_idx)[0]), int(pt(s_idx)[1]))
                    e_pt = (int(pt(e_idx)[0]), int(pt(e_idx)[1]))
                    dx = e_pt[0] - s_pt[0]
                    dy = e_pt[1] - s_pt[1]
                    # [FIX] Draw to 75% of shoulder→elbow distance (was 45%).
                    # 45% was too conservative: target garment sleeves were wider than
                    # original, leaving shoulder gaps in the composite output.
                    mid_pt = (int(s_pt[0] + dx * 0.75), int(s_pt[1] + dy * 0.75))
                    cv2.line(inpaint_mask, s_pt, mid_pt, 255, int(28 * scale_factor))

            # FIX #3: Protect the LOWER arm (elbow→wrist) but NOT the upper arm near the sleeve.
            # Build a forearm-only mask using the elbow/wrist keypoints.
            forearm_protect = np.zeros_like(parse_array, dtype=np.uint8)
            for e_idx, w_idx in [(3, 4), (6, 7)]:   # (RElbow→RWrist), (LElbow→LWrist)
                if pt(e_idx)[0] > 1 and pt(w_idx)[0] > 1:
                    e_pt = (int(pt(e_idx)[0]), int(pt(e_idx)[1]))
                    w_pt = (int(pt(w_idx)[0]), int(pt(w_idx)[1]))
                    cv2.line(forearm_protect, e_pt, w_pt, 255, int(24 * scale_factor))
            forearm_protect = cv2.dilate(forearm_protect, np.ones((9, 9), np.uint8), iterations=1)

            # Combine: protect face/hair/neck + forearms (below elbow)
            arms_protect = cv2.bitwise_or(
                ((parse_array == 14) | (parse_array == 15)).astype(np.uint8) * 255,
                forearm_protect
            )
            skin_protect = cv2.bitwise_or(skin_protect_base, forearm_protect)
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
    
    # SILHOUETTE LOCK: 
    silhouette = (parse_array > 0).astype(np.uint8) * 255
    mask_expanded = cv2.bitwise_and(mask_expanded, silhouette)
    
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