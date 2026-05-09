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

def get_mask_location(model_type, category, model_parse: Image.Image, keypoint: dict, width=384, height=512):
    import cv2
    import numpy as np
    
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    print(" -> Constructing High-Precision Mask (Strict Clothing Boundaries)...")

    # 1. Base Mask: ONLY Upper clothes (4) and Dress (7)
    # لا ندمج الأذرع هنا أبداً لمنع تحول الجلد إلى قماش
    inpaint_mask = ((parse_array == 4) | (parse_array == 7)).astype(np.uint8) * 255

    # 2. Add Neck area (to allow collar changes)
    pose_data = np.array(keypoint["pose_keypoints_2d"]).reshape((-1, 2))
    scale_factor = height / 512.0
    pt = lambda idx: np.multiply(tuple(pose_data[idx][:2]), scale_factor)

    if pt(1)[0] > 1: # Neck keypoint
        cv2.circle(inpaint_mask, (int(pt(1)[0]), int(pt(1)[1])), int(15 * scale_factor), 255, -1)

    # 3. Protect Identity & Skin (Face, Hair, Hat, Bare Arms)
    # نحمي الأذرع والوجه صراحةً حتى لو تمدد القناع لا يغطيهم
    skin_protect = ((parse_array == 11) | (parse_array == 2) | (parse_array == 1) | (parse_array == 14) | (parse_array == 15)).astype(np.uint8) * 255
    
    # 4. Protect Bottoms (Pants, Skirts)
    bottoms = ((parse_array == 5) | (parse_array == 6) | (parse_array == 9) | (parse_array == 10) | (parse_array == 12) | (parse_array == 13)).astype(np.uint8) * 255
    
    # 5. Dilation: Give the UNet room to draw the shirt slightly larger
    # --- [PRODUCTION FIX] Silhouette-Locked Dilation ---
    mask_expanded = cv2.dilate(inpaint_mask, np.ones((15, 15), np.uint8), iterations=1)
    
    # SILHOUETTE LOCK: 
    # We strictly prevent the mask from going outside the human body boundary (Labels > 0)
    # This kills the "Gray Wings" effect on the background.
    silhouette = (parse_array > 0).astype(np.uint8) * 255
    mask_expanded = cv2.bitwise_and(mask_expanded, silhouette)
    
    # Apply protections (طرح مناطق الحماية من القناع المتمدد)
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
    print(f" -> Optimized Mask: {percentage:.1f}% coverage. Clothing only, arms protected.")

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