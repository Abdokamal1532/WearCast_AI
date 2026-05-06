import pdb

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
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
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
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    # 2. High-Accuracy Pose-Guided Mask Generation (Optimized for "T-shirt Only")
    print(" -> Constructing High-Precision Mask (Target: Tshirt Only)...")
    
    # Labels to TARGET for generation (core clothes)
    # 4: upper_clothes, 7: dress, 17: scarf/accessory
    target_area = (parse_array == 4).astype(np.float32) + \
                  (parse_array == 7).astype(np.float32) + \
                  (parse_array == 17).astype(np.float32)

    # Labels to PROTECT (Face, Hair, and Arms as much as possible)
    head_only = (parse_array == 1).astype(np.float32) + \
                (parse_array == 3).astype(np.float32) + \
                (parse_array == 11).astype(np.float32)
    
    # Arm labels to protect
    arms_labels = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)
    
    # Pose keypoints
    pose_data = np.array(keypoint["pose_keypoints_2d"]).reshape((-1, 2))
    # Correct scale factor calculation
    scale_factor = height / 512.0
    pt = lambda idx: np.multiply(tuple(pose_data[idx][:2]), scale_factor)
    
    # OpenPose: 2=RShoulder, 5=LShoulder, 3=RElbow, 6=LElbow
    # OpenPose: 2=RShoulder, 5=LShoulder, 3=RElbow, 6=LElbow, 4=RWrist, 7=LWrist
    s_r, s_l = pt(2), pt(5)   # Shoulders
    e_r, e_l = pt(3), pt(6)   # Elbows
    w_r, w_l = pt(4), pt(7)   # Wrists
    
    # ============================================================
    # SLEEVE-SUPPORT OPTIMIZATION: CONVEX HULL (Shoulders + Elbows)
    # ============================================================
    valid = lambda p: p[0] > 1 and p[1] > 1
    
    # Moderate lateral padding for T-shirt sleeves (35px for better coverage)
    ARM_PAD = int(35 / 512 * height) 
    
    # 3. Create Arm Masks separately (to avoid filling the gap between arm and torso)
    inpaint_mask = (target_area > 0).astype(np.uint8) * 255
    
    def add_arm_mask_local(shoulder, elbow, side_pad):
        if valid(shoulder) and valid(elbow):
            # Create a local hull for this arm only
            arm_pts = []
            # Shoulder points (deltoid)
            arm_pts.append([shoulder[0] - side_pad, shoulder[1]])
            arm_pts.append([shoulder[0] + side_pad, shoulder[1]])
            arm_pts.append([shoulder[0], shoulder[1] - side_pad // 2])
            # Elbow points
            arm_pts.append([elbow[0] - side_pad, elbow[1]])
            arm_pts.append([elbow[0] + side_pad, elbow[1]])
            
            arm_mask = np.zeros_like(inpaint_mask)
            cv2.fillConvexPoly(arm_mask, np.array(arm_pts, dtype=np.int32), 255)
            return arm_mask
        return None

    # Add Left and Right arm masks independently
    l_arm = add_arm_mask_local(s_l, e_l, ARM_PAD)
    r_arm = add_arm_mask_local(s_r, e_r, ARM_PAD)
    
    if l_arm is not None: inpaint_mask = cv2.bitwise_or(inpaint_mask, l_arm)
    if r_arm is not None: inpaint_mask = cv2.bitwise_or(inpaint_mask, r_arm)
    
    # Constrain mask to body silhouette (prevent background bleed)
    body_mask = ((parse_array > 0) & (parse_array != 16)).astype(np.uint8) * 255
    body_mask_dilated = cv2.dilate(body_mask, np.ones((9, 9), np.uint8), iterations=1)
    inpaint_mask = cv2.bitwise_and(inpaint_mask, body_mask_dilated)
    
    # 4. Forearm Protection (Exclude area below elbows)
    # Erase everything below the elbow to protect the skin
    if valid(e_l) and valid(w_l):
        cv2.line(inpaint_mask, (int(e_l[0]), int(e_l[1])), (int(w_l[0]), int(w_l[1])), 0, thickness=int(ARM_PAD * 2.5))
    if valid(e_r) and valid(w_r):
        cv2.line(inpaint_mask, (int(e_r[0]), int(e_r[1])), (int(w_r[0]), int(w_r[1])), 0, thickness=int(ARM_PAD * 2.5))

    # Neck region: Use pose keypoints instead of non-existent parse label 18
    if valid(pt(1)):  # Neck keypoint
        neck_pt = pt(1)
        neck_box_y_top = int(neck_pt[1]) - int(ARM_PAD * 0.5)
        neck_box_y_bot = int(neck_pt[1]) + int(ARM_PAD * 1.0)
        neck_box_x_left = int(neck_pt[0]) - ARM_PAD
        neck_box_x_right = int(neck_pt[0]) + ARM_PAD
        cv2.rectangle(inpaint_mask,
                      (neck_box_x_left, neck_box_y_top),
                      (neck_box_x_right, neck_box_y_bot), 255, -1)
    
    # 5. Hole Filling and Refinement

    # Hole fill + largest-contour refinement
    def hole_fill_local(img):
        img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
        img_copy = img.copy()
        mask_fill = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(img, mask_fill, (0, 0), 255)
        img_inverse = cv2.bitwise_not(img)
        dst_fill = cv2.bitwise_or(img_copy, img_inverse)
        return dst_fill

    def refine_mask_local(mask_ref):
        cnts, _ = cv2.findContours(mask_ref.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        areas = [abs(cv2.contourArea(c, True)) for c in cnts]
        refined_mask = np.zeros_like(mask_ref).astype(np.uint8)
        if len(areas) != 0:
            idx_max = areas.index(max(areas))
            cv2.drawContours(refined_mask, cnts, idx_max, color=255, thickness=-1)
        return refined_mask

    filled = hole_fill_local(np.where(inpaint_mask, 255, 0).astype(np.uint8))
    dst = refine_mask_local(filled)
    
    # Smoothing & Refinement
    # We use a slight dilation to ensure the old garment is fully covered
    kernel_dilate = np.ones((7, 7), np.uint8)
    mask_hard = cv2.dilate(dst.astype(np.uint8), kernel_dilate, iterations=1)
    
    # Open operation to remove small noise
    kernel_open = np.ones((5, 5), np.uint8)
    mask_hard = cv2.morphologyEx(mask_hard, cv2.MORPH_OPEN, kernel_open)
    
    # Feather the edges slightly (5x5 instead of 11x11) 
    # This keeps the center of the mask SOLID (1.0) to prevent ghosting
    mask_soft = cv2.GaussianBlur(mask_hard.astype(np.float32), (5, 5), 0)
    inpaint_mask_soft = np.clip(mask_soft / 255.0, 0, 1)

    percentage = 100 * np.sum(dst > 0) / (width * height)
    print(f" -> Optimized Mask: {percentage:.1f}% coverage. Restricted to torso/shoulders.")
    
    return Image.fromarray(mask_hard), Image.fromarray((inpaint_mask_soft * 255).astype(np.uint8))

def smart_resize(img: Image.Image, target_size=(768, 1024), fill_color=(255, 255, 255)):
    """
    Resizes image while maintaining aspect ratio, adding padding to reach target_size.
    This prevents the 'stretched' look common in naive resizing.
    """
    img_w, img_h = img.size
    target_w, target_h = target_size
    
    # Calculate aspect ratios
    aspect = img_w / img_h
    target_aspect = target_w / target_h
    
    if aspect > target_aspect:
        # Image is wider than target: fit to width
        new_w = target_w
        new_h = int(target_w / aspect)
    else:
        # Image is taller than target: fit to height
        new_h = target_h
        new_w = int(target_h * aspect)
        
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Create background and paste
    new_img = Image.new("RGB", target_size, fill_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    
    return new_img
