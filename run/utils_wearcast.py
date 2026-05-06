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
    # 4: upper_clothes, 7: dress, 17: scarf
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
    s_r, s_l = pt(2), pt(5)   # Shoulders
    e_r, e_l = pt(3), pt(6)   # Elbows
    
    # ============================================================
    # SLEEVE-SUPPORT OPTIMIZATION: CONVEX HULL (Shoulders + Elbows)
    # ============================================================
    hull_pts = []
    valid = lambda p: p[0] > 1 and p[1] > 1
    
    # Moderate lateral padding for T-shirt sleeves (35px for better coverage)
    ARM_PAD = int(35 / 512 * height) 
    
    # Add more points around shoulders to ensure full deltoid coverage
    for p in [s_r, s_l, e_r, e_l]:
        if valid(p):
            hull_pts.append([p[0] + ARM_PAD, p[1]])
            hull_pts.append([p[0] - ARM_PAD, p[1]])
            hull_pts.append([p[0], p[1] - ARM_PAD // 2]) # Top of shoulder
    
    inpaint_mask = target_area.copy()
    torso_pixels = np.column_stack(np.where(target_area > 0))
    
    if len(torso_pixels) > 5 and len(hull_pts) >= 3:
        torso_xy = torso_pixels[:, [1, 0]]
        if len(torso_xy) > 600:
            idx = np.random.choice(len(torso_xy), 600, replace=False)
            torso_xy = torso_xy[idx]
        
        all_pts = np.vstack([torso_xy, np.array(hull_pts)]).astype(np.float32)
        hull = cv2.convexHull(all_pts)
        hull_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull.astype(np.int32), 255)
        
        inpaint_mask = np.logical_or(inpaint_mask, hull_mask / 255.0).astype(np.float32)
    elif len(torso_pixels) > 5:
        inpaint_mask = cv2.dilate(inpaint_mask, np.ones((15, 15), np.uint8), iterations=1)

    # Neck region: Conservative (1px dilation)
    neck_area = (parse_array == 18).astype(np.float32)
    neck_tight = cv2.dilate(neck_area, np.ones((1, 1), np.uint8), iterations=1)
    inpaint_mask = np.logical_or(inpaint_mask, neck_tight).astype(np.float32)

    # FINAL PROTECTION: Remove head AND forearm area
    inpaint_mask = np.logical_and(inpaint_mask, np.logical_not(head_only)).astype(np.float32)
    
    # Forearm Protection: Only protect arms BELOW the elbow level to allow sleeves
    forearm_protection = np.zeros_like(arms_labels)
    for e_pt in [e_r, e_l]:
        if valid(e_pt):
            elbow_y = int(e_pt[1])
            forearm_protection[elbow_y + 20:, :] = 1
    
    arms_to_protect = np.logical_and(arms_labels, forearm_protection)
    inpaint_mask = np.logical_and(inpaint_mask, np.logical_not(arms_to_protect * 0.95)).astype(np.float32)
    
    # Smooth with small kernel
    inpaint_mask = cv2.dilate(inpaint_mask, np.ones((5, 5), np.uint8), iterations=1)

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
    
    mask_soft = cv2.GaussianBlur(dst.astype(np.float32), (11, 11), 3) 
    inpaint_mask_soft = np.clip(mask_soft / 255.0, 0, 1)

    return Image.fromarray(dst), Image.fromarray((inpaint_mask_soft * 255).astype(np.uint8))

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
