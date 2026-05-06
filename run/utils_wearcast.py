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
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    print(" -> Constructing High-Precision Mask (T-shirt Torso Only)...")

    # ================================================================
    # STEP 1: Base mask = only the segmented upper clothing pixels
    # Labels: 4=upper_clothes, 7=dress, 17=scarf/accessory
    # We do NOT include arm labels (14, 15) here.
    # ================================================================
    target_area = (
        (parse_array == 4).astype(np.uint8) |
        (parse_array == 7).astype(np.uint8) |
        (parse_array == 17).astype(np.uint8)
    )
    inpaint_mask = target_area.astype(np.uint8) * 255

    # ================================================================
    # STEP 2: Add a SMALL shoulder-cap region only
    #   We extend from the shoulder keypoint by only 1/3 of the
    #   shoulder→elbow vector — this covers the sleeve cap of a
    #   T-shirt WITHOUT reaching the forearm.
    # ================================================================
    pose_data = np.array(keypoint["pose_keypoints_2d"]).reshape((-1, 2))
    scale_factor = height / 512.0
    pt = lambda idx: np.multiply(tuple(pose_data[idx][:2]), scale_factor)

    # OpenPose indices: 2=RShoulder, 5=LShoulder, 3=RElbow, 6=LElbow
    s_r, s_l = pt(2), pt(5)   # Shoulders
    e_r, e_l = pt(3), pt(6)   # Elbows

    valid = lambda p: p[0] > 1 and p[1] > 1

    # Sleeve cap width — keep it tight (12px at 512h)
    SLEEVE_PAD = int(12 / 512 * height)

    def add_sleeve_cap(shoulder, elbow):
        """
        Creates a small convex polygon covering only the sleeve cap:
        from the shoulder to 1/3 of the way toward the elbow.
        This covers a T-shirt short sleeve without touching the forearm.
        """
        if not (valid(shoulder) and valid(elbow)):
            return None

        shoulder = np.array(shoulder, dtype=np.float32)
        elbow    = np.array(elbow,    dtype=np.float32)

        # Sleeve endpoint = shoulder + 1/3 * (elbow - shoulder)
        sleeve_end = shoulder + (1.0 / 3.0) * (elbow - shoulder)

        # Direction perpendicular to shoulder→elbow for width
        direction = elbow - shoulder
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return None
        perp = np.array([-direction[1], direction[0]]) / length

        pts = np.array([
            shoulder + perp * SLEEVE_PAD,
            shoulder - perp * SLEEVE_PAD,
            sleeve_end - perp * SLEEVE_PAD,
            sleeve_end + perp * SLEEVE_PAD,
        ], dtype=np.int32)

        cap_mask = np.zeros_like(inpaint_mask)
        cv2.fillConvexPoly(cap_mask, pts, 255)
        return cap_mask

    left_cap  = add_sleeve_cap(s_l, e_l)
    right_cap = add_sleeve_cap(s_r, e_r)

    if left_cap  is not None: inpaint_mask = cv2.bitwise_or(inpaint_mask, left_cap)
    if right_cap is not None: inpaint_mask = cv2.bitwise_or(inpaint_mask, right_cap)

    # ================================================================
    # STEP 3: Neck notch — add a small rectangle at the neckline
    #   so the collar area is included in the mask.
    # ================================================================
    if valid(pt(1)):  # Neck keypoint
        neck_pt = pt(1)
        NECK_PAD = int(15 / 512 * height)
        neck_top  = int(neck_pt[1]) - int(NECK_PAD * 0.5)
        neck_bot  = int(neck_pt[1]) + int(NECK_PAD * 1.0)
        neck_left = int(neck_pt[0]) - NECK_PAD
        neck_right= int(neck_pt[0]) + NECK_PAD
        cv2.rectangle(inpaint_mask, (neck_left, neck_top), (neck_right, neck_bot), 255, -1)

    # ================================================================
    # STEP 4: Constrain mask to body silhouette
    #   Only allow mask where the parser sees body/clothes.
    #   CRITICAL: do NOT include arm labels (14, 15) here.
    #   This is what prevented the old mask from spreading down the arms.
    # ================================================================
    torso_constraint = (
        (parse_array == 4)  |   # upper_clothes
        (parse_array == 7)  |   # dress
        (parse_array == 17) |   # scarf
        (parse_array == 11) |   # head/neck
        (parse_array == 2)  |   # hair
        (parse_array == 1)      # hat
    ).astype(np.uint8) * 255

    # Small dilation to catch seam pixels (3×3 only, not 15×15)
    torso_constraint = cv2.dilate(torso_constraint, np.ones((3, 3), np.uint8), iterations=1)

    # For the sleeve cap areas only, we need a slightly wider constraint
    # that includes just the shoulder portion of the arm label
    shoulder_only = (
        (parse_array == 14) |   # left_arm  (shoulder portion only after erosion)
        (parse_array == 15)     # right_arm
    ).astype(np.uint8) * 255
    # Erode arm labels heavily so only the shoulder attachment remains
    shoulder_only = cv2.erode(shoulder_only, np.ones((20, 20), np.uint8), iterations=1)
    shoulder_only = cv2.dilate(shoulder_only, np.ones((5, 5), np.uint8), iterations=1)

    combined_constraint = cv2.bitwise_or(torso_constraint, shoulder_only)
    inpaint_mask = cv2.bitwise_and(inpaint_mask, combined_constraint)

    # ================================================================
    # STEP 5: Also constrain to overall body (prevent background bleed)
    # ================================================================
    body_mask = ((parse_array > 0) & (parse_array != 16)).astype(np.uint8) * 255
    body_mask_dilated = cv2.dilate(body_mask, np.ones((5, 5), np.uint8), iterations=1)
    inpaint_mask = cv2.bitwise_and(inpaint_mask, body_mask_dilated)

    # ================================================================
    # STEP 6: Hole fill + largest-contour refinement
    # ================================================================
    def hole_fill_local(img):
        img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
        img_copy = img.copy()
        mask_fill = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(img, mask_fill, (0, 0), 255)
        img_inverse = cv2.bitwise_not(img)
        return cv2.bitwise_or(img_copy, img_inverse)

    def refine_mask_local(mask_ref):
        cnts, _ = cv2.findContours(mask_ref.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        areas = [abs(cv2.contourArea(c, True)) for c in cnts]
        refined = np.zeros_like(mask_ref).astype(np.uint8)
        if areas:
            cv2.drawContours(refined, cnts, areas.index(max(areas)), color=255, thickness=-1)
        return refined

    filled = hole_fill_local(np.where(inpaint_mask, 255, 0).astype(np.uint8))
    dst = refine_mask_local(filled)

    # ================================================================
    # STEP 7: Final morphology — keep dilation SMALL (3×3)
    #   Old code used 7×7 which was expanding the mask too aggressively.
    # ================================================================
    kernel_dilate = np.ones((3, 3), np.uint8)
    mask_hard = cv2.dilate(dst.astype(np.uint8), kernel_dilate, iterations=1)

    # Remove small noise
    kernel_open = np.ones((5, 5), np.uint8)
    mask_hard = cv2.morphologyEx(mask_hard, cv2.MORPH_OPEN, kernel_open)

    # --- FIX #7: Organic Smoothing (Anti-Box) ---
    # We blur the mask heavily and re-threshold to create natural rounded edges
    # that follow body curves rather than rigid OpenPose polygons.
    mask_hard = cv2.GaussianBlur(mask_hard, (15, 15), 0)
    mask_hard = (mask_hard > 127).astype(np.uint8) * 255

    # Feather edges slightly for smooth blending
    mask_soft = cv2.GaussianBlur(mask_hard.astype(np.float32), (5, 5), 0)
    inpaint_mask_soft = np.clip(mask_soft / 255.0, 0, 1)

    percentage = 100 * np.sum(dst > 0) / (width * height)
    print(f" -> Optimized Mask: {percentage:.1f}% coverage. Restricted to torso/sleeve-caps only.")

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