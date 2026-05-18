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
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)

    if category == 'dresses':
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body' or category == 'upperbody':
        parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                        (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'lower_body':
        parse_mask = (parse_array == 6).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    else:
        raise NotImplementedError

    # Tight organic semantic masking to prevent background blurring and grey box artifacts
    if category == 'dresses' or category == 'upper_body' or category == 'upperbody':
        # Snug 1-iteration dilation for the clothing region to prevent edge gaps
        parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=1)
        
<<<<<<< HEAD

        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            im_arms_right = arms_right
        else:
            wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
            arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            im_arms_left = arms_left
        else:
            wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
            arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
        hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
        parser_mask_fixed += hands_left + hands_right

    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
    
    # [FIX] Do not dilate clothes mask to achieve exact typical mask boundary
    # parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    
    if category == 'dresses' or category == 'upper_body' or category == 'upperbody':
=======
        # Neck protection/mask
>>>>>>> b076da8e4021f4d082bed9944ed09a1082b4a818
        neck_mask = (parse_array == 18).astype(np.float32)
        # [FIX] Do not dilate neck mask to prevent background edge artifacts
        # neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
<<<<<<< HEAD
        
        # [FIX] Only mask arms for long-sleeved garments, and keep it typical/exact to arm boundaries (no background leakage)
        if is_long_sleeve:
            im_arms_left_np = np.array(im_arms_left).astype(np.float32) / 255.0
            im_arms_right_np = np.array(im_arms_right).astype(np.float32) / 255.0
            arm_mask = np.logical_or(
                np.logical_and(im_arms_left_np, arms_left),
                np.logical_and(im_arms_right_np, arms_right)
            ).astype(np.float32)
            parse_mask = np.logical_or(parse_mask, arm_mask)
=======

        # Snug semantic arms (No giant straight drawn tube lines!)
        im_arms_left = arms_left
        im_arms_right = arms_right
        arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=1)
        parse_mask = np.logical_or(parse_mask, arm_mask)

    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray
>>>>>>> b076da8e4021f4d082bed9944ed09a1082b4a818

    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray


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