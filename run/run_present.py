"""
WearCast AI — Full Pipeline Presentation Runner
================================================
Run this on Kaggle to capture EVERY detail from every step of the pipeline
and save all outputs, visualisations, and metrics to:

    <PROJECT_ROOT>/run_output_present/

Phases captured:
  Phase 0  — Model loading & hardware diagnostics
  Phase 1  — Human Parsing & Pose Estimation
  Phase 2  — Latent Processing & Fusion (CLIP + VAE)
  Phase 3  — Generation & Reconstruction (Denoising Diffusion)
  Phase 4  — Post-processing & Compositing

Usage (Kaggle notebook cell):
    !python run/run_present.py \
        --person  path/to/person.jpg \
        --garment path/to/garment.jpg
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Bootstrap — silence noise before any heavy imports
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, json, time, warnings, textwrap
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

try:
    import transformers
    if not hasattr(transformers, "EncoderDecoderCache"):
        class EncoderDecoderCache: pass
        transformers.EncoderDecoderCache = EncoderDecoderCache
    import transformers.utils
    if not hasattr(transformers.utils, "FLAX_WEIGHTS_NAME"):
        transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
except ImportError:
    pass

try:
    from diffusers.utils import logging as _dl
    _dl.set_verbosity_error()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Path setup
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Standard library + third-party (all installed on Kaggle/T4)
# ─────────────────────────────────────────────────────────────────────────────
import argparse, math, shutil, datetime, traceback
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Output directory helpers
# ─────────────────────────────────────────────────────────────────────────────
OUT = PROJECT_ROOT / "run_output_present"

def mk(*parts):
    """Create directory and return its Path."""
    p = OUT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p

DIRS = {
    "root"          : mk(),
    "phase0_model"  : mk("phase0_model_loading"),
    "phase1_parse"  : mk("phase1_human_parsing_pose"),
    "phase2_latent" : mk("phase2_latent_processing_fusion"),
    "phase3_gen"    : mk("phase3_generation_diffusion"),
    "phase4_post"   : mk("phase4_postprocessing"),
    "summary"       : mk("summary"),
}

LOG_PATH = OUT / "full_pipeline_log.txt"
log_file = open(LOG_PATH, "w", encoding="utf-8")

def log(msg: str):
    """Dual-write: console + log file."""
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

def save_img(img, directory, filename):
    """Save PIL image and return the path."""
    p = Path(directory) / filename
    img.save(str(p))
    log(f"   [SAVED] {p.relative_to(PROJECT_ROOT)}")
    return p

def save_json(data: dict, directory, filename):
    """Save dict as pretty JSON and return the path."""
    p = Path(directory) / filename
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log(f"   [SAVED] {p.relative_to(PROJECT_ROOT)}")
    return p

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────
LABEL_NAMES = {
    0:'Background', 1:'Hat', 2:'Hair', 3:'Sunglasses', 4:'UpperClothes',
    5:'Skirt', 6:'Pants', 7:'Dress', 8:'Belt', 9:'LeftShoe', 10:'RightShoe',
    11:'Face', 12:'LeftLeg', 13:'RightLeg', 14:'LeftArm', 15:'RightArm',
    16:'Bag', 17:'Scarf', 18:'Neck',
}

LABEL_COLORS = {
    0: (20,20,20),    1: (255,128,0),   2: (200,100,50),  3: (100,200,255),
    4: (0,200,0),     5: (100,0,200),   6: (0,100,200),   7: (200,0,200),
    8: (128,128,0),   9: (0,128,128),   10:(128,0,128),   11:(255,200,100),
    12:(0,255,128),   13:(128,255,0),   14:(255,50,50),    15:(50,50,255),
    16:(200,200,0),   17:(0,200,200),   18:(255,150,200),
}

KP_NAMES = ['Nose','Neck','RShoulder','RElbow','RWrist',
            'LShoulder','LElbow','LWrist',
            'RHip','RKnee','RAnkle','LHip','LKnee','LAnkle',
            'REye','LEye','REar','LEar']

SKELETON = [
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
    (0,14),(14,16),(0,15),(15,17),
]

SKEL_COLORS = [
    (255,100,0),(255,140,0),(255,180,0),(255,220,0),
    (0,200,100),(0,230,140),(0,255,180),
    (100,100,255),(140,140,255),(180,180,255),
    (255,0,200),(255,100,200),(255,150,200),
    (255,255,0),(255,200,0),(200,255,0),(150,255,0),
]


def draw_skeleton(img_pil: Image.Image, keypoints_2d: list) -> Image.Image:
    """Draw coloured skeleton + keypoint dots on a copy of img_pil."""
    vis = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)
    pts = np.array(keypoints_2d).reshape(-1, 2)

    w, h = vis.size
    sx = w / 384.0   # keypoints were computed at 384×512
    sy = h / 512.0

    # Bones
    for i, (a, b) in enumerate(SKELETON):
        if a < len(pts) and b < len(pts):
            xa, ya = pts[a][0]*sx, pts[a][1]*sy
            xb, yb = pts[b][0]*sx, pts[b][1]*sy
            if xa > 1 and ya > 1 and xb > 1 and yb > 1:
                color = SKEL_COLORS[i % len(SKEL_COLORS)]
                draw.line([(xa, ya), (xb, yb)], fill=color, width=4)

    # Joints
    r = 6
    for i, (x, y) in enumerate(pts):
        x, y = x*sx, y*sy
        if x > 1 and y > 1:
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(255,255,255), outline=(0,0,0), width=2)

    return vis


def draw_parse_colormap(parse_arr: np.ndarray, size=None) -> Image.Image:
    """Convert integer label array to a colourful segmentation map."""
    h, w = parse_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in LABEL_COLORS.items():
        rgb[parse_arr == label] = color
    img = Image.fromarray(rgb)
    if size:
        img = img.resize(size, Image.NEAREST)
    return img


def make_legend(labels_present: list) -> Image.Image:
    """Create a small legend image for the parse colour map."""
    n = len(labels_present)
    W, row_h = 260, 26
    H = n * row_h + 10
    leg = Image.new("RGB", (W, H), (30, 30, 30))
    draw = ImageDraw.Draw(leg)
    for i, lbl in enumerate(labels_present):
        y = 5 + i * row_h
        c = LABEL_COLORS.get(lbl, (200, 200, 200))
        draw.rectangle([8, y+3, 28, y+row_h-3], fill=c)
        draw.text((36, y+4), f"{lbl:2d}  {LABEL_NAMES.get(lbl, '?')}", fill=(255,255,255))
    return leg


def titled_strip(*images, title: str, bg=(15,15,25), gap=12, title_h=40) -> Image.Image:
    """Stitch images side-by-side with a dark title bar."""
    max_h = max(im.height for im in images)
    total_w = sum(im.width for im in images) + gap * (len(images)-1)
    strip = Image.new("RGB", (total_w, max_h + title_h), bg)
    draw = ImageDraw.Draw(strip)
    draw.rectangle([0,0,total_w, title_h-4], fill=(30,30,50))
    draw.text((10, 8), title, fill=(200,220,255))
    x = 0
    for im in images:
        strip.paste(im.convert("RGB"), (x, title_h))
        x += im.width + gap
    return strip


def overlay_label(img: Image.Image, text: str, pos="bottom",
                  bg=(0,0,0,160), fg=(255,255,255)) -> Image.Image:
    """Add a semi-transparent label banner."""
    out = img.convert("RGBA")
    ov = Image.new("RGBA", out.size, (0,0,0,0))
    d = ImageDraw.Draw(ov)
    w, h = out.size
    bh = 32
    y0 = h - bh if pos == "bottom" else 0
    d.rectangle([0, y0, w, y0+bh], fill=bg)
    d.text((8, y0+6), text, fill=fg)
    return Image.alpha_composite(out, ov).convert("RGB")


def tensor_summary(t, label="") -> dict:
    """Return a dict of stats for a torch tensor."""
    if not isinstance(t, torch.Tensor):
        return {"type": str(type(t))}
    f = t.float()
    return {
        "label"  : label,
        "shape"  : list(t.shape),
        "dtype"  : str(t.dtype),
        "device" : str(t.device),
        "min"    : round(float(f.min()), 5),
        "max"    : round(float(f.max()), 5),
        "mean"   : round(float(f.mean()), 5),
        "std"    : round(float(f.std()), 5),
    }


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
    return float(10 * np.log10(255**2 / max(mse, 1e-6)))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main presentation runner
# ─────────────────────────────────────────────────────────────────────────────
def run(person_path: str, garment_path: str):
    global_start = time.time()
    timeline = []   # list of {phase, event, t}

    def tick(phase, event):
        t = time.time() - global_start
        timeline.append({"phase": phase, "event": event, "elapsed_s": round(t, 3)})
        log(f"\n{'─'*60}")
        log(f"[{t:7.2f}s]  {phase}  ▶  {event}")
        log('─'*60)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 0 — Hardware & Model Loading
    # ──────────────────────────────────────────────────────────────────────────
    tick("PHASE-0", "Hardware diagnostics & model loading")

    hw = {
        "timestamp"    : datetime.datetime.now().isoformat(),
        "python"       : sys.version,
        "torch"        : torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        hw.update({
            "gpu_name"      : props.name,
            "gpu_vram_gb"   : round(props.total_memory / 1e9, 2),
            "cuda_version"  : torch.version.cuda,
            "cudnn_version" : str(torch.backends.cudnn.version()),
        })
    save_json(hw, DIRS["phase0_model"], "hardware_info.json")
    for k, v in hw.items():
        log(f"   {k:25s}: {v}")

    # Load model
    from wearcast.inference_wearcast_hd import WearCastHD
    from run.utils_wearcast import smart_resize

    tick("PHASE-0", "Instantiating WearCastHD model")
    t_load = time.time()
    model = WearCastHD(0)
    hw["model_load_time_s"] = round(time.time() - t_load, 2)
    log(f"   Model loaded in {hw['model_load_time_s']}s")

    if torch.cuda.is_available():
        hw["vram_after_load_gb"] = round(torch.cuda.memory_allocated(0)/1e9, 3)
    save_json(hw, DIRS["phase0_model"], "hardware_info.json")

    # ──────────────────────────────────────────────────────────────────────────
    # Load input images
    # ──────────────────────────────────────────────────────────────────────────
    tick("PHASE-0", "Loading input images")
    person_orig  = Image.open(person_path).convert("RGB")
    garment_orig = Image.open(garment_path).convert("RGB")

    person_img  = smart_resize(person_orig)
    garment_img = smart_resize(garment_orig)

    save_img(person_orig,  DIRS["phase0_model"], "input_person_original.jpg")
    save_img(garment_orig, DIRS["phase0_model"], "input_garment_original.jpg")
    save_img(person_img,   DIRS["phase0_model"], "input_person_resized_768x1024.jpg")
    save_img(garment_img,  DIRS["phase0_model"], "input_garment_resized_768x1024.jpg")

    input_strip = titled_strip(
        overlay_label(person_img,  "Person  (768×1024)"),
        overlay_label(garment_img, "Garment (768×1024)"),
        title="PHASE 0 — Input Images"
    )
    save_img(input_strip, DIRS["phase0_model"], "inputs_side_by_side.jpg")

    input_meta = {
        "person_path"       : person_path,
        "garment_path"      : garment_path,
        "person_orig_size"  : list(person_orig.size),
        "garment_orig_size" : list(garment_orig.size),
        "person_resized"    : list(person_img.size),
        "garment_resized"   : list(garment_img.size),
    }
    save_json(input_meta, DIRS["phase0_model"], "input_metadata.json")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1 — Human Parsing & Pose Estimation
    # ──────────────────────────────────────────────────────────────────────────
    tick("PHASE-1", "Loading Human Parsing (SCHP) & OpenPose models")

    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from run.utils_wearcast import get_mask_location, analyze_sleeve_length

    gpu_id_int = int(model.gpu_id.split(":")[1])
    t1 = time.time()
    parsing_model = Parsing(gpu_id_int)
    openpose_model = OpenPose(gpu_id_int)
    log(f"   Parsing + OpenPose loaded in {time.time()-t1:.2f}s")

    # ── 1a  Garment matting ──────────────────────────────────────────────────
    tick("PHASE-1", "Garment background removal (rembg / U2-Net)")
    t_mat = time.time()
    garm_proc, garm_mask = model.remove_garment_background_rembg(garment_img)
    t_mat = time.time() - t_mat

    is_long_sleeve = analyze_sleeve_length(garm_mask)
    log(f"   rembg time     : {t_mat:.2f}s")
    log(f"   is_long_sleeve : {is_long_sleeve}")
    log(f"   garment fg px  : {int(garm_mask.sum())} / {garm_mask.size} ({100*garm_mask.mean():.1f}%)")

    garm_mask_vis = Image.fromarray((garm_mask * 255).astype(np.uint8))
    save_img(garm_proc,     DIRS["phase1_parse"], "p1a_garment_bg_removed.jpg")
    save_img(garm_mask_vis, DIRS["phase1_parse"], "p1a_garment_fg_mask.jpg")

    mat_strip = titled_strip(
        overlay_label(garment_img,  "Original garment"),
        overlay_label(garm_proc,    "BG removed (rembg)"),
        overlay_label(garm_mask_vis,"Foreground mask"),
        title="PHASE 1a — Garment Matting (U2-Net rembg)"
    )
    save_img(mat_strip, DIRS["phase1_parse"], "p1a_garment_matting_strip.jpg")

    matting_meta = {
        "rembg_time_s"         : round(t_mat, 3),
        "is_long_sleeve"       : bool(is_long_sleeve),
        "fg_pixel_count"       : int(garm_mask.sum()),
        "total_pixel_count"    : int(garm_mask.size),
        "fg_coverage_percent"  : round(100*float(garm_mask.mean()), 2),
        "garment_proc_size"    : list(garm_proc.size),
    }
    save_json(matting_meta, DIRS["phase1_parse"], "p1a_matting_metadata.json")

    # ── 1b  OpenPose ─────────────────────────────────────────────────────────
    tick("PHASE-1", "OpenPose keypoint detection")
    t0 = time.time()
    keypoints = openpose_model(person_img)
    t_pose = time.time() - t0

    pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape(-1, 2)
    log(f"   OpenPose time  : {t_pose:.2f}s")
    log(f"   Keypoints      : {len(pose_data)}")
    for i, (name, pt) in enumerate(zip(KP_NAMES, pose_data)):
        log(f"     KP[{i:02d}] {name:12s}: ({pt[0]:.1f}, {pt[1]:.1f})")

    # Skeleton visualisation
    skeleton_vis = draw_skeleton(person_img, keypoints["pose_keypoints_2d"])
    save_img(skeleton_vis, DIRS["phase1_parse"], "p1b_openpose_skeleton.jpg")

    # Heatmap-style point-only view
    heatmap_vis = person_img.copy().convert("RGBA")
    ov = Image.new("RGBA", heatmap_vis.size, (0,0,0,0))
    d2 = ImageDraw.Draw(ov)
    w_img, h_img = person_img.size
    sx = w_img / 384.0; sy = h_img / 512.0
    for i, (x, y) in enumerate(pose_data):
        if x > 1 and y > 1:
            cx, cy = x*sx, y*sy
            r = 10
            d2.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255,80,80,220))
            d2.text((cx+r+2, cy-8), KP_NAMES[i] if i < len(KP_NAMES) else str(i),
                    fill=(255,255,0,255))
    heatmap_vis = Image.alpha_composite(heatmap_vis, ov).convert("RGB")
    save_img(heatmap_vis, DIRS["phase1_parse"], "p1b_openpose_keypoints_labeled.jpg")

    pose_strip = titled_strip(
        overlay_label(person_img,   "Person input"),
        overlay_label(skeleton_vis, "Skeleton"),
        overlay_label(heatmap_vis,  "Labelled keypoints"),
        title="PHASE 1b — OpenPose Skeleton & Keypoints"
    )
    save_img(pose_strip, DIRS["phase1_parse"], "p1b_openpose_strip.jpg")

    pose_meta = {
        "inference_time_s": round(t_pose, 3),
        "keypoint_count"  : int(len(pose_data)),
        "keypoints"       : {
            KP_NAMES[i]: {"x": round(float(pose_data[i][0]),2),
                          "y": round(float(pose_data[i][1]),2)}
            for i in range(min(len(KP_NAMES), len(pose_data)))
        },
    }
    save_json(pose_meta, DIRS["phase1_parse"], "p1b_pose_keypoints.json")

    # ── 1c  Human Parsing ────────────────────────────────────────────────────
    tick("PHASE-1", "Human Parsing — semantic segmentation (SCHP)")
    t0 = time.time()
    model_parse, face_mask = parsing_model(person_img)
    t_parse = time.time() - t0

    parse_arr    = np.array(model_parse)
    labels_found = [int(l) for l in np.unique(parse_arr)]
    log(f"   Parsing time   : {t_parse:.2f}s")
    log(f"   Output size    : {model_parse.size}")
    log(f"   Labels found   : {labels_found}")
    label_stats = {}
    for l in labels_found:
        cnt = int(np.sum(parse_arr == l))
        pct = round(100*cnt/parse_arr.size, 2)
        log(f"     [{l:2d}] {LABEL_NAMES.get(l,'?'):20s}: {cnt:7d} px  ({pct}%)")
        label_stats[str(l)] = {"name": LABEL_NAMES.get(l,"?"), "pixels": cnt, "percent": pct}

    parse_color = draw_parse_colormap(parse_arr, size=person_img.size)
    legend      = make_legend(labels_found)
    save_img(parse_color, DIRS["phase1_parse"], "p1c_parse_colormap.jpg")
    save_img(legend,      DIRS["phase1_parse"], "p1c_parse_legend.png")

    # Blend parse overlay on person
    blend = Image.blend(person_img.convert("RGB"), parse_color, 0.5)
    save_img(blend, DIRS["phase1_parse"], "p1c_parse_overlay_blend.jpg")

    parse_strip = titled_strip(
        overlay_label(person_img,  "Person input"),
        overlay_label(parse_color, "Segmentation map"),
        overlay_label(blend,       "Overlay (50%)"),
        title="PHASE 1c — Human Parsing Segmentation"
    )
    save_img(parse_strip, DIRS["phase1_parse"], "p1c_parse_strip.jpg")

    parse_meta = {
        "inference_time_s" : round(t_parse, 3),
        "output_size"      : list(model_parse.size),
        "labels"           : label_stats,
    }
    save_json(parse_meta, DIRS["phase1_parse"], "p1c_parse_metadata.json")

    # ── 1d  Mask generation ──────────────────────────────────────────────────
    tick("PHASE-1", "Inpainting mask generation (get_mask_location)")

    mask_hard, mask_gray = get_mask_location(
        "hd", "upperbody", model_parse, keypoints,
        width=384, height=512, is_long_sleeve=is_long_sleeve
    )
    mask_np      = np.array(mask_hard)
    mask_pixels  = int(np.sum(mask_np > 127))
    total_pixels = int(mask_np.size)
    log(f"   Mask pixels    : {mask_pixels} / {total_pixels}  ({100*mask_pixels/total_pixels:.2f}%)")

    # Upscale mask
    mask_hr = mask_hard.resize((768, 1024), Image.BILINEAR)
    mask_hr_np = (np.array(mask_hr) > 127).astype(np.uint8) * 255
    mask_hr    = Image.fromarray(mask_hr_np)

    # Gray-area masked person (what UNet receives)
    mask_gray_np = mask_hr_np.astype(np.uint8) // 2   # 127 gray
    mask_gray_pil = Image.fromarray(mask_gray_np)
    person_masked = Image.composite(mask_gray_pil, person_img, mask_hr)

    save_img(mask_hard,    DIRS["phase1_parse"], "p1d_mask_hard_384x512.jpg")
    save_img(mask_hr,      DIRS["phase1_parse"], "p1d_mask_hard_768x1024.jpg")
    save_img(person_masked,DIRS["phase1_parse"], "p1d_person_masked_gray.jpg")

    # Coloured mask overlay
    mask_red = Image.new("RGB", person_img.size, (255,50,50))
    mask_ov  = Image.composite(mask_red, person_img, mask_hr)
    save_img(mask_ov, DIRS["phase1_parse"], "p1d_mask_overlay_red.jpg")

    mask_strip = titled_strip(
        overlay_label(person_img,    "Person"),
        overlay_label(mask_hr,       "Binary mask (white=erase)"),
        overlay_label(mask_ov,       "Mask overlay"),
        overlay_label(person_masked, "UNet input (127-gray)"),
        title="PHASE 1d — Inpainting Mask Generation"
    )
    save_img(mask_strip, DIRS["phase1_parse"], "p1d_mask_strip.jpg")

    mask_meta = {
        "is_long_sleeve"       : bool(is_long_sleeve),
        "mask_size_original"   : list(mask_hard.size),
        "mask_size_upscaled"   : list(mask_hr.size),
        "mask_pixels_set"      : mask_pixels,
        "total_pixels"         : total_pixels,
        "mask_coverage_percent": round(100*mask_pixels/total_pixels, 2),
    }
    save_json(mask_meta, DIRS["phase1_parse"], "p1d_mask_metadata.json")

    # Full Phase-1 summary strip
    p1_summary = titled_strip(
        overlay_label(person_img,    "Input Person"),
        overlay_label(skeleton_vis,  "OpenPose"),
        overlay_label(parse_color,   "SCHP Parsing"),
        overlay_label(person_masked, "Masked Input"),
        overlay_label(garm_proc,     "Garment (rembg)"),
        title="PHASE 1 SUMMARY — Human Parsing & Pose Estimation"
    )
    save_img(p1_summary, DIRS["phase1_parse"], "PHASE1_SUMMARY.jpg")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Latent Processing & Fusion
    # ──────────────────────────────────────────────────────────────────────────
    tick("PHASE-2", "CLIP Vision Encoding")

    with torch.no_grad():
        # ── 2a  CLIP encoding ─────────────────────────────────────────────────
        clip_inputs = model.auto_processor(images=garm_proc, return_tensors="pt").to(model.gpu_id)
        clip_pixel_vals = clip_inputs.data["pixel_values"]   # [1, 3, 224, 224]

        # Visualise the CLIP input crop (224×224)
        clip_np = clip_pixel_vals[0].cpu().permute(1,2,0).numpy()
        clip_np = ((clip_np - clip_np.min()) / (clip_np.max()-clip_np.min()) * 255).astype(np.uint8)
        clip_crop_vis = Image.fromarray(clip_np)
        save_img(clip_crop_vis, DIRS["phase2_latent"], "p2a_clip_input_224x224.jpg")

        clip_outputs = model.image_encoder(clip_pixel_vals)
        image_embeds = clip_outputs.image_embeds.unsqueeze(1)  # [1,1,768]

        clip_meta = {
            "pixel_values"  : tensor_summary(clip_pixel_vals,  "pixel_values"),
            "image_embeds"  : tensor_summary(image_embeds,     "image_embeds"),
            "projection_dim": int(model.image_encoder.config.projection_dim),
            "hidden_size"   : int(model.image_encoder.config.hidden_size),
        }
        log(f"   CLIP pixel_values  : {list(clip_pixel_vals.shape)}")
        log(f"   CLIP image_embeds  : {list(image_embeds.shape)}  dtype={image_embeds.dtype}")
        save_json(clip_meta, DIRS["phase2_latent"], "p2a_clip_embedding_stats.json")

        # Embed vector bar chart (first 64 dims)
        emb_np = image_embeds[0,0].float().cpu().numpy()
        emb_img_w, emb_img_h = 800, 200
        emb_vis = Image.new("RGB", (emb_img_w, emb_img_h), (20,20,30))
        d_emb = ImageDraw.Draw(emb_vis)
        show_dims = min(64, len(emb_np))
        bar_w = emb_img_w // show_dims
        mn, mx = emb_np[:show_dims].min(), emb_np[:show_dims].max()
        for j in range(show_dims):
            v = (emb_np[j] - mn) / (mx - mn + 1e-8)
            bh = int(v * (emb_img_h - 20))
            x0 = j * bar_w
            hue_r = int(50 + 200 * (j / show_dims))
            hue_b = int(255 - 200 * (j / show_dims))
            d_emb.rectangle([x0, emb_img_h-bh-5, x0+bar_w-2, emb_img_h-5],
                             fill=(hue_r, 100, hue_b))
        d_emb.text((5,5), f"CLIP Image Embedding — first {show_dims} of {len(emb_np)} dims", fill=(200,220,255))
        save_img(emb_vis, DIRS["phase2_latent"], "p2a_clip_embedding_barplot.jpg")

        # ── 2b  Text / Prompt Embeddings ─────────────────────────────────────
        tick("PHASE-2", "Prompt embedding (CLIP text encoder, 2-token OOTDiffusion format)")
        prompt_embeds = model.text_encoder(
            model.tokenize_captions([""], 2).to(model.gpu_id)
        )[0]
        prompt_embeds[:, 1:] = image_embeds[:]   # inject visual embedding
        log(f"   prompt_embeds shape: {list(prompt_embeds.shape)}")
        log(f"   prompt_embeds dtype: {prompt_embeds.dtype}")

        pe_meta = tensor_summary(prompt_embeds, "prompt_embeds")
        pe_meta["architecture"] = "OOTDiffusion 2-token: [SOS, image_embed]"
        save_json(pe_meta, DIRS["phase2_latent"], "p2b_prompt_embedding_stats.json")

        # ── 2c  VAE Encoding ─────────────────────────────────────────────────
        tick("PHASE-2", "VAE Encoding — garment & person latents")
        vae = model.pipe.vae

        # ── Garment VAE ──
        garm_tensor  = model.pipe.image_processor.preprocess(garment_img).to(device=model.gpu_id, dtype=vae.dtype)
        garm_dist    = vae.encode(garm_tensor).latent_dist
        garm_latent  = garm_dist.mode()  # [1, 4, 128, 96]

        # round-trip decode
        garm_decode  = vae.decode(garm_latent).sample
        garm_rt_np   = ((garm_decode[0].float().cpu().clamp(-1,1)+1)/2*255).byte().permute(1,2,0).numpy()
        garm_rt_img  = Image.fromarray(garm_rt_np)
        garm_orig_rs = np.array(garment_img.resize((garm_rt_np.shape[1], garm_rt_np.shape[0]))).astype(np.float32)
        garm_psnr    = psnr(garm_orig_rs, garm_rt_np.astype(np.float32))

        save_img(garm_rt_img, DIRS["phase2_latent"], "p2c_garment_vae_roundtrip.jpg")

        # ── Person VAE ──
        person_tensor = model.pipe.image_processor.preprocess(person_masked).to(device=model.gpu_id, dtype=vae.dtype)
        person_dist   = vae.encode(person_tensor).latent_dist
        person_latent = person_dist.mode()  # [1, 4, 128, 96]

        person_decode = vae.decode(person_latent).sample
        person_rt_np  = ((person_decode[0].float().cpu().clamp(-1,1)+1)/2*255).byte().permute(1,2,0).numpy()
        person_rt_img = Image.fromarray(person_rt_np)
        save_img(person_rt_img, DIRS["phase2_latent"], "p2c_person_masked_vae_roundtrip.jpg")

        # ── Latent visualisations (channel-wise) ─────────────────────────────
        def latent_channel_vis(latent_t: torch.Tensor, label: str) -> Image.Image:
            """Render each of the 4 latent channels side-by-side."""
            ch = latent_t[0].float().cpu().numpy()   # [4, H, W]
            imgs = []
            for c in range(ch.shape[0]):
                d = ch[c]
                d_norm = ((d - d.min()) / (d.max()-d.min()+1e-8) * 255).astype(np.uint8)
                cm = cv2.applyColorMap(d_norm, cv2.COLORMAP_VIRIDIS)
                cm_rgb = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
                imgs.append(Image.fromarray(cm_rgb))
            combined = titled_strip(*imgs, title=f"Latent channels  [{label}]  shape={list(latent_t.shape)}")
            return combined

        garm_lat_vis   = latent_channel_vis(garm_latent,   "garment_latent")
        person_lat_vis = latent_channel_vis(person_latent, "person_latent")
        save_img(garm_lat_vis,   DIRS["phase2_latent"], "p2c_garment_latent_4ch.jpg")
        save_img(person_lat_vis, DIRS["phase2_latent"], "p2c_person_latent_4ch.jpg")

        vae_meta = {
            "vae_scaling_factor"   : float(vae.config.scaling_factor),
            "vae_latent_channels"  : int(vae.config.latent_channels),
            "garment": {
                "input_tensor"     : tensor_summary(garm_tensor,  "garm_tensor"),
                "latent"           : tensor_summary(garm_latent,  "garm_latent"),
                "roundtrip_psnr_dB": round(garm_psnr, 2),
                "psnr_quality"     : "excellent" if garm_psnr > 35 else "good" if garm_psnr > 30 else "acceptable",
            },
            "person_masked": {
                "input_tensor"     : tensor_summary(person_tensor, "person_tensor"),
                "latent"           : tensor_summary(person_latent, "person_latent"),
            },
        }
        save_json(vae_meta, DIRS["phase2_latent"], "p2c_vae_encoding_stats.json")
        log(f"   Garment VAE PSNR  : {garm_psnr:.1f} dB")

        vae_strip = titled_strip(
            overlay_label(garment_img,  "Garment input"),
            overlay_label(garm_rt_img,  f"VAE decode PSNR={garm_psnr:.1f}dB"),
            overlay_label(person_img,   "Person input"),
            overlay_label(person_rt_img,"Person VAE decode"),
            title="PHASE 2c — VAE Round-trip Fidelity Check"
        )
        save_img(vae_strip, DIRS["phase2_latent"], "p2c_vae_roundtrip_strip.jpg")

        # Full Phase-2 summary strip
        p2_summary = titled_strip(
            overlay_label(clip_crop_vis,  "CLIP input (224²)"),
            overlay_label(emb_vis,        "CLIP embedding"),
            overlay_label(garm_lat_vis.resize((garm_lat_vis.width//2, garm_lat_vis.height//2)), "Garment latent"),
            overlay_label(person_lat_vis.resize((person_lat_vis.width//2, person_lat_vis.height//2)), "Person latent"),
            title="PHASE 2 SUMMARY — Latent Processing & Fusion"
        )
        save_img(p2_summary, DIRS["phase2_latent"], "PHASE2_SUMMARY.jpg")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3 — Denoising Diffusion (UNet)
    # ──────────────────────────────────────────────────────────────────────────
    tick("PHASE-3", "Setting up diffusion — scheduler & callback")

    steps_data = []   # per-step log
    step_latent_imgs = []

    def step_callback(step: int, t, latents):
        """Called after every denoising step — log + optionally decode."""
        now = time.time()
        ts_val = int(t) if hasattr(t, "__int__") else float(t)
        entry = {
            "step"       : step,
            "timestep"   : ts_val,
            "elapsed_s"  : round(now - global_start, 3),
            "latent"     : tensor_summary(latents, f"step_{step}_latents"),
        }
        steps_data.append(entry)
        log(f"   [STEP {step:3d}]  t={ts_val:4d}  "
            f"lat_mean={latents.float().mean().item():.4f}  "
            f"lat_std={latents.float().std().item():.4f}")

        # Decode every 5th step (and first/last) to make an animation strip
        if step % 5 == 0 or step == 0:
            try:
                scaled = latents / model.pipe.vae.config.scaling_factor
                dec = model.pipe.vae.decode(scaled.to(dtype=model.pipe.vae.dtype)).sample
                img_np = ((dec[0].float().cpu().clamp(-1,1)+1)/2*255).byte().permute(1,2,0).numpy()
                step_img = Image.fromarray(img_np).resize((192, 256))
                step_img = overlay_label(step_img, f"Step {step}  t={ts_val}")
                step_latent_imgs.append(step_img)
            except Exception:
                pass

    # Run the full pipeline call (with output_dir pointing to Phase 3 folder)
    tick("PHASE-3", "Running WearCast pipeline (30 denoising steps)")
    t_gen_start = time.time()
    images = model(
        model_type   = "hd",
        category     = "upperbody",
        image_garm   = garment_img,
        image_vton   = person_img,
        mask         = None,
        image_ori    = person_img,
        num_samples  = 1,
        num_steps    = 30,
        image_scale  = 2.5,
        seed         = -1,
        callback     = step_callback,
        callback_steps = 1,
        output_dir   = str(DIRS["phase3_gen"]),
    )
    t_gen_total = time.time() - t_gen_start
    log(f"\n   Total generation time: {t_gen_total:.2f}s")
    log(f"   Steps recorded      : {len(steps_data)}")

    save_json(steps_data, DIRS["phase3_gen"], "p3_per_step_latent_stats.json")

    gen_meta = {
        "total_time_s"   : round(t_gen_total, 2),
        "num_steps"      : 30,
        "steps_recorded" : len(steps_data),
        "avg_step_s"     : round(t_gen_total / max(1, len(steps_data)), 3),
    }
    save_json(gen_meta, DIRS["phase3_gen"], "p3_generation_metadata.json")

    # ── Step-evolution strip ─────────────────────────────────────────────────
    if step_latent_imgs:
        evo_w = sum(im.width for im in step_latent_imgs) + 8*(len(step_latent_imgs)-1)
        evo_h = step_latent_imgs[0].height + 50
        evo_strip = Image.new("RGB", (evo_w, evo_h), (15,15,25))
        d_evo = ImageDraw.Draw(evo_strip)
        d_evo.text((8, 8), f"PHASE 3 — Denoising Evolution  ({len(step_latent_imgs)} snapshots)", fill=(200,220,255))
        x = 0
        for im in step_latent_imgs:
            evo_strip.paste(im, (x, 40))
            x += im.width + 8
        save_img(evo_strip, DIRS["phase3_gen"], "p3_denoising_evolution_strip.jpg")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4 — Post-processing & Final Output
    # ──────────────────────────────────────────────────────────────────────────
    tick("PHASE-4", "Collecting final outputs & post-processing diagnostics")

    # The model.__call__ already saved debug images to DIRS["phase3_gen"]
    # Move them to the correct phase folders for cleaner organisation
    raw_unet_path = DIRS["phase3_gen"] / "debug_phase4_raw_unet_output.jpg"
    final_path    = DIRS["phase3_gen"] / "debug_final_output.jpg"
    comparison_path = DIRS["phase3_gen"] / "debug_phase4_comparison.jpg"

    final_image = images[0]
    save_img(final_image, DIRS["phase4_post"], "p4_final_result.jpg")

    # Copy debug images from model call to phase4 folder
    for debug_fname in [
        "debug_phase4_raw_unet_output.jpg",
        "debug_phase4_comparison.jpg",
        "debug_phase4_feather_mask.jpg",
        "debug_final_output.jpg",
        "debug_final_unet_mask.jpg",
        "debug_phase3_masked_person.jpg",
        "debug_phase2_vae_roundtrip.jpg",
        "debug_phase2_clip_bg_replaced.jpg",
    ]:
        src = DIRS["phase3_gen"] / debug_fname
        if src.exists():
            shutil.copy(src, DIRS["phase4_post"] / debug_fname)
            log(f"   [COPY] {debug_fname}")

    # ── Quality metrics ───────────────────────────────────────────────────────
    final_np  = np.array(final_image).astype(np.float32)
    person_np = np.array(person_img.resize(final_image.size, Image.BICUBIC)).astype(np.float32)
    garm_np   = np.array(garment_img.resize(final_image.size, Image.BICUBIC)).astype(np.float32)

    ssim_approx = float(1 - np.mean(np.abs(final_np - person_np)) / 255)

    quality_meta = {
        "output_size"           : list(final_image.size),
        "mean_pixel_diff_person": round(float(np.mean(np.abs(final_np - person_np))), 2),
        "ssim_approx_vs_person" : round(ssim_approx, 4),
        "total_pipeline_time_s" : round(time.time() - global_start, 2),
        "generation_time_s"     : round(t_gen_total, 2),
    }
    save_json(quality_meta, DIRS["phase4_post"], "p4_quality_metrics.json")
    for k, v in quality_meta.items():
        log(f"   {k:35s}: {v}")

    # ── Big 5-panel final comparison ─────────────────────────────────────────
    raw_unet_img = Image.open(raw_unet_path) if raw_unet_path.exists() else final_image

    final_strip = titled_strip(
        overlay_label(person_img,   "① Person Input"),
        overlay_label(garment_img,  "② Garment Input"),
        overlay_label(garm_proc,    "③ BG Removed"),
        overlay_label(raw_unet_img, "④ Raw UNet Output"),
        overlay_label(final_image,  "⑤ FINAL RESULT"),
        title="PHASE 4 — Post-processing & Final Result"
    )
    save_img(final_strip, DIRS["phase4_post"], "PHASE4_SUMMARY.jpg")

    # ──────────────────────────────────────────────────────────────────────────
    # SUMMARY — Master collage & timeline
    # ──────────────────────────────────────────────────────────────────────────
    tick("SUMMARY", "Building master summary & timeline")

    total_time = round(time.time() - global_start, 2)
    timeline.append({"phase": "DONE", "event": "Pipeline complete", "elapsed_s": total_time})
    save_json(timeline, DIRS["summary"], "full_timeline.json")
    save_json({"total_time_s": total_time, **hw, **quality_meta},
              DIRS["summary"], "run_summary.json")

    # Master 2-row collage
    row1 = titled_strip(
        overlay_label(person_img,   "Person"),
        overlay_label(garment_img,  "Garment"),
        overlay_label(skeleton_vis, "OpenPose"),
        overlay_label(parse_color,  "SCHP Parse"),
        title="WearCast AI — Row 1: Inputs & Phase 1"
    )
    row2 = titled_strip(
        overlay_label(person_masked, "Masked Input"),
        overlay_label(garm_proc,     "Garment (rembg)"),
        overlay_label(raw_unet_img,  "UNet Raw"),
        overlay_label(final_image,   "✅ Final Try-on"),
        title=f"WearCast AI — Row 2: Processing & Output  [{total_time:.0f}s total]"
    )
    target_w = max(row1.width, row2.width)
    master = Image.new("RGB", (target_w, row1.height + row2.height + 8), (10,10,20))
    master.paste(row1, (0, 0))
    master.paste(row2, (0, row1.height + 8))
    save_img(master, DIRS["summary"], "MASTER_COLLAGE.jpg")

    # ── Step-by-step log summary ──────────────────────────────────────────────
    log("\n" + "="*70)
    log("  WearCast Presentation — ALL OUTPUT SAVED")
    log("="*70)
    log(f"  Output folder : {OUT}")
    log(f"  Total time    : {total_time}s")
    log("")
    log("  Folders:")
    for k, p in DIRS.items():
        n = len(list(p.iterdir()))
        log(f"    {k:20s}: {p.relative_to(PROJECT_ROOT)}  ({n} files)")
    log("="*70)

    log_file.close()
    print(f"\n✅ Done!  All outputs → {OUT}")
    return str(OUT)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WearCast AI — Full Pipeline Presentation Runner"
    )
    parser.add_argument(
        "--person",
        type=str,
        default=str(PROJECT_ROOT / "images" / "person.jpg"),
        help="Path to the person/model image"
    )
    parser.add_argument(
        "--garment",
        type=str,
        default=str(PROJECT_ROOT / "images" / "garment.jpg"),
        help="Path to the garment image"
    )
    args = parser.parse_args()

    if not Path(args.person).exists():
        # Try any jpg in project root as fallback
        fallbacks = list(PROJECT_ROOT.glob("*.jpg")) + list(PROJECT_ROOT.glob("images/*.jpg"))
        if fallbacks:
            args.person = str(fallbacks[0])
            print(f"[WARN] --person not found, using fallback: {args.person}")
        else:
            print("[ERROR] No person image found. Provide --person path")
            sys.exit(1)

    if not Path(args.garment).exists():
        fallbacks = [p for p in (list(PROJECT_ROOT.glob("*.jpg")) + list(PROJECT_ROOT.glob("images/*.jpg")))
                     if str(p) != args.person]
        if fallbacks:
            args.garment = str(fallbacks[0])
            print(f"[WARN] --garment not found, using fallback: {args.garment}")
        else:
            print("[ERROR] No garment image found. Provide --garment path")
            sys.exit(1)

    run(args.person, args.garment)
