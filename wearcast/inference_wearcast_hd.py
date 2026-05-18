from pathlib import Path
import sys
import os
import torch
import torch._dynamo
import numpy as np
from PIL import Image, ImageDraw
import cv2
import random
import time
from rembg import remove, new_session
import kornia

import transformers.utils
if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
    transformers.utils.FLAX_WEIGHTS_NAME = 'flax_model.msgpack'

from wearcast.pipelines_wearcast.pipeline_wearcast import WearCastPipeline
from wearcast.pipelines_wearcast.unet_garm_2d_condition import UNetGarm2DConditionModel
from wearcast.pipelines_wearcast.unet_vton_2d_condition import UNetVton2DConditionModel
from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer

# Absolute Pathing
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()  # WearCast_AI root
VIT_PATH = os.path.join(PROJECT_ROOT, "checkpoints/clip-vit-large-patch14")
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints/ootd")

# Add project root to path for run.utils_wearcast import
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from run.utils_wearcast import get_mask_location as get_mask_location_pro


def _dbg_tensor(label, t):
    """Print a compact one-liner tensor summary."""
    if isinstance(t, torch.Tensor):
        mn, mx, mu = t.float().min().item(), t.float().max().item(), t.float().mean().item()
        print(f"   [DBG] {label:40s} | shape={list(t.shape)} dtype={t.dtype} dev={t.device} | min={mn:.4f} max={mx:.4f} mean={mu:.4f}")
    else:
        print(f"   [DBG] {label:40s} | type={type(t)}")

# NOTE: match_histograms and laplacian_pyramid_blend were removed.
# They over-corrected the UNet output and introduced halo/glow artifacts.
# OOTDiffusion uses simple Gaussian-feathered mask compositing — we do the same.


class WearCastHD:

    def __init__(self, gpu_id):
        self.gpu_id = 'cuda:' + str(gpu_id)

        print("=" * 70)
        print(f"[WearCastHD.__init__] Initialising on device: {self.gpu_id}")
        print(f"[WearCastHD.__init__] MODEL_PATH : {MODEL_PATH}")
        print(f"[WearCastHD.__init__] VIT_PATH   : {VIT_PATH}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(gpu_id)
            print(f"[WearCastHD.__init__] GPU Name   : {props.name}")
            print(f"[WearCastHD.__init__] VRAM Total : {props.total_memory / 1e9:.1f} GB")
        print("=" * 70)

        print(f"Loading components from {MODEL_PATH}...")

        # 1. Load VAE (float16 — matches original OOTDiffusion training)
        print("[WearCastHD] Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            MODEL_PATH,
            subfolder="vae",
            torch_dtype=torch.float16,
        )
        print(f"[WearCastHD] VAE loaded. Scaling factor: {vae.config.scaling_factor}  |  latent_channels={vae.config.latent_channels}")

        # 2. Load UNets
        print("[WearCastHD] Loading UNet-Garm (Safetensors)...")
        unet_garm = UNetGarm2DConditionModel.from_pretrained(
            MODEL_PATH,
            subfolder="unet_garm",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        print(f"[WearCastHD] UNet-Garm loaded. in_channels={unet_garm.config.in_channels}  cross_attention_dim={unet_garm.config.cross_attention_dim}")

        print("[WearCastHD] Loading UNet-Vton (Safetensors)...")
        unet_vton = UNetVton2DConditionModel.from_pretrained(
            MODEL_PATH,
            subfolder="unet_vton",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        print(f"[WearCastHD] UNet-Vton loaded. in_channels={unet_vton.config.in_channels}  cross_attention_dim={unet_vton.config.cross_attention_dim}")

        # 3. Load Text Encoder and Tokenizer
        print("[WearCastHD] Loading CLIP Tokenizer...")
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
        print(f"[WearCastHD] Tokenizer loaded. model_max_length={self.tokenizer.model_max_length}")

        print("[WearCastHD] Loading CLIP Text Encoder...")
        self.text_encoder = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder").to(self.gpu_id)
        print(f"[WearCastHD] Text Encoder loaded. hidden_size={self.text_encoder.config.hidden_size}  dtype={next(self.text_encoder.parameters()).dtype}")

        # 4. Load CLIP Vision & Scheduler
        print("[WearCastHD] Loading AutoProcessor (CLIP)...")
        self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
        print(f"[WearCastHD] AutoProcessor loaded.")

        print("[WearCastHD] Loading CLIPVisionModelWithProjection...")
        # Keep CLIP in float32 — matches original OOTDiffusion. float16 loses embedding precision.
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.gpu_id)
        print(f"[WearCastHD] CLIPVisionModelWithProjection loaded.  hidden_size={self.image_encoder.config.hidden_size}  projection_dim={self.image_encoder.config.projection_dim}")

        # Load scheduler
        print("[WearCastHD] Loading DDPMScheduler...")
        from diffusers import DDPMScheduler
        scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
        print(f"[WearCastHD] Scheduler loaded. type={type(scheduler).__name__}  num_train_timesteps={scheduler.config.num_train_timesteps}")

        # 5. Initialize Pipeline DIRECTLY
        print("Assembling WearCast Pipeline...")
        self.pipe = WearCastPipeline(
            vae=vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            scheduler=scheduler,
            feature_extractor=self.auto_processor,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.gpu_id)
        print(f"[WearCastHD] Pipeline assembled. VAE scale factor={self.pipe.vae_scale_factor}")

        # Use UniPC scheduler (what OOTD was designed for)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        print(f"[WearCastHD] Scheduler replaced with UniPCMultistepScheduler.")
        
        # --- PERFORMANCE OPTIMIZATION ---
        print("[WearCastHD] Initializing rembg session globally...")
        self.rembg_session = new_session("u2net")

        print("[WearCastHD] Compiling UNet for massive speedup (First run will be slow, subsequent runs 15-20s)...")
        try:
            # [PERFORMANCE FIX v2.1] 
            # 1. Allow integer attributes on modules without recompiling (crucial for block indices)
            # 2. Increase recompile limit to accommodate all 16 Transformer blocks
            torch.backends.cudnn.benchmark = True
            torch._dynamo.config.recompile_limit = 24 
            
            self.pipe.unet_vton = torch.compile(self.pipe.unet_vton, mode="default", fullgraph=False)
            print("[WearCastHD] UNet compilation scheduled (Dynamic Indexing enabled).")
        except Exception as e:
            print(f"[WearCastHD] torch.compile failed: {e}. Proceeding without compilation.")

        print("=" * 70)

    def tokenize_captions(self, captions, max_length):
        inputs = self.tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def apply_logo_warping(self, image_garm, person_keypoints, generated_image, target_mask):
        """
        [ULTIMATE PRODUCTION] UNet-Guided Feature Transfer
        Automatically detects features (logos/patterns) and uses the UNet's
        generated output to guide their perfect placement and warping.
        """
        import cv2
        from PIL import Image
        import numpy as np
        
        garm_np = np.array(image_garm.convert('RGB'))
        gen_np  = np.array(generated_image.convert('RGB'))
        
        # Ensure target_mask is 2D
        mask_np = np.array(target_mask)
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0]
            
        mask_np = mask_np.astype(np.float32) 
        
        h_gen, w_gen = gen_np.shape[:2]
        h_garm, w_garm = garm_np.shape[:2]

        # 1. SMART LOGO EXTRACTION (Source)
        garm_f32 = garm_np.astype(np.float32)
        bg_mask = np.all(garm_f32 >= 240, axis=-1)
        fg_pixels = garm_f32[~bg_mask]
        
        if len(fg_pixels) < 100:
             return generated_image
             
        base_color = np.median(fg_pixels, axis=0)
        color_diff = np.linalg.norm(garm_f32 - base_color, axis=-1)
        color_diff[bg_mask] = 0
        
        # Adaptive thresholding
        # --- LOGO EXTRACTION (Adaptive Floor) ---
        # FIX: Lowered floor to 12.0 and multiplier to 0.5 to capture thin patterns (like 'LOVE' dental tools)
        std_diff = np.std(color_diff[color_diff > 0])
        thresh = max(12.0, std_diff * 0.5)
        raw_logo_mask = (color_diff > thresh).astype(np.uint8) * 255
        
        y_indices, x_indices = np.where(raw_logo_mask > 0)
        if len(y_indices) < 50:
            print(f" -> [LOGO_WARP] No distinct features found at threshold {thresh:.1f}. Skipping.")
            return generated_image

        # 2. FEATURE TRACKING (Target Discovery via UNet Output)
        gen_f32 = gen_np.astype(np.float32)
        gen_fg_mask = mask_np > 0.5
        
        if not np.any(gen_fg_mask):
            return generated_image
            
        gen_base_color = np.median(gen_f32[gen_fg_mask], axis=0)
        gen_color_diff = np.linalg.norm(gen_f32 - gen_base_color, axis=-1)
        gen_color_diff[~gen_fg_mask] = 0
        
        # Look for where the UNet placed the pattern
        gen_logo_mask = (gen_color_diff > (thresh * 0.6)).astype(np.uint8) * 255
        gy, gx = np.where(gen_logo_mask > 0)
        
        # 3. COORDINATE MAPPING
        src_y_all, src_x_all = np.where(~bg_mask)
        src_bbox = [np.min(src_x_all), np.min(src_y_all), np.max(src_x_all), np.max(src_y_all)]
        
        logo_bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]
        
        tgt_y_all, tgt_x_all = np.where(gen_fg_mask)
        tgt_bbox = [np.min(tgt_x_all), np.min(tgt_y_all), np.max(tgt_x_all), np.max(tgt_y_all)]
        
        # Dynamic Alignment: Use UNet's placement if robust, else fallback to global
        if len(gy) > 200:
            print(" -> [LOGO_WARP] AI-Guided placement detected. Aligning graphics to UNet features (Aspect-Ratio preserved).")
            tx_min, tx_max = np.min(gx), np.max(gx)
            ty_min, ty_max = np.min(gy), np.max(gy)
            
            # Enforce aspect ratio of original logo
            logo_w = logo_bbox[2] - logo_bbox[0]
            logo_h = logo_bbox[3] - logo_bbox[1]
            
            cx, cy = (tx_min + tx_max) / 2.0, (ty_min + ty_max) / 2.0
            
            # Use width scale as primary anchor (logos usually scale by chest width)
            scale = (tx_max - tx_min) / float(logo_w) if logo_w > 0 else 1.0
            
            dst_w = logo_w * scale
            dst_h = logo_h * scale
            
            # Center the new aspect-ratio-corrected bounding box over the UNet's placement
            dtx_min = cx - dst_w / 2.0
            dtx_max = cx + dst_w / 2.0
            dty_min = cy - dst_h / 2.0
            dty_max = cy + dst_h / 2.0
            
            dst_pts = np.float32([
                [dtx_min, dty_min], [dtx_max, dty_min],
                [dtx_max, dty_max], [dtx_min, dty_max]
            ])
            src_pts = np.float32([
                [logo_bbox[0], logo_bbox[1]], [logo_bbox[2], logo_bbox[1]],
                [logo_bbox[2], logo_bbox[3]], [logo_bbox[0], logo_bbox[3]]
            ])
        else:
            print(" -> [LOGO_WARP] Using global anatomical mapping to preserve original proportions.")
            src_pts = np.float32([
                [src_bbox[0], src_bbox[1]], [src_bbox[2], src_bbox[1]],
                [src_bbox[2], src_bbox[3]], [src_bbox[0], src_bbox[3]]
            ])
            dst_pts = np.float32([
                [tgt_bbox[0], tgt_bbox[1]], [tgt_bbox[2], tgt_bbox[1]],
                [tgt_bbox[2], tgt_bbox[3]], [tgt_bbox[0], tgt_bbox[3]]
            ])

        # 4. WARP & BLEND
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_garm = cv2.warpPerspective(garm_np, M, (w_gen, h_gen), flags=cv2.INTER_LANCZOS4)
        
        logo_alpha_src = cv2.dilate(raw_logo_mask, np.ones((3, 3), np.uint8), iterations=1)
        logo_alpha_src = cv2.GaussianBlur(logo_alpha_src, (3, 3), 0).astype(np.float32) / 255.0
        warped_alpha = cv2.warpPerspective(logo_alpha_src, M, (w_gen, h_gen), flags=cv2.INTER_LANCZOS4)
        
        warped_alpha = warped_alpha * mask_np
        alpha_3d = np.stack([warped_alpha]*3, axis=-1)
        
        result = gen_np * (1.0 - alpha_3d) + warped_garm * alpha_3d

        print(f" -> [LOGO_WARP] Transfer successful. Pattern coverage: {np.mean(warped_alpha)*100:.2f}%")
        return Image.fromarray(result.astype(np.uint8))

    def match_histograms_lab(self, source, reference_fg_vals, mask, strength=1.0):
        """
        Sophisticated color correction using 1D histogram matching in LAB space.
        Matches the lighting/color distribution of the generated garment to the reference.
        """
        import cv2
        mask_bool = mask > 0.1
        if not np.any(mask_bool) or len(reference_fg_vals) == 0 or strength <= 0:
            return source

        # Convert source to LAB
        src_lab = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        matched_lab = src_lab.copy()

        # Match each channel (L, A, B)
        for i in range(3):
            s_vals = src_lab[mask_bool, i]
            r_vals = reference_fg_vals[:, i]
            
            # Simple histogram matching via percentile mapping
            s_quantiles = np.percentile(s_vals, np.linspace(0, 100, 64))
            r_quantiles = np.percentile(r_vals, np.linspace(0, 100, 64))
            
            # Interpolate source values to reference quantiles
            matched_vals = np.interp(s_vals, s_quantiles, r_quantiles)
            # Blend with original based on strength
            matched_lab[mask_bool, i] = (1.0 - strength) * s_vals + strength * matched_vals

        # Convert back to RGB
        result = cv2.cvtColor(np.clip(matched_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        return result.astype(np.float32)

    def apply_frequency_blending(self, generated, original, skin_mask):
        """
        Preserves original skin texture (pores, hair) while using AI-generated lighting.
        Uses frequency separation: result = gen_low_freq + ori_high_freq
        """
        import cv2
        # Use a subtle sigma (2.0) to capture fine details without haloing
        sigma = 2.0
        
        # Extract details (High Frequency) from original
        ori_low = cv2.GaussianBlur(original, (0, 0), sigma)
        ori_high = original - ori_low
        
        # Extract base colors (Low Frequency) from generated
        gen_low = cv2.GaussianBlur(generated, (0, 0), sigma)
        
        # Reconstruct: AI color + Original detail
        blended = gen_low + ori_high
        
        # Constrain to skin mask
        mask_3d = np.repeat(skin_mask[:, :, np.newaxis], 3, axis=2)
        # Apply blending with a slight opacity (0.7) to allow some AI lighting variation
        strength = 0.7
        final = blended * (mask_3d * strength) + generated * (1.0 - mask_3d * strength)
        
        return np.clip(final, 0, 255).astype(np.float32)

    def generate_garment_caption(self, category, image_garm):
        """
        Generates a descriptive prompt for the UNet. 
        In the future, this should call a VLM like Florence-2.
        """
        base_desc = {
            'upperbody': "a person wearing a top garment, highly detailed",
            'lowerbody': "a person wearing pants, highly detailed",
            'dress': "a person wearing a dress, highly detailed"
        }
        return base_desc.get(category, "a person wearing clothes")

    def remove_garment_background_pro(self, image_pil):
        """
        Advanced background removal using GrabCut + Centrality Prior.
        Handles white-on-white cases much better than simple thresholding.
        """
        import cv2
        img = np.array(image_pil).astype(np.uint8)
        h, w = img.shape[:2]
        
        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 1. Initialize mask with centrality prior
        mask = np.zeros((h, w), np.uint8)
        # Assume middle 90% is probably foreground, edges are definitely background
        rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        try:
            # 2. Run GrabCut (Fast 2-iteration pass)
            cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
            # 3. Create final binary mask (1 and 3 are foreground)
            mask_bin = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        except:
            # Fallback to threshold if GrabCut fails
            print(" -> [MATTING] GrabCut failed, falling back to threshold.")
            mask_bin = (np.mean(img, axis=-1) < 250).astype(np.uint8)

        # 4. Morphological cleaning
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        
        # 5. Apply mask and fill background with neutral gray [128, 128, 128]
        # Neutral gray is best for CLIP Vision encoder to ignore the background.
        res = img.copy()
        res[mask_bin == 0] = [128, 128, 128]
        
        return Image.fromarray(res), mask_bin

    def remove_garment_background_rembg(self, image_pil):
        """
        AI-powered background removal using rembg (U2-Net).
        Handles complex backgrounds and white-on-white cases perfectly.
        """
        import numpy as np
        from PIL import Image
        print(" -> [MATTING] Initializing rembg (U2-Net)...")
        try:
            # 1. AI Background Removal
            res_pil = remove(image_pil, session=self.rembg_session)
            # 2. Extract alpha channel as mask
            res_np = np.array(res_pil)
            # Check if RGBA
            if res_np.shape[2] == 4:
                mask_bin = (res_np[:, :, 3] > 10).astype(np.uint8)
                
                # --- FIX #19: Garment Alpha Erosion (Halo Removal) ---
                # Removes the thin white outline often left by rembg
                kernel_erode = np.ones((3, 3), np.uint8) # Slightly stronger for safety
                mask_bin = cv2.erode(mask_bin, kernel_erode, iterations=1)
            else:
                print(" -> [MATTING] Warning: rembg did not return alpha channel. Falling back.")
                return self.remove_garment_background_pro(image_pil)
            
            # 3. Fill background with neutral gray for CLIP
            img_rgb = np.array(res_pil.convert("RGB"))
            bg_color = np.array([128, 128, 128], dtype=np.uint8)
            res_final = np.where(mask_bin[:, :, np.newaxis] == 0, bg_color, img_rgb)
            
            return Image.fromarray(res_final), mask_bin
        except Exception as e:
            print(f" -> [MATTING] rembg failed: {e}. Falling back to GrabCut.")
            return self.remove_garment_background_pro(image_pil)

    def __call__(
        self,
        model_type='hd',
        category='upperbody',
        image_garm=None,
        image_vton=None,
        mask=None,
        image_ori=None,
        num_samples=1,
        num_steps=20,
        image_scale=1.0,
        seed=-1,
        callback=None,
        callback_steps=1,
        output_dir=None,
    ):
        # Ensure output directory exists if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"   [WearCast] Debug images will be saved to: {output_dir}")

        print("\n" + "=" * 70)
        print("[WearCastHD.__call__] Starting inference...")
        print(f"[WearCastHD.__call__] model_type ={model_type}")
        print(f"[WearCastHD.__call__] category   ={category}")
        print(f"[WearCastHD.__call__] num_samples ={num_samples}")
        print(f"[WearCastHD.__call__] num_steps   ={num_steps} (UI; may be overridden by auto-params)")
        print(f"[WearCastHD.__call__] image_scale ={image_scale} (UI; may be overridden by auto-params)")
        print(f"[WearCastHD.__call__] seed (raw)  ={seed}")
        print(f"[WearCastHD.__call__] mask passed ={'YES' if mask is not None else 'NO (auto-generate)'}")

        if image_garm is not None:
            print(f"[WearCastHD.__call__] image_garm  : size={image_garm.size}  mode={image_garm.mode}")
        if image_vton is not None:
            print(f"[WearCastHD.__call__] image_vton  : size={image_vton.size}  mode={image_vton.mode}")
        if image_ori is not None:
            print(f"[WearCastHD.__call__] image_ori   : size={image_ori.size}  mode={image_ori.mode}")

        if seed == -1:
            random.seed(time.time())
            seed = random.randint(0, 2147483647)
        print('Initial seed: ' + str(seed))
        generator = torch.manual_seed(seed)

        # =========================================================
        # PRE-PHASE: Garment Matting & Analysis
        # =========================================================
        def debug_save(img, name):
            if output_dir:
                img.save(os.path.join(output_dir, name))
            else:
                img.save(name)

        print(" -> [MATTING] Running AI-powered garment extraction (rembg)...")
        t_mat = time.time()
        garm_proc, garm_mask = self.remove_garment_background_rembg(image_garm)
        self._cached_garm_mask = garm_mask
        print(f" -> [MATTING] Extraction complete in {time.time() - t_mat:.2f}s")
        
        from run.utils_wearcast import analyze_sleeve_length
        is_long_sleeve = analyze_sleeve_length(garm_mask)

        # =========================================================
        # PHASE 1 — AI Preprocessing (OpenPose + HumanParsing + Mask)
        # =========================================================
        if mask is None:
            print(f"\n[WearCast] Phase 1/4: Starting AI Preprocessing...")
            from preprocess.humanparsing.run_parsing import Parsing
            from preprocess.openpose.run_openpose import OpenPose

            if not hasattr(self, 'parsing_model'):
                print(" -> Loading Human-Parsing model checkpoints...")
                self.parsing_model = Parsing(int(self.gpu_id.split(':')[1]))
                print(" -> Loading OpenPose model checkpoints...")
                self.openpose_model = OpenPose(int(self.gpu_id.split(':')[1]))
            else:
                print(" -> Reusing cached Parsing and OpenPose models.")

            print(" -> Running OpenPose inference (Human Keypoints)...")
            t0 = time.time()
            keypoints = self.openpose_model(image_vton)
            self._cached_keypoints = keypoints # Store for later warping
            t1 = time.time()
            pose_data = np.array(keypoints["pose_keypoints_2d"])
            print(f"    [POSE] Keypoints count : {len(keypoints['pose_keypoints_2d'])}")
            print(f"    [POSE] pose_data shape  : {pose_data.shape}")
            print(f"    [POSE] Inference time   : {t1 - t0:.2f}s")
            # Print each named keypoint
            kp_names = ['Nose','Neck','RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist',
                        'RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','REye','LEye','REar','LEar']
            for idx, (name, pt) in enumerate(zip(kp_names, keypoints["pose_keypoints_2d"])):
                print(f"    [POSE]   KP[{idx:02d}] {name:12s}: ({pt[0]:.1f}, {pt[1]:.1f})")

            print(" -> Running Parsing inference (Semantic Segmentation)...")
            t0 = time.time()
            model_parse, face_mask = self.parsing_model(image_vton)
            t1 = time.time()
            parse_arr = np.array(model_parse)
            unique_labels, counts = np.unique(parse_arr, return_counts=True)
            print(f"    [PARSE] Output size     : {model_parse.size}")
            print(f"    [PARSE] Inference time  : {t1 - t0:.2f}s")
            print(f"    [PARSE] Detected classes (label: pixel_count):")
            label_names = {
                0:'Background', 1:'Hat', 2:'Hair', 3:'Sunglasses', 4:'UpperClothes',
                5:'Skirt', 6:'Pants', 7:'Dress', 8:'Belt', 9:'LeftShoe', 10:'RightShoe',
                11:'Face', 12:'LeftLeg', 13:'RightLeg', 14:'LeftArm', 15:'RightArm',
                16:'Bag', 17:'Scarf', 18:'Neck(refined)'
            }
            for l, c in zip(unique_labels, counts):
                name = label_names.get(l, f"Label {l}")
                print(f"       Label {l:2d} ({name:20s}): {c:7d} px ({100*c/parse_arr.size:3.1f}%)")

            # 11: Face, 14: LeftArm, 15: RightArm, 18: Neck
            self._skin_mask = ((parse_arr == 11) | (parse_arr == 14) | (parse_arr == 15) | (parse_arr == 18)).astype(np.float32)
            self._cached_parse = parse_arr
            print(f" -> [SKIN] Cached skin mask and silhouette for identity preservation.")

            # 2. Sophisticated Mask Generation
            # Use the Professional High-Precision mask logic from utils_wearcast
            print(" -> [WEARCAST] Using Professional T-shirt Mask Logic (Organic Smoothing)...")
            # get_mask_location_pro returns (mask_255, mask_gray_127)
            mask, mask_gray = get_mask_location_pro(
                model_type, 
                category, 
                model_parse, 
                keypoints,
                width=384,
                height=512,
                is_long_sleeve=is_long_sleeve
            )
            self._cached_hard_mask = mask

            # Diagnostic: Check mask density
            mask_np = np.array(mask)
            mask_pixels = np.sum(mask_np > 127)
            total_pixels = mask_np.size
            print(f" -> Mask Diagnostic: {mask_pixels} pixels marked for replacement ({100*mask_pixels/total_pixels:.2f}% of image)")
            print(f" -> Mask raw size: {mask.size}  |  dtype={mask_np.dtype}  |  min={mask_np.min()}  max={mask_np.max()}")

            mask = mask.resize((768, 1024), Image.BILINEAR)  # Smooth upscale
            # Re-binarize after bilinear to prevent grey artifacts
            mask_np_clean = np.array(mask)
            mask_np_clean = (mask_np_clean > 127).astype(np.uint8) * 255
            mask = Image.fromarray(mask_np_clean)
            print(f" -> Mask after bilinear resize and re-binarization: {mask.size}")
            
            # --- FIX: EXACT OOTD PREPROCESSING ---
            # OOTD requires the person image to have the garment area replaced with 127-gray BEFORE encoding.
            mask_gray_np = (mask_np_clean > 127).astype(np.uint8) * 127
            mask_gray = Image.fromarray(mask_gray_np)
            image_vton = Image.composite(mask_gray, image_vton, mask)
            print(f" -> [PREPROCESS] Person image (image_vton) garment area replaced with 127-gray")

            # Save mask debug images
            debug_save(mask, "debug_phase1_hard_mask.jpg")
            print(f" -> [SAVED] Hard mask (255-binary) saved")
            debug_save(mask_gray, "debug_phase1_soft_mask.jpg")
            print(f" -> [SAVED] Soft mask (127-gray) saved")

            print(" -> Preprocessing Stage: SUCCESS")

        # Auto-detect garment complexity and choose optimal params
        is_complex = self.detect_garment_complexity(image_garm)
        auto_params = self.get_optimal_params(category, is_complex)
        
        # PRO-PRIORITY: Always use at least 30 steps for Typically-level quality,
        # unless the user explicitly requested a very high value.
        final_steps = max(30, num_steps if num_steps > 30 else auto_params["num_steps"])
        final_scale = image_scale if image_scale > 0 else auto_params["image_scale"]
        
        print(f"\n[WearCast] Pro-Engine: complex={is_complex} | auto_params={auto_params}")
        print(f"[WearCast] Quality Lock: steps={final_steps} (Professional Mode) | guidance_scale={final_scale}")


        # =========================================================
        # PHASE 2 — CLIP Vision Encoding
        # =========================================================
        print(f"\n[WearCast] Phase 2/4: Encoding Inputs (VAE & Multi-Scale CLIP Vision)...")
        with torch.no_grad():
            from PIL import ImageEnhance

            # --- 2a. AI Garment Matting (rembg) ---
            # Extraction already completed in PRE-PHASE
            debug_save(garm_proc, "debug_phase2_clip_bg_replaced.jpg")

            # Save the original garment for reference comparison
            debug_save(image_garm, "debug_phase2_garment_original.jpg")
            print(f" -> [SAVED] Original garment saved to: debug_phase2_garment_original.jpg")

            # --- 2b. CLIP input preparation ---
            # Use the processed garment directly for CLIP encoding.
            # Heavy sharpness/contrast enhancement was removed because it
            # over-emphasises edges and textures, causing stiff/puffy artifacts.
            garm_enhanced = garm_proc
            debug_save(garm_enhanced, "debug_phase2_clip_input.jpg")
            print(f" -> [SAVED] CLIP input garment saved to: debug_phase2_clip_input.jpg")

            # --- 2c. CLIP Encoding (Standard Global — float32, matches original OOTD) ---
            clip_inputs = self.auto_processor(images=garm_enhanced, return_tensors="pt").to(device=self.gpu_id)
            with torch.no_grad():
                # Original OOTD does NOT cast to float16 here — CLIP needs float32 precision
                clip_outputs = self.image_encoder(clip_inputs.data['pixel_values'])
                image_embeds = clip_outputs.image_embeds.unsqueeze(1) # [1, 1, 768]
            
            # --- 2d. Prompt Construction (OOTDiffusion Standard 2-token format) ---
            print(" -> [PROMPT] Using OOTDiffusion 2-token architecture.")
            prompt_embeds = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0]
            
            # Inject global image embedding into token index 1 (OOTDiffusion Standard)
            prompt_embeds[:, 1:] = image_embeds[:]
            print(f" -> Prompt embeddings shape: {list(prompt_embeds.shape)} (Strict 2-token format)")
            _dbg_tensor("prompt_embeds", prompt_embeds)

            # --- 2d. VAE Garment Fidelity Check ---
            # No pre-sharpening: the VAE was trained on natural images.
            # Pre-sharpening pushes it out-of-distribution and creates ringing artifacts.
            print(f"\n   [VAE FIDELITY] Encoding garment through VAE and decoding back...")
            garm_tensor = self.pipe.image_processor.preprocess(image_garm).to(device=self.gpu_id, dtype=self.pipe.vae.dtype)
            garm_latent = self.pipe.vae.encode(garm_tensor).latent_dist.mode()
            garm_roundtrip = self.pipe.vae.decode(garm_latent).sample
            # Undo: [-1,1] -> [0,255]
            garm_rt_np = ((garm_roundtrip[0].float().cpu().clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
            debug_save(Image.fromarray(garm_rt_np), "debug_phase2_vae_roundtrip.jpg")
            # Compute PSNR between original and roundtrip
            garm_orig_resized = np.array(image_garm.resize((garm_rt_np.shape[1], garm_rt_np.shape[0]))).astype(np.float32)
            mse_val = np.mean((garm_orig_resized - garm_rt_np.astype(np.float32))**2)
            psnr_val = 10 * np.log10(255**2 / max(mse_val, 1e-6))
            print(f"   [VAE FIDELITY] Garment VAE round-trip PSNR: {psnr_val:.1f} dB (>30=good, >35=excellent)")
            print(f"   [VAE FIDELITY] [SAVED] debug_phase2_vae_roundtrip.jpg")

            # =========================================================
            # PHASE 3 — Denoising Diffusion
            # =========================================================
            print(f"\n[WearCast] Phase 3/4: Starting Denoising Diffusion (U-Net)...")

            # Save final mask diagnostic
            mask_diagnostic = (np.array(mask) * 255).astype(np.uint8)
            debug_save(Image.fromarray(mask_diagnostic), "debug_final_unet_mask.jpg")
            print(f" -> [SAVED] Final UNet mask saved to: debug_final_unet_mask.jpg")

            # Save masked person image (what the UNet receives as context)
            mask_np_vis = np.array(mask.resize(image_vton.size, Image.NEAREST))
            vton_np = np.array(image_vton)
            masked_person_vis = vton_np.copy()
            masked_person_vis[mask_np_vis > 127] = [0, 0, 0]  # black out masked region
            debug_save(Image.fromarray(masked_person_vis), "debug_phase3_masked_person.jpg")
            print(f" -> [SAVED] Masked person input saved to: debug_phase3_masked_person.jpg")

            # --- Pass original image directly (matching original OOTDiffusion) ---
            # Context inpainting was removed: it distorted the image_ori reference and
            # caused the SDEdit blending to produce smeared patches.
            debug_save(image_ori, "debug_phase3_ori_context.jpg")
            print(" -> [CONTEXT] Using original image directly (no inpainting — matches OOTD architecture).")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated(0) / 1e9
                print(f" -> GPU VRAM before UNet call: {mem_before:.2f} GB")

            t_unet_start = time.time()
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                image_garm=garm_proc,    # Garment with 128-gray background (OOTD requirement)
                image_vton=image_vton,   # Masked person image (127-gray replacement)
                mask=mask,
                image_ori=image_ori,     # Original person image (no inpainting)
                num_inference_steps=final_steps,
                image_guidance_scale=final_scale,
                num_images_per_prompt=num_samples,
                generator=generator,
                callback=callback,
                callback_steps=callback_steps,
            ).images
            t_unet_end = time.time()
            raw_generated = images[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated(0) / 1e9
                print(f" -> GPU VRAM after  UNet call: {mem_after:.2f} GB")
            print(f" -> Denoising loop elapsed    : {t_unet_end - t_unet_start:.2f}s")
            print(f" -> Pipeline returned {len(images)} image(s)")

        # =====================================================================
        # Phase 4: Final Post-processing (OOTDiffusion-style paste-back)
        # =====================================================================
        # The UNet was trained to produce realistic try-on results that look
        # correct when composited with simple mask-based paste-back. Complex
        # corrections (histogram matching, Laplacian pyramid, Poisson cloning)
        # fight against the model and create halo/glow artifacts.
        # =====================================================================
        print(f"\n[WearCast] Phase 4/4: Final Post-processing (clean paste-back)...")
        raw_generated = images[0]
        debug_save(raw_generated, "debug_phase4_raw_unet_output.jpg")
        print(f" -> [SAVED] Raw UNet output saved to: debug_phase4_raw_unet_output.jpg")
        print(f" -> UNet Generated Target Size: {raw_generated.size}")

        gen_arr = np.array(raw_generated).astype(np.float32)
        ori_arr = np.array(image_ori).astype(np.float32)
        if ori_arr.shape[:2] != gen_arr.shape[:2]:
            ori_arr = np.array(image_ori.resize(raw_generated.size, Image.BICUBIC)).astype(np.float32)

        # --- FIX #1: Dynamic Output Masking (The Cape Killer) ---
        # Instead of using the input mask (which has a huge background "box"),
        # we re-parse the generated image to find the ACTUAL shirt pixels.
        print(" -> Running Dynamic Re-Parsing for clean mask...")
        t_reparse = time.time()
        parse_new_pil, _ = self.parsing_model(raw_generated)
        parse_new = np.array(parse_new_pil.resize(raw_generated.size, Image.NEAREST))
        
        # Label 4 = upper_clothes, Label 7 = dress
        # Also include arm labels (14=LeftArm, 15=RightArm) when the original
        # inpaint mask covered the arms (long/short sleeve garments).
        # Without this, translucent/low-contrast sleeves in the UNet output are
        # re-classified as bare skin and get cut from the composite mask.
        _orig_mask_coverage = 0.0
        if hasattr(self, '_cached_hard_mask'):
            _hm = np.array(self._cached_hard_mask)
            _orig_mask_coverage = np.mean(_hm > 0)

        _include_arms = _orig_mask_coverage > 0.03  # arms present if mask > 3% coverage above torso baseline ~15%
        
        # We need TWO masks:
        # 1. color_transfer_mask: strictly the garment (4, 7). We do NOT want to color-transfer skin pixels.
        # 2. composite_mask: garment + generated skin (14, 15, 18). We MUST paste the newly generated
        #    skin back over the original image, otherwise the new sleeves won't align with the old arms.
        color_transfer_mask = ((parse_new == 4) | (parse_new == 7)).astype(np.float32)
        composite_mask = ((parse_new == 4) | (parse_new == 7) | (parse_new == 14) | (parse_new == 15) | (parse_new == 18)).astype(np.float32)
        
        # --- FIX #20: Strict Dynamic Bounding (GPU Accelerated) ---
        with torch.no_grad():
            comp_mask_t = torch.from_numpy(composite_mask).unsqueeze(0).unsqueeze(0).to(self.gpu_id)
            color_mask_t = torch.from_numpy(color_transfer_mask).unsqueeze(0).unsqueeze(0).to(self.gpu_id)
            
            if hasattr(self, '_cached_hard_mask'):
                input_mask_np = np.array(self._cached_hard_mask.resize(raw_generated.size, Image.NEAREST)).astype(np.float32) / 255.0
                input_mask_t = torch.from_numpy(input_mask_np).unsqueeze(0).unsqueeze(0).to(self.gpu_id)
                
                # Dilate the hard mask to ensure we don't clip the generated sleeves
                dilation_px = 15 if _include_arms else 7
                kernel_relax = torch.ones(dilation_px, dilation_px, device=self.gpu_id)
                relaxed_mask_t = kornia.morphology.dilation(input_mask_t, kernel_relax)
                
                # Only apply silhouette clipping for torso-only garments.
                if not _include_arms and hasattr(self, '_cached_parse'):
                    sil = (self._cached_parse > 0).astype(np.float32)
                    sil_t = torch.from_numpy(sil).unsqueeze(0).unsqueeze(0).to(self.gpu_id)
                    sil_t = F.interpolate(sil_t, size=(raw_generated.size[1], raw_generated.size[0]), mode='nearest')
                    relaxed_mask_t = relaxed_mask_t * sil_t
                    print(" -> [STRICT] Applied tight silhouette boundary constraint (torso-only, GPU).")
                else:
                    print(" -> [STRICT] Skipped silhouette clipping (arm-inclusive mask — large dilation used instead).")
                    
                comp_mask_t = comp_mask_t * relaxed_mask_t

            # Refine the composite mask slightly
            comp_mask_t = kornia.morphology.dilation(comp_mask_t, torch.ones(5, 5, device=self.gpu_id))
            
            alpha = comp_mask_t[0, 0].cpu().numpy()
            
            # Feather the composite mask for smooth blending
            alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
            alpha = np.clip(alpha, 0, 1)
            binary_mask = (alpha > 0.5).astype(np.uint8) * 255
            
            color_alpha = color_mask_t[0, 0].cpu().numpy()
            color_alpha = cv2.GaussianBlur(color_alpha, (7, 7), 0)
            color_alpha = np.clip(color_alpha, 0, 1)
            
        print(f" -> Dynamic mask generated. Reparse time: {time.time() - t_reparse:.2f}s")
        
        # Save feather mask for debugging
        debug_save(Image.fromarray((alpha * 255).astype(np.uint8)), "debug_phase4_feather_mask.jpg")
        print(" -> Pro-Feather mask: (dynamic alpha generated)")
        # --- Pre-compositing diagnostics ---
        mask_bool = binary_mask > 0.5
        gen_in_mask = gen_arr[mask_bool]
        gen_lum_mean = float((gen_in_mask[:, 0]*0.299 + gen_in_mask[:, 1]*0.587 + gen_in_mask[:, 2]*0.114).mean()) if len(gen_in_mask) > 0 else 128.0
        
        ori_final = np.array(image_ori.resize(raw_generated.size, Image.BICUBIC).convert('RGB')).astype(np.float32)

        # --- Color Transfer ---
        print(" -> [COLOR] Evaluating selective color transfer...")
        garm_rgb = np.array(image_garm.convert('RGB'))
        bg_mask_g = np.all(garm_rgb >= 240, axis=-1)
        fg_g = garm_rgb[~bg_mask_g]
        garm_lum = float((fg_g[:, 0]*0.299 + fg_g[:, 1]*0.587 + fg_g[:, 2]*0.114).mean()) if len(fg_g) > 0 else 255.0
        
        IS_DARK_GARMENT = garm_lum < 50.0
        if IS_DARK_GARMENT:
            print(f" -> [COLOR] Dark garment detected (lum={garm_lum:.1f}). Applying color transfer.")
            gen_arr = self.apply_statistical_color_transfer(gen_arr, image_garm, color_alpha)
        else:
            print(f" -> [COLOR] Light/Medium garment (lum={garm_lum:.1f}). Color transfer disabled.")

        # --- Final Compositing ---
        t_post = time.time()
        print(f" -> [COMPOSITE] Blending generated garment onto original background...")

        # --- PURE SEMANTIC ALPHA BLENDING ---
        # Expand alpha to 3 channels for broadcasting [H, W, 1] -> [H, W, 3]
        alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        
        # Simple math: output = generated * alpha + original * (1 - alpha)
        print(" -> [COMPOSITE] Using Pure Semantic Alpha Blending (clean paste-back)...")
        final_np = gen_arr * alpha_3d + ori_final * (1.0 - alpha_3d)
        
        final_image = Image.fromarray(np.clip(final_np, 0, 255).astype(np.uint8))
        
        # --- Logo Warping ---
        if is_complex:
            print(" -> [GRAPHICS] Applying Logo Warping to restore graphics...")
            final_image = self.apply_logo_warping(image_garm, self._cached_keypoints, final_image, color_alpha)
            final_np = np.array(final_image).astype(np.float32)
            
        print(f" -> [COMPOSITE] Success. Elapsed: {time.time() - t_post:.2f}s")

        # Post-composite sharpening REMOVED.
        # With the upstream fixes (VAE float16, CLIP float32, raw garment input),
        # the UNet output should be sharp enough natively. Sharpening creates ringing artifacts.
        final_image = Image.fromarray(np.clip(final_np, 0, 255).astype(np.uint8))



        # --- Post-compositing diagnostics ---
        final_np = np.array(final_image).astype(np.float32)
        final_in_mask = final_np[mask_bool]
        final_lum = float((final_in_mask[:, 0]*0.299 + final_in_mask[:, 1]*0.587 + final_in_mask[:, 2]*0.114).mean()) if len(final_in_mask) > 0 else 0.0
        print(f"   [DIAG] Final composited luminance: {final_lum:.1f} (recovered from {gen_lum_mean:.1f})")

        # Save 3-panel comparison: Garment | Raw UNet | Final
        comparison_w = raw_generated.size[0] * 3
        comparison = Image.new('RGB', (comparison_w, raw_generated.size[1]))
        comparison.paste(image_garm.resize(raw_generated.size), (0, 0))
        comparison.paste(raw_generated, (raw_generated.size[0], 0))
        comparison.paste(final_image, (raw_generated.size[0] * 2, 0))
        debug_save(comparison, "debug_phase4_comparison.jpg")
        print(f" -> [SAVED] 3-panel comparison: Garment | Raw UNet | Final")

        debug_save(final_image, "debug_final_output.jpg")
        print(f" -> [SAVED] Final result saved to: debug_final_output.jpg")

        print("\n[WearCast] SUCCESS: Inference completed successfully!")
        print("=" * 70)

        return [final_image]

    def detect_garment_complexity(self, image_garm):
        """
        [DISABLED] Logo warping is destructive for standard garments.
        The UNet generates patterns and logos accurately natively.
        """
        print("   [COMPLEXITY] TPS Warping: DISABLED (returning False always)")
        return False

    def get_optimal_params(self, category, is_complex_garment):
        if is_complex_garment:
            # [FIX PATTERN CLARITY] Complex/patterned garments:
            # Raised image_scale from 2.0 → 2.5 to strengthen garment conditioning.
            # Higher guidance = UNet follows source garment more faithfully,
            # which reduces pattern distortion from the VAE encode/decode cycle.
            return {"num_steps": 30, "image_scale": 2.5}
        else:
            # Simple solid garments: 30 steps, 2.0 scale (lower scale = smoother, natural drape)
            return {"num_steps": 30, "image_scale": 2.0}

    def apply_statistical_color_transfer(self, gen_arr, image_garm, alpha_mask):
        """
        Content-Aware Statistical Color Transfer in LAB space.
        GPU-Accelerated via Kornia for maximum performance.
        """
        print(" -> [COLOR] Applying Content-Aware LAB Color Transfer (GPU Accelerated)...")
        
        # 1. Prepare target garment pixels
        garm_arr = np.array(image_garm.resize((gen_arr.shape[1], gen_arr.shape[0]))).astype(np.float32)
        if hasattr(self, '_cached_garm_mask'):
            garm_mask_resized = np.array(Image.fromarray(self._cached_garm_mask * 255).resize((gen_arr.shape[1], gen_arr.shape[0]), Image.NEAREST))
            valid_garm = garm_mask_resized > 127
        else:
            valid_garm = ~np.all(garm_arr >= 240, axis=-1)
            
        if valid_garm.sum() < 100:
            print("   [COLOR] Not enough valid garment pixels to calculate stats. Skipping transfer.")
            return gen_arr
            
        mask_bool = alpha_mask > 0.05
        if mask_bool.sum() < 100:
            return gen_arr
            
        # 2. Convert to Tensors and move to GPU
        with torch.no_grad():
            gen_tensor = torch.from_numpy(gen_arr).permute(2, 0, 1).unsqueeze(0).to(self.gpu_id) / 255.0
            garm_tensor = torch.from_numpy(garm_arr).permute(2, 0, 1).unsqueeze(0).to(self.gpu_id) / 255.0
            
            # 3. Convert to LAB color space via Kornia (Note: Kornia L is 0-100)
            gen_lab = kornia.color.rgb_to_lab(gen_tensor)
            garm_lab = kornia.color.rgb_to_lab(garm_tensor)
            
            # Extract valid pixels
            valid_garm_t = torch.from_numpy(valid_garm).to(self.gpu_id)
            mask_bool_t = torch.from_numpy(mask_bool).to(self.gpu_id)
            
            target_pixels = garm_lab[0].permute(1, 2, 0)[valid_garm_t] # N x 3
            source_pixels = gen_lab[0].permute(1, 2, 0)[mask_bool_t] # M x 3
            
            source_median = source_pixels.median(dim=0).values
            rough_shift = target_pixels.median(dim=0).values - source_median
            
            target_repr = torch.zeros(3, device=self.gpu_id)
            for ch in range(3):
                if rough_shift[ch] < -2.0: # scaled for 0-100 L scale
                    target_repr[ch] = torch.quantile(target_pixels[:, ch], 0.20)
                elif rough_shift[ch] > 2.0:
                    target_repr[ch] = torch.quantile(target_pixels[:, ch], 0.80)
                else:
                    target_repr[ch] = target_pixels[:, ch].median()
                    
            print(f"   [COLOR] Target Rep (directional pct): L={target_repr[0]:.1f}, a={target_repr[1]:.1f}, b={target_repr[2]:.1f}")
            print(f"   [COLOR] Source Median: L={source_median[0]:.1f}, a={source_median[1]:.1f}, b={source_median[2]:.1f}")

            # 4. Asymmetric Gamut-Aware L-Channel Transfer
            # CAP: Limit L-shift to ±12 L-units (Kornia 0-100 scale, ~±30 in 0-255 scale).
            # Without this cap, white shirts get shifted to L=96+ (washed-out / pure white),
            # and black shirts get crushed to L<5 (all detail lost). 
            # ADAPTIVE: For very bright targets (L > 85), raise cap to 15.0 to ensure
            # white garments don't look gray. For very dark targets (L < 20), raise cap to 25.0
            # so black shirts aren't left dark gray.
            MAX_L_SHIFT = 15.0 if target_repr[0] > 85.0 else (25.0 if target_repr[0] < 20.0 else 12.0)
            corrected_lab = gen_lab.clone()
            source_l = gen_lab[:, 0:1, :, :]
            raw_shift_l = target_repr[0] - source_median[0]
            shift_l = torch.clamp(raw_shift_l, -MAX_L_SHIFT, MAX_L_SHIFT)
            print(f"   [COLOR] L-shift: raw={raw_shift_l:.1f}, clamped={shift_l:.1f} (cap=+/-{MAX_L_SHIFT})")
            
            if shift_l < 0:
                multiplier_l = torch.where(
                    source_l <= source_median[0],
                    source_l / (source_median[0] + 1e-6),
                    torch.ones_like(source_l)
                )
                logo_protect = torch.clamp((82.0 - source_l) / 19.6, 0.0, 1.0) # 82 corresponds to ~210 in 0-255
                multiplier_l = torch.min(multiplier_l, logo_protect)
            else:
                multiplier_l = torch.where(
                    source_l >= source_median[0],
                    (100.0 - source_l) / (100.0 - source_median[0] + 1e-6),
                    torch.ones_like(source_l)
                )
                logo_protect = torch.clamp(source_l / 19.6, 0.0, 1.0) # 19.6 corresponds to ~50 in 0-255
                multiplier_l = torch.min(multiplier_l, logo_protect)
                
            corrected_lab[:, 0:1, :, :] = source_l + shift_l * multiplier_l
            
            # A/B Channels Color Transfer with Logo Protection
            ab_diff = torch.sqrt(torch.sum((gen_lab[:, 1:, :, :] - source_median[1:].view(1, 2, 1, 1))**2, dim=1, keepdim=True))
            color_transfer_weight = torch.clamp((15.7 - ab_diff) / 7.8, 0.0, 1.0) # Scaled for Kornia LAB A/B scale (-128 to 127 vs OpenCV 0 to 255)
            
            for ch in [1, 2]:
                shift_c = target_repr[ch] - source_median[ch]
                corrected_lab[:, ch:ch+1, :, :] += shift_c * color_transfer_weight
                
            # Convert back to RGB
            corrected_rgb_t = kornia.color.lab_to_rgb(corrected_lab) * 255.0
            corrected_rgb_t = torch.clamp(corrected_rgb_t, 0.0, 255.0)
            
            corrected_rgb = corrected_rgb_t[0].permute(1, 2, 0).cpu().numpy()
            
            # Blend into original array based on mask
            result = gen_arr.copy()
            result[mask_bool] = corrected_rgb[mask_bool]
            
            print("   [COLOR] Content-Aware LAB transfer complete.")
            return result

    def laplacian_pyramid_blend(self, generated, original, mask_soft, levels=5):
        """
        Seamlessly blend using Gaussian/Laplacian pyramids.
        """
        gen_np = np.array(generated).astype(np.float32)
        ori_np = np.array(original).astype(np.float32)
        msk_np = np.array(mask_soft).astype(np.float32) / 255.0

        if msk_np.ndim == 2:
            msk_np = np.stack([msk_np] * 3, axis=-1)

        print(f"   [LAPBLEND] gen={gen_np.shape}  ori={ori_np.shape}  mask={msk_np.shape}  levels={levels}")

        def build_gaussian_pyramid(img, levels):
            pyramid = [img]
            for _ in range(levels - 1):
                img = cv2.pyrDown(img)
                pyramid.append(img)
            return pyramid

        def build_laplacian_pyramid(img, levels):
            gauss = build_gaussian_pyramid(img, levels)
            lap = []
            for i in range(levels - 1):
                up = cv2.pyrUp(gauss[i + 1], dstsize=(gauss[i].shape[1], gauss[i].shape[0]))
                lap.append(gauss[i] - up)
            lap.append(gauss[-1])
            return lap

        lap_gen  = build_laplacian_pyramid(gen_np,  levels)
        lap_ori  = build_laplacian_pyramid(ori_np,  levels)
        gauss_msk = build_gaussian_pyramid(msk_np, levels)

        blended_pyramid = []
        for i in range(levels):
            msk_level = cv2.resize(gauss_msk[i], (lap_gen[i].shape[1], lap_gen[i].shape[0]))
            if msk_level.ndim == 2:
                msk_level = np.stack([msk_level] * 3, axis=-1)
            blended = lap_gen[i] * msk_level + lap_ori[i] * (1 - msk_level)
            blended_pyramid.append(blended)
            print(f"   [LAPBLEND] Level {i}: gen_lap={lap_gen[i].shape}  msk_level={msk_level.shape}")

        result = blended_pyramid[-1]
        for i in range(levels - 2, -1, -1):
            result = cv2.pyrUp(result, dstsize=(blended_pyramid[i].shape[1], blended_pyramid[i].shape[0]))
            result = result + blended_pyramid[i]

        print(f"   [LAPBLEND] Final result shape: {result.shape}")
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))