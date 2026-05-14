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

        # 1. Load VAE
        print("[WearCastHD] Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            MODEL_PATH,
            subfolder="vae",
            torch_dtype=torch.float32,
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
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.gpu_id).half()
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
            print(" -> [LOGO_WARP] AI-Guided placement detected. Aligning graphics to UNet features.")
            tx_min, tx_max = np.min(gx), np.max(gx)
            ty_min, ty_max = np.min(gy), np.max(gy)
            
            # Add small padding to prevent edge cropping
            pw, ph = (tx_max - tx_min) * 0.05, (ty_max - ty_min) * 0.05
            dst_pts = np.float32([
                [tx_min - pw, ty_min - ph], [tx_max + pw, ty_min - ph],
                [tx_max + pw, ty_max + ph], [tx_min - pw, ty_max + ph]
            ])
            src_pts = np.float32([
                [logo_bbox[0], logo_bbox[1]], [logo_bbox[2], logo_bbox[1]],
                [logo_bbox[2], logo_bbox[3]], [logo_bbox[0], logo_bbox[3]]
            ])
        else:
            print(" -> [LOGO_WARP] Using global anatomical mapping.")
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

            # --- 2c. CLIP Encoding (Standard Global) ---
            # Reverted to standard global embedding to prevent UNet out-of-distribution hallucinations
            clip_inputs = self.auto_processor(images=garm_enhanced, return_tensors="pt").to(device=self.gpu_id)
            with torch.no_grad():
                clip_outputs = self.image_encoder(clip_inputs.data['pixel_values'].to(dtype=torch.float16))
                image_embeds = clip_outputs.image_embeds.unsqueeze(1) # [1, 1, 768]
            
            # --- 2d. Advanced Prompt Construction ---
            # CRITICAL FIX 1: OOTDiffusion strictly requires a 2-token sequence ["", Image_Embed].
            # Feeding 77 tokens causes the UNet to hallucinate the "Gray Blob".
            print(" -> [PROMPT] Bypassing text prompt to restore OOTDiffusion 2-token architecture.")
            prompt_embeds = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0].to(dtype=torch.float16) # [1, 2, 768]
            
            # Inject global image embedding into token index 1 (OOTDiffusion Standard)
            prompt_embeds[:, 1:2, :] = image_embeds[:]
            print(f" -> Prompt embeddings shape: {list(prompt_embeds.shape)} (Restored strict 2-token format)")
            _dbg_tensor("prompt_embeds", prompt_embeds)

            # --- 2d. VAE Garment Fidelity Check ---
            # Encode garment through VAE and decode back to check reconstruction quality
            print(f"\n   [VAE FIDELITY] Encoding garment through VAE and decoding back...")
            # FIX #6: Adaptive sharpening before VAE encoding to preserve fine graphic/text details.
            # Complex garments (prints, text, logos) get stronger sharpening to compensate
            # for the VAE's natural blurring tendency (PSNR ~27 dB for this checkpoint).
            # Simple garments stay gentle to avoid edge halos on fabric folds.
            from PIL import ImageFilter
            if is_complex:
                # Strong sharpening for graphic/patterned garments
                image_garm_for_vae = image_garm.filter(ImageFilter.UnsharpMask(radius=1.5, percent=160, threshold=2))
            else:
                # Gentle sharpening for solid-color garments
                image_garm_for_vae = image_garm.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
            garm_tensor = self.pipe.image_processor.preprocess(image_garm_for_vae).to(device=self.gpu_id, dtype=torch.float32)
            garm_latent = self.pipe.vae.encode(garm_tensor).latent_dist.mode()
            garm_roundtrip = self.pipe.vae.decode(garm_latent).sample  # mode() returns raw latents, decode directly
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

            # [FIX: GHOSTING + COLOR CONTAMINATION]
            # Neutralize original garment pixels inside the mask region before sending
            # to the UNet as context (image_ori). Without this fix, the UNet sees the
            # OLD clothing's color and geometry as a strong prior, causing:
            #   - Ghosting: old garment shape bleeds into generated result (Run 3)
            #   - Color contamination: dark original garment contaminates white target (Run 2)
            # We replace the masked region with neutral grey (128, 128, 128) — the
            # training-distribution neutral value — with a soft boundary blur.
            _mask_np_ctx = np.array(mask.resize(image_ori.size, Image.NEAREST)).astype(np.float32) / 255.0
            # Soft boundary: gaussian blur on the mask boundary
            _ctx_alpha = cv2.GaussianBlur(_mask_np_ctx, (0, 0), 8.0)
            _ctx_alpha = np.clip(_ctx_alpha, 0.0, 1.0)
            _ori_np_full = np.array(image_ori).astype(np.float32)
            _neutral_fill = np.full_like(_ori_np_full, 128.0)
            _ori_ctx_np = (
                _ori_np_full * (1.0 - _ctx_alpha[:, :, np.newaxis])
                + _neutral_fill * _ctx_alpha[:, :, np.newaxis]
            )
            image_ori_constrained = Image.fromarray(np.clip(_ori_ctx_np, 0, 255).astype(np.uint8))
            debug_save(image_ori_constrained, "debug_phase3_ori_context.jpg")
            print(" -> [CONTEXT] Neutralized garment region in UNet context (Ghosting + Color Contamination fix).")
            print(f" -> [CONTEXT] Mask coverage in context: {100.0*float(_ctx_alpha.mean()):.2f}% neutralized.")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated(0) / 1e9
                print(f" -> GPU VRAM before UNet call: {mem_before:.2f} GB")

            t_unet_start = time.time()
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                image_garm=garm_proc,
                image_vton=image_vton,
                mask=mask,
                image_ori=image_ori_constrained, # USE CONSTRAINED SILHOUETTE
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
        if _include_arms:
            dynamic_mask = ((parse_new == 4) | (parse_new == 7) | (parse_new == 14) | (parse_new == 15)).astype(np.float32)
        else:
            dynamic_mask = ((parse_new == 4) | (parse_new == 7)).astype(np.float32)
        
        # --- FIX #20: Strict Dynamic Bounding ---
        # Reduces relaxation kernel and ensures result is strictly inside the silhouette.
        if hasattr(self, '_cached_hard_mask'):
            input_mask_np = np.array(self._cached_hard_mask.resize(raw_generated.size, Image.NEAREST)).astype(np.float32) / 255.0
            # Reduced to 5x5 for tighter fit
            kernel_relax = np.ones((5, 5), np.uint8)
            relaxed_mask = cv2.dilate(input_mask_np, kernel_relax, iterations=1)
            
            # MUST stay within person silhouette
            if hasattr(self, '_cached_parse'):
                sil = (self._cached_parse > 0).astype(np.float32)
                sil = cv2.resize(sil, raw_generated.size, interpolation=cv2.INTER_NEAREST)
                relaxed_mask = relaxed_mask * sil
                
            dynamic_mask = dynamic_mask * relaxed_mask
            print(" -> [STRICT] Applied tight silhouette boundary constraint.")

        # Refine the dynamic mask slightly
        dynamic_mask = cv2.dilate(dynamic_mask, np.ones((5, 5), np.uint8), iterations=1)
        
        # Binarize
        binary_mask = (dynamic_mask > 0.5).astype(np.uint8)
        # FIX #4: Seamless Photographic Alpha Blend
        # 1. Very gentle erosion to avoid bleeds but keep boundary natural
        kernel_erode = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(binary_mask, kernel_erode, iterations=1) 
        
        # 2. Narrower photographic feathering instead of jagged cutouts (prevents background halo)
        feather_sigma = 3.0
        alpha = cv2.GaussianBlur(eroded_mask.astype(np.float32), (0, 0), feather_sigma)
        
        # FIX: Restore the solid core so thin sleeves don't become translucent from the blur
        core_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1).astype(np.float32)
        alpha = np.maximum(alpha, core_mask)
        alpha = np.clip(alpha, 0.0, 1.0)
        print(f" -> Dynamic mask generated. Reparse time: {time.time() - t_reparse:.2f}s")

        # Save feather mask for debugging
        debug_save(Image.fromarray((alpha * 255).astype(np.uint8)), "debug_phase4_feather_mask.jpg")
        print(f" -> Pro-Feather mask: sigma={feather_sigma}px (eroded for tight fit)")
        # --- Pre-compositing diagnostics ---
        mask_bool = binary_mask > 0.5
        gen_in_mask = gen_arr[mask_bool]
        gen_lum_mean = float((gen_in_mask[:, 0]*0.299 + gen_in_mask[:, 1]*0.587 + gen_in_mask[:, 2]*0.114).mean()) if len(gen_in_mask) > 0 else 128.0
        
        ori_final = np.array(image_ori.resize(raw_generated.size, Image.BICUBIC).convert('RGB')).astype(np.float32)

        # We use Reinhard LAB Statistical Color Transfer to mathematically enforce the target color
        # without destroying the AI's 3D variance (wrinkles, shadows, highlights).
        gen_arr = self.apply_statistical_color_transfer(gen_arr, image_garm, alpha)

        # --- Final Compositing ---
        # We gently paste the AI-generated garment onto the ORIGINAL background to preserve face/bg crispness.
        t_post = time.time()
        print(f" -> [COMPOSITE] Blending generated garment onto original background...")

        # --- STUDIO MASTER: AI-Native Rendering ---
        # [MASTER PLAN] We NO LONGER manually warp logos if we want AI-Native results.
        # This prevents "Double Logos" and allows the UNet to render graphics organically.
        # if is_complex:
        #    print(" -> [LOGO] Patterned/Graphic garment detected. Applying TPS Warping...")
        #    blended_pil = Image.fromarray(np.clip(gen_arr, 0, 255).astype(np.uint8))
        #    blended_pil = self.apply_logo_warping(image_garm, self._cached_keypoints, blended_pil, alpha)
        #    gen_arr = np.array(blended_pil).astype(np.float32)

        # --- FIX #12: Apply High-Frequency Skin Blending ---
        if hasattr(self, '_skin_mask'):
            # Resize skin mask to match output
            skin_mask_final = cv2.resize(self._skin_mask, raw_generated.size, interpolation=cv2.INTER_LINEAR)
            
            # Prevent skin texture from bleeding onto newly generated fabric
            # Since alpha represents the new garment, we subtract it from the skin mask
            skin_mask_final = skin_mask_final * (1.0 - alpha)
            
            gen_arr = self.apply_frequency_blending(gen_arr, ori_final, skin_mask_final)
            print(" -> [SKIN] Frequency separation complete (restored skin texture on bare areas).")
        
        # Expand alpha to 3 channels for broadcasting [H, W, 1] -> [H, W, 3]
        alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        
        # Alpha Blend: result = (new * alpha) + (old * (1 - alpha))
        final_np = (gen_arr * alpha_3d) + (ori_final * (1.0 - alpha_3d))
        final_image = Image.fromarray(np.clip(final_np, 0, 255).astype(np.uint8))
        
        print(f" -> [COMPOSITE] Success. Elapsed: {time.time() - t_post:.2f}s")

        # --- [FIX PATTERN CLARITY] Post-Composite Adaptive Sharpening ---
        # The VAE encode/decode cycle loses high-frequency detail (PSNR ~27 dB).
        # We recover edge sharpness by applying an UnsharpMask ONLY on the garment region.
        # Strength scales with garment complexity: stronger for prints/text, gentle for solids.
        if is_complex:
            _sharp_radius  = 1.2
            _sharp_percent = 140
            _sharp_thresh  = 2
        else:
            _sharp_radius  = 0.8
            _sharp_percent = 110
            _sharp_thresh  = 3
        _final_pil    = Image.fromarray(np.clip(final_np, 0, 255).astype(np.uint8))
        _final_sharp  = _final_pil.filter(ImageFilter.UnsharpMask(
            radius=_sharp_radius, percent=_sharp_percent, threshold=_sharp_thresh
        ))
        # Blend sharpened result ONLY within the garment mask (alpha_3d), preserve background as-is
        _sharp_np  = np.array(_final_sharp).astype(np.float32)
        final_np   = _sharp_np * alpha_3d + final_np * (1.0 - alpha_3d)
        final_image = Image.fromarray(np.clip(final_np, 0, 255).astype(np.uint8))
        print(f" -> [SHARP] Post-composite sharpening applied (complex={is_complex}, r={_sharp_radius}, p={_sharp_percent}%).")



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
        [UPDATED] Highly sensitive detection for logos, text, and patterns.
        """
        import cv2
        import numpy as np
        garm_np = np.array(image_garm)
        bg_mask = np.all(garm_np >= 240, axis=-1)
        fg_pixels = garm_np[~bg_mask]

        if len(fg_pixels) < 500:
            return False

        # 1. Color variance (Lowered threshold from 38.0 to 20.0)
        color_std = np.std(fg_pixels, axis=0).mean()
        is_patterned = color_std > 20.0

        # 2. Interior Edge Density (For text and thin logos)
        gray = cv2.cvtColor(garm_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100) # More sensitive edge detection
        
        # We only care about edges INSIDE the shirt (ignore the outer boundary)
        fg_mask_img = (~bg_mask).astype(np.uint8) * 255
        interior_mask = cv2.erode(fg_mask_img, np.ones((15, 15), np.uint8), iterations=1)
        interior_edges = cv2.bitwise_and(edges, interior_mask)
        
        # Lowered threshold from 0.06 (6%) to 0.015 (1.5%)
        edge_density = np.sum(interior_edges > 0) / max(1, np.sum(interior_mask > 0))
        is_complex_shape = edge_density > 0.015 

        print(f"   [COMPLEXITY] color_std={color_std:.2f} | interior_edge_density={edge_density:.4f} | TPS_WARPING_TRIGGERED={is_patterned or is_complex_shape}")
        return is_patterned or is_complex_shape

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
        FIX #1: Uses directional percentiles (not median) to find the true fabric color,
        so that logos/graphics don't contaminate the target measurement.
        FIX #2: Returns corrected gen_arr WITHOUT doing an alpha blend — the caller
        performs the single authoritative composite. Eliminates the double-blend bug
        that caused garments to look transparent/pasted.
        """
        print(" -> [COLOR] Applying Content-Aware LAB Color Transfer...")
        
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
            
        # 2. Convert to LAB color space
        gen_lab = cv2.cvtColor(np.clip(gen_arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        garm_lab = cv2.cvtColor(np.clip(garm_arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # 3. FIX #1: Directional-Percentile Target Measurement
        # -------------------------------------------------------
        # OLD: np.median(target_pixels) — biased by white/dark logos on opposite-tone shirts.
        # NEW: When darkening (target < source), use 20th-pct to get true dark-fabric tone.
        #      When brightening (target > source), use 80th-pct to get true light-fabric tone.
        # This removes the logo-contamination effect that was making black shirts appear grey.
        target_pixels = garm_lab[valid_garm]
        source_pixels = gen_lab[mask_bool]
        source_median = np.median(source_pixels, axis=0)

        # Determine shift direction per channel before choosing percentile
        rough_shift = np.median(target_pixels, axis=0) - source_median
        target_repr = np.zeros(3)
        for ch in range(3):
            if rough_shift[ch] < -5:      # Darkening: use lower percentile (true dark fabric)
                target_repr[ch] = np.percentile(target_pixels[:, ch], 20)
            elif rough_shift[ch] > 5:     # Brightening: use upper percentile (true light fabric)
                target_repr[ch] = np.percentile(target_pixels[:, ch], 80)
            else:                          # Neutral: use median (no strong shift needed)
                target_repr[ch] = np.median(target_pixels[:, ch])

        target_median = target_repr  # use the directional representative value
        target_std = np.std(target_pixels, axis=0) + 1e-6
        source_std = np.std(source_pixels, axis=0) + 1e-6
        
        print(f"   [COLOR] Target Rep (directional pct): L={target_median[0]:.1f}, a={target_median[1]:.1f}, b={target_median[2]:.1f}")
        print(f"   [COLOR] Source Median: L={source_median[0]:.1f}, a={source_median[1]:.1f}, b={source_median[2]:.1f}")

        # 4. Asymmetric Gamut-Aware L-Channel Transfer
        corrected_lab = gen_lab.copy()
        source_l = gen_lab[:, :, 0]
        shift_l = target_median[0] - source_median[0]
        
        if shift_l < 0:
            # Darkening (e.g. Black shirt)
            # Shadows approach 0 — decay the shift to avoid full crush.
            # Highlights are moving away from 255 — full shift preserves 3D fold depth.
            multiplier_l = np.where(
                source_l <= source_median[0],
                source_l / (source_median[0] + 1e-6),  # Proportional decay for shadows
                1.0  # Full shift for highlights
            )
            # Protect pure-white logo pixels (L > 180) — avoid turning them grey
            logo_protect = np.clip((210.0 - source_l) / 50.0, 0.0, 1.0)
            multiplier_l = np.minimum(multiplier_l, logo_protect)
        else:
            # Brightening (e.g. White shirt)
            # Highlights approach 255 — decay to prevent blowout.
            # Shadows are moving away from 0 — full shift preserves 3D fold depth.
            multiplier_l = np.where(
                source_l >= source_median[0],
                (255.0 - source_l) / (255.0 - source_median[0] + 1e-6),  # Decay highlights
                1.0  # Full shift for shadows
            )
            # Protect pure-black logo pixels (L < 60) — avoid brightening them
            logo_protect = np.clip((source_l - 30.0) / 50.0, 0.0, 1.0)
            multiplier_l = np.minimum(multiplier_l, logo_protect)
            
        shifted_l = source_l + shift_l * multiplier_l
        
        # FIX #1b: Reduced shadow soft-clip floor from L=30 → L=10
        # OLD floor of 30 was preventing true black (L≈5-15) from ever being reached.
        high_mask = shifted_l > 220
        if high_mask.sum() > 0:
            excess = shifted_l[high_mask] - 220
            shifted_l[high_mask] = 220 + 35.0 * (1.0 - np.exp(-excess / 35.0))
            
        # Shadows: compress below 10 (not 30), allowing near-black fabric
        low_mask = shifted_l < 10
        if low_mask.sum() > 0:
            deficit = 10 - shifted_l[low_mask]
            shifted_l[low_mask] = 10 - 10.0 * (1.0 - np.exp(-deficit / 10.0))
            
        corrected_lab[mask_bool, 0] = shifted_l[mask_bool]
        
        # A and B Channels (Color)
        # Protect colorful graphics using pure A/B distance from the source neutral
        ab_diff = np.sqrt(np.sum((gen_lab[:, :, 1:] - source_median[1:])**2, axis=2))
        color_transfer_weight = np.clip((40.0 - ab_diff) / 20.0, 0.0, 1.0)
        
        for i in [1, 2]:
            source_c = gen_lab[:, :, i]
            shift_c = target_median[i] - source_median[i]
            shifted_c = source_c + shift_c * color_transfer_weight
            corrected_lab[mask_bool, i] = shifted_c[mask_bool]
            
        corrected_lab = np.clip(corrected_lab, 0, 255)
        
        # FIX #2: Return corrected pixels WITHOUT an alpha blend.
        # Previously this function blended internally AND the caller blended again,
        # causing a double-composite that made garments look semi-transparent.
        # Now: only pixels inside the mask region are updated; caller does the one blend.
        corrected_rgb = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
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