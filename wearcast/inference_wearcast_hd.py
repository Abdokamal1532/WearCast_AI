from pathlib import Path
import sys
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
import random
import time

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


def _dbg_tensor(label, t):
    """Print a compact one-liner tensor summary."""
    if isinstance(t, torch.Tensor):
        mn, mx, mu = t.float().min().item(), t.float().max().item(), t.float().mean().item()
        print(f"   [DBG] {label:40s} | shape={list(t.shape)} dtype={t.dtype} dev={t.device} | min={mn:.4f} max={mx:.4f} mean={mu:.4f}")
    else:
        print(f"   [DBG] {label:40s} | type={type(t)}")


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

        from transformers import CLIPVisionModel, CLIPVisionModelWithProjection
        print("[WearCastHD] Loading CLIPVisionModel (for hidden states)...")
        self.image_encoder = CLIPVisionModel.from_pretrained(VIT_PATH).to(self.gpu_id).half()
        print(f"[WearCastHD] CLIPVisionModel loaded.  hidden_size={self.image_encoder.config.hidden_size}  num_hidden_layers={self.image_encoder.config.num_hidden_layers}")

        print("[WearCastHD] Extracting pretrained visual_projection...")
        clip_full = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH)
        self.visual_projection = clip_full.visual_projection.to(self.gpu_id).half()
        del clip_full
        print(f"[WearCastHD] visual_projection extracted. projection_dim={self.visual_projection.out_features if hasattr(self.visual_projection, 'out_features') else 'N/A'}")

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

        # Swap to DPM-Solver++
        from diffusers import DPMSolverMultistepScheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True,
            final_sigmas_type="sigma_min",
        )
        print(f"[WearCastHD] Scheduler replaced with DPMSolverMultistepScheduler.")
        print("=" * 70)

    def tokenize_captions(self, captions, max_length):
        inputs = self.tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def __call__(self,
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
    ):
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
            for lbl, cnt in zip(unique_labels, counts):
                name = label_names.get(int(lbl), f'Unknown({lbl})')
                print(f"       Label {lbl:2d} ({name:20s}): {cnt:6d} px ({100*cnt/parse_arr.size:.1f}%)")

            # 2. Sophisticated Mask Generation
            print(" -> Constructing sophisticated in-painting mask...")
            hard_mask, mask = self.get_mask_location(model_type, category, model_parse, keypoints)

            # Diagnostic: Check mask density
            mask_np = np.array(mask)
            mask_pixels = np.sum(mask_np > 127)
            total_pixels = mask_np.size
            print(f" -> Mask Diagnostic: {mask_pixels} pixels marked for replacement ({100*mask_pixels/total_pixels:.2f}% of image)")
            print(f" -> Mask raw size: {mask.size}  |  dtype={mask_np.dtype}  |  min={mask_np.min()}  max={mask_np.max()}")

            mask = mask.resize((768, 1024), Image.NEAREST)
            print(f" -> Mask after resize: {mask.size}")

            # Save mask debug image
            debug_mask_path = "debug_phase1_soft_mask.jpg"
            mask.save(debug_mask_path)
            print(f" -> [SAVED] Soft mask saved to: {debug_mask_path}")

            hard_mask_resized = hard_mask.resize((768, 1024), Image.NEAREST)
            debug_hard_mask_path = "debug_phase1_hard_mask.jpg"
            hard_mask_resized.save(debug_hard_mask_path)
            print(f" -> [SAVED] Hard mask saved to: {debug_hard_mask_path}")

            print(" -> Preprocessing Stage: SUCCESS")

        # =========================================================
        # PHASE 1.5 — Garment Auto-Analysis
        # =========================================================
        print(f"\n[WearCast] Phase 1.5: Garment Complexity Analysis...")
        is_complex = self.detect_garment_complexity(image_garm)
        opt_params = self.get_optimal_params(category, is_complex)
        final_steps = opt_params["num_steps"]
        final_scale = opt_params["image_scale"]
        print(f" -> is_complex     : {is_complex}")
        print(f" -> Auto-params    : steps={final_steps}, scale={final_scale}, complex={is_complex}")

        # =========================================================
        # PHASE 2 — CLIP Vision Encoding
        # =========================================================
        print(f"\n[WearCast] Phase 2/4: Encoding Inputs (VAE & Multi-Scale CLIP Vision)...")
        with torch.no_grad():
            from PIL import ImageEnhance
            garm_np = np.array(image_garm.copy())
            bg_mask = np.all(garm_np >= 240, axis=-1)
            bg_coverage = bg_mask.mean()
            print(f" -> Garment BG analysis: {100*bg_coverage:.1f}% near-white pixels (threshold=5%)")

            if bg_coverage > 0.05:
                print(f" -> White product background detected ({100*bg_coverage:.1f}% coverage), replacing with mid-gray for CLIP...")
                garm_np_proc = garm_np.copy()
                garm_np_proc[bg_mask] = [160, 160, 160]
                garm_proc = Image.fromarray(garm_np_proc)
            else:
                garm_proc = image_garm.copy()

            # === ACCURACY FIX 1: Stronger sharpening for richer CLIP detail ===
            garm_enhanced = ImageEnhance.Sharpness(garm_proc).enhance(2.5)
            garm_enhanced = ImageEnhance.Contrast(garm_enhanced).enhance(1.6)

            # ---- MULTI-SCALE CROPS (4 crops for richer garment coverage) ----
            w, h = garm_enhanced.size
            cx, cy = w // 2, h // 2
            crop_full    = garm_enhanced.copy()
            crop_center  = garm_enhanced.crop([
                int(cx - w * 0.30), int(cy - h * 0.30),
                int(cx + w * 0.30), int(cy + h * 0.30)
            ])
            crop_upper   = garm_enhanced.crop([0, 0, w, int(h * 0.40)])
            # 4th crop: focuses on the main graphic/print region (center vertical band)
            crop_graphic = garm_enhanced.crop([int(w*0.10), int(h*0.15), int(w*0.90), int(h*0.85)])
            print(f" -> Multi-scale crops: full={crop_full.size}, center={crop_center.size}, upper={crop_upper.size}, graphic={crop_graphic.size}")

            # ---- ENCODE CONTEXT SCALES & PROJECT ----
            # === ACCURACY FIX 2: stride 32→16 doubles spatial token coverage (8→16 patch tokens)
            def encode_crop(img, name="?"):
                inputs = self.auto_processor(images=img, return_tensors="pt").to(self.gpu_id)
                pixel_vals = inputs.data['pixel_values'].to(dtype=torch.float16)
                print(f"    [CLIP] Encoding '{name}': pixel_values={list(pixel_vals.shape)} dtype={pixel_vals.dtype}")
                outputs = self.image_encoder(pixel_values=pixel_vals, output_hidden_states=True)
                hidden = outputs.last_hidden_state   # [1, 257, 1024]
                print(f"    [CLIP]   last_hidden_state={list(hidden.shape)}")

                patch_tokens = hidden[:, 1:, :]              # [1, 256, 1024]
                projected_patches = self.visual_projection(patch_tokens)  # [1, 256, 768]
                sampled = projected_patches[:, ::16, :]      # [1, 16, 768] (was ::32 = 8 tokens)

                cls_proj = self.visual_projection(hidden[:, 0:1, :])  # [1, 1, 768]
                result = torch.cat([cls_proj, sampled], dim=1)        # [1, 17, 768]
                print(f"    [CLIP]   final embedding={list(result.shape)}")
                return result

            feat_full    = encode_crop(crop_full,    "full")
            feat_center  = encode_crop(crop_center,  "center")
            feat_upper   = encode_crop(crop_upper,   "upper")
            feat_graphic = encode_crop(crop_graphic, "graphic")

            # === Graphic crop weight boosted 30%→40% for better print/text fidelity ===
            # full(25%) + center(20%) + upper(15%) + graphic(40%) = 100%
            garment_features = (feat_full * 0.25) + (feat_center * 0.20) + (feat_upper * 0.15) + (feat_graphic * 0.40)
            print(f" -> Fused garment_features: {list(garment_features.shape)}")

            text_emb = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0].to(dtype=torch.float16)
            print(f" -> Text embedding (null): {list(text_emb.shape)}")
            prompt_embeds = torch.cat([text_emb, garment_features], dim=1)
            print(f" -> Multi-Scale CLIP Embedding shape: {prompt_embeds.shape}")
            _dbg_tensor("prompt_embeds", prompt_embeds)

            # =========================================================
            # PHASE 3 — Denoising Diffusion
            # =========================================================
            print(f"\n[WearCast] Phase 3/4: Starting Denoising Diffusion (U-Net)...")

            # Save final mask diagnostic
            mask_diagnostic = (np.array(mask) * 255).astype(np.uint8)
            Image.fromarray(mask_diagnostic).save("debug_final_unet_mask.jpg")
            print(f" -> [SAVED] Final UNet mask saved to: debug_final_unet_mask.jpg")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated(0) / 1e9
                print(f" -> GPU VRAM before UNet call: {mem_before:.2f} GB")

            t_unet_start = time.time()
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                image_garm=image_garm,
                image_vton=image_vton,
                mask=mask,
                image_ori=image_ori,
                num_inference_steps=final_steps,
                image_guidance_scale=final_scale,
                guidance_scale=9.0,   # boosted from 7.5 — stronger CLIP/garment feature adherence
                num_images_per_prompt=num_samples,
                generator=generator,
            ).images
            t_unet_end = time.time()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated(0) / 1e9
                print(f" -> GPU VRAM after  UNet call: {mem_after:.2f} GB")
            print(f" -> Denoising loop elapsed    : {t_unet_end - t_unet_start:.2f}s")
            print(f" -> Pipeline returned {len(images)} image(s)")

            # =========================================================
            # PHASE 4 — Post-Processing
            # =========================================================
            print(f"\n[WearCast] Phase 4/4: Final Post-processing...")

            raw_generated = images[0]
            raw_generated.save("debug_raw_unet_output.jpg")
            print(f" -> [SAVED] Raw UNet output saved to: debug_raw_unet_output.jpg")
            print(f" -> UNet Generated Target Size: {raw_generated.size}")

            # --- Step 1: Color Correction ---
            print(" -> Step 1/2: Local color correction...")
            t0 = time.time()
            color_fixed = self.local_color_correction(
                generated=raw_generated,
                original_garment=image_garm,
                mask_hard=self._cached_hard_mask if hasattr(self, '_cached_hard_mask') else mask
            )
            print(f" -> [SUCCESS] Color correction completed in {time.time()-t0:.2f}s")
            color_fixed.save("debug_color_corrected.jpg")
            print(f" -> [SAVED] Color-corrected image saved to: debug_color_corrected.jpg")

            # --- Step 2: Strict Background-Clean Alpha Composite ---
            # === FIX B: REPLACES Laplacian blend — eliminates white halo ===
            # Root cause of halo: Laplacian pyramid at 5 levels, coarsest level (64x48) covers
            # 16×16 original pixels per cell, spreading the blend zone 16–32px outside the mask.
            # Solution: (A) strictly copy original background into generated image BEFORE blend;
            #           (B) apply only a narrow 9px soft alpha at the exact boundary.
            print(" -> Step 2/2: Strict background-clean alpha compositing...")

            # Resolve the hard (binary) mask at full output resolution
            if hasattr(self, '_cached_hard_mask'):
                mask_np_hard = np.array(self._cached_hard_mask.resize(raw_generated.size, Image.NEAREST))
            else:
                mask_np_hard = (np.array(mask.resize(raw_generated.size, Image.NEAREST)) > 127).astype(np.uint8) * 255
            print(f" -> Hard mask: shape={mask_np_hard.shape}  white_px={(mask_np_hard>127).sum()}  coverage={100*(mask_np_hard>127).mean():.1f}%")

            t0 = time.time()

            gen_arr = np.array(color_fixed).astype(np.float32)
            ori_arr = np.array(image_ori).astype(np.float32)
            if ori_arr.shape[:2] != gen_arr.shape[:2]:
                ori_arr = np.array(image_ori.resize(color_fixed.size, Image.BICUBIC)).astype(np.float32)
                print(f"   [COMPOSITE] Resized original to {ori_arr.shape[:2]}")

            # A: Strict background replacement
            #    Every pixel OUTSIDE the hard mask is copied directly from the original.
            #    This completely eliminates any white-shirt bleed into the background.
            hard_f   = (mask_np_hard > 127).astype(np.float32)   # binary 0/1 float
            hard_3d  = np.stack([hard_f] * 3, axis=-1)
            gen_clean = gen_arr * hard_3d + ori_arr * (1.0 - hard_3d)
            print(f"   [COMPOSITE] A: Background replaced outside mask  coverage={hard_f.mean()*100:.1f}%")

            # B: Soft-feather alpha blend at the boundary only (13px gaussian on binary mask = ~13px blend zone)
            alpha    = cv2.GaussianBlur(mask_np_hard.astype(np.float32), (13, 13), 3.5) / 255.0  # was (9,9),2.5
            alpha_3d = np.stack([alpha] * 3, axis=-1)
            final_arr = gen_clean * alpha_3d + ori_arr * (1.0 - alpha_3d)
            print(f"   [COMPOSITE] B: Alpha boundary blend done  max={alpha.max():.3f}  mean={alpha.mean():.4f}")

            # === USM Sharpening: applied only inside hard mask region ===
            # Sharpens graphic edges (text, patterns) without affecting background
            # Formula: sharpened = (1+a)*orig - a*blurred  where a=0.5
            blurred_arr  = cv2.GaussianBlur(final_arr.astype(np.float32), (3, 3), 0.8)
            sharp_arr    = np.clip(1.3 * final_arr - 0.3 * blurred_arr, 0, 255)  # amount=0.3 (was 0.5)
            # Blend: inside mask -> sharpened; outside mask -> original (gradual via hard_3d)
            final_arr = sharp_arr * hard_3d + final_arr * (1.0 - hard_3d)
            print(f"   [COMPOSITE] USM sharpening applied inside mask (amount=0.3, k=3×3)")

            final_image = Image.fromarray(np.clip(final_arr, 0, 255).astype(np.uint8))
            print(f"   [COMPOSITE] Completed in {time.time()-t0:.2f}s")
            final_image.save("debug_final_output.jpg")
            print(f" -> [SAVED] Final result saved to: debug_final_output.jpg")

            print("\n[WearCast] SUCCESS: Inference completed successfully!")
            print("=" * 70)

        return [final_image]

    def detect_garment_complexity(self, image_garm):
        """
        Detect if garment is complex (pattern, logo, ruffle) using variance and edges.
        """
        garm_np = np.array(image_garm)

        # Remove white background first
        bg_mask = np.all(garm_np >= 240, axis=-1)
        fg_pixels = garm_np[~bg_mask]

        if len(fg_pixels) < 500:
            print(f"   [COMPLEXITY] Only {len(fg_pixels)} foreground pixels — defaulting to simple.")
            return False

        color_std = np.std(fg_pixels, axis=0).mean()
        is_patterned = color_std > 38.0

        gray = cv2.cvtColor(garm_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        is_complex_shape = edge_density > 0.06

        print(f"   [COMPLEXITY] FG pixel count={len(fg_pixels)} | color_std={color_std:.2f} (>38=patterned:{is_patterned}) | edge_density={edge_density:.4f} (>0.06=complex:{is_complex_shape})")
        return is_patterned or is_complex_shape

    def get_optimal_params(self, category, is_complex_garment):
        if is_complex_garment:
            # Graphic/patterned: needs more steps AND higher image conditioning for print fidelity
            return {"num_steps": 40, "image_scale": 3.0}  # was 2.5 — stronger garment adherence
        else:
            # Simple solid garment: slightly fewer steps, strong guidance is fine
            return {"num_steps": 30, "image_scale": 3.5}  # was 3.0

    def local_color_correction(self, generated, original_garment, mask_hard):
        """
        Match overall S/V stats of the generated garment to the original garment,
        restricted precisely to the mask boundary, and gently capped.
        """
        print(f"   [COLOR] Starting correction. Gen shape: {generated.size}, Garm shape: {original_garment.size}")
        gen_np  = np.array(generated).astype(np.float32)
        garm_np = np.array(original_garment.resize(generated.size)).astype(np.float32)
        msk_np  = np.array(mask_hard.resize(generated.size, Image.NEAREST)).astype(np.float32) / 255.0

        result = gen_np.copy()

        gen_hsv  = cv2.cvtColor(gen_np.astype(np.uint8),  cv2.COLOR_RGB2HSV).astype(np.float32)
        garm_hsv = cv2.cvtColor(garm_np.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

        mask_bool = msk_np > 0.5
        print(f"   [COLOR] Mask interior area: {mask_bool.sum()} pixels  ({100*mask_bool.mean():.1f}% of image)")

        if mask_bool.sum() < 100:
            print("   [COLOR] Skipped: Mask interior area too small (<100px).")
            return generated

        # Reference clothing background check
        bg_garm  = np.all(garm_np >= 238, axis=-1)
        valid_garm = ~bg_garm
        print(f"   [COLOR] Valid garment pixels (non-BG): {valid_garm.sum()}  ({100*valid_garm.mean():.1f}%)")

        if valid_garm.sum() < 100:
            valid_garm = np.ones_like(bg_garm)
            print("   [COLOR] Warning: Product image had almost no valid foreground pixels. Using entire image stats.")

        for channel, name in zip([1, 2], ["Saturation", "Value / Brightness"]):
            gen_mean  = gen_hsv[mask_bool, channel].mean()
            garm_mean = garm_hsv[valid_garm, channel].mean()

            gen_std  = gen_hsv[mask_bool,  channel].std() + 1e-6
            garm_std = garm_hsv[valid_garm, channel].std() + 1e-6

            if channel == 2:  # V — boost only, but cap aggressively to prevent glowing white shirt
                ratio_std  = max(min(garm_std / gen_std, 1.10), 1.0)   # was 1.3 max — narrowed
                # Cap shift: product-image brightness gap is expected (studio vs body shadows)
                # 0.35 * gap capped at 22 prevents unnatural shirt glow while still correcting hue
                shift_mean = min((garm_mean - gen_mean) * 0.35, 22.0)  # was * 0.85 uncapped
            else:             # S — boost graphic colors without tinting the white shirt body
                ratio_std  = min(garm_std / gen_std, 1.25)  # slightly reduced from 1.30
                shift_mean = (garm_mean - gen_mean) * 0.55  # was 0.75 — less aggressive

            print(f"   [COLOR] {name:<18}: Ref Mean={garm_mean:.1f}, Gen Mean={gen_mean:.1f} | Ref Std={garm_std:.1f}, Gen Std={gen_std:.1f} | Ratio={ratio_std:.2f}, Shift={shift_mean:.1f}")

            corrected_ch = gen_hsv[:, :, channel].copy()
            if channel == 1:  # S — split: colored pixels get full boost, white shirt body gets minimal push
                # White shirt body = bright (V>195) AND nearly unsaturated (S<=20)
                # Applying full shift there adds a color cast that makes the shirt look transparent/bluish
                is_white_body = (gen_hsv[:, :, 2] > 195) & (gen_hsv[:, :, 1] <= 20)
                colored_px    = mask_bool & ~is_white_body
                white_px      = mask_bool &  is_white_body
                corrected_ch[colored_px] = corrected_ch[colored_px] * ratio_std + shift_mean
                # White areas: at most +1.5 saturation so they stay clean white
                corrected_ch[white_px]   = corrected_ch[white_px] + min(max(shift_mean * 0.08, 0), 1.5)
                print(f"   [COLOR]   S split: colored_px={colored_px.sum()}  white_body_px={white_px.sum()}")
            else:  # V — apply uniformly to all mask pixels
                corrected_ch[mask_bool] = corrected_ch[mask_bool] * ratio_std + shift_mean
            gen_hsv[:, :, channel] = np.clip(corrected_ch, 0, 255)

        corrected_rgb = cv2.cvtColor(gen_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        blend_mask    = cv2.GaussianBlur(msk_np, (61, 61), 15)
        blend_mask_3d = np.stack([blend_mask]*3, axis=-1)

        result_float = corrected_rgb * blend_mask_3d + gen_np * (1.0 - blend_mask_3d)
        print("   [COLOR] Execution sequence entirely completed.")
        return Image.fromarray(np.clip(result_float, 0, 255).astype(np.uint8))

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

    def extend_arm_mask(self, wrist, elbow, scale):
        wrist = elbow + scale * (wrist - elbow)
        return wrist

    def hole_fill(self, img):
        img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
        img_copy = img.copy()
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(img, mask, (0, 0), 255)
        img_inverse = cv2.bitwise_not(img)
        dst = cv2.bitwise_or(img_copy, img_inverse)
        return dst

    def refine_mask(self, mask):
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        area = []
        for j in range(len(contours)):
            a_d = cv2.contourArea(contours[j], True)
            area.append(abs(a_d))
        refined = np.zeros_like(mask).astype(np.uint8)
        if len(area) != 0:
            i = area.index(max(area))
            cv2.drawContours(refined, contours, i, color=255, thickness=-1)
        print(f"   [REFINE_MASK] contours found={len(area)}  largest_area={max(area) if area else 0:.0f} px²")
        return refined

    def get_mask_location(self, model_type, category, model_parse: Image.Image, keypoint: dict, width=384, height=512):
        print(f"\n   [MASK_GEN] get_mask_location: model_type={model_type}  category={category}  target_size=({width},{height})")
        im_parse = model_parse.resize((width, height), Image.NEAREST)
        parse_array = np.array(im_parse)
        print(f"   [MASK_GEN] parse_array shape={parse_array.shape}  unique_vals={np.unique(parse_array).tolist()}")

        print(" -> Constructing Smart Category Mask (Off-Shoulder/Crop-Top Aware)...")

        pose_data = np.array(keypoint["pose_keypoints_2d"]).reshape((-1, 2))
        scale = height / 512.0
        pt = lambda idx: np.multiply(pose_data[idx][:2], scale)

        # ALL keypoints needed
        nose  = pt(0)
        neck  = pt(1)
        s_r, s_l = pt(2), pt(5)
        e_r, e_l = pt(3), pt(6)
        w_r, w_l = pt(4), pt(7)
        hip_r = pt(8)
        hip_l = pt(11)

        print(f"   [MASK_GEN] Key points (scaled by {scale:.3f}):")
        print(f"              Nose={nose.tolist()}  Neck={neck.tolist()}")
        print(f"              Shoulder R={s_r.tolist()}  L={s_l.tolist()}")
        print(f"              Elbow    R={e_r.tolist()}  L={e_l.tolist()}")
        print(f"              Wrist    R={w_r.tolist()}  L={w_l.tolist()}")
        print(f"              Hip      R={hip_r.tolist()}  L={hip_l.tolist()}")

        # ---- STEP 1: Base semantic mask per category ----
        if category == 'upperbody':
            target_labels = [4, 7]
            arm_labels    = [14, 15]
        elif category == 'lowerbody':
            target_labels = [5, 6, 12, 13]
            arm_labels    = []
        elif category == 'dress':
            target_labels = [4, 5, 6, 7]
            arm_labels    = [14, 15]

        garment_mask = np.zeros((height, width), dtype=np.float32)
        for label in target_labels:
            px_count = (parse_array == label).sum()
            print(f"   [MASK_GEN] Label {label}: {px_count} pixels")
            garment_mask += (parse_array == label).astype(np.float32)

        # === Widen garment dilation to 7x7 to recover mask area lost after arm-clip Fix A ===
        k_size = max(int((height * 0.015)) | 1, 7)   # was: int(h*0.01)|1 → 5px; now min 7px
        print(f"   [MASK_GEN] Garment dilation kernel size: {k_size}x{k_size}")
        kernel_garment = np.ones((k_size, k_size), np.uint8)
        garment_mask = cv2.dilate(garment_mask, kernel_garment, iterations=1)

        arm_mask = np.zeros((height, width), dtype=np.float32)
        for label in arm_labels:
            px_count = (parse_array == label).sum()
            print(f"   [MASK_GEN] Arm Label {label}: {px_count} pixels")
            arm_mask += (parse_array == label).astype(np.float32)

        # === FIX A: Clip arm mask to sleeve region only (prevents forearm ghosting/artifacts) ===
        # Strategy: only keep arm pixels from the shoulder down to 75% of the shoulder→elbow distance.
        # For a short-sleeve T-shirt this safely covers the sleeve but excludes the forearm.
        # Long-sleeve shirts have no exposed arm skin (label 14/15 is covered by label 4) so this is safe.
        if len(arm_labels) > 0 and category == 'upperbody':
            min_sh_y     = min(s_r[1], s_l[1])            # highest shoulder point
            avg_elbow_y  = (e_r[1] + e_l[1]) / 2.0
            sleeve_hem_y = int(min_sh_y + 0.75 * (avg_elbow_y - min_sh_y))
            sleeve_hem_y = max(0, min(height - 1, sleeve_hem_y))
            arm_mask[sleeve_hem_y:, :] = 0
            print(f"   [MASK_GEN] FIX A — Short-sleeve clip: arm mask zeroed at Y>={sleeve_hem_y} "
                  f"(shoulder_y={min_sh_y:.0f}, elbow_y={avg_elbow_y:.0f})")
        else:
            print(f"   [MASK_GEN] No sleeve-hem clip applied (arm_labels={arm_labels}, cat={category}).")

        base_mask = np.clip(garment_mask + arm_mask, 0, 1)
        print(f"   [MASK_GEN] base_mask coverage: {100*(base_mask > 0).mean():.1f}%")

        # ---- STEP 2: Detect garment sub-type ----
        upper_clothes_rows = np.where((parse_array == 4).any(axis=1))[0]
        if len(upper_clothes_rows) > 0:
            garment_top_y = upper_clothes_rows.min()
            shoulder_y    = min(s_r[1], s_l[1])
            is_off_shoulder = garment_top_y > (shoulder_y + int(20 / 512 * height))
            print(f"   [MASK GEOM] Garment Top Y={garment_top_y}, Shoulder Y={shoulder_y:.1f} -> Off-Shoulder={is_off_shoulder}")
        else:
            is_off_shoulder = False
            print("   [MASK GEOM] No upper clothes detected in parse mask.")

        garment_pixels_y = np.where(base_mask > 0)[0]
        is_crop_top = False
        if len(garment_pixels_y) > 0:
            garment_bottom_y = np.max(garment_pixels_y)
            hip_y = (hip_r[1] + hip_l[1]) / 2
            if hip_y > 0:
                is_crop_top = garment_bottom_y < (hip_y * 0.85)
                print(f"   [MASK GEOM] Garment Bottom Y={garment_bottom_y}, Hip Y={hip_y:.1f} -> Crop-Top={is_crop_top}")

        # ---- STEP 3: Build hull points ----
        # === ACCURACY FIX 5: Wider mask coverage around shoulder/sleeve boundary ===
        ARM_PAD    = int(20 / 512 * height)  # was 12px — prevents transparency at sleeve edges
        SLEEVE_PAD = int(18 / 512 * height)  # was 12px
        hull_pts = []

        if is_off_shoulder:
            hull_pts += [
                [s_r[0] + ARM_PAD, s_r[1] + int(20/512*height)],
                [s_l[0] - ARM_PAD, s_l[1] + int(20/512*height)],
                [e_r[0] + SLEEVE_PAD, e_r[1]],
                [e_l[0] - SLEEVE_PAD, e_l[1]],
            ]
            protect_labels = [1, 2, 11, 18]
        else:
            hull_pts += [
                [s_r[0] + ARM_PAD, s_r[1]],
                [s_l[0] - ARM_PAD, s_l[1]],
                [neck[0], neck[1] + int(5/512*height)],
            ]
            # === FIX: Add torso-width (hip-level) hull points to create full trapezoid coverage ===
            # Without these, the hull is only a tiny shoulder-neck triangle (3pts) → 10.9% coverage.
            # Adding hip-width anchors at waist level fills the gap from garment_bottom_y→waist_cutoff
            # and eliminates the semi-transparent shirt body effect.
            hip_y_val = (hip_r[1] + hip_l[1]) / 2
            if hip_y_val > 0:
                # Shirt hem at 91% of hip_y — places hem just above the belt line (was 97% = at belt)
                hull_anchor_y = int(hip_y_val * 0.91)
                # Use actual hip x-positions for the hem (narrower than shoulders — natural shirt taper)
                # hip_r[0]≈163, hip_l[0]≈241  vs  s_r[0]+PAD≈156, s_l[0]-PAD≈240 at shoulder
                hem_r_x = int(hip_r[0] + ARM_PAD * 0.5)  # right hem (hip x + small pad)
                hem_l_x = int(hip_l[0] - ARM_PAD * 0.5)  # left  hem (hip x - small pad)
                hull_pts += [
                    [hem_r_x, hull_anchor_y],  # right hem
                    [hem_l_x, hull_anchor_y],  # left  hem
                ]
                print(f"   [MASK_GEN] Added hip hull pts at Y={hull_anchor_y} "
                      f"(R_x={hem_r_x}, L_x={hem_l_x}) "
                      f"[taper: shoulder_w={int(s_l[0]-ARM_PAD)-int(s_r[0]+ARM_PAD)}px "
                      f"→ hem_w={hem_l_x-hem_r_x}px]")
            has_sleeves = (parse_array == 14).sum() + (parse_array == 15).sum() > 200
            print(f"   [MASK_GEN] has_sleeves={has_sleeves}  (arm px sum={(parse_array==14).sum()+(parse_array==15).sum()})")
            if has_sleeves:
                pass  # Do NOT add elbow/wrist to hull (prevents ghost cape)
            protect_labels = [1, 2, 11]

        print(f"   [MASK_GEN] hull_pts (raw): {hull_pts}")

        # ---- STEP 4: Crop top lower boundary ----
        if is_crop_top:
            garment_bottom_y = np.max(garment_pixels_y)
            cutoff = int(garment_bottom_y * 1.05)
            base_mask[cutoff:, :] = 0
            print(f"   [MASK_GEN] Crop-top cutoff applied at Y={cutoff}")

        hip_y = (hip_r[1] + hip_l[1]) / 2
        if category == 'upperbody' and hip_y > 0:
            waist_cutoff = int(hip_y + int(25 / 512 * height))
            base_mask[waist_cutoff:, :] = 0
            print(f"   [MASK GEOM] Upperbody waist cutoff applied at Y={waist_cutoff}")

        # ---- STEP 5: Convex hull from core torso ----
        core_mask = np.zeros((height, width), dtype=np.float32)
        if category == 'upperbody':
            for label in [4, 7]:
                core_mask += (parse_array == label).astype(np.float32)
        else:
            core_mask = base_mask.copy()

        if is_crop_top:
            core_mask[int(garment_bottom_y * 1.05):, :] = 0
        if category == 'upperbody' and hip_y > 0:
            core_mask[waist_cutoff:, :] = 0

        core_pixels = np.column_stack(np.where(core_mask > 0))
        valid = lambda p: p[0] > 1 and p[1] > 1
        valid_hull_pts = [p for p in hull_pts if valid(p)]
        print(f"   [MASK_GEN] core_pixels={len(core_pixels)}  valid_hull_pts={len(valid_hull_pts)}")

        if len(core_pixels) > 5 and len(valid_hull_pts) >= 3:
            core_xy = core_pixels[:, [1, 0]]
            if len(core_xy) > 800:
                idx = np.random.choice(len(core_xy), 800, replace=False)
                core_xy = core_xy[idx]
                print(f"   [MASK_GEN] core_xy downsampled to 800 random pts")

            all_pts = np.vstack([core_xy, np.array(valid_hull_pts)]).astype(np.float32)
            hull = cv2.convexHull(all_pts)
            hull_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull.astype(np.int32), 255)
            hull_coverage = 100*(hull_mask > 0).mean()
            print(f"   [MASK_GEN] Convex hull built. Hull coverage: {hull_coverage:.1f}%")
            inpaint_mask = np.logical_or(base_mask, hull_mask / 255.0).astype(np.float32)
        else:
            print(f"   [MASK_GEN] Skipped convex hull (not enough points). Using raw base_mask.")
            inpaint_mask = base_mask.copy()

        # ---- STEP 6: Protect face/hair/skin ----
        protect_mask = np.zeros((height, width), dtype=np.float32)
        for label in protect_labels:
            px = (parse_array == label).sum()
            print(f"   [MASK_GEN] Protect label {label}: {px} pixels")
            protect_mask += (parse_array == label).astype(np.float32)
        protect_mask = np.clip(protect_mask, 0, 1)
        inpaint_mask = np.logical_and(inpaint_mask, np.logical_not(protect_mask)).astype(np.float32)
        print(f"   [MASK_GEN] After protection: inpaint_mask coverage={100*(inpaint_mask>0).mean():.1f}%")

        # === Widened dilation 5→9px: filling gaps at shirt boundary (main cause of semi-transparent shirt) ===
        kernel_close  = np.ones((9, 9), np.uint8)
        kernel_dilate = np.ones((9, 9), np.uint8)   # was 5x5
        inpaint_mask_u8 = (inpaint_mask * 255).astype(np.uint8)
        inpaint_mask_u8 = cv2.morphologyEx(inpaint_mask_u8, cv2.MORPH_CLOSE, kernel_close)
        inpaint_mask_u8 = cv2.dilate(inpaint_mask_u8, kernel_dilate, iterations=1)

        filled = self.hole_fill(inpaint_mask_u8)
        dst    = self.refine_mask(filled)
        print(f"   [MASK_GEN] After morphology+fill: dst coverage={100*(dst>0).mean():.1f}%")

        # ---- STEP 8: Soft feathered edge ----
        # === ACCURACY FIX 6: Tighter feather (33px→21px) → sharper mask edges → less boundary bleed ===
        mask_soft = cv2.GaussianBlur(dst.astype(np.float32), (21, 21), 7)
        inpaint_mask_soft = np.clip(mask_soft / 255.0, 0, 1)

        percentage = 100 * np.sum(dst > 0) / (width * height)
        print(f" -> Smart Category Mask: {percentage:.1f}% | OffShoulder={is_off_shoulder} | CropTop={is_crop_top}")

        self._cached_hard_mask = Image.fromarray(dst)
        return Image.fromarray(dst), Image.fromarray((inpaint_mask_soft * 255).astype(np.uint8))
