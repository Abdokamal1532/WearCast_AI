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
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute() # WearCast_AI root
VIT_PATH = os.path.join(PROJECT_ROOT, "checkpoints/clip-vit-large-patch14")
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints/ootd")

class WearCastHD:

    def __init__(self, gpu_id):
        self.gpu_id = 'cuda:' + str(gpu_id)

        print(f"Loading components from {MODEL_PATH}...")
        
        # 1. Load VAE
        vae = AutoencoderKL.from_pretrained(
            MODEL_PATH,
            subfolder="vae",
            torch_dtype=torch.float16,
        )

        # 2. Load UNets (using Safetensors)
        unet_garm = UNetGarm2DConditionModel.from_pretrained(
            MODEL_PATH,
            subfolder="unet_garm",
            torch_dtype=torch.float16,
            use_safetensors=True, 
        )
        unet_vton = UNetVton2DConditionModel.from_pretrained(
            MODEL_PATH,
            subfolder="unet_vton",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        # 3. Load Text Encoder and Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
        ).to(self.gpu_id)
        
        # 4. Load CLIP Vision & Scheduler
        self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
        from transformers import CLIPVisionModel, CLIPVisionModelWithProjection
        
        # Load base vision model for hidden states (Phase 2 fix)
        self.image_encoder = CLIPVisionModel.from_pretrained(VIT_PATH).to(self.gpu_id).half()
        
        # Load the pretrained projection matrix safely
        clip_full = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH)
        self.visual_projection = clip_full.visual_projection.to(self.gpu_id).half()
        del clip_full
        
        # Load scheduler explicitly to avoid from_config auto-load lists
        from diffusers import DDPMScheduler
        scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

        # 5. Initialize Pipeline DIRECTLY (Bypass broken loaders)
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

        # Ensure we use DPM-Solver++ for highest quality inference in fewer steps (Phase 3)
        from diffusers import DPMSolverMultistepScheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True,
            final_sigmas_type="sigma_min",
        )

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
        if seed == -1:
            random.seed(time.time())
            seed = random.randint(0, 2147483647)
        print('Initial seed: ' + str(seed))
        generator = torch.manual_seed(seed)

        # Automated Mask Generation (If not provided)
        if mask is None:
            print(f"[WearCast] Phase 1/4: Starting AI Preprocessing...")
            from preprocess.humanparsing.run_parsing import Parsing
            from preprocess.openpose.run_openpose import OpenPose
            
            # Initialize preprocessors if they don't exist
            if not hasattr(self, 'parsing_model'):
                print(" -> Loading Human-Parsing model checkpoints...")
                self.parsing_model = Parsing(self.gpu_id)
                print(" -> Loading OpenPose model checkpoints...")
                self.openpose_model = OpenPose(self.gpu_id)
            
            # 1. Run Preprocessors
            print(" -> Running OpenPose inference (Human Keypoints)...")
            keypoints = self.openpose_model(image_vton)
            print(" -> Running Parsing inference (Semantic Segmentation)...")
            model_parse, _ = self.parsing_model(image_vton)
            
            # 2. Sophisticated Mask Generation
            print(" -> Constructing sophisticated in-painting mask...")
            _, mask = self.get_mask_location(model_type, category, model_parse, keypoints)
            
            # Diagnostic: Check mask density
            mask_np = np.array(mask)
            mask_pixels = np.sum(mask_np > 127)
            total_pixels = mask_np.size
            print(f" -> Mask Diagnostic: {mask_pixels} pixels marked for replacement ({100*mask_pixels/total_pixels:.2f}% of image)")
            
            mask = mask.resize((768, 1024), Image.NEAREST)
            print(" -> Preprocessing Stage: SUCCESS")

        # Phase 3: Adaptive diffusion parameters setup
        is_complex = self.detect_garment_complexity(image_garm)
        opt_params = self.get_optimal_params(category, is_complex)
        
        # Merge explicitly passed params if available, otherwise fallback to optimal
        final_steps = opt_params["num_steps"]
        final_scale = opt_params["image_scale"]
        print(f" -> Auto-params: steps={final_steps}, scale={final_scale}, complex={is_complex}")

        print(f"[WearCast] Phase 2/4: Encoding Inputs (VAE & Multi-Scale CLIP Vision)...")
        with torch.no_grad():
            from PIL import ImageEnhance
            garm_np = np.array(image_garm.copy())
            bg_mask = np.all(garm_np >= 240, axis=-1)
            
            if bg_mask.mean() > 0.05:
                print(f" -> White product background detected ({100*bg_mask.mean():.1f}% coverage), replacing with mid-gray for CLIP...")
                garm_np_proc = garm_np.copy()
                garm_np_proc[bg_mask] = [160, 160, 160]
                garm_proc = Image.fromarray(garm_np_proc)
            else:
                garm_proc = image_garm.copy()
                
            garm_enhanced = ImageEnhance.Sharpness(garm_proc).enhance(1.8)
            garm_enhanced = ImageEnhance.Contrast(garm_enhanced).enhance(1.3)

            # ---- MULTI-SCALE CROPS ----
            w, h = garm_enhanced.size
            crop_full = garm_enhanced.copy()
            cx, cy = w // 2, h // 2
            crop_center = garm_enhanced.crop([
                int(cx - w * 0.30), int(cy - h * 0.30),
                int(cx + w * 0.30), int(cy + h * 0.30)
            ])
            crop_upper = garm_enhanced.crop([0, 0, w, int(h * 0.40)])

            # ---- ENCODE CONTEXT SCALES & PROJECT ----
            def encode_crop(img):
                inputs = self.auto_processor(images=img, return_tensors="pt").to(self.gpu_id)
                pixel_vals = inputs.data['pixel_values'].to(dtype=torch.float16)
                outputs = self.image_encoder(pixel_values=pixel_vals, output_hidden_states=True)
                hidden = outputs.last_hidden_state  # [1, 257, 1024]
                
                # Use pretrained visual projection
                patch_tokens = hidden[:, 1:, :] # [1, 256, 1024]
                projected_patches = self.visual_projection(patch_tokens) # [1, 256, 768]
                sampled = projected_patches[:, ::32, :] # [1, 8, 768]
                
                cls_proj = self.visual_projection(hidden[:, 0:1, :]) # [1, 1, 768]
                
                return torch.cat([cls_proj, sampled], dim=1) # [1, 9, 768]

            feat_full   = encode_crop(crop_full)
            feat_center = encode_crop(crop_center)
            feat_upper  = encode_crop(crop_upper)

            garment_features = (feat_full * 0.5) + (feat_center * 0.3) + (feat_upper * 0.2)
            
            # Text encoder is just 2 tokens, concatenate with the 9 garment tokens
            text_emb = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0].to(dtype=torch.float16)
            prompt_embeds = torch.cat([text_emb, garment_features], dim=1)
            
            print(f" -> Multi-Scale CLIP Embedding shape: {prompt_embeds.shape}")

            print(f"[WearCast] Phase 3/4: Starting Denoising Diffusion (U-Net)...")
            
            # Phase A Diagnostic: Save the final mask being sent to the AI
            mask_diagnostic = (np.array(mask) * 255).astype(np.uint8)
            Image.fromarray(mask_diagnostic).save("debug_final_unet_mask.jpg")

            images = self.pipe(prompt_embeds=prompt_embeds,
                        image_garm=image_garm,
                        image_vton=image_vton, 
                        mask=mask,
                        image_ori=image_ori,
                        num_inference_steps=final_steps,
                        image_guidance_scale=final_scale,
                        num_images_per_prompt=num_samples,
                        generator=generator,
            ).images
            
            print(f"[WearCast] Phase 4/4: Final Post-processing...")
            
            raw_generated = images[0]
            # Phase A Diagnostic: Save raw UNet output to see if ghosting is in the AI generation
            raw_generated.save("debug_raw_unet_output.jpg")
            print(f" -> UNet Generated Target Size: {raw_generated.size}")

            print(" -> Step 1/2: Local color correction...")
            # Use the HARD mask for checking color stats so we don't include UNET background grey
            color_fixed = self.local_color_correction(
                generated=raw_generated,
                original_garment=image_garm,
                mask_hard=self._cached_hard_mask if hasattr(self, '_cached_hard_mask') else mask
            )
            print(" -> [SUCCESS] Color correction completed.")

            print(" -> Step 2/2: Laplacian pyramid blending...")
            # Grab the hard mask or fall back to thresholded mask if passed externally
            if hasattr(self, '_cached_hard_mask'):
                mask_np_hard = np.array(self._cached_hard_mask.resize(raw_generated.size, Image.NEAREST))
            else:
                mask_np_hard = (np.array(mask.resize(raw_generated.size, Image.NEAREST)) > 127).astype(np.uint8) * 255
            
            print(f" -> Resolved hard mask shape: {mask_np_hard.shape}")
            
            # Step A: Tight blend mask (7px feather only, not 21px)
            mask_soft_blend = Image.fromarray(
                cv2.GaussianBlur(mask_np_hard.astype(np.float32), (7, 7), 2).astype(np.uint8)
            )

            # Step B: Pure Laplacian blend (levels=3 stays)
            # Since the convex hull webbing is destroyed, laplacian will no longer hallucinate
            # background ghosts. We remove the hard boolean clipping to allow the anatomical 
            # arms to blend perfectly without creating jagged step-function chunks of flesh!
            lap_result = self.laplacian_pyramid_blend(
                generated=color_fixed,
                original=image_ori,
                mask_soft=mask_soft_blend,
                levels=3
            )
            
            final_image = lap_result

            print("[WearCast] SUCCESS: Inference completed successfully!")

        return [final_image]

    def detect_garment_complexity(self, image_garm):
        """
        Detect if garment is complex (pattern, logo, ruffle) using variance and edges.
        """
        garm_np = np.array(image_garm)
        
        # Remove white background first (same threshold as CLIP preprocessing)
        bg_mask = np.all(garm_np >= 240, axis=-1)
        fg_pixels = garm_np[~bg_mask]
        
        if len(fg_pixels) < 500:
            return False  # Fallback: not enough garment pixels
        
        # Compute std ONLY on foreground garment pixels
        color_std = np.std(fg_pixels, axis=0).mean()
        is_patterned = color_std > 38.0  # Lower threshold since bg is excluded
        
        # Edge density on full image (shape complexity — bg doesn't hurt this)
        gray = cv2.cvtColor(garm_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        is_complex_shape = edge_density > 0.06  # Lower threshold too
        
        return is_patterned or is_complex_shape

    def get_optimal_params(self, category, is_complex_garment):
        if is_complex_garment:
            return {"num_steps": 25, "image_scale": 2.0}
        else:
            return {"num_steps": 20, "image_scale": 2.5}

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

        gen_hsv = cv2.cvtColor(gen_np.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        garm_hsv = cv2.cvtColor(garm_np.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

        mask_bool = msk_np > 0.5
        if mask_bool.sum() < 100:
            print("   [COLOR] Skipped: Mask interior area too small (<100px).")
            return generated

        # Reference clothing background check
        bg_garm = np.all(garm_np >= 238, axis=-1)
        valid_garm = ~bg_garm

        if valid_garm.sum() < 100:
            valid_garm = np.ones_like(bg_garm)
            print("   [COLOR] Warning: Product image had almost no valid foreground pixels. Using entire image stats.")

        for channel, name in zip([1, 2], ["Saturation", "Value / Brightness"]):
            gen_mean = gen_hsv[mask_bool, channel].mean()
            garm_mean = garm_hsv[valid_garm, channel].mean()
            
            gen_std = gen_hsv[mask_bool, channel].std() + 1e-6
            garm_std = garm_hsv[valid_garm, channel].std() + 1e-6
            
            if channel == 2:  # V (Brightness) — ONLY ever boost, never compress
                ratio_std = max(min(garm_std / gen_std, 1.3), 1.0)
                shift_mean = (garm_mean - gen_mean) * 0.85
            else:            # S (Saturation) — gentle bidirectional correction
                ratio_std = min(garm_std / gen_std, 1.2)
                shift_mean = (garm_mean - gen_mean) * 0.6
            
            print(f"   [COLOR] {name:<18}: Ref Mean={garm_mean:.1f}, Gen Mean={gen_mean:.1f} | Ratio={ratio_std:.2f}, Shift={shift_mean:.1f}")
            
            corrected_ch = gen_hsv[:, :, channel].copy()
            corrected_ch[mask_bool] = corrected_ch[mask_bool] * ratio_std + shift_mean
            gen_hsv[:, :, channel] = np.clip(corrected_ch, 0, 255)

        corrected_rgb = cv2.cvtColor(gen_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # Smooth transition at mask boundary
        blend_mask = cv2.GaussianBlur(msk_np, (61, 61), 15)
        blend_mask_3d = np.stack([blend_mask]*3, axis=-1)

        result_float = corrected_rgb * blend_mask_3d + gen_np * (1.0 - blend_mask_3d)
        print("   [COLOR] Execution sequence entirely completed.")
        return Image.fromarray(np.clip(result_float, 0, 255).astype(np.uint8))

    def laplacian_pyramid_blend(self, generated, original, mask_soft, levels=5):
        """
        Seamlessly blend using Gaussian/Laplacian pyramids.
        """
        gen_np  = np.array(generated).astype(np.float32)
        ori_np  = np.array(original).astype(np.float32)
        msk_np  = np.array(mask_soft).astype(np.float32) / 255.0
        
        if msk_np.ndim == 2:
            msk_np = np.stack([msk_np] * 3, axis=-1)

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

        result = blended_pyramid[-1]
        for i in range(levels - 2, -1, -1):
            result = cv2.pyrUp(result, dstsize=(blended_pyramid[i].shape[1], blended_pyramid[i].shape[0]))
            result = result + blended_pyramid[i]

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
        return refined

    def get_mask_location(self, model_type, category, model_parse: Image.Image, keypoint: dict, width=384, height=512):
        im_parse = model_parse.resize((width, height), Image.NEAREST)
        parse_array = np.array(im_parse)

        # 2. High-Accuracy Pose-Guided & Category-Aware Mask
        print(" -> Constructing Smart Category Mask (Off-Shoulder/Crop-Top Aware)...")
        
        pose_data = np.array(keypoint["pose_keypoints_2d"]).reshape((-1, 2))
        scale = height / 512.0
        pt = lambda idx: np.multiply(pose_data[idx][:2], scale)

        # ALL keypoints needed
        nose      = pt(0)
        neck      = pt(1)
        s_r, s_l  = pt(2), pt(5)   # Shoulders
        e_r, e_l  = pt(3), pt(6)   # Elbows
        w_r, w_l  = pt(4), pt(7)   # Wrists
        hip_r     = pt(8)
        hip_l     = pt(11)

        # ---- STEP 1: Base semantic mask per category ----
        if category == 'upperbody':
            target_labels = [4, 7]           # upper_clothes, dress
            arm_labels    = [14, 15]         # left_arm, right_arm
            
        elif category == 'lowerbody':
            target_labels = [5, 6, 12, 13]  # pants, skirt, left_leg, right_leg
            arm_labels    = []
            
        elif category == 'dress':
            target_labels = [4, 5, 6, 7]    # full body coverage
            arm_labels    = [14, 15]

        # Phase B: Dynamic dilation of the clothing mask to eat edges
        garment_mask = np.zeros((height, width), dtype=np.float32)
        for label in target_labels:
            garment_mask += (parse_array == label).astype(np.float32)
        
        # Calculate dynamic kernel size (approx 1% of height, must be odd)
        k_size = int((height * 0.01)) | 1  
        kernel_garment = np.ones((k_size, k_size), np.uint8)
        garment_mask = cv2.dilate(garment_mask, kernel_garment, iterations=1)

        # Add the arms normally (don't dilate arms or they eat the background/shoulders)
        arm_mask = np.zeros((height, width), dtype=np.float32)
        for label in arm_labels:
            arm_mask += (parse_array == label).astype(np.float32)

        base_mask = np.clip(garment_mask + arm_mask, 0, 1)

        # ---- STEP 2: Detect garment sub-type from parse geometry ----
        upper_clothes_rows = np.where((parse_array == 4).any(axis=1))[0]
        if len(upper_clothes_rows) > 0:
            garment_top_y = upper_clothes_rows.min()
            shoulder_y    = min(s_r[1], s_l[1])   # highest shoulder keypoint
            # Off-shoulder if garment starts MORE than 20px below the shoulder
            is_off_shoulder = garment_top_y > (shoulder_y + int(20 / 512 * height))
            print(f"   [MASK GEOM] Garment Top Y={garment_top_y}, Shoulder Y={shoulder_y:.1f} -> Off-Shoulder={is_off_shoulder}")
        else:
            is_off_shoulder = False
            print("   [MASK GEOM] No upper clothes detected in parse mask.")
        
        # Detect if it's a crop top (garment bottom above hip line)
        garment_pixels_y = np.where(base_mask > 0)[0]
        is_crop_top = False
        if len(garment_pixels_y) > 0:
            garment_bottom_y = np.max(garment_pixels_y)
            hip_y = (hip_r[1] + hip_l[1]) / 2
            if hip_y > 0: # Ensure pose detected hips
                is_crop_top = garment_bottom_y < (hip_y * 0.85)
                print(f"   [MASK GEOM] Garment Bottom Y={garment_bottom_y}, Hip Y={hip_y:.1f} -> Crop-Top={is_crop_top}")

        # ---- STEP 3: Build hull points based on garment type ----
        ARM_PAD = int(12 / 512 * height)
        SLEEVE_PAD = int(12 / 512 * height)
        hull_pts = []

        if is_off_shoulder:
            # For off-shoulder: hull goes BELOW shoulder keypoints
            hull_pts += [
                [s_r[0] + ARM_PAD, s_r[1] + int(20/512*height)],  # below right shoulder
                [s_l[0] - ARM_PAD, s_l[1] + int(20/512*height)],  # below left shoulder
                [e_r[0] + SLEEVE_PAD, e_r[1]],
                [e_l[0] - SLEEVE_PAD, e_l[1]],
            ]
            protect_labels = [1, 2, 11, 18]  # face, hair, neck, skin at shoulder
        else:
            # Standard: include full shoulder
            hull_pts += [
                [s_r[0] + ARM_PAD, s_r[1]],
                [s_l[0] - ARM_PAD, s_l[1]],
                [neck[0], neck[1] + int(5/512*height)],  # just below neck
            ]
            # Check sleeve length
            has_sleeves = (parse_array == 14).sum() + (parse_array == 15).sum() > 200
            if has_sleeves:
                # CRITICAL FIX: Do NOT add elbow and wrist keypoints to hull_pts!
                # base_mask already explicitly contains the semantic arm labels (14 and 15).
                # Adding them to convexHull draws a straight line from the elbow/wrist to the hip,
                # which encompasses the empty background air between the arm and the torso,
                # causing the UNet to generate a gray "ghost cape" hallucination.
                pass
            protect_labels = [1, 2, 11]  # face, hair, neck only

        # ---- STEP 4: Crop top lower boundary ----
        if is_crop_top:
            garment_bottom_y = np.max(garment_pixels_y)
            base_mask[int(garment_bottom_y * 1.05):, :] = 0

        # For upperbody: enforce hard waist cutoff even for long shirts
        # We do NOT want the UNet generating shirt content over the jeans
        hip_y = (hip_r[1] + hip_l[1]) / 2
        if category == 'upperbody' and hip_y > 0:
            waist_cutoff = int(hip_y + int(25 / 512 * height))  # hip + 25px margin
            base_mask[waist_cutoff:, :] = 0
            print(f"   [MASK GEOM] Upperbody waist cutoff applied at Y={waist_cutoff}")

        # ---- STEP 5: Build final convex hull purely from the core torso ----
        # DO NOT include arms (14, 15) in the hull, or it will draw polygons across the background air!
        core_mask = np.zeros((height, width), dtype=np.float32)
        if category == 'upperbody':
            for label in [4, 7]:  # upper_clothes, dress
                core_mask += (parse_array == label).astype(np.float32)
        else:
            core_mask = base_mask.copy()
            
        # Apply the exact same waist/crop-top cutoffs to the core_mask!
        if is_crop_top:
            core_mask[int(garment_bottom_y * 1.05):, :] = 0
        if category == 'upperbody' and hip_y > 0:
            core_mask[waist_cutoff:, :] = 0

        core_pixels = np.column_stack(np.where(core_mask > 0))
        
        valid = lambda p: p[0] > 1 and p[1] > 1
        valid_hull_pts = [p for p in hull_pts if valid(p)]

        if len(core_pixels) > 5 and len(valid_hull_pts) >= 3:
            core_xy = core_pixels[:, [1, 0]]  # (row,col) -> (x,y)
            if len(core_xy) > 800:
                idx = np.random.choice(len(core_xy), 800, replace=False)
                core_xy = core_xy[idx]
             
            all_pts = np.vstack([core_xy, np.array(valid_hull_pts)]).astype(np.float32)
            hull = cv2.convexHull(all_pts)
            hull_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull.astype(np.int32), 255)
            
            # Combine the smooth torso hull with the raw arms from the base_mask
            inpaint_mask = np.logical_or(base_mask, hull_mask / 255.0).astype(np.float32)
        else:
            inpaint_mask = base_mask.copy()

        # ---- STEP 6: Protect face/hair/skin ----
        protect_mask = np.zeros((height, width), dtype=np.float32)
        for label in protect_labels:
            protect_mask += (parse_array == label).astype(np.float32)
        protect_mask = np.clip(protect_mask, 0, 1)
        inpaint_mask = np.logical_and(inpaint_mask, np.logical_not(protect_mask)).astype(np.float32)

        # ---- STEP 7: Morphological cleanup ----
        kernel_close = np.ones((9, 9), np.uint8)
        kernel_dilate = np.ones((5, 5), np.uint8)
        inpaint_mask_u8 = (inpaint_mask * 255).astype(np.uint8)
        inpaint_mask_u8 = cv2.morphologyEx(inpaint_mask_u8, cv2.MORPH_CLOSE, kernel_close)
        inpaint_mask_u8 = cv2.dilate(inpaint_mask_u8, kernel_dilate, iterations=1)
        
        filled = self.hole_fill(inpaint_mask_u8)
        dst = self.refine_mask(filled)

        # ---- STEP 8: Soft feathered edge (32px instead of 8px) ----
        mask_soft = cv2.GaussianBlur(dst.astype(np.float32), (33, 33), 11)
        inpaint_mask_soft = np.clip(mask_soft / 255.0, 0, 1)

        percentage = 100 * np.sum(dst > 0) / (width * height)
        print(f" -> Smart Category Mask: {percentage:.1f}% | OffShoulder={is_off_shoulder} | CropTop={is_crop_top}")

        self._cached_hard_mask = Image.fromarray(dst)
        return Image.fromarray(dst), Image.fromarray((inpaint_mask_soft * 255).astype(np.uint8))
