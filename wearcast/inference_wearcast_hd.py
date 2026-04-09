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
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.gpu_id)
        
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

        # Ensure we use UniPC for inference as per original OOTD
        from diffusers import UniPCMultistepScheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

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

        print(f"[WearCast] Phase 2/4: Encoding Inputs (VAE & CLIP Vision)...")
        with torch.no_grad():
            # -------------------------------------------------------
            # GARMENT PREPROCESSING: Remove white product background
            # so CLIP can CLEARLY distinguish white garment fabric
            # from the white product-shot background.
            #
            # Without this: white-shirt-on-white-bg → CLIP encodes
            # ambiguous features → model generates GRAY instead of WHITE.
            # With this: pure white bg → replaced with mid-gray (160) →
            # CLIP sees white fabric as a distinct signal.
            # -------------------------------------------------------
            from PIL import ImageEnhance
            garm_np = np.array(image_garm.copy())
            # Detect near-white background: all channels >= 240
            bg_mask = np.all(garm_np >= 240, axis=-1)
            if bg_mask.mean() > 0.05:  # Only if > 5% of image is white bg
                print(f" -> White product background detected ({100*bg_mask.mean():.1f}% coverage), replacing with mid-gray for CLIP...")
                garm_np_proc = garm_np.copy()
                garm_np_proc[bg_mask] = [160, 160, 160]  # Neutral gray
                garm_proc = Image.fromarray(garm_np_proc)
            else:
                garm_proc = image_garm.copy()
            # Sharpness + Contrast boost for richer texture features
            garm_enhanced = ImageEnhance.Sharpness(garm_proc).enhance(1.8)
            garm_enhanced = ImageEnhance.Contrast(garm_enhanced).enhance(1.3)
            print(f" -> Garment CLIP features extracted with enhanced preprocessing.")

            # Explicitly cast to half (float16) for T4 GPU compatibility
            prompt_image = self.auto_processor(images=garm_enhanced, return_tensors="pt").to(device=self.gpu_id)
            prompt_image = self.image_encoder(prompt_image.data['pixel_values'].to(dtype=torch.float16)).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            
            prompt_embeds = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0].to(dtype=torch.float16)
            prompt_embeds[:, 1:] = prompt_image[:]
            print(f" -> CLIP Embedding shape: {prompt_embeds.shape}")

            print(f"[WearCast] Phase 3/4: Starting Denoising Diffusion (U-Net)...")
            images = self.pipe(prompt_embeds=prompt_embeds,
                        image_garm=image_garm,
                        image_vton=image_vton, 
                        mask=mask,
                        image_ori=image_ori,
                        num_inference_steps=num_steps,
                        image_guidance_scale=image_scale,
                        num_images_per_prompt=num_samples,
                        generator=generator,
            ).images
            print(f"[WearCast] Phase 4/4: Final Post-processing...")
            print("[WearCast] SUCCESS: Inference completed successfully!")

        return images

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

        # 2. High-Accuracy Pose-Guided Mask Generation
        print(" -> Constructing High-Accuracy Mask (Convex Hull + Pose-Guided)...")
        
        # Labels to TARGET for generation (core clothes)
        # 4: upper_clothes, 7: dress, 17: scarf, 14: left_arm, 15: right_arm (include arms for sleeves)
        target_area = (parse_array == 4).astype(np.float32) + \
                      (parse_array == 7).astype(np.float32) + \
                      (parse_array == 17).astype(np.float32)

        # Labels to PROTECT (Face, Hair only — NOT neck, NOT arms)
        head_only = (parse_array == 1).astype(np.float32) + \
                    (parse_array == 2).astype(np.float32) + \
                    (parse_array == 11).astype(np.float32)
        
        # Pose keypoints
        pose_data = np.array(keypoint["pose_keypoints_2d"]).reshape((-1, 2))
        pt = lambda idx: np.multiply(tuple(pose_data[idx][:2]), height / 512.0)
        # OpenPose: 2=RShoulder, 5=LShoulder, 3=RElbow, 6=LElbow, 4=RWrist, 7=LWrist
        s_r, s_l = pt(2), pt(5)   # Shoulders
        e_r, e_l = pt(3), pt(6)   # Elbows
        
        # ============================================================
        # ROOT-CAUSE FIX: CONVEX HULL GARMENT MASK
        # Drawing disconnected sleeve LINES creates "cold shoulder" holes.
        # Instead, we collect all boundary points (torso pixels + pose
        # keypoints) and wrap them in ONE solid convex polygon.
        # This guarantees anatomical continuity - no disconnected regions.
        # ============================================================
        
        # Seed points from semantic parsing
        torso_pixels = np.column_stack(np.where(target_area > 0))  # (row, col)
        
        # Keypoint boundary points (x=col, y=row) with lateral padding
        ARM_PAD = int(38 / 512 * height)
        valid = lambda p: p[0] > 1 and p[1] > 1
        
        hull_pts = []
        for p in [s_r, s_l, e_r, e_l]:
            if valid(p):
                # p is (x, y) where x=col, y=row
                hull_pts.append([p[0] + ARM_PAD, p[1]])
                hull_pts.append([p[0] - ARM_PAD, p[1]])
        
        # Build the garment mask
        inpaint_mask = target_area.copy()
        
        if len(torso_pixels) > 5 and len(hull_pts) >= 3:
            # torso_pixels: (row, col) -> convert to (col, row) = (x, y)
            torso_xy = torso_pixels[:, [1, 0]]
            
            # Subsample for speed
            if len(torso_xy) > 600:
                idx = np.random.choice(len(torso_xy), 600, replace=False)
                torso_xy = torso_xy[idx]
            
            all_pts = np.vstack([torso_xy, np.array(hull_pts)]).astype(np.float32)
            
            # Convex hull wraps everything into one solid region
            hull = cv2.convexHull(all_pts)
            hull_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull.astype(np.int32), 255)
            
            # Union: parsing labels + convex hull
            inpaint_mask = np.logical_or(inpaint_mask, hull_mask / 255.0).astype(np.float32)
        elif len(torso_pixels) > 5:
            # Fallback: just dilate the torso heavily if pose is missing
            inpaint_mask = cv2.dilate(inpaint_mask, np.ones((31, 31), np.uint8), iterations=2)

        # Add neck region - MINIMAL expansion (just 3px) to avoid choker artifact
        # We only want to barely expose the collar area, NOT create a visible neck band
        neck_area = (parse_array == 18).astype(np.float32)
        neck_expanded = cv2.dilate(neck_area, np.ones((3, 3), np.uint8), iterations=1)
        inpaint_mask = np.logical_or(inpaint_mask, neck_expanded).astype(np.float32)

        # FINAL GUARD: protect only face/hair — allow arms and neck
        inpaint_mask = np.logical_and(inpaint_mask, np.logical_not(head_only)).astype(np.float32)
        
        # Light dilation to smooth jagged edges from the convex hull
        inpaint_mask = cv2.dilate(inpaint_mask, np.ones((11, 11), np.uint8), iterations=1)

        # Hole fill + largest-contour refinement
        filled = self.hole_fill(np.where(inpaint_mask, 255, 0).astype(np.uint8))
        dst = self.refine_mask(filled)
        
        # Soft feathered mask for smooth final blending at skin edges
        mask_soft = cv2.GaussianBlur(dst.astype(np.float32), (21, 21), 7)
        inpaint_mask_soft = np.clip(mask_soft / 255.0, 0, 1)

        # Diagnostic
        percentage = 100 * np.sum(dst > 0) / (width * height)
        print(f" -> Convex Hull Mask: {percentage:.1f}% coverage. Solid connected region, no holes.")
        
        return Image.fromarray(dst), Image.fromarray((inpaint_mask_soft * 255).astype(np.uint8))
