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

        # Auto-detect garment complexity and choose optimal params
        is_complex = self.detect_garment_complexity(image_garm)
        auto_params = self.get_optimal_params(category, is_complex)
        final_steps = auto_params["num_steps"]    # use auto-detected optimal steps
        final_scale = auto_params["image_scale"]   # use auto-detected optimal guidance
        print(f"\n[WearCast] Auto-detect: complex={is_complex} | auto_params={auto_params}")
        print(f"[WearCast] Using params: steps={final_steps} (UI={num_steps}), guidance_scale={final_scale} (UI={image_scale})")

        # =========================================================
        # PHASE 2 — CLIP Vision Encoding
        # =========================================================
        print(f"\n[WearCast] Phase 2/4: Encoding Inputs (VAE & Multi-Scale CLIP Vision)...")
        with torch.no_grad():
            from PIL import ImageEnhance

            # --- 2a. Adaptive Background Detection ---
            garm_np = np.array(image_garm.copy())

            # Try multiple thresholds to find the product-photo background
            for bg_thresh in [250, 245, 240]:
                bg_mask = np.all(garm_np >= bg_thresh, axis=-1)
                bg_coverage = bg_mask.mean()
                print(f"   [BG-DETECT] Threshold>={bg_thresh}: {100*bg_coverage:.1f}% pixels")
                if bg_coverage > 0.05:
                    break

            print(f" -> Garment BG analysis: {100*bg_coverage:.1f}% BG pixels (selected threshold>={bg_thresh})")

            if bg_coverage > 0.05:
                print(f" -> White product background detected ({100*bg_coverage:.1f}%), replacing with neutral gray for CLIP...")
                garm_np_proc = garm_np.copy()
                garm_np_proc[bg_mask] = [170, 170, 170]
                garm_proc = Image.fromarray(garm_np_proc)
                garm_proc.save("debug_phase2_clip_bg_replaced.jpg")
                print(f" -> [SAVED] CLIP input (BG replaced) saved to: debug_phase2_clip_bg_replaced.jpg")
            else:
                print(f" -> No significant white BG detected. Using original garment for CLIP.")
                garm_proc = image_garm.copy()

            # Save the original garment for reference comparison
            image_garm.save("debug_phase2_garment_original.jpg")
            print(f" -> [SAVED] Original garment saved to: debug_phase2_garment_original.jpg")

            # --- 2b. CLIP input preparation ---
            # Use the processed garment directly for CLIP encoding.
            # Heavy sharpness/contrast enhancement was removed because it
            # over-emphasises edges and textures, causing stiff/puffy artifacts.
            garm_enhanced = garm_proc
            garm_enhanced.save("debug_phase2_clip_input.jpg")
            print(f" -> [SAVED] CLIP input garment saved to: debug_phase2_clip_input.jpg")

            # --- 2c. CLIP Encoding ---
            prompt_image = self.auto_processor(images=garm_enhanced, return_tensors="pt").to(device=self.gpu_id)
            prompt_image = self.image_encoder(prompt_image.data['pixel_values'].to(dtype=torch.float16)).image_embeds
            clip_norm = prompt_image.float().norm().item()
            clip_std  = prompt_image.float().std().item()
            prompt_image = prompt_image.unsqueeze(1)  # [1, 768] -> [1, 1, 768]
            print(f" -> CLIP image_embeds shape: {list(prompt_image.shape)} | norm={clip_norm:.2f} | std={clip_std:.4f}")

            prompt_embeds = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0].to(dtype=torch.float16)
            prompt_embeds[:, 1:] = prompt_image[:]
            print(f" -> Prompt embeddings shape: {list(prompt_embeds.shape)} (2 tokens: null_text + garment_CLIP)")
            _dbg_tensor("prompt_embeds", prompt_embeds)

            # --- 2d. VAE Garment Fidelity Check ---
            # Encode garment through VAE and decode back to check reconstruction quality
            print(f"\n   [VAE FIDELITY] Encoding garment through VAE and decoding back...")
            garm_tensor = self.pipe.image_processor.preprocess(image_garm).to(device=self.gpu_id, dtype=torch.float16)
            garm_latent = self.pipe.vae.encode(garm_tensor).latent_dist.mode()
            garm_roundtrip = self.pipe.vae.decode(garm_latent).sample  # mode() returns raw latents, decode directly
            # Undo: [-1,1] -> [0,255]
            garm_rt_np = ((garm_roundtrip[0].float().cpu().clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
            Image.fromarray(garm_rt_np).save("debug_phase2_vae_roundtrip.jpg")
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
            Image.fromarray(mask_diagnostic).save("debug_final_unet_mask.jpg")
            print(f" -> [SAVED] Final UNet mask saved to: debug_final_unet_mask.jpg")

            # Save masked person image (what the UNet receives as context)
            mask_np_vis = np.array(mask.resize(image_vton.size, Image.NEAREST))
            vton_np = np.array(image_vton)
            masked_person_vis = vton_np.copy()
            masked_person_vis[mask_np_vis > 127] = [0, 0, 0]  # black out masked region
            Image.fromarray(masked_person_vis).save("debug_phase3_masked_person.jpg")
            print(f" -> [SAVED] Masked person input saved to: debug_phase3_masked_person.jpg")

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
        raw_generated.save("debug_phase4_raw_unet_output.jpg")
        print(f" -> [SAVED] Raw UNet output saved to: debug_phase4_raw_unet_output.jpg")
        print(f" -> UNet Generated Target Size: {raw_generated.size}")

        gen_arr = np.array(raw_generated).astype(np.float32)
        ori_arr = np.array(image_ori).astype(np.float32)
        if ori_arr.shape[:2] != gen_arr.shape[:2]:
            ori_arr = np.array(image_ori.resize(raw_generated.size, Image.BICUBIC)).astype(np.float32)

        # --- Build soft compositing mask ---
        # Get the binary mask used during inpainting
        if hasattr(self, '_cached_hard_mask'):
            mask_pil = self._cached_hard_mask.resize(raw_generated.size, Image.BILINEAR)
        else:
            mask_pil = mask.resize(raw_generated.size, Image.BILINEAR)

        mask_np = np.array(mask_pil).astype(np.float32)
        if mask_np.max() > 1.0:
            mask_np = mask_np / 255.0

        # Binarize then apply thin Gaussian feather (2px sigma)
        # This is exactly what OOTDiffusion does — a thin, smooth transition
        binary_mask = (mask_np > 0.5).astype(np.float32)
        feather_sigma = 2.0
        alpha = cv2.GaussianBlur(binary_mask, (0, 0), feather_sigma)
        alpha = np.clip(alpha, 0.0, 1.0)

        # Save feather mask for debugging
        Image.fromarray((alpha * 255).astype(np.uint8)).save("debug_phase4_feather_mask.jpg")
        print(f" -> Feather mask: sigma={feather_sigma}px")

        # --- Pre-compositing diagnostics ---
        mask_bool = binary_mask > 0.5
        gen_in_mask = gen_arr[mask_bool]
        garm_arr = np.array(image_garm.resize(raw_generated.size)).astype(np.float32)
        garm_fg_mask = np.all(garm_arr >= 240, axis=-1)
        garm_fg_valid = ~garm_fg_mask
        gen_lum_mean = float((gen_in_mask[:, 0]*0.299 + gen_in_mask[:, 1]*0.587 + gen_in_mask[:, 2]*0.114).mean()) if len(gen_in_mask) > 0 else 128.0
        garm_lum_mean = float((garm_arr[garm_fg_valid, 0]*0.299 + garm_arr[garm_fg_valid, 1]*0.587 + garm_arr[garm_fg_valid, 2]*0.114).mean()) if garm_fg_valid.sum() > 0 else 128.0
        lum_gap = garm_lum_mean - gen_lum_mean
        print(f"   [DIAG] Raw UNet luminance: {gen_lum_mean:.1f} | Reference: {garm_lum_mean:.1f} | Gap: {lum_gap:.1f}")

        # --- Luminance recovery REMOVED ---
        # The UNet naturally adapts lighting for on-body realism.
        # Recovering luminance toward studio lighting created unnatural brightness mismatch.
        print(f"   [CORR] Luminance recovery: DISABLED (let UNet handle lighting naturally)")

        # Subtle saturation boost to counteract VAE color dampening
        sat_boost = 1.05
        gen_uint8 = np.clip(gen_arr, 0, 255).astype(np.uint8)
        gen_hsv = cv2.cvtColor(gen_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        gen_hsv[:, :, 1] = np.clip(gen_hsv[:, :, 1] * sat_boost, 0, 255)
        gen_arr = cv2.cvtColor(gen_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        print(f"   [CORR] Saturation boost: +{int((sat_boost-1)*100)}%")

        # --- Alpha composite (OOTDiffusion-style paste-back) ---
        t_composite = time.time()
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        final_arr = gen_arr * alpha_3ch + ori_arr * (1.0 - alpha_3ch)
        final_image = Image.fromarray(np.clip(final_arr, 0, 255).astype(np.uint8))
        print(f" -> Alpha compositing completed in {time.time()-t_composite:.3f}s")

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
        comparison.save("debug_phase4_comparison.jpg")
        print(f" -> [SAVED] 3-panel comparison: Garment | Raw UNet | Final")

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
            # Complex garments: use moderate guidance to preserve garment detail
            # while allowing natural fabric draping. OOTDiffusion docs recommend
            # image_guidance_scale 2.0-2.5; >2.5 causes color distortion and puffy artifacts.
            return {"num_steps": 25, "image_scale": 2.0}
        else:
            # Simple/solid garments: gentle guidance for natural draping and fit.
            # Lower value lets the UNet adapt the garment to body shape freely.
            return {"num_steps": 20, "image_scale": 1.5}

    def local_color_correction(self, generated, original_garment, mask_hard):
        """
        Match overall H/S/V stats of the generated garment to the original garment,
        restricted precisely to the mask boundary, and gently capped.
        """
        print(f"   [COLOR] Starting correction. Gen shape: {generated.size}, Garm shape: {original_garment.size}")
        gen_np  = np.array(generated).astype(np.float32)
        garm_np = np.array(original_garment.resize(generated.size)).astype(np.float32)
        msk_np  = np.array(mask_hard.resize(generated.size, Image.NEAREST)).astype(np.float32) / 255.0

        gen_hsv  = cv2.cvtColor(gen_np.astype(np.uint8),  cv2.COLOR_RGB2HSV).astype(np.float32)
        garm_hsv = cv2.cvtColor(garm_np.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

        mask_bool = msk_np > 0.5
        print(f"   [COLOR] Mask interior area: {mask_bool.sum()} pixels  ({100*mask_bool.mean():.1f}% of image)")

        if mask_bool.sum() < 100:
            print("   [COLOR] Skipped: Mask interior area too small (<100px).")
            return generated

        bg_garm  = np.all(garm_np >= 238, axis=-1)
        valid_garm = ~bg_garm
        print(f"   [COLOR] Valid garment pixels (non-BG): {valid_garm.sum()}  ({100*valid_garm.mean():.1f}%)")

        if valid_garm.sum() < 100:
            valid_garm = np.ones_like(bg_garm)
            print("   [COLOR] Warning: Product image had almost no valid foreground pixels. Using entire image stats.")

        ref_lum_mean  = garm_hsv[valid_garm, 2].mean()   # reference V mean (brightness)
        ref_sat_mean  = garm_hsv[valid_garm, 1].mean()   # reference S mean (saturation)
        is_white_garm = ref_lum_mean > 180 and ref_sat_mean < 60  # garment is primarily white/light
        print(f"   [COLOR] Garment profile: ref_lum={ref_lum_mean:.1f}, ref_sat={ref_sat_mean:.1f}, is_white={is_white_garm}")

        # ── Channel 2: V (Brightness) ──────────────────────────────────────────
        # KEY FIX: Weight the brightness boost by (1 - normalized_saturation).
        # Vivid graphic elements (high-S: blue jeans illustration, red text) get weight~0 → no brightness change.
        # White shirt background (low-S) gets weight~1 → full brightness boost.
        # This PRESERVES graphic colors while whitening the shirt background.
        gen_mean_v  = gen_hsv[mask_bool, 2].mean()
        garm_mean_v = garm_hsv[valid_garm, 2].mean()
        gen_std_v   = gen_hsv[mask_bool,  2].std() + 1e-6
        garm_std_v  = garm_hsv[valid_garm, 2].std() + 1e-6
        ratio_v     = max(min(garm_std_v / gen_std_v, 1.30), 1.00)  # floor=1.0: NEVER compress brightness
        raw_shift_v = (garm_mean_v - gen_mean_v) * 0.70
        # FIX: V shift cap at 30 — needs headroom for UNet's ~55pt brightness deficit
        shift_v     = min(max(raw_shift_v, 0.0), 30.0)              # only boost, never darken; cap at 30
        print(f"   [COLOR] {'Value / Brightness':<18}: Ref Mean={garm_mean_v:.1f}, Gen Mean={gen_mean_v:.1f} | Ref Std={garm_std_v:.1f}, Gen Std={gen_std_v:.1f} | Ratio={ratio_v:.2f}, Shift={shift_v:.1f}")
        corrected_v = gen_hsv[:, :, 2].copy()
        # Saturation-weighted boost: high-S (colorful) pixels get <10% of shift; low-S (white) get 100%
        sat_mask     = gen_hsv[:, :, 1]  # 0–255 saturation map
        # FIX: Tightest saturation guard S/50 — pixels with S>50 are graphic/
        # illustration elements and must NOT receive any brightness boost. This prevents the
        # blue scribbles, red text, and yellow figure from being washed out.
        sat_weight   = np.clip(1.0 - sat_mask / 50.0, 0.0, 1.0)   # S<15→weight~1.0, S>50→weight~0.0
        v_boost_map  = sat_weight * shift_v                          # per-pixel boost amount
        corrected_v[mask_bool] = corrected_v[mask_bool] * ratio_v + v_boost_map[mask_bool]
        gen_hsv[:, :, 2] = np.clip(corrected_v, 0, 255)

        # ── Channel 1: S (Saturation) ──────────────────────────────────────────
        gen_mean_s  = gen_hsv[mask_bool, 1].mean()
        garm_mean_s = garm_hsv[valid_garm, 1].mean()
        gen_std_s   = gen_hsv[mask_bool,  1].std() + 1e-6
        garm_std_s  = garm_hsv[valid_garm, 1].std() + 1e-6
        if is_white_garm:
            # White garment: compress saturation gently (cap ratio at 0.85)
            # FIX: Reduced from 1.00→0.85 and shift 1.20→0.80 to prevent washing out colored elements
            ratio_s  = min(garm_std_s / gen_std_s, 0.85)          # cap at 0.85: gentler compression
            shift_s  = (garm_mean_s - gen_mean_s) * 0.80          # gentler pull toward low-S white
        else:
            # Colorful garment: allow expansion for vibrancy
            ratio_s  = min(garm_std_s / gen_std_s, 1.60)
            shift_s  = (garm_mean_s - gen_mean_s) * 0.90
        print(f"   [COLOR] {'Saturation':<18}: Ref Mean={garm_mean_s:.1f}, Gen Mean={gen_mean_s:.1f} | Ref Std={garm_std_s:.1f}, Gen Std={gen_std_s:.1f} | Ratio={ratio_s:.2f}, Shift={shift_s:.1f}")
        corrected_s = gen_hsv[:, :, 1].copy()
        corrected_s[mask_bool] = corrected_s[mask_bool] * ratio_s + shift_s
        gen_hsv[:, :, 1] = np.clip(corrected_s, 0, 255)

        # ── Channel 0: H (Hue) — pull toward reference hue for white-base garments ──
        # FIX 3: Hue shift is now ONLY applied to low-saturation pixels (S < 40), i.e.
        # the white shirt body. High-saturation pixels (blue collar, red "FOLLOW THE SUN"
        # text, yellow illustration) keep their original hue untouched.
        if is_white_garm:
            gen_mean_h  = gen_hsv[mask_bool, 0].mean()
            garm_mean_h = garm_hsv[valid_garm, 0].mean()
            hue_shift   = np.clip((garm_mean_h - gen_mean_h) * 0.60, -15, 15)  # gentle hue correction (max ±15)
            corrected_h = gen_hsv[:, :, 0].copy()
            # Only shift low-saturation (near-white shirt) pixels — protect vivid graphic colors
            sat_in_mask = gen_hsv[:, :, 1][mask_bool]  # saturation at mask pixels
            # FIX: Tightened from S<40 to S<25 — only truly white pixels get hue shift
            low_sat_sel = sat_in_mask < 25             # S < 25 → truly near-white shirt background
            mask_rows, mask_cols = np.where(mask_bool)
            corrected_h[mask_rows[low_sat_sel], mask_cols[low_sat_sel]] += hue_shift
            gen_hsv[:, :, 0] = np.clip(corrected_h, 0, 180)  # OpenCV H range is 0–180
            low_sat_pct = 100 * low_sat_sel.sum() / max(mask_bool.sum(), 1)
            print(f"   [COLOR] {'Hue':<18}: Ref Mean={garm_mean_h:.1f}, Gen Mean={gen_mean_h:.1f} | Shift={hue_shift:.1f} | Applied to {low_sat_pct:.1f}% of mask (S<40 only)")

        corrected_rgb = cv2.cvtColor(gen_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # Luminance rescue disabled — brightness shift already handles white correction.
        # The rescue was over-modifying 70%+ of mask pixels.
        if False and ref_lum_mean > 200:  # reference is a white/light garment
            mask_r = corrected_rgb[mask_bool, 0]
            mask_g = corrected_rgb[mask_bool, 1]
            mask_b = corrected_rgb[mask_bool, 2]
            rgb_lum      = mask_r * 0.299 + mask_g * 0.587 + mask_b * 0.114
            rgb_sum      = mask_r + mask_g + mask_b + 1e-6
            rgb_min      = np.minimum(np.minimum(mask_r, mask_g), mask_b)
            # Desaturation fraction: 0=pure gray/white, 1=fully saturated color
            rgb_sat_frac = 1.0 - 3.0 * rgb_min / rgb_sum
            # Only rescue pixels that are: dark AND low-saturation (near-white, not colorful)
            # FIX 2 (part 2): Tighten luminance rescue threshold 220→210 to avoid boosting
            # near-white graphic highlight regions (e.g. white shirt background inside illustration)
            dark_and_light = (rgb_lum < 210) & (rgb_sat_frac < 0.30)  # low-S guard: S<30% → near-white
            dark_idx = np.where(mask_bool)
            dark_sel = dark_and_light
            if dark_sel.sum() > 0:
                boost = np.clip((220 - rgb_lum[dark_sel]) * 0.6, 0, 30)
                corrected_rgb[dark_idx[0][dark_sel], dark_idx[1][dark_sel]] = np.clip(
                    corrected_rgb[dark_idx[0][dark_sel], dark_idx[1][dark_sel]] + boost[:, None], 0, 255
                )
                print(f"   [COLOR] RGB luminance rescue: {dark_sel.sum()} near-white dark pixels boosted to target=230 (ref_lum={ref_lum_mean:.1f})")

        # === Final pass: flatten residual blue in NEAR-WHITE pixels only ===
        # GUARD: Only apply to pixels that are both: (1) blue-biased AND (2) already bright/near-white.
        # Dark pixels belong to the graphic/illustration — their blue is intentional (blue jeans art, scribbles).
        if is_white_garm:
            mask_pix_r = corrected_rgb[mask_bool, 0]
            mask_pix_g = corrected_rgb[mask_bool, 1]
            mask_pix_b = corrected_rgb[mask_bool, 2]
            lum_pix    = (mask_pix_r * 0.299 + mask_pix_g * 0.587 + mask_pix_b * 0.114)
            blue_bias  = mask_pix_b - (mask_pix_r + mask_pix_g) / 2.0
            # FIX: Raised lum guard from 210→230 — only desaturate truly white-area blue tinting,
            # not blue collar/trim which is legitimately blue but bright
            blue_sel   = (blue_bias > 10) & (lum_pix > 230)
            if blue_sel.sum() > 0:
                # FIX: Reduced max blend from 0.35→0.20 for gentler correction
                blend_str = np.clip(blue_bias[blue_sel] / 50.0, 0, 0.20)  # gentle: max 20% desaturation
                r_idx, c_idx = np.where(mask_bool)
                r_blue = r_idx[blue_sel]
                c_blue = c_idx[blue_sel]
                lum_b  = lum_pix[blue_sel]
                for ch in range(3):
                    ch_vals = corrected_rgb[r_blue, c_blue, ch]
                    corrected_rgb[r_blue, c_blue, ch] = ch_vals * (1 - blend_str) + lum_b * blend_str
                print(f"   [COLOR] Blue-desaturation (near-white only): {blue_sel.sum()} px (guarded from {(blue_bias>10).sum()} total blue-biased)")

        # Blend zone: 19px tight feather — avoids the 51px halo while still blending mask boundary
        blend_mask    = cv2.GaussianBlur(msk_np, (19, 19), 4)   # tightened: 51→19px kills white halo glow
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
        print(f"\n   [MASK_GEN] ============================================================")
        print(f"   [MASK_GEN] get_mask_location: model_type={model_type}  category={category}  target_size=({width},{height})")
        im_parse = model_parse.resize((width, height), Image.NEAREST)
        parse_array = np.array(im_parse)
        print(f"   [MASK_GEN] parse_array shape={parse_array.shape}  unique_vals={np.unique(parse_array).tolist()}")

        print(" -> Constructing Smart Category Mask (Off-Shoulder/Crop-Top Aware)...")

        pose_data = np.array(keypoint["pose_keypoints_2d"]).reshape((-1, 2))
        scale = height / 512.0
        pt = lambda idx: np.multiply(pose_data[idx][:2], scale)

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
        else:
            target_labels = [4, 7]
            arm_labels    = [14, 15]

        garment_mask = np.zeros((height, width), dtype=np.float32)
        for label in target_labels:
            px_count = (parse_array == label).sum()
            print(f"   [MASK_GEN] Label {label}: {px_count} pixels")
            garment_mask += (parse_array == label).astype(np.float32)

        # GARMENT DILATION: 7px minimum to fill boundary gaps (reduced from 13 to prevent puffy over-expansion)
        k_size = max(int((height * 0.014)) | 1, 7)
        print(f"   [MASK_GEN] Garment dilation kernel size: {k_size}x{k_size}")
        kernel_garment = np.ones((k_size, k_size), np.uint8)
        garment_mask_pre_dilate = garment_mask.copy()
        garment_mask = cv2.dilate(garment_mask, kernel_garment, iterations=1)
        print(f"   [MASK_GEN] Garment mask coverage before dilation: {100*(garment_mask_pre_dilate>0).mean():.1f}%")
        print(f"   [MASK_GEN] Garment mask coverage after  dilation: {100*(garment_mask>0).mean():.1f}%")

        arm_mask = np.zeros((height, width), dtype=np.float32)
        for label in arm_labels:
            px_count = (parse_array == label).sum()
            print(f"   [MASK_GEN] Arm Label {label}: {px_count} pixels")
            arm_mask += (parse_array == label).astype(np.float32)

        # ---- ARM MASK: Keep entire arm from shoulder to elbow (NO forearm clipping) ----
        # ROOT CAUSE of shoulder holes: clipping arm_mask at 90% shoulder→elbow removes 
        # the shoulder seam pixel band, creating a visible gap/hole at the shoulder.
        # FIX: Keep ALL arm pixels up to the elbow, but NOT beyond (avoids forearm ghost).
        if len(arm_labels) > 0 and category == 'upperbody':
            arm_px_before = (arm_mask > 0).sum()
            avg_elbow_y = (e_r[1] + e_l[1]) / 2.0
            # Use elbow Y + small padding: covers full sleeve including hem
            sleeve_cutoff_y = int(avg_elbow_y + int(5 / 512 * height))
            sleeve_cutoff_y = max(0, min(height - 1, sleeve_cutoff_y))
            arm_mask[sleeve_cutoff_y:, :] = 0
            arm_px_after = (arm_mask > 0).sum()
            print(f"   [MASK_GEN] ARM CLIP: zeroed arm below Y={sleeve_cutoff_y} (elbow_y={avg_elbow_y:.0f})")
            print(f"   [MASK_GEN] ARM CLIP: arm pixels before={arm_px_before}  after={arm_px_after}  removed={arm_px_before-arm_px_after}")
        else:
            print(f"   [MASK_GEN] No arm clip applied (arm_labels={arm_labels}, cat={category}).")

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
        # SHOULDER_OUT: how far outward the hull extends from shoulder keypoint
        # Must be LARGER than actual shoulder to cover shirt fabric at the seam
        ARM_PAD      = int(20 / 512 * height)   # inward pad from shoulder toward neck
        SLEEVE_PAD   = int(16 / 512 * height)   # pad at elbow level for sleeves (reduced from 24 to prevent puffy elbows)
        # FIX: Reduced from 42→30 to prevent boxy/puffy shoulder over-extension
        SHOULDER_OUT = int(30 / 512 * height)   # outward extension past shoulder keypoint
        SLEEVE_LAT   = int(22 / 512 * height)   # lateral extension for sleeve hull (reduced from 32)

        print(f"   [MASK_GEN] Hull padding params: ARM_PAD={ARM_PAD} SLEEVE_PAD={SLEEVE_PAD} SHOULDER_OUT={SHOULDER_OUT} SLEEVE_LAT={SLEEVE_LAT}")
        hull_pts = []

        if is_off_shoulder:
            hull_pts += [
                [s_r[0] + ARM_PAD,    s_r[1] + int(20/512*height)],
                [s_l[0] - ARM_PAD,    s_l[1] + int(20/512*height)],
                [e_r[0] + SLEEVE_PAD, e_r[1]],
                [e_l[0] - SLEEVE_PAD, e_l[1]],
            ]
            protect_labels = [1, 2, 11, 18]
            print(f"   [MASK_GEN] Off-shoulder hull: 4 pts")
        else:
            hull_pts += [
                [s_r[0] - SHOULDER_OUT, s_r[1]],          # right shoulder — FAR outward
                [s_r[0] + ARM_PAD,      s_r[1]],          # right shoulder — inward
                [s_l[0] + SHOULDER_OUT, s_l[1]],          # left shoulder  — FAR outward
                [s_l[0] - ARM_PAD,      s_l[1]],          # left shoulder  — inward
                [neck[0],               neck[1] + int(2/512*height)],   # neck base
            ]
            print(f"   [MASK_GEN] Shoulder hull pts:")
            print(f"              R-out=[{s_r[0]-SHOULDER_OUT:.0f},{s_r[1]:.0f}]  R-in=[{s_r[0]+ARM_PAD:.0f},{s_r[1]:.0f}]")
            print(f"              L-out=[{s_l[0]+SHOULDER_OUT:.0f},{s_l[1]:.0f}]  L-in=[{s_l[0]-ARM_PAD:.0f},{s_l[1]:.0f}]")

            # Hip-level anchor: fills torso body of shirt
            hip_y_val = (hip_r[1] + hip_l[1]) / 2
            if hip_y_val > 0:
                hull_anchor_y = int(hip_y_val * 0.97)
                hip_pad = int(12 / 512 * height)
                hull_pts += [
                    [hip_r[0] + hip_pad, hull_anchor_y],
                    [hip_l[0] - hip_pad, hull_anchor_y],
                ]
                print(f"   [MASK_GEN] Hip hull pts: Y={hull_anchor_y}  R_x={hip_r[0]+hip_pad:.0f}  L_x={hip_l[0]-hip_pad:.0f}")

            # Sleeve hull: extend outward at mid-sleeve to cover short sleeve fabric
            has_sleeves = (parse_array == 14).sum() + (parse_array == 15).sum() > 200
            print(f"   [MASK_GEN] has_sleeves={has_sleeves}  (arm_px={(parse_array==14).sum()+(parse_array==15).sum()})")
            if has_sleeves:
                # Mid-sleeve Y: 50% down shoulder→elbow (not 65%) — shorter sleeves mean we need the hull earlier
                sleeve_y_r = int(s_r[1] + 0.50 * (e_r[1] - s_r[1]))
                sleeve_y_l = int(s_l[1] + 0.50 * (e_l[1] - s_l[1]))
                # Also add a second point near shoulder for upper sleeve
                upper_sleeve_y_r = int(s_r[1] + 0.20 * (e_r[1] - s_r[1]))
                upper_sleeve_y_l = int(s_l[1] + 0.20 * (e_l[1] - s_l[1]))
                hull_pts += [
                    [s_r[0] - SLEEVE_LAT, upper_sleeve_y_r],  # RIGHT upper sleeve
                    [s_r[0] - SLEEVE_LAT, sleeve_y_r],        # RIGHT lower sleeve
                    [s_l[0] + SLEEVE_LAT, upper_sleeve_y_l],  # LEFT  upper sleeve
                    [s_l[0] + SLEEVE_LAT, sleeve_y_l],        # LEFT  lower sleeve
                ]
                print(f"   [MASK_GEN] Sleeve hull: R_upper=[{s_r[0]-SLEEVE_LAT:.0f},{upper_sleeve_y_r}]  R_lower=[{s_r[0]-SLEEVE_LAT:.0f},{sleeve_y_r}]")
                print(f"   [MASK_GEN] Sleeve hull: L_upper=[{s_l[0]+SLEEVE_LAT:.0f},{upper_sleeve_y_l}]  L_lower=[{s_l[0]+SLEEVE_LAT:.0f},{sleeve_y_l}]")
            protect_labels = [1, 2, 11]

        print(f"   [MASK_GEN] Total hull_pts: {len(hull_pts)}")
        for i, pt_val in enumerate(hull_pts):
            print(f"   [MASK_GEN]   hull_pt[{i}]: {pt_val}")

        # ---- STEP 4: Waist / Crop-top cutoff ----
        if is_crop_top and len(garment_pixels_y) > 0:
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

        if is_crop_top and len(garment_pixels_y) > 0:
            core_mask[int(np.max(garment_pixels_y) * 1.05):, :] = 0
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
            all_pts = np.vstack([core_xy, np.array(valid_hull_pts)]).astype(np.float32)
            hull = cv2.convexHull(all_pts)
            hull_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull.astype(np.int32), 255)
            hull_coverage = 100*(hull_mask > 0).mean()
            print(f"   [MASK_GEN] Convex hull built. Hull coverage: {hull_coverage:.1f}%")
            # DIAGNOSTIC: check if shoulder region is covered by hull
            sh_y_min = int(min(s_r[1], s_l[1])) - 5
            sh_y_max = int(min(s_r[1], s_l[1])) + 15
            sh_x_min = int(min(s_r[0], s_l[0]) - SHOULDER_OUT) - 5
            sh_x_max = int(max(s_r[0], s_l[0]) + SHOULDER_OUT) + 5
            sh_y_min = max(0, sh_y_min); sh_y_max = min(height-1, sh_y_max)
            sh_x_min = max(0, sh_x_min); sh_x_max = min(width-1,  sh_x_max)
            shoulder_region = hull_mask[sh_y_min:sh_y_max, sh_x_min:sh_x_max]
            print(f"   [MASK_GEN] SHOULDER REGION CHECK (Y={sh_y_min}:{sh_y_max}, X={sh_x_min}:{sh_x_max}): "
                  f"coverage={100*(shoulder_region>0).mean():.1f}%  "
                  f"({'GOOD - covered' if (shoulder_region>0).mean() > 0.3 else 'WARNING - low coverage - SHOULDER HOLE RISK'})")
            inpaint_mask = np.logical_or(base_mask, hull_mask / 255.0).astype(np.float32)
        else:
            print(f"   [MASK_GEN] WARNING: Skipped convex hull (not enough points). Using raw base_mask.")
            inpaint_mask = base_mask.copy()

        # ---- STEP 6: Protect face/hair/neck ----
        protect_mask = np.zeros((height, width), dtype=np.float32)
        for label in protect_labels:
            px = (parse_array == label).sum()
            print(f"   [MASK_GEN] Protect label {label}: {px} pixels")
            protect_mask += (parse_array == label).astype(np.float32)
        protect_mask = np.clip(protect_mask, 0, 1)
        coverage_before_protect = 100*(inpaint_mask>0).mean()
        inpaint_mask = np.logical_and(inpaint_mask, np.logical_not(protect_mask)).astype(np.float32)
        coverage_after_protect = 100*(inpaint_mask>0).mean()
        print(f"   [MASK_GEN] After protection: {coverage_before_protect:.1f}% -> {coverage_after_protect:.1f}%  (removed {coverage_before_protect-coverage_after_protect:.1f}%)")

        # ---- SHOULDER BRIDGE FILL: Guarantee shoulder-seam coverage ----
        # Draw thick lines between neck and shoulders, and circles at shoulders,
        # to ensure no gaps at the seams, even if shoulders are tilted.
        if category == 'upperbody' and not is_off_shoulder:
            thickness = int(35 / 512 * height)
            circle_radius = int(35 / 512 * height)
            
            bridge_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Draw circles at shoulders to cover the deltoid/seam area completely
            cv2.circle(bridge_mask, (int(s_r[0]), int(s_r[1])), circle_radius, 1, -1)
            cv2.circle(bridge_mask, (int(s_l[0]), int(s_l[1])), circle_radius, 1, -1)
            
            # Draw thick lines from neck to shoulders to fill the trapezius/collarbone area
            cv2.line(bridge_mask, (int(neck[0]), int(neck[1])), (int(s_r[0]), int(s_r[1])), 1, thickness)
            cv2.line(bridge_mask, (int(neck[0]), int(neck[1])), (int(s_l[0]), int(s_l[1])), 1, thickness)
            
            inpaint_mask = np.logical_or(inpaint_mask, bridge_mask).astype(np.float32)
            
            # Re-apply protection to avoid painting over face/hair
            inpaint_mask = np.logical_and(inpaint_mask, np.logical_not(protect_mask)).astype(np.float32)
            print(f"   [MASK_GEN] SHOULDER BRIDGE: Applied geometric bridge with radius {circle_radius} and thickness {thickness}")

        # ---- STEP 7: Morphology close + dilate to fill gaps ----
        # FIX: Reduced close 17→11, dilate 13→7 to prevent puffy over-expansion
        kernel_close  = np.ones((11, 11), np.uint8)
        kernel_dilate = np.ones((7, 7), np.uint8)
        inpaint_mask_u8 = (inpaint_mask * 255).astype(np.uint8)
        coverage_before_morph = 100*(inpaint_mask_u8>0).mean()
        inpaint_mask_u8 = cv2.morphologyEx(inpaint_mask_u8, cv2.MORPH_CLOSE, kernel_close)
        inpaint_mask_u8 = cv2.dilate(inpaint_mask_u8, kernel_dilate, iterations=1)
        coverage_after_morph = 100*(inpaint_mask_u8>0).mean()
        print(f"   [MASK_GEN] Morphology: before={coverage_before_morph:.1f}%  after={coverage_after_morph:.1f}%  gained={coverage_after_morph-coverage_before_morph:.1f}%")

        # DIAGNOSTIC: check shoulder area after morphology
        sh_check = inpaint_mask_u8[sh_y_min:sh_y_max, sh_x_min:sh_x_max]
        print(f"   [MASK_GEN] POST-MORPH SHOULDER CHECK: coverage={100*(sh_check>0).mean():.1f}%  "
              f"({'GOOD' if (sh_check>0).mean() > 0.5 else 'STILL LOW - SHOULDER HOLE WILL APPEAR'})")

        filled = self.hole_fill(inpaint_mask_u8)
        dst    = self.refine_mask(filled)
        print(f"   [MASK_GEN] After hole_fill+refine: dst coverage={100*(dst>0).mean():.1f}%")

        # DIAGNOSTIC: final shoulder check
        sh_final = dst[sh_y_min:sh_y_max, sh_x_min:sh_x_max]
        print(f"   [MASK_GEN] FINAL SHOULDER CHECK: coverage={100*(sh_final>0).mean():.1f}%  "
              f"({'GOOD' if (sh_final>0).mean() > 0.5 else 'PROBLEM: shoulder not masked - HOLE WILL APPEAR'})")

        # ---- STEP 8: Soft feathered edge (tighter: 15px/σ5 instead of 21px/σ7) ----
        mask_soft = cv2.GaussianBlur(dst.astype(np.float32), (15, 15), 5)
        inpaint_mask_soft = np.clip(mask_soft / 255.0, 0, 1)

        percentage = 100 * np.sum(dst > 0) / (width * height)
        print(f" -> Smart Category Mask: {percentage:.1f}% | OffShoulder={is_off_shoulder} | CropTop={is_crop_top}")
        print(f"   [MASK_GEN] ============================================================")

        self._cached_hard_mask = Image.fromarray(dst)
        return Image.fromarray(dst), Image.fromarray((inpaint_mask_soft * 255).astype(np.uint8))