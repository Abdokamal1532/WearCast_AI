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
            mask, _ = self.get_mask_location(model_type, category, model_parse, keypoints)
            mask = mask.resize((768, 1024), Image.NEAREST)
            print(" -> Preprocessing Stage: SUCCESS")

        print(f"[WearCast] Phase 2/4: Encoding Inputs (VAE & CLIP Vision)...")
        with torch.no_grad():
            # Explicitly cast to half (float16) for T4 GPU compatibility
            prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").to(device=self.gpu_id)
            prompt_image = self.image_encoder(prompt_image.data['pixel_values'].to(dtype=torch.float16)).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            
            prompt_embeds = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0].to(dtype=torch.float16)
            prompt_embeds[:, 1:] = prompt_image[:]

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
        label_map = {"background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4, "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10, "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15, "bag": 16, "scarf": 17}
        im_parse = model_parse.resize((width, height), Image.NEAREST)
        parse_array = np.array(im_parse)
        arm_width = 60

        parse_head = (parse_array == 1).astype(np.float32) + (parse_array == 3).astype(np.float32) + (parse_array == 11).astype(np.float32)
        parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + (parse_array == label_map["right_shoe"]).astype(np.float32) + (parse_array == label_map["hat"]).astype(np.float32) + (parse_array == label_map["sunglasses"]).astype(np.float32) + (parse_array == label_map["bag"]).astype(np.float32)
        parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)
        arms_left = (parse_array == 14).astype(np.float32)
        arms_right = (parse_array == 15).astype(np.float32)
        parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        pose_data = np.array(keypoint["pose_keypoints_2d"]).reshape((-1, 2))
        im_arms_left, im_arms_right = Image.new('L', (width, height)), Image.new('L', (width, height))
        arms_draw_left, arms_draw_right = ImageDraw.Draw(im_arms_left), ImageDraw.Draw(im_arms_right)
        
        # Scaling pose points
        pt = lambda idx: np.multiply(tuple(pose_data[idx][:2]), height / 512.0)
        s_r, s_l, e_r, e_l, w_r, w_l = pt(2), pt(5), pt(3), pt(6), pt(4), pt(7)
        ARM_LINE_WIDTH = int(arm_width / 512 * height)

        if w_r[0] > 1. or w_r[1] > 1.:
            w_r_ext = self.extend_arm_mask(w_r, e_r, 1.2)
            arms_draw_right.line(np.concatenate((s_r, e_r, w_r_ext)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_right.arc([s_r[0]-ARM_LINE_WIDTH//2, s_r[1]-ARM_LINE_WIDTH//2, s_r[0]+ARM_LINE_WIDTH//2, s_r[1]+ARM_LINE_WIDTH//2], 0, 360, 'white', ARM_LINE_WIDTH//2)
        else: im_arms_right = arms_right

        if w_l[0] > 1. or w_l[1] > 1.:
            w_l_ext = self.extend_arm_mask(w_l, e_l, 1.2)
            arms_draw_left.line(np.concatenate((w_l_ext, e_l, s_l)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_left.arc([s_l[0]-ARM_LINE_WIDTH//2, s_l[1]-ARM_LINE_WIDTH//2, s_l[0]+ARM_LINE_WIDTH//2, s_l[1]+ARM_LINE_WIDTH//2], 0, 360, 'white', ARM_LINE_WIDTH//2)
        else: im_arms_left = arms_left

        hands_left, hands_right = np.logical_and(np.logical_not(im_arms_left), arms_left), np.logical_and(np.logical_not(im_arms_right), arms_right)
        parser_mask_fixed = parser_mask_fixed + hands_left + hands_right + parse_head
        parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
        neck_mask = cv2.dilate((parse_array == 18).astype(np.float32), np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)
        parse_mask += np.logical_or(parse_mask, arm_mask)
        parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
        
        inpaint_mask = 1 - np.logical_or(parse_mask, parser_mask_fixed)
        dst = self.refine_mask(self.hole_fill(np.where(inpaint_mask, 255, 0).astype(np.uint8)))
        inpaint_mask = dst / 255.0
        return Image.fromarray((inpaint_mask * 255).astype(np.uint8)), Image.fromarray((inpaint_mask * 127).astype(np.uint8))
