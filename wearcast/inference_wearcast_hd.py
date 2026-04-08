from pathlib import Path
import sys
import os
import torch
import numpy as np
from PIL import Image
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

# Absolute Pathing to avoid HF Repo confusion
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute() # WearCast_AI root
VIT_PATH = os.path.join(PROJECT_ROOT, "checkpoints/clip-vit-large-patch14")
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints/ootd")

class WearCastHD:

    def __init__(self, gpu_id):
        self.gpu_id = 'cuda:' + str(gpu_id)

        vae = AutoencoderKL.from_pretrained(
            MODEL_PATH,
            subfolder="vae",
            torch_dtype=torch.float16,
        )

        unet_garm = UNetGarm2DConditionModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            use_safetensors=False, 
        )
        unet_vton = UNetVton2DConditionModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            use_safetensors=False,
        )

        self.pipe = WearCastPipeline.from_pretrained(
            MODEL_PATH,
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=False,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.gpu_id)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.gpu_id)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
        ).to(self.gpu_id)


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

        with torch.no_grad():
            prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").to(self.gpu_id)
            prompt_image = self.image_encoder(prompt_image.data['pixel_values']).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            
            # Simplified for Men's Half-body (HD)
            prompt_embeds = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0]
            prompt_embeds[:, 1:] = prompt_image[:]

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

        return images
