# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Yuhao Xu for WearCast (https://github.com/levihsu/WearCast)
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from transformers import AutoProcessor, CLIPVisionModelWithProjection

from .unet_vton_2d_condition import UNetVton2DConditionModel
from .unet_garm_2d_condition import UNetGarm2DConditionModel

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def preprocess(image):
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class WearCastPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    r"""
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "vton_latents"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet_garm: UNetGarm2DConditionModel,
        unet_vton: UNetVton2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image_garm: PipelineImageInput = None,
        image_vton: PipelineImageInput = None,
        mask: PipelineImageInput = None,
        image_ori: PipelineImageInput = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be repainted according to `prompt`. Can also accept
                image latents as `image`, but if passing latents directly it is not encoded again.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            image_guidance_scale (`float`, *optional*, defaults to 1.5):
                Push the generated image towards the initial `image`. Image guidance scale is enabled by setting
                `image_guidance_scale > 1`. Higher image guidance scale encourages generated images that are closely
                linked to the source `image`, usually at the expense of lower image quality. This pipeline requires a
                value of at least `1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Check inputs
        self.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        print("\n" + "=" * 60)
        print("[PIPELINE.__call__] Starting WearCastPipeline...")
        print(f"[PIPELINE] guidance_scale         = {guidance_scale}")
        print(f"[PIPELINE] image_guidance_scale   = {image_guidance_scale}")
        print(f"[PIPELINE] num_inference_steps    = {num_inference_steps}")
        print(f"[PIPELINE] num_images_per_prompt  = {num_images_per_prompt}")
        print(f"[PIPELINE] do_classifier_free_guidance = {self.do_classifier_free_guidance}")
        print(f"[PIPELINE] output_type            = {output_type}")

        if (image_vton is None) or (image_garm is None):
            raise ValueError("`image` input cannot be undefined.")

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        print(f"[PIPELINE] batch_size = {batch_size}")

        device = self._execution_device
        print(f"[PIPELINE] execution_device = {device}")
        self.to(device)

        # Force-move all primary input tensors to the pipeline's device
        if image_garm is not None:
             if isinstance(image_garm, torch.Tensor): image_garm = image_garm.to(device=device)
        if image_vton is not None:
             if isinstance(image_vton, torch.Tensor): image_vton = image_vton.to(device=device)
        if mask is not None:
             if isinstance(mask, torch.Tensor): mask = mask.to(device=device)
        if image_ori is not None:
             if isinstance(image_ori, torch.Tensor): image_ori = image_ori.to(device=device)

        # check if scheduler is in sigmas space
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

        # 2. Encode input prompt
        print("[PIPELINE] Encoding prompt embeddings...")
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        print(f"[PIPELINE] prompt_embeds after _encode_prompt: shape={list(prompt_embeds.shape)} dtype={prompt_embeds.dtype} device={prompt_embeds.device}")

        # Ensure prompt_embeds is in the correct precision for the U-Nets (float16)
        if prompt_embeds.dtype != torch.float16:
            print(f"[DEBUG] Early Cast: prompt_embeds from {prompt_embeds.dtype} to torch.float16")
            prompt_embeds = prompt_embeds.to(dtype=torch.float16)

        print(f"[WearCast] Phase 3/5: Encoding Garment and Mask...")
        # 3. Preprocess image
        print("[PIPELINE] Pre-processing images through VaeImageProcessor...")
        image_garm = self.image_processor.preprocess(image_garm)
        image_vton = self.image_processor.preprocess(image_vton)
        image_ori  = self.image_processor.preprocess(image_ori)
        print(f"[PIPELINE] image_garm preprocessed: shape={list(image_garm.shape)} dtype={image_garm.dtype} range=[{image_garm.min().item():.2f},{image_garm.max().item():.2f}]")
        print(f"[PIPELINE] image_vton preprocessed: shape={list(image_vton.shape)} dtype={image_vton.dtype} range=[{image_vton.min().item():.2f},{image_vton.max().item():.2f}]")
        print(f"[PIPELINE] image_ori  preprocessed: shape={list(image_ori.shape)} dtype={image_ori.dtype} range=[{image_ori.min().item():.2f},{image_ori.max().item():.2f}]")

        # Save raw pixel-space info for final compositing BEFORE mask binarization
        image_ori_pixels = (image_ori.clone().float() + 1.0) / 2.0  # [-1,1] -> [0,1]
        image_ori_pixels = image_ori_pixels.clamp(0, 1).to(device)

        print("[PIPELINE] Processing mask (soft + hard)...")
        mask = np.array(mask)
        print(f"[PIPELINE] Raw mask from PIL: shape={mask.shape} dtype={mask.dtype} min={mask.min()} max={mask.max()} nonzero={np.sum(mask>0)}")

        # Soft mask for pixel-space composite
        mask_soft_np = mask.astype(np.float32) / 255.0
        import cv2 as _cv2
        # === ACCURACY FIX 9: Tighter pixel-space mask blur (9→5px, σ3→σ2) → sharper compositing edges ===
        mask_soft_np = _cv2.GaussianBlur(mask_soft_np, (5, 5), 2)
        mask_soft_np = np.clip(mask_soft_np, 0, 1)
        mask_pixel = torch.tensor(mask_soft_np, device=device, dtype=torch.float32)
        mask_pixel = mask_pixel.reshape(1, 1, mask_pixel.size(-2), mask_pixel.size(-1))
        mask_pixel = torch.nn.functional.interpolate(mask_pixel, size=image_ori_pixels.shape[-2:], mode='bilinear', align_corners=False)
        print(f"[PIPELINE] mask_pixel (soft, pixel-space): shape={list(mask_pixel.shape)}")

        # Hard mask for latent-space operations
        # === ACCURACY FIX 10: Raise binarization threshold 110→127 → stricter latent mask boundary ===
        mask[mask < 127] = 0
        mask[mask >= 127] = 255
        print(f"[PIPELINE] Hard-binarized mask (thresh=127): white_px={np.sum(mask==255)} black_px={np.sum(mask==0)} coverage={100*np.mean(mask==255):.1f}%")
        mask = torch.tensor(mask, device=device, dtype=prompt_embeds.dtype)
        mask = mask / 255.0
        mask = mask.reshape(-1, 1, mask.size(-2), mask.size(-1))
        print(f"[PIPELINE] mask tensor: shape={list(mask.shape)} dtype={mask.dtype} device={mask.device}")

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        print(f"[PIPELINE] Scheduler: {type(self.scheduler).__name__}  |  total timesteps={len(timesteps)}")
        print(f"[PIPELINE] Timestep range: {timesteps[0].item():.0f} -> {timesteps[-1].item():.0f}")

        # 5. Prepare Image latents
        print("\n[PIPELINE] --- Preparing Garment Latents ---")
        garm_latents = self.prepare_garm_latents(
            image_garm,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
            generator,
        )
        print(f"[PIPELINE] garm_latents: shape={list(garm_latents.shape)} dtype={garm_latents.dtype} range=[{garm_latents.float().min().item():.4f},{garm_latents.float().max().item():.4f}]")

        print(f"\n[WearCast Phase 3] Preparing Person & Mask Latents (Guidance={self.do_classifier_free_guidance})...")
        print("[PIPELINE] --- Preparing VTON / Mask / Original Latents ---")
        vton_latents, mask_latents, image_ori_latents = self.prepare_vton_latents(
            image_vton,
            mask,
            image_ori,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
            generator,
        )
        print(f"   [LATS] VTON Latents    : {vton_latents.shape} range=[{vton_latents.float().min().item():.4f},{vton_latents.float().max().item():.4f}]")
        print(f"   [LATS] Mask Latents    : {mask_latents.shape} range=[{mask_latents.float().min().item():.4f},{mask_latents.float().max().item():.4f}]")
        print(f"   [LATS] OrigImg Latents : {image_ori_latents.shape} range=[{image_ori_latents.float().min().item():.4f},{image_ori_latents.float().max().item():.4f}]")

        # 1. Default height and width from unet
        height, width = vton_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width  = width  * self.vae_scale_factor
        print(f"[PIPELINE] Effective image resolution: {width} x {height}")

        print(f"\n[WearCast Phase 4] Preparing Noise Latents...")
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        print(f"   [LATS] Initial Noise Latents: {latents.shape} dtype={latents.dtype} init_sigma={self.scheduler.init_noise_sigma:.4f}")
        print(f"   [LATS] Noise stats: min={latents.float().min().item():.4f} max={latents.float().max().item():.4f} std={latents.float().std().item():.4f}")

        noise = latents.clone()

        print(f"\n[WearCast Phase 5] Starting Main Denoising Loop ({num_inference_steps} steps)...")
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        print(f"[PIPELINE] num_warmup_steps = {num_warmup_steps}  |  scheduler.order = {self.scheduler.order}")

        # Pre-run Garment UNet once
        garm_prompt_embeds = prompt_embeds.chunk(2)[-1] if self.do_classifier_free_guidance else prompt_embeds
        print(f"   [UNET GARM] Running preliminary Garment Spatial Attention... (Encoder embeds: {garm_prompt_embeds.shape})")
        print(f"   [UNET GARM] garm_latents shape={list(garm_latents.shape)} dtype={garm_latents.dtype} device={garm_latents.device}")

        _, spatial_attn_outputs = self.unet_garm(
            garm_latents,
            0,
            encoder_hidden_states=garm_prompt_embeds,
            return_dict=False,
        )
        if isinstance(spatial_attn_outputs, list):
            print(f"   [UNET GARM] Spatial Attention execution successful.  outputs={len(spatial_attn_outputs)} tensors")
            for si, s in enumerate(spatial_attn_outputs[:3]):
                print(f"   [UNET GARM]   attn[{si}]: shape={list(s.shape)} dtype={s.dtype}")
            if len(spatial_attn_outputs) > 3:
                print(f"   [UNET GARM]   ... ({len(spatial_attn_outputs)-3} more)")
        else:
            print(f"   [UNET GARM] Spatial Attention execution successful.  output shape={list(spatial_attn_outputs.shape)}")

        # Phase 4: Starting Denoising Loop
        print("\n[PIPELINE] ==== DENOISING LOOP START ====")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i % 5 == 0 or i == num_inference_steps - 1:
                    print(f" -> Denoising Step {i}/{num_inference_steps} (Timestep {t.item():.1f})   latent_std={latents.float().std().item():.4f}")

                if i == 0:
                    print(f"[DEBUG] Loop Start Shapes: latents={list(latents.shape)}, prompt_embeds={list(prompt_embeds.shape)}")
                    print(f"[DEBUG] Loop Start Shapes: vton_latents={list(vton_latents.shape)}, mask={list(mask_latents.shape)}")

                # Smart Guidance: Selective Doubling
                # Only double tensors if they don't already match the guidance batch (prompt_embeds)
                target_batch = prompt_embeds.shape[0]
                
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance and latents.shape[0] < target_batch else latents
                vton_latents_input = torch.cat([vton_latents] * 2) if self.do_classifier_free_guidance and vton_latents.shape[0] < target_batch else vton_latents
                mask_latents_input = torch.cat([mask_latents] * 2) if self.do_classifier_free_guidance and mask_latents.shape[0] < target_batch else mask_latents
                image_ori_latents_input = torch.cat([image_ori_latents] * 2) if self.do_classifier_free_guidance and image_ori_latents.shape[0] < target_batch else image_ori_latents
                
                # Smart Doubling for Spatial Attention Features
                if self.do_classifier_free_guidance:
                    if isinstance(spatial_attn_outputs, list):
                        if spatial_attn_outputs[0].shape[0] < target_batch:
                            spatial_attn_inputs = [torch.cat([s] * 2) for s in spatial_attn_outputs]
                        else:
                            spatial_attn_inputs = spatial_attn_outputs
                    else:
                        if spatial_attn_outputs.shape[0] < target_batch:
                            spatial_attn_inputs = torch.cat([spatial_attn_outputs] * 2)
                        else:
                            spatial_attn_inputs = spatial_attn_outputs
                else:
                    spatial_attn_inputs = spatial_attn_outputs
                
                if i == 0:
                    print(f"[DEBUG] Doubled Shapes: latent_model_input={list(latent_model_input.shape)}, vton_input={list(vton_latents_input.shape)}")
                
                # Ensure dtype compatibility for U-Net input
                if latent_model_input.dtype != self.unet_vton.dtype:
                    latent_model_input = latent_model_input.to(dtype=self.unet_vton.dtype)

                # Scaled input for the scheduler
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Full 13-channel VTON concatenation
                # Final Batch Alignment Check (Safety Net)
                expected_batch = prompt_embeds.shape[0]
                if i == 0:
                    print(f"[DEBUG] Pre-Concat Shapes: scaled_noise={list(scaled_latent_model_input.shape)}, vton_masked={list(vton_latents_input.shape)}")
                    print(f"[DEBUG] Pre-Concat Shapes: mask={list(mask_latents_input.shape)}, image_ori={list(image_ori_latents_input.shape)}")

                if latent_model_input.shape[0] != vton_latents_input.shape[0]:
                    print(f"[FIX] Re-aligning batch: latent({latent_model_input.shape[0]}) vs vton({vton_latents_input.shape[0]})")
                    if vton_latents_input.shape[0] < expected_batch:
                         vton_latents_input = torch.cat([vton_latents_input] * (expected_batch // vton_latents_input.shape[0]))
                    if mask_latents_input.shape[0] < expected_batch:
                         mask_latents_input = torch.cat([mask_latents_input] * (expected_batch // mask_latents_input.shape[0]))
                    if image_ori_latents_input.shape[0] < expected_batch:
                         image_ori_latents_input = torch.cat([image_ori_latents_input] * (expected_batch // image_ori_latents_input.shape[0]))

                # IMPORTANT: OOTD-HD expects 8 channels (4 noise + 4 masked person)
                # DO NOT add the mask (1) or original image (4) to the concatenation if in_channels is 8.
                latent_vton_model_input = torch.cat([
                    scaled_latent_model_input, 
                    vton_latents_input
                ], dim=1)

                if i == 0:
                    print(f"[DEBUG] FINAL U-Net Input Shape: {list(latent_vton_model_input.shape)}")
                    print(f"[DEBUG] Spatial Attn Inputs Count: {len(spatial_attn_inputs)}")
                    if len(spatial_attn_inputs) > 0:
                        print(f"[DEBUG] First Spatial Attn Shape: {list(spatial_attn_inputs[0].shape)}")

                # predict the noise residual
                try:
                    noise_pred = self.unet_vton(
                        latent_vton_model_input,
                        spatial_attn_inputs,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                except Exception as e:
                    print(f"[CRITICAL ERROR] UNet Forward Failed: {str(e)}")
                    print(f" -> latent_vton_model_input device: {latent_vton_model_input.device}, dtype: {latent_vton_model_input.dtype}")
                    print(f" -> prompt_embeds device: {prompt_embeds.device}, dtype: {prompt_embeds.dtype}")
                    raise e

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. So we need to compute the
                # predicted_original_sample here if we are using a karras style scheduler.
                if scheduler_is_in_sigma_space:
                    step_index = (self.scheduler.timesteps == t).nonzero()[0].item()
                    sigma = self.scheduler.sigmas[step_index]
                    if sigma.device != latent_model_input.device:
                        sigma = sigma.to(latent_model_input.device)
                    noise_pred = latent_model_input - sigma * noise_pred

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_text_image, noise_pred_text = noise_pred.chunk(2)
                    if i == 0:
                        print(f"[DEBUG] Guidance: chunk shapes = {list(noise_pred_text_image.shape)}, {list(noise_pred_text.shape)}")
                    noise_pred = (
                        noise_pred_text
                        + self.image_guidance_scale * (noise_pred_text_image - noise_pred_text)
                    )

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. But the scheduler.step function
                # expects the noise_pred and computes the predicted_original_sample internally. So we
                # need to overwrite the noise_pred here such that the value of the computed
                # predicted_original_sample is correct.
                if scheduler_is_in_sigma_space:
                    if sigma.device != noise_pred.device:
                        sigma = sigma.to(noise_pred.device)
                    noise_pred = (noise_pred - latents) / (-sigma)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # =====================================================================
                # RESTORED SDEdit latent blending (OOTDiffusion default)
                # Mixing noisy original-image latents at every step ensures the output
                # matches the person's original identity and background perfectly.
                # =====================================================================
                init_latents_proper = image_ori_latents
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.add_noise(
                        image_ori_latents, noise, torch.tensor([noise_timestep], dtype=torch.long, device=latents.device)
                    )
                latents = (1 - mask_latents) * init_latents_proper + mask_latents * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    vton_latents = callback_outputs.pop("vton_latents", vton_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        print("\n[PIPELINE] ==== DENOISING LOOP COMPLETE ==== ")
        print(f"[PIPELINE] Final latents: shape={list(latents.shape)} dtype={latents.dtype} std={latents.float().std().item():.4f}")

        if not output_type == "latent":
            print(f"[PIPELINE] Decoding latents via VAE (scaling_factor={self.vae.config.scaling_factor})...")
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            print(f"[PIPELINE] VAE decoded image: shape={list(image.shape)} dtype={image.dtype} range=[{image.float().min().item():.4f},{image.float().max().item():.4f}]")
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        print(f"[PIPELINE] Postprocessed: {len(image)} image(s)  size={image[0].size if hasattr(image[0], 'size') else 'N/A'}")

        # Offload all models
        self.maybe_free_model_hooks()
        print("[PIPELINE] Pipeline call complete.")
        print("=" * 60)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_garm_latents(
        self, image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            image_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            print(f"[DEBUG] prepare_garm_latents: input shape={image.shape}, dtype={image.dtype} range=[{image.float().min().item():.4f},{image.float().max().item():.4f}]")
            if image.dtype != self.vae.dtype:
                print(f"[DEBUG]  -> Casting garment image from {image.dtype} to {self.vae.dtype}")
                image = image.to(dtype=self.vae.dtype)

            if isinstance(generator, list):
                image_latents = [self.vae.encode(image[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
                image_latents = torch.cat(image_latents, dim=0)
            else:
                enc_output = self.vae.encode(image)
                print(f"[DEBUG] prepare_garm_latents: VAE latent_dist mean={list(enc_output.latent_dist.mean.shape)} std_min={enc_output.latent_dist.std.min().item():.4f}")
                image_latents = enc_output.latent_dist.mode()

            print(f"[DEBUG] prepare_garm_latents: output latents shape={image_latents.shape}, dtype={image_latents.dtype} range=[{image_latents.float().min().item():.4f},{image_latents.float().max().item():.4f}]")

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        # No doubling here; we handle it in the main denoising loop for consistency
        return image_latents
    
    def prepare_vton_latents(
        self, image, mask, image_ori, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)
        image_ori = image_ori.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        mask = mask.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            image_latents = image
            image_ori_latents = image_ori
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            # Masking the reference image: We MUST zero out the area to be replaced 
            # so the model knows what to fill in.
            if mask.device != image.device:
                mask = mask.to(image.device)
            
            mask_vae = torch.nn.functional.interpolate(mask, size=image.shape[-2:])
            
            # --- HIGH-ACCURACY FIX: BINARIZATION ---
            # We must use a HARD (binary) mask for the input to zero out old clothes.
            # Otherwise, soft edges in the mask leave "ghost" pixels of the old white tank top.
            mask_vae_bin = (mask_vae > 0.5).to(dtype=image.dtype)
            one_minus_mask = torch.ones_like(mask_vae_bin, device=image.device) - mask_vae_bin
            image_masked = image * one_minus_mask
            
            if image_masked.dtype != self.vae.dtype:
                image_masked = image_masked.to(dtype=self.vae.dtype)

            print(f"[DEBUG] prepare_vton_latents: image_masked range=[{image_masked.float().min().item():.4f},{image_masked.float().max().item():.4f}]  masked_region_zero={((image_masked==0).float().mean()*100):.1f}%")
            enc_vton = self.vae.encode(image_masked)
            image_latents = enc_vton.latent_dist.mode()
            print(f"[DEBUG] prepare_vton_latents: masked person latents shape={image_latents.shape}  range=[{image_latents.float().min().item():.4f},{image_latents.float().max().item():.4f}]")

            if image_ori.dtype != self.vae.dtype:
                print(f"[DEBUG]  -> Casting original image from {image_ori.dtype} to {self.vae.dtype}")
                image_ori = image_ori.to(dtype=self.vae.dtype)

            enc_ori = self.vae.encode(image_ori)
            image_ori_latents = enc_ori.latent_dist.mode()
            # Scale ONLY the original latents used for background blending to prevent SDEdit explosion.
            # DO NOT scale image_latents (vton conditioning) as OOTDiffusion UNet expects it unscaled.
            image_ori_latents = image_ori_latents * self.vae.config.scaling_factor
            print(f"[DEBUG] prepare_vton_latents: original latents shape={image_ori_latents.shape}  range=[{image_ori_latents.float().min().item():.4f},{image_ori_latents.float().max().item():.4f}] (after scaling)")

        mask = torch.nn.functional.interpolate(
            mask, size=(image_latents.size(-2), image_latents.size(-1))
        )
        mask = mask.to(device=device, dtype=dtype)

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            mask = torch.cat([mask] * additional_image_per_prompt, dim=0)
            image_ori_latents = torch.cat([image_ori_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)
            mask = torch.cat([mask], dim=0)
            image_ori_latents = torch.cat([image_ori_latents], dim=0)

        # No doubling here; we handle it in the main denoising loop for consistency and to avoid double-doubling
        # Final Device Anchor: Force all return tensors back to GPU
        image_latents = image_latents.to(device=device)
        mask = mask.to(device=device)
        image_ori_latents = image_ori_latents.to(device=device)

        return image_latents, mask, image_ori_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_freeu
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet_vton.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet_vton.disable_freeu()

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self.image_guidance_scale >= 1.0
