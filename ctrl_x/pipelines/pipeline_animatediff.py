import numpy as np, torch, PIL, inspect
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel, UNetMotionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.models.unets.unet_motion_model import MotionAdapter
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.free_init_utils import FreeInitMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.animatediff.pipeline_output import AnimateDiffPipelineOutput

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import\
    rescale_noise_cfg, retrieve_latents, retrieve_timesteps
from diffusers import AnimateDiffPipeline

from ctrl_x.utils import preprocess_video, noise_prev
from ctrl_x.utils.sdxl import register_attr
from ctrl_x.utils.utils import *


# ***
BATCH_ORDER = [
    "structure_uncond", "appearance_uncond", "uncond", "structure_cond", "appearance_cond", "cond",
]

def get_last_control_i(control_schedule, num_inference_steps):
    if control_schedule is None:
        return num_inference_steps, num_inference_steps
    
    def max_(l):
        if len(l) == 0:
            return 0.0
        return max(l)

    structure_max = 0.0
    appearance_max = 0.0
    for block in control_schedule.values():
        if isinstance(block, list):  # Handling mid_block
            block = {0: block}
        for layer in block.values():
            structure_max = max(structure_max, max_(layer[0] + layer[1]))
            appearance_max = max(appearance_max, max_(layer[2]))

    structure_i = round(num_inference_steps * structure_max)
    appearance_i = round(num_inference_steps * appearance_max)
    return structure_i, appearance_i

# ***


class CtrlXAnimateDiffPipeline(AnimateDiffPipeline):  # diffusers==0.28.0

    def prepare_latents(
        self, image, batch_size, num_videos_per_prompt, num_channels_latents, 
        num_frames, height, width,
        dtype, device, generator=None, noise=None,
    ):
        batch_size = batch_size * num_videos_per_prompt

        if noise is None:
            shape = (
                batch_size,
                num_channels_latents,
                num_frames,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor
            )
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            noise = noise * self.scheduler.init_noise_sigma  # Starting noise, need to scale
        else:
            noise = noise.to(device)

        if image is None:
            return noise, None

        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )
        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:  # Image already in latents form, C=4
            init_latents = image

        else: # otherwise, pass through VAE
            # Make sure the VAE is in float32 mode, as it overflows in float16
            if self.vae.config.force_upcast:
                image = image.to(torch.float32)
                self.vae.to(torch.float32)

            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            if self.vae.config.force_upcast:
                self.vae.to(dtype)

            init_latents = init_latents.to(dtype)
            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # Expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        # !
        # ensure each are of shape [b, c, f, h, w]
        init_latents = init_latents.permute(1, 0, 2, 3).unsqueeze(dim=0)
        assert noise.ndim==5 and init_latents.ndim==5, "the shapes of tensors noise and init_latents are not (b, c, f, h, w)"
        return noise, init_latents


    @torch.no_grad()
    def __call__(
        self,

        prompt: Union[str, List[str]] = None,
        structure_prompt: Optional[Union[str, List[str]]] = None,       # *** 
        appearance_prompt: Optional[Union[str, List[str]]] = None,      # *** 
        positive_prompt: Optional[Union[str, List[str]]] = None,        # *** 
        negative_prompt: Optional[Union[str, List[str]]] = None,        # *** 

        structure_image: Optional[PipelineImageInput] = None,           # *** 
        appearance_image: Optional[PipelineImageInput] = None,          # ***  
                                                                        

        num_frames: Optional[int] = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,

        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        structure_guidance_scale: Optional[float] = None,    # *** 
        appearance_guidance_scale: Optional[float] = None,   # ***
        guidance_rescale : float = 0.0,                      # ***

        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,

        latents: Optional[torch.Tensor] = None,
        structure_latents: Optional[torch.Tensor] = None,
        appearance_latents: Optional[torch.Tensor] = None,

        self_recurrence_schedule: Optional[List[int]] = [],  # ***
        control_schedule: Optional[Dict] = None,             # ***

        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):

        # callback stuff - don't touch
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        print(height, width)

        # ***
        original_size = (height, width)
        target_size = (height, width)
        # ***

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._structure_guidance_scale = structure_guidance_scale
        self._appearance_guidance_scale = appearance_guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # ***
        self._denoising_end = None
        self._denoising_start = None
        self._interrupt = False
        # ***

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3.0 Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        # *** 
        # add positive prompt to (+control, +appearance, -structure) prompts
        if positive_prompt is not None and positive_prompt != "":
            prompt = prompt + ", " + positive_prompt
            if appearance_prompt is not None and appearance_prompt != "":
                appearance_prompt = appearance_prompt + ", " + positive_prompt
        # *** 

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 3.1 Structure prompt embeddings
        if structure_prompt is not None and structure_prompt != "":
            structure_prompt_embeds, negative_structure_prompt_embeds = self.encode_prompt(
                structure_prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt = negative_prompt if structure_image is None else "",
                prompt_embeds=None, # TODO: should be structure_prompt_embeds
                negative_prompt_embeds=None,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )
            structure_prompt_embeds = torch.cat([negative_structure_prompt_embeds, structure_prompt_embeds], 
                                                dim=0).to(device)
        else:
            structure_prompt_embeds = prompt_embeds


        # 3.2 appearance prompt embeddings
        if appearance_prompt is not None and appearance_prompt != "":
            appearance_prompt_embeds, negative_appearance_prompt_embeds = self.encode_prompt(
                appearance_prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt = negative_prompt if appearance_image is None else "",
                prompt_embeds=None, # TODO: should be appearance_prompt_embeds
                negative_prompt_embeds=None,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )
            appearance_prompt_embeds = torch.cat([negative_appearance_prompt_embeds, appearance_prompt_embeds], 
                                                dim=0).to(device)
        else:
            appearance_prompt_embeds = prompt_embeds

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_videos_per_prompt,
                self.do_classifier_free_guidance,
            )

        # TODO: add additional time id as kwargs
        #   ***


        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents, _ = self.prepare_latents(
            None,
            batch_size,
            num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # ***
        from diffusers.image_processor import VaeImageProcessor
        self.image_processor = VaeImageProcessor()
        # ***

        # 5.1 Prepare structure latents
        if structure_image is not None:
            structure_image = preprocess_video(  # Center crop + resize
                structure_image, self.image_processor, height=height, width=width, resize_mode="crop"
            )
            _, clean_structure_latents = self.prepare_latents(
                structure_image, batch_size, num_videos_per_prompt, num_channels_latents, 
                num_frames, height, width,
                prompt_embeds.dtype, device, generator, structure_latents,
            )
        else:
            clean_structure_latents = None
        structure_latents = latents if structure_latents is None else structure_latents

        # 5.2 Prepare appearance latents
        if appearance_image is not None:
            appearance_image = preprocess_video(  # Center crop + resize
                appearance_image, self.image_processor, height=height, width=width, resize_mode="crop"
            )
            _, clean_appearance_latents = self.prepare_latents(
                appearance_image, batch_size, num_videos_per_prompt, num_channels_latents, 
                num_frames, height, width,
                prompt_embeds.dtype, device, generator, appearance_latents,
            )
        else:
            clean_appearance_latents = None
        appearance_latents = latents if appearance_latents is None else appearance_latents

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # ***
        # 7.1 Get batch order
        batch_order = deepcopy(BATCH_ORDER)
        if structure_image is not None:  # If image is provided, not generating, so CFG not needed
            batch_order.remove("structure_uncond")
        if appearance_image is not None:
            batch_order.remove("appearance_uncond")

        structure_control_stop_i, appearance_control_stop_i = get_last_control_i(control_schedule, num_inference_steps)
        if self_recurrence_schedule is None:
            self_recurrence_schedule = [0] * num_inference_steps
        # ***

        num_free_init_iters = self._free_init_num_iters if self.free_init_enabled else 1
        for free_init_iter in range(num_free_init_iters):
            if self.free_init_enabled:
                latents, timesteps = self._apply_free_init(
                    latents, free_init_iter, num_inference_steps, device, latents.dtype, generator
                )

            self._num_timesteps = len(timesteps)
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

            # 8. Denoising loop
            with self.progress_bar(total=self._num_timesteps) as progress_bar:
                for i, t in enumerate(timesteps):

                    # ***
                    # if not generating structure/appearance, 
                    # drop after last control for memory savings
                    if i == structure_control_stop_i:  
                        if "structure_uncond" not in batch_order:
                            batch_order.remove("structure_cond")
                    if i == appearance_control_stop_i:
                        if "appearance_uncond" not in batch_order:
                            batch_order.remove("appearance_cond")
                    # ***


                    # *** 
                    # assume CFG, scale inputs
                    register_attr(self, t=t.item(), do_control=True, batch_order=batch_order)
                    latent_model_input = self.scheduler.scale_model_input(latents, t)
                    structure_latent_model_input = self.scheduler.scale_model_input(structure_latents, t)
                    appearance_latent_model_input = self.scheduler.scale_model_input(appearance_latents, t)
                    # ***


                    # ***
                    all_latent_model_input = {
                        "structure_uncond": structure_latent_model_input[0:1],
                        "appearance_uncond": appearance_latent_model_input[0:1],
                        "uncond": latent_model_input[0:1],
                        "structure_cond": structure_latent_model_input[0:1],
                        "appearance_cond": appearance_latent_model_input[0:1],
                        "cond": latent_model_input[0:1],
                    }

                    all_prompt_embeds = {
                        "structure_uncond": structure_prompt_embeds[0:1],
                        "appearance_uncond": appearance_prompt_embeds[0:1],
                        "uncond": prompt_embeds[0:1],
                        "structure_cond": structure_prompt_embeds[1:2],
                        "appearance_cond": appearance_prompt_embeds[1:2],
                        "cond": prompt_embeds[1:2],
                    }

                    concat_latent_model_input = batch_dict_to_tensor(all_latent_model_input, batch_order)
                    concat_prompt_embeds = batch_dict_to_tensor(all_prompt_embeds, batch_order)
                    # ***

                    # predict the noise residual
                    concat_noise_pred = self.unet(
                        concat_latent_model_input,
                        t,
                        encoder_hidden_states=concat_prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                    all_noise_pred = batch_tensor_to_dict(concat_noise_pred, batch_order)

                    # ***
                    # CFG 
                    noise_pred = all_noise_pred["uncond"] +\
                            self.guidance_scale * (all_noise_pred["cond"] - all_noise_pred["uncond"])

                    structure_noise_pred = all_noise_pred["structure_cond"]\
                        if "structure_cond" in batch_order else noise_pred
                    if "structure_uncond" in all_noise_pred:
                        structure_noise_pred = all_noise_pred["structure_uncond"] +\
                            self.structure_guidance_scale * (structure_noise_pred - all_noise_pred["structure_uncond"])
                
                    appearance_noise_pred = all_noise_pred["appearance_cond"]\
                        if "appearance_cond" in batch_order else noise_pred
                    if "appearance_uncond" in all_noise_pred:
                        appearance_noise_pred = all_noise_pred["appearance_uncond"] +\
                            self._appearance_guidance_scale * (appearance_noise_pred - all_noise_pred["appearance_uncond"])
                    # ***

                    # ***
                    # guidance rescaling
                    if self._guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(
                            noise_pred, all_noise_pred["cond"], guidance_rescale=self._guidance_rescale
                        )
                    if "structure_uncond" in all_noise_pred:
                        structure_noise_pred = rescale_noise_cfg(
                            structure_noise_pred, all_noise_pred["structure_cond"],
                            guidance_rescale=self._guidance_rescale
                        )
                    if "appearance_uncond" in all_noise_pred:
                        appearance_noise_pred = rescale_noise_cfg(
                            appearance_noise_pred, all_noise_pred["appearance_cond"],
                            guidance_rescale=self._guidance_rescale
                        )
                    # ***

                    # ***
                    # compute the previous noisy sample x_t -> x_t-1
                    concat_noise_pred = torch.cat(
                        [structure_noise_pred, appearance_noise_pred, noise_pred], dim=0,
                    )
                    concat_latents = torch.cat(
                        [structure_latents, appearance_latents, latents], dim=0,
                    )
                    structure_latents, appearance_latents, latents = self.scheduler.step(
                        concat_noise_pred, t, concat_latents, **extra_step_kwargs,
                    ).prev_sample.chunk(3)
                    # ***


                    # ***
                    if clean_structure_latents is not None:
                        structure_latents = noise_prev(self.scheduler, t, clean_structure_latents)
                    if clean_appearance_latents is not None:
                        appearance_latents = noise_prev(self.scheduler, t, clean_appearance_latents)
                    # ***


                    # ***
                    # self-recurrence
                    for _ in range(self_recurrence_schedule[i]):
                        if hasattr(self.scheduler, "_step_index"):  # For fancier schedulers
                            self.scheduler._step_index -= 1  # TODO: Does this actually work?
                        
                        t_prev = 0 if i + 1 >= num_inference_steps else timesteps[i + 1]
                        latents = noise_t2t(self.scheduler, t_prev, t, latents)
                        latent_model_input = torch.cat([latents] * 2)
                        
                        register_attr(self, t=t.item(), do_control=False, batch_order=["uncond", "cond"])
                        
                        # Predict the noise residual
                        noise_pred_uncond, noise_pred_ = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states = prompt_embeds,
                            cross_attention_kwargs = self.cross_attention_kwargs,
                            added_cond_kwargs = added_cond_kwargs,
                        ).sample.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_ - noise_pred_uncond)
                    
                        if self._guidance_rescale > 0.0:
                            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_, guidance_rescale=self._guidance_rescale)
                        
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    # ***


                    # callback stuff
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        # "Reconstruction"
        if clean_structure_latents is not None:
            structure_latents = clean_structure_latents
        if clean_appearance_latents is not None:
            appearance_latents = clean_appearance_latents

        # 9. Post processing
        if output_type == "latent":
            video = latents
        else:
            video_tensor = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video_tensor, output_type=output_type)

        # 10. Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return AnimateDiffPipelineOutput(frames=video)