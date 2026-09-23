# SPDX-License-Identifier: Apache-2.0
import math

import torch
from PIL import Image

from sglang.multimodal_gen.runtime.cache.dpcache import (
    DPCacheRequestSignature,
    DPCacheState,
    check_schedule_matches,
    checkpoint_identity,
    config_digest,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import build_layout
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    calculate_linear_shift,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.vision import load_image

logger = init_logger(__name__)

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
SYSTEM_TEMPLATE = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"


def collapse_image_slots(hidden, input_ids, image_token_id):
    image_mask = input_ids == image_token_id
    keep = ~image_mask
    keep[0] = True
    keep[1:] |= image_mask[1:] & ~image_mask[:-1]
    return hidden[keep], image_mask[keep]


class QwenImage21InputValidationStage(InputValidationStage):
    def load_condition_image(self, image):
        return load_image(image, convert_method=lambda image: image.convert("RGBA"))

    def preprocess_condition_image(
        self, batch, server_args, condition_image_width, condition_image_height
    ):
        # one model-owned resize is shared by the VLM and VAE in the encoding stage
        return None

    def forward(self, batch, server_args):
        if batch.prompt is None:
            raise ValueError(
                "Qwen-Image 2.1 requires a prompt to build image-token positions"
            )
        batch = super().forward(batch, server_args)
        if batch.height % 32 or batch.width % 32:
            raise ValueError("Qwen-Image 2.1 height and width must be divisible by 32")
        return batch


class QwenImage21EncodingStage(PipelineStage):
    def __init__(self, text_encoder, processor, vae, scheduler):
        super().__init__()
        self.text_encoder, self.processor, self.vae, self.scheduler = (
            text_encoder,
            processor,
            vae,
            scheduler,
        )
        self.text_encoder.model.visual.fp32_position_interpolation = False
        self.text_encoder.model.visual.rotary_pos_emb.recompute_on_device_change = True
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        system_message = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        ]
        self.drop_idx = len(
            processor.apply_chat_template(
                system_message, tokenize=True, return_dict=False
            )[0]
        )

    def component_uses(self, server_args, stage_name=None):
        name = self._component_stage_name(stage_name)
        return [
            # preserve the loader's mixed weight and rotary buffer dtypes
            ComponentUse(name, "text_encoder"),
            ComponentUse(name, "vae", target_dtype=torch.bfloat16),
        ]

    def encode_prompt(self, prompt, images, device):
        prefix = " ".join(
            f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>"
            for i in range(len(images))
        )
        text = (
            SYSTEM_TEMPLATE
            + f"<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
        )
        kwargs = dict(
            text=[text], padding=True, padding_side="left", return_tensors="pt"
        )
        if images:
            vision_images = []
            for image in images:
                if image.mode == "RGBA":
                    # vision conditioning uses white compositing; the VAE keeps RGBA
                    white = Image.new("RGB", image.size, (255, 255, 255))
                    white.paste(image, mask=image.getchannel("A"))
                    image = white
                vision_images.append(image)
            kwargs["images"] = vision_images
        inputs = self.processor(**kwargs).to(device)
        with self.use_declared_component(
            component_name="text_encoder", module=self.text_encoder
        ) as encoder:
            outputs = encoder(
                **inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1
            )
            # the checkpoint expects Transformers 4.57's pre-final-norm hidden state
            final_hidden = outputs.hidden_states[-1]
        valid = inputs.attention_mask[0].bool()
        hidden = final_hidden[0, valid][self.drop_idx :]
        ids = inputs.input_ids[0, valid][self.drop_idx :]
        return collapse_image_slots(hidden, ids, self.image_token_id)

    def forward(self, batch, server_args):
        config = server_args.pipeline_config
        ac = config.vae_config.arch_config
        device = get_local_torch_device()
        images = batch.condition_image
        images = (
            [] if images is None else images if isinstance(images, list) else [images]
        )
        resized, shapes, conditions = [], [], []
        area = batch.height * batch.width
        image_mode = "RGBA" if ac.in_channels == 4 else "RGB"
        for image in images:
            if not isinstance(image, Image.Image):
                image = load_image(
                    image, convert_method=lambda image: image.convert(image_mode)
                )
            width = max(
                32, round(math.sqrt(area * image.width / image.height) / 32) * 32
            )
            height = max(
                32, round(math.sqrt(area * image.height / image.width) / 32) * 32
            )
            resized.append(
                image.convert(image_mode).resize(
                    (width, height), Image.Resampling.LANCZOS
                )
            )
            shapes.append((1, height // 16, width // 16))
        if resized:
            with self.use_declared_component(
                component_name="vae", module=self.vae
            ) as vae:
                vae.use_tiling = config.vae_tiling
                for image in resized:
                    pixels = torch.frombuffer(
                        bytearray(image.tobytes()), dtype=torch.uint8
                    ).reshape(image.height, image.width, ac.in_channels)
                    # preserve the reference's batch stride for identical cuDNN convolution rounding
                    pixels = (
                        pixels[None].permute(0, 3, 1, 2).unsqueeze(2).float() / 255.0
                    )
                    pixels = (2 * pixels - 1).to(device=device, dtype=torch.bfloat16)
                    latent = vae.encode(pixels).mode()
                    mean = latent.new_tensor(ac.latents_mean).view(1, ac.z_dim, 1, 1, 1)
                    std = latent.new_tensor(ac.latents_std).view(1, ac.z_dim, 1, 1, 1)
                    conditions.append(
                        ((latent - mean) / std).flatten(2).transpose(1, 2)
                    )
        shapes.append((1, batch.height // 16, batch.width // 16))
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        negatives = (
            batch.negative_prompt
            if isinstance(batch.negative_prompt, list)
            else [batch.negative_prompt] * len(prompts)
        )
        sample_count = len(prompts) * batch.num_outputs_per_prompt
        condition_latents = (
            torch.cat(conditions, dim=1).expand(sample_count, -1, -1)
            if conditions
            else None
        )
        for negative in [False, True] if batch.do_classifier_free_guidance else [False]:
            embeds, masks, layouts = [], [], []
            for prompt in negatives if negative else prompts:
                with set_forward_context(
                    current_timestep=None, attn_metadata=None, forward_batch=batch
                ):
                    hidden, slots = self.encode_prompt(prompt, resized, device)
                layout = build_layout(
                    slots.tolist(), shapes, config.dit_config.axes_dims_rope, device
                )
                for _ in range(batch.num_outputs_per_prompt):
                    embeds.append(hidden)
                    layouts.append(layout)
            max_length = max(x.shape[0] for x in embeds)
            for x in embeds:
                masks.append(torch.arange(max_length, device=device) < x.shape[0])
            packed = torch.stack(
                [
                    torch.nn.functional.pad(x, (0, 0, 0, max_length - x.shape[0]))
                    for x in embeds
                ]
            )
            mask = torch.stack(masks)
            if negative:
                batch.negative_prompt_embeds = [packed]
                batch.negative_prompt_embeds_mask = [mask]
                batch.negative_prompt_seq_lens = [mask.sum(1).tolist()]
            else:
                batch.prompt_embeds = [packed]
                batch.prompt_embeds_mask = [mask]
                batch.prompt_seq_lens = [mask.sum(1).tolist()]
            batch.extra["qwen21_negative" if negative else "qwen21_positive"] = dict(
                layouts=layouts,
                condition_latents=condition_latents,
                prefix_caches=[
                    [{} for _ in range(config.dit_config.num_layers)]
                    for _ in range(sample_count)
                ],
            )
        sched = self.scheduler.config
        batch.extra["qwen21_mu"] = calculate_linear_shift(
            (batch.height // 16) * (batch.width // 16),
            base_seq_len=sched.get("base_image_seq_len", 256),
            max_seq_len=sched.get("max_image_seq_len", 4096),
            base_shift=sched.get("base_shift", 0.5),
            max_shift=sched.get("max_shift", 1.15),
        )
        return batch


def prepare_qwen21_mu(batch, server_args):
    return "mu", batch.extra["qwen21_mu"]


DPCACHE_BRANCHES = ("qwen21_positive", "qwen21_negative")


def _has_active_lora(module):
    return any(
        isinstance(layer, BaseLayerWithLoRA)
        and (layer.merged or not layer.disable_lora)
        for layer in module.modules()
    )


class QwenImage21DenoisingStage(DenoisingStage):
    def forward(self, batch, server_args):
        states = self._attach_dpcache(batch, server_args)
        if not states:
            return super().forward(batch, server_args)
        try:
            batch = super().forward(batch, server_args)
            if not batch.is_warmup:
                logger.info(
                    "DPCache: %s",
                    ", ".join(
                        f"{name} full={state.num_full} predicted={state.num_predicted}"
                        for name, state in states.items()
                    ),
                )
            return batch
        finally:
            # post_denoising_loop drops the branch kwargs; also clean up on error
            for name in states:
                batch.extra.get(name, {}).pop("dpcache_state", None)

    def _attach_dpcache(self, batch, server_args):
        """Give each branch of this request its own fresh DPCache state."""
        schedule = batch.sampling_params.dpcache_schedule
        if schedule is None:
            return {}
        if self._cache_dit_enabled and not self._cache_dit_requested_for_batch(batch):
            # a previous request left the wrapper mounted; this one runs without it
            self._unmount_cache_dit()
        self._check_dpcache_supported(batch, server_args)
        full_steps = check_schedule_matches(
            schedule, self.dpcache_signature(batch, server_args)
        )
        if batch.extra.get("qwen21_positive") is None:
            raise RuntimeError("DPCache needs the Qwen-Image 2.1 branch kwargs")
        states = {}
        for name in DPCACHE_BRANCHES:
            branch = batch.extra.get(name)
            if branch is not None:
                branch["dpcache_state"] = DPCacheState(full_steps, len(batch.timesteps))
                states[name] = branch["dpcache_state"]
        return states

    def dpcache_signature(self, batch, server_args):
        """The request settings a DPCache schedule is calibrated for."""
        return DPCacheRequestSignature(
            pipeline=type(server_args.pipeline_config).__name__,
            checkpoint=checkpoint_identity(server_args.model_path),
            num_inference_steps=len(batch.timesteps),
            height=batch.height,
            width=batch.width,
            guidance_scale=float(batch.guidance_scale),
            do_classifier_free_guidance=bool(batch.do_classifier_free_guidance),
            quality=batch.quality,
            attention_backend=server_args.attention_backend,
            dtype=str(self.transformer.proj_out.weight.dtype).removeprefix("torch."),
            scheduler=type(batch.scheduler).__name__,
            scheduler_config_sha256=config_digest(batch.scheduler.config),
            timesteps=tuple(batch.timesteps.float().cpu().tolist()),
            sigmas=tuple(batch.scheduler.sigmas.float().cpu().tolist()),
        )

    def _check_dpcache_supported(self, batch, server_args):
        # v1 is validated only for single-image BF16 text-to-image without CFG;
        # anything else would compose approximations or change what was calibrated.
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        unsupported = {
            "torch compile": server_args.enable_torch_compile,
            "breakable CUDA graphs": server_args.enable_breakable_cuda_graph,
            "Cache-DiT": self._cache_dit_requested_for_batch(batch),
            "TeaCache": batch.enable_teacache,
            "Spectrum": batch.enable_spectrum,
            "skip-softmax attention": batch.skip_softmax_params is not None,
            "attention backend override": batch.attention_backend_override is not None,
            "classifier-free guidance": batch.do_classifier_free_guidance,
            "non-lossless quality": batch.quality != "lossless",
            "reference images": batch.condition_image is not None,
            "more than one image per request": len(prompts) != 1
            or batch.num_outputs_per_prompt != 1,
            "LoRA": _has_active_lora(self.transformer),
            "non-BF16 transformer": self.transformer.proj_out.weight.dtype
            != torch.bfloat16,
            "CFG parallel": server_args.enable_cfg_parallel,
            "sequence/tensor parallel": get_sp_world_size() > 1
            or get_tp_world_size() > 1,
        }
        enabled = [name for name, on in unsupported.items() if on]
        if enabled:
            raise ValueError(f"dpcache_schedule cannot be combined with {enabled}")

    def _bcg_pad_prompt_kwargs(
        self, call_kwargs, current_model=None, force_bucket=None
    ):
        # Prefill runs eagerly. Later steps use exact-length prefix KV, so text
        # padding only creates duplicate graphs without enabling more replay.
        return call_kwargs

    def _predict_noise(
        self,
        current_model,
        latent_model_input,
        timestep,
        target_dtype,
        guidance,
        **kwargs,
    ):
        caches = kwargs["prefix_caches"]
        if caches is not None and not caches[0][0]:
            # prefill is request-specific; graph replay must only see populated cache tensors
            return current_model(
                hidden_states=latent_model_input, timestep=timestep, **kwargs
            )
        return super()._predict_noise(
            current_model,
            latent_model_input,
            timestep,
            target_dtype,
            guidance,
            **kwargs,
        )
