# Copyright 2026 Qwen-Image Team and The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    can_use_fused_complex_rope,
    can_use_fused_layernorm_modulate,
    can_use_fused_scaled_silu_mul,
    can_use_fused_scaled_silu_mul_fp8,
    can_use_fused_silu_mul,
    can_use_rmsnorm_preserve_reduction,
    fused_complex_rope,
    fused_layernorm_modulate,
    fused_scaled_silu_mul,
    fused_scaled_silu_mul_fp8,
    fused_silu_mul_bitexact,
    residual_gate_add,
    rmsnorm_preserve_reduction,
    tensors_equal,
)
from sglang.kernels.ops.diffusion.attention.sageattn_prequant_triton import (
    can_use_sage_prequant_attention,
    sage_prequant_attention,
)
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_kv_triton import (
    can_use_qknorm_complex_rope_kv,
    qknorm_complex_rope_kv,
)
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_triton import (
    can_use_qknorm_complex_rope,
    qknorm_complex_rope,
)
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.layers.layernorm import RMSNorm

logger = init_logger(__name__)
_ROPE_FUSION = BitExactFusionGate("Qwen-Image 2.1 complex RoPE")
_SILU_MUL_FUSION = BitExactFusionGate("Qwen-Image 2.1 SiLU-mul")
_QK_ROPE_FUSION = BitExactFusionGate("Qwen-Image 2.1 Q/K RMSNorm + complex RoPE")
_KV_ROPE_FUSION = BitExactFusionGate("Qwen-Image 2.1 K RMSNorm + RoPE + KV packing")
_QK_NORM_FUSION = BitExactFusionGate("Qwen-Image 2.1 Q/K RMSNorm")
_MODULATION_FUSION = BitExactFusionGate("Qwen-Image 2.1 LayerNorm modulation")
_SAGE_PREQUANT_FUSION = BitExactFusionGate(
    "Qwen-Image 2.1 SageAttention2 operands from the Q/K/V kernels"
)
_QKV_SCALE_FUSION = BitExactFusionGate(
    "Qwen-Image 2.1 QKV FP8 GEMM scale pass in the SageAttention2 Q/K/V kernels"
)
_FFN_SCALE_FUSION = BitExactFusionGate(
    "Qwen-Image 2.1 gate/up FP8 GEMM scale pass in the SiLU-mul"
)
_DOWN_INPUT_QUANT_FUSION = BitExactFusionGate(
    "Qwen-Image 2.1 down-projection FP8 input quantization in the SiLU-mul"
)


def build_layout(image_slots, image_shapes, axes_dims, device):
    """Expand each condition-image slot to its complete latent grid before denoising."""
    indices, image_indices, positions, segments = [], [], [], []
    cursor = position = image_index = 0
    for text_index, is_image in enumerate(image_slots):
        if not is_image:
            indices.append(text_index)
            positions.append((position, position, position))
            position += 1
            continue
        start = len(indices)
        if start > cursor:
            segments.append((cursor, start, False))
        _, height, width = image_shapes[image_index]
        for h in range(-(height - height // 2), height // 2):
            for w in range(-(width - width // 2), width // 2):
                indices.append(text_index)
                image_indices.append(len(indices) - 1)
                positions.append((position, h, w))
        segments.append((start, len(indices), True))
        cursor = len(indices)
        position += max(height, width)
        image_index += 1
    if image_index != len(image_shapes) - 1:
        raise ValueError("condition-image slots do not match image_shapes")
    if len(indices) > cursor:
        segments.append((cursor, len(indices), False))
    prefix_len = len(indices)
    _, height, width = image_shapes[-1]
    for h in range(-(height - height // 2), height // 2):
        for w in range(-(width - width // 2), width // 2):
            positions.append((position, h, w))
    pos = torch.tensor(positions, device=device, dtype=torch.float32)
    angles = torch.cat(
        [
            pos[:, axis : axis + 1]
            * (10000.0 ** (-torch.arange(0, dim, 2, device=device).float() / dim))
            for axis, dim in enumerate(axes_dims)
        ],
        dim=-1,
    )
    rope = torch.polar(torch.ones_like(angles), angles)
    return dict(
        encoder_seq_len=len(image_slots),
        text_indices=torch.tensor(indices, device=device, dtype=torch.long),
        image_indices=torch.tensor(image_indices, device=device, dtype=torch.long),
        prefix_rope=rope[:prefix_len],
        target_rope=rope[prefix_len:],
        segments=tuple(segments),
    )


def apply_rope(x, rope):
    fused = None
    if can_use_fused_complex_rope(x, rope) and _ROPE_FUSION.can_attempt_once():
        fused = fused_complex_rope(x, rope)
        if _ROPE_FUSION.verified:
            return fused
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)
    if fused is not None:
        return _ROPE_FUSION.accept_or_fallback(fused, out, logger=logger)
    return out


def apply_qk_norm(x, norm):
    fused = None
    if (
        can_use_rmsnorm_preserve_reduction(x, norm.weight)
        and _QK_NORM_FUSION.can_attempt_once()
    ):
        fused = rmsnorm_preserve_reduction(x, norm.weight, norm.variance_epsilon)
        if _QK_NORM_FUSION.verified:
            return fused
    out = norm(x)
    if fused is not None:
        return _QK_NORM_FUSION.accept_or_fallback(fused, out, logger=logger)
    return out


def apply_qk_norm_rope(x, norm, rope):
    fused = None
    if (
        can_use_qknorm_complex_rope(x, norm.weight, rope)
        and _QK_ROPE_FUSION.can_attempt_once()
    ):
        fused = qknorm_complex_rope(x, norm.weight, rope, norm.variance_epsilon)
        if _QK_ROPE_FUSION.verified:
            return fused
    out = apply_rope(apply_qk_norm(x, norm), rope)
    if fused is not None:
        return _QK_ROPE_FUSION.accept_or_fallback(fused, out, logger=logger)
    return out


def apply_modulation(x, norm, scale):
    fused = None
    if (
        can_use_fused_layernorm_modulate(x, scale.squeeze(1), None)
        and _MODULATION_FUSION.can_attempt_once()
    ):
        fused = fused_layernorm_modulate(x, scale.squeeze(1), None, norm.eps)
        if _MODULATION_FUSION.verified:
            return fused
    out = norm(x) * (1 + scale)
    if fused is not None:
        return _MODULATION_FUSION.accept_or_fallback(fused, out, logger=logger)
    return out


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        scale = self.weight.float() + 1
        value = x.float()
        return (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
            * scale
        ).to(x.dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_dim, dim, eps):
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_dim, eps)
        self.in_layer = nn.Linear(context_dim, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(
            nn.functional.gelu(self.in_layer(self.text_norm(x)), approximate="tanh")
        )


class QwenImage21TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_embedder = nn.Module()
        self.timestep_embedder.linear_1 = nn.Linear(256, dim, bias=False)
        self.timestep_embedder.linear_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, t, dtype):
        freq = torch.exp(
            -math.log(10000) * torch.arange(128, device=t.device).float() / 128
        )
        angles = t.float()[:, None] * 1000 * freq
        x = torch.cat([angles.cos(), angles.sin()], dim=-1).to(dtype)
        return self.timestep_embedder.linear_2(
            nn.functional.silu(self.timestep_embedder.linear_1(x))
        )


def _deferred_scale_linear(linear, x, quantized_input=None):
    """The FP8 linear's unscaled product and its scales, or None.

    Only for a plain FP8 linear (no LoRA wrapper, no bias, one rank) on SM120's
    cuBLASLt route. Returns ``(unscaled [M, N], row_scale [M], col_scale [N],
    quantized_input)``; the consumer applies ``row_scale[m] * col_scale[n]`` in
    fp32 as it loads, which replaces the route's separate scale pass.
    """
    if (
        type(linear) not in (ColumnParallelLinear, MergedColumnParallelLinear)
        or linear.bias is not None
        or get_tp_world_size() != 1
        or not isinstance(linear.quant_method, Fp8LinearMethod)
    ):
        return None
    return linear.quant_method.apply_deferred_scale(linear, x, quantized_input)


def _takes_per_token_quantized_input(linear):
    """Whether linear can consume an input its producer quantized per token."""
    return (
        type(linear) is RowParallelLinear
        and linear.bias is None
        and get_tp_world_size() == 1
        and isinstance(linear.quant_method, Fp8LinearMethod)
    )


def _uses_per_token_quant_warp_kernel(tokens, device):
    """Whether sglang_per_token_quant_fp8 takes its warp kernel for this many
    tokens (per_token_quant_fp8.cuh: at least 2 x SMs x 8). The CTA kernel it
    uses below that has no zero-scale guard, so an all-zero row there comes out
    as 448 instead of 0; the fused SiLU-mul reproduces the warp kernel only."""
    return tokens >= torch.cuda.get_device_properties(device).multi_processor_count * 16


def _quantized_equal(a, b):
    """Bitwise equality of two (FP8 values, fp32 scales) pairs."""
    return torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8)) and torch.equal(
        a[1], b[1]
    )


def _scale_qkv_view(t, row_scale, col_scale):
    """A [B, S, H, D] view of an unscaled GEMM product, scaled and rounded to
    bf16 as the GEMM route's scale pass does: ``bf16(x * (sa * sb))``."""
    batch, seq, heads, dim = t.shape
    scale = row_scale.view(batch, seq, 1, 1) * col_scale.view(1, 1, heads, dim)
    return (t.float() * scale).to(t.dtype)


def _scale_rows_cols(x, row_scale, col_scale):
    """``bf16(x * (row_scale[m] * col_scale[n]))`` of an [M, N] unscaled product."""
    return (x.float() * (row_scale[:, None] * col_scale[None, :])).to(x.dtype)


class QwenImage21FeedForward(nn.Module):
    def __init__(self, dim, ratio, quant_config, prefix):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim,
            dim * ratio,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.gate_layer = ColumnParallelLinear(
            dim,
            dim * ratio,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_layer",
        )
        self.out = RowParallelLinear(
            dim * ratio,
            dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out",
        )

    def _deferred_mlp(self, x):
        """out(silu(gate) * value) from unscaled gate/value GEMMs, or None.

        Both GEMMs share one quantization of x; the scaled SiLU-mul applies
        their scales on load with the scale pass's rounding. Runs once the
        plain SiLU-mul fusion is verified, and is checked bitwise against the
        scale pass + plain SiLU-mul on first sight.
        """
        gate = _deferred_scale_linear(self.gate_layer, x)
        if gate is None:
            return None
        value = _deferred_scale_linear(self.proj, x, quantized_input=gate[3])
        if value is None:
            return None
        args = (gate[0], value[0], gate[1], gate[2], value[2])
        if not can_use_fused_scaled_silu_mul(*args):
            return None
        out = self._silu_mul_fp8_down(args)
        if out is None:
            hidden = fused_scaled_silu_mul(*args)
            if not _FFN_SCALE_FUSION.verified:
                reference = fused_silu_mul_bitexact(
                    _scale_rows_cols(gate[0], gate[1], gate[2]),
                    _scale_rows_cols(value[0], value[1], value[2]),
                )
                hidden = _FFN_SCALE_FUSION.accept_or_fallback(
                    hidden, reference, logger=logger
                )
            out = self.out(hidden)[0]
        return out.view(*x.shape[:-1], out.shape[-1])

    def _silu_mul_fp8_down(self, args):
        """The down linear fed by a SiLU-mul that writes its FP8 input, or None.

        The SiLU-mul emits the per-token FP8 values and scales the down
        linear would compute from its bf16 output, so the bf16 intermediate
        and the separate quantization pass disappear. Runs once the bf16
        deferred path is verified, and is checked bitwise against it followed
        by the per-token quantization on first sight.
        """
        if not (
            _FFN_SCALE_FUSION.verified
            and can_use_fused_scaled_silu_mul_fp8(*args)
            and _uses_per_token_quant_warp_kernel(args[0].shape[0], args[0].device)
            and _takes_per_token_quantized_input(self.out)
            and _DOWN_INPUT_QUANT_FUSION.can_attempt_once()
        ):
            return None
        quantized = fused_scaled_silu_mul_fp8(*args)
        if not _DOWN_INPUT_QUANT_FUSION.verified:
            reference = sglang_per_token_quant_fp8(fused_scaled_silu_mul(*args))
            quantized = _DOWN_INPUT_QUANT_FUSION.accept_or_fallback(
                quantized, reference, equal=_quantized_equal, logger=logger
            )
        return self.out.quant_method.apply_per_token_quantized(self.out, *quantized)

    def forward(self, x):
        if _SILU_MUL_FUSION.verified and _FFN_SCALE_FUSION.can_attempt_once():
            out = self._deferred_mlp(x)
            if out is not None:
                return out
        gate, value = self.gate_layer(x)[0], self.proj(x)[0]
        fused = None
        if can_use_fused_silu_mul(gate, value) and _SILU_MUL_FUSION.can_attempt_once():
            fused = fused_silu_mul_bitexact(gate, value)
            if _SILU_MUL_FUSION.verified:
                return self.out(fused)[0]
        hidden = nn.functional.silu(gate) * value
        if fused is not None:
            hidden = _SILU_MUL_FUSION.accept_or_fallback(fused, hidden, logger=logger)
        return self.out(hidden)[0]


class QwenImage21Attention(nn.Module):
    def __init__(self, ac, quant_config, prefix):
        super().__init__()
        dim = ac.hidden_size
        self.heads = ac.num_attention_heads // get_tp_world_size()
        self.head_dim = ac.attention_head_dim
        # One N = 3 * dim GEMM (and one activation quantization) instead of three.
        # The checkpoint's to_q/to_k/to_v are merged at load (param_names_mapping).
        self.to_qkv = MergedColumnParallelLinear(
            dim,
            [dim] * 3,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_qkv",
        )
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    dim,
                    dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0",
                )
            ]
        )
        self.norm_q = RMSNorm(
            self.head_dim, ac.eps, cast_x_before_out_mul=True, force_native=True
        )
        self.norm_k = RMSNorm(
            self.head_dim, ac.eps, cast_x_before_out_mul=True, force_native=True
        )
        backends = QwenImage21Transformer2DModel._supported_attention_backends
        self.local_attn = LocalAttention(
            self.heads, self.head_dim, supported_attention_backends=backends
        )
        self.target_attn = USPAttention(
            self.heads, self.head_dim, supported_attention_backends=backends
        )

    def project_qkv(self, x):
        # Views into the merged output: each token's heads stay contiguous, and
        # tokens are 3 * dim apart, which the fused Q/K kernels accept.
        qkv = self.to_qkv(x)[0].unflatten(-1, (3, self.heads, self.head_dim))
        return qkv.unbind(-3)

    def _deferred_qkv(self, x):
        """Unscaled q, k, v views plus (row_scale, col_scale) for the SA2 prequant path, or None.

        Only once that path is verified: its Q/K/V kernels apply the scales on
        load, which replaces the GEMM route's separate scale pass.
        """
        if not (
            get_sp_world_size() == 1
            and self.target_attn.backend == AttentionBackendEnum.SAGE_ATTN
            and _SAGE_PREQUANT_FUSION.verified
            and _QKV_SCALE_FUSION.can_attempt_once()
        ):
            return None
        deferred = _deferred_scale_linear(self.to_qkv, x)
        if deferred is None:
            return None
        unscaled, row_scale, col_scale, _ = deferred
        qkv = unscaled.view(*x.shape[:-1], 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(-3)
        return q, k, v, row_scale, col_scale

    def qkv(self, x, rope):
        q, k, v = self.project_qkv(x)
        return (
            apply_qk_norm_rope(q, self.norm_q, rope),
            apply_qk_norm_rope(k, self.norm_k, rope),
            # The prefix V becomes the cached V prefix, which the fused K/V
            # packing kernel reads as a dense buffer.
            v.contiguous(),
        )

    def attend_sample(
        self, q, k, v, rope, prefix, prefix_rope, segments, cache, scales=None
    ):
        """scales: (row_scale [S], qkv_col_scale [3 * H * D]) when q, k, v are
        views of an unscaled QKV GEMM product (_deferred_qkv), else None."""
        if cache:
            kp, vp = cache["key"], cache["value"]
            prefix_output = None
        else:
            qp, kp, vp = self.qkv(prefix, prefix_rope)
            outputs = []
            # text runs are causal; image blocks see the entire preceding sequence and themselves
            for start, end, is_image in segments:
                mask = None
                if not is_image:
                    mask = (
                        torch.arange(end, device=q.device)[None, :]
                        <= torch.arange(start, end, device=q.device)[:, None]
                    )
                    mask = mask[None, None]
                outputs.append(
                    self.local_attn(
                        qp[:, start:end], kp[:, :end], vp[:, :end], attn_mask=mask
                    )
                )
            prefix_output = self.to_out[0](torch.cat(outputs, dim=1).flatten(2))[0]
            if cache is not None:
                cache.update(key=kp, value=vp)
        can_prequant = (
            get_sp_world_size() == 1
            and self.target_attn.backend == AttentionBackendEnum.SAGE_ATTN
            and can_use_sage_prequant_attention(
                q, self.norm_q.weight, k, self.norm_k.weight, rope, v, kp, vp
            )
        )
        if scales is not None:
            if can_prequant and _SAGE_PREQUANT_FUSION.verified:
                row_scale, qkv_col_scale = scales
                out = sage_prequant_attention(
                    q,
                    self.norm_q.weight,
                    self.norm_q.variance_epsilon,
                    k,
                    self.norm_k.weight,
                    self.norm_k.variance_epsilon,
                    rope,
                    v,
                    kp,
                    vp,
                    self.target_attn.softmax_scale,
                    row_scale=row_scale,
                    qkv_col_scale=qkv_col_scale,
                )
                if not _QKV_SCALE_FUSION.verified:
                    width = self.heads * self.head_dim
                    qs, ks, vs = (
                        _scale_qkv_view(
                            t, row_scale, qkv_col_scale[i * width : (i + 1) * width]
                        )
                        for i, t in enumerate((q, k, v))
                    )
                    reference = sage_prequant_attention(
                        qs,
                        self.norm_q.weight,
                        self.norm_q.variance_epsilon,
                        ks,
                        self.norm_k.weight,
                        self.norm_k.variance_epsilon,
                        rope,
                        vs,
                        kp,
                        vp,
                        self.target_attn.softmax_scale,
                    )
                    out = _QKV_SCALE_FUSION.accept_or_fallback(
                        out, reference, logger=logger
                    )
                return out, prefix_output
            width = self.heads * self.head_dim
            row_scale, qkv_col_scale = scales
            q, k, v = (
                _scale_qkv_view(
                    t, row_scale, qkv_col_scale[i * width : (i + 1) * width]
                )
                for i, t in enumerate((q, k, v))
            )
        if can_prequant and _SAGE_PREQUANT_FUSION.can_attempt_once():
            fused = sage_prequant_attention(
                q,
                self.norm_q.weight,
                self.norm_q.variance_epsilon,
                k,
                self.norm_k.weight,
                self.norm_k.variance_epsilon,
                rope,
                v,
                kp,
                vp,
                self.target_attn.softmax_scale,
            )
            if _SAGE_PREQUANT_FUSION.verified:
                return fused, prefix_output
            reference = self.attend_target(q, k, v, rope, kp, vp)
            out = _SAGE_PREQUANT_FUSION.accept_or_fallback(
                fused, reference, logger=logger
            )
            return out, prefix_output
        return self.attend_target(q, k, v, rope, kp, vp), prefix_output

    def attend_target(self, q, k, v, rope, kp, vp):
        q = apply_qk_norm_rope(q, self.norm_q, rope)
        packed = None
        if (
            get_sp_world_size() == 1
            and can_use_qknorm_complex_rope_kv(k, self.norm_k.weight, rope, v, kp, vp)
            and _KV_ROPE_FUSION.can_attempt_once()
        ):
            packed = qknorm_complex_rope_kv(
                k, self.norm_k.weight, rope, v, kp, vp, self.norm_k.variance_epsilon
            )
            if not _KV_ROPE_FUSION.verified:
                reference = (
                    torch.cat([kp, apply_rope(apply_qk_norm(k, self.norm_k), rope)], 1),
                    torch.cat([vp, v], 1),
                )
                packed = _KV_ROPE_FUSION.accept_or_fallback(
                    packed,
                    reference,
                    equal=tensors_equal,
                    logger=logger,
                )
        if packed is not None:
            return self.target_attn(q, *packed)
        k = apply_qk_norm_rope(k, self.norm_k, rope)
        return self.target_attn.forward_with_replicated_kv_prefix(q, kp, vp, k, v)

    def forward(self, x, ropes, prefixes, layouts, caches):
        # batch target projections while retaining each sample's unpadded prefix
        deferred = self._deferred_qkv(x)
        if deferred is None:
            q, k, v = self.project_qkv(x)
            row_scale = col_scale = None
        else:
            q, k, v, row_scale, col_scale = deferred
        seq = x.shape[1]
        outputs, prefix_outputs = [], []
        for sample, layout in enumerate(layouts):
            scales = None
            if row_scale is not None:
                scales = (row_scale[sample * seq : (sample + 1) * seq], col_scale)
            out, prefix_out = self.attend_sample(
                q[sample : sample + 1],
                k[sample : sample + 1],
                v[sample : sample + 1],
                ropes[sample],
                prefixes[sample],
                layout["prefix_rope"],
                layout["segments"],
                caches[sample],
                scales,
            )
            outputs.append(out)
            prefix_outputs.append(prefix_out)
        return self.to_out[0](torch.cat(outputs).flatten(2))[0], prefix_outputs


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, ac, quant_config, prefix, layer_id):
        super().__init__()
        self._layer_id = layer_id
        self.img_norm1 = nn.LayerNorm(
            ac.hidden_size, eps=ac.eps, elementwise_affine=False
        )
        self.img_norm2 = nn.LayerNorm(
            ac.hidden_size, eps=ac.eps, elementwise_affine=False
        )
        self.attn = QwenImage21Attention(ac, quant_config, f"{prefix}.attn")
        self.img_mlp = QwenImage21FeedForward(
            ac.hidden_size, ac.mlp_ratio, quant_config, f"{prefix}.img_mlp"
        )

    def forward(
        self,
        hidden_states,
        modulation,
        prefix_states,
        prefix_modulation,
        layouts,
        ropes,
        caches,
    ):
        # Cache-DiT's UnifiedBlocks forwards the same args to every layer.
        # Slice here so prefix KV stays per-layer after that wrap.
        caches = [cache[self._layer_id] for cache in caches]
        scale1, gate1, scale2, gate2 = modulation
        prefixes = [
            apply_modulation(
                state["hidden_states"], self.img_norm1, prefix_modulation[0]
            )
            if not cache
            else None
            for state, cache in zip(prefix_states, caches, strict=True)
        ]
        attention, prefix_attentions = self.attn(
            apply_modulation(hidden_states, self.img_norm1, scale1),
            ropes,
            prefixes,
            layouts,
            caches,
        )
        hidden_states = residual_gate_add(hidden_states, attention, gate1)
        hidden_states = residual_gate_add(
            hidden_states,
            self.img_mlp(apply_modulation(hidden_states, self.img_norm2, scale2)),
            gate2,
        )
        for state, attention in zip(prefix_states, prefix_attentions, strict=True):
            if attention is not None:
                _, pg1, ps2, pg2 = prefix_modulation
                prefix = residual_gate_add(state["hidden_states"], attention, pg1)
                state["hidden_states"] = residual_gate_add(
                    prefix,
                    self.img_mlp(apply_modulation(prefix, self.img_norm2, ps2)),
                    pg2,
                )
        return hidden_states


class QwenImage21OutputNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x, temb):
        return apply_modulation(
            x, self.norm, self.linear(nn.functional.silu(temb))[:, None]
        )


class QwenImage21Transformer2DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    _supported_attention_backends = {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.SAGE_ATTN,
        AttentionBackendEnum.SAGE_ATTN_3,
        AttentionBackendEnum.TORCH_SDPA,
    }
    _fsdp_shard_conditions = [
        lambda name, module: isinstance(module, QwenImage21TransformerBlock)
    ]
    _compile_conditions = _fsdp_shard_conditions
    layer_names = ["transformer_blocks"]
    # Checkpoint weights and Diffusers LoRA A/B tensors of to_q/to_k/to_v both
    # map to shards of the merged to_qkv.
    param_names_mapping = {
        rf"^(transformer_blocks\.\d+\.attn)\.to_{name}\.(.+)$": (
            r"\1.to_qkv.\2",
            index,
            3,
        )
        for index, name in enumerate("qkv")
    }
    packed_modules_mapping = {"to_qkv": ["to_q", "to_k", "to_v"]}

    def __init__(self, config, hf_config, quant_config=None, **kwargs):
        super().__init__(config, hf_config=hf_config, **kwargs)
        ac = self.config
        if ac.patch_size != 1 or not ac.causal_condition or not ac.causal_block:
            raise ValueError(
                "Qwen-Image 2.1 requires patch_size=1, causal_condition=True and causal_block=True"
            )
        self.hidden_size = ac.hidden_size
        self.num_attention_heads = ac.num_attention_heads
        self.num_channels_latents = ac.in_channels
        self.img_in = nn.Linear(ac.in_channels, ac.hidden_size, bias=False)
        self.txt_in = QwenImage21TextProjection(
            ac.context_in_dim, ac.hidden_size, ac.eps
        )
        self.time_text_embed = QwenImage21TimeEmbedding(ac.hidden_size)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(ac.hidden_size, ac.hidden_size * 4, bias=False)
        )
        self.num_layers = ac.num_layers
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(
                    ac, quant_config, f"transformer_blocks.{i}", i
                )
                for i in range(ac.num_layers)
            ]
        )
        self.norm_out = QwenImage21OutputNorm(ac.hidden_size, ac.eps)
        self.proj_out = nn.Linear(ac.hidden_size, ac.out_channels, bias=False)

    def prepare_modulation(self, temb):
        # All blocks share these gates. Preserve the native tanh and its dtype,
        # but compute it once per timestep instead of once per block.
        scale1, gate1, scale2, gate2 = self.modulation(temb)[:, None].chunk(4, dim=-1)
        return scale1, gate1.tanh(), scale2, gate2.tanh()

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        layouts,
        condition_latents=None,
        prefix_caches=None,
        **kwargs,
    ):
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]
        sp = get_sp_world_size()
        target_len = hidden_states.shape[1]
        if target_len % sp:
            raise ValueError(
                f"target token count {target_len} must be divisible by SP degree {sp}"
            )
        local_len = target_len // sp
        rank = get_sp_parallel_rank()
        start, end = rank * local_len, (rank + 1) * local_len
        images = self.img_in(hidden_states[:, start:end])
        temb = self.time_text_embed((timestep.to(images.dtype) / 1000), images.dtype)
        modulation = self.prepare_modulation(temb)
        prefix_modulation = None
        if prefix_caches is None or any(not cache[0] for cache in prefix_caches):
            zero_temb = self.time_text_embed(
                timestep.new_zeros(1).to(images.dtype), images.dtype
            )
            prefix_modulation = self.prepare_modulation(zero_temb)
        if prefix_caches is None:
            prefix_caches = [[None] * self.num_layers for _ in layouts]
        prefix_states, ropes = [], []
        for sample, layout in enumerate(layouts):
            prefix = None
            if not prefix_caches[sample][0]:
                prefix = self.txt_in(
                    encoder_hidden_states[
                        sample : sample + 1, : layout["encoder_seq_len"]
                    ]
                ).index_select(1, layout["text_indices"])
                if condition_latents is not None:
                    prefix[:, layout["image_indices"]] = self.img_in(
                        condition_latents[sample : sample + 1]
                    )
            prefix_states.append({"hidden_states": prefix})
            ropes.append(layout["target_rope"][start:end])
        # Same extras for every block so Cache-DiT's UnifiedBlocks wrap is valid.
        # Each block slices prefix_caches by _layer_id. Visit once per layer for
        # layerwise offload.
        for block in self.transformer_blocks:
            images = block(
                images,
                modulation,
                prefix_states,
                prefix_modulation,
                layouts,
                ropes,
                prefix_caches,
            )
        output = self.proj_out(self.norm_out(images, temb))
        if sp > 1:
            output = sequence_model_parallel_all_gather(output, dim=1)
        return output


EntryClass = QwenImage21Transformer2DModel
