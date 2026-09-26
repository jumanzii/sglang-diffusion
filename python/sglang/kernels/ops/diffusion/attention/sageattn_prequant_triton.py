# SPDX-License-Identifier: Apache-2.0
"""SageAttention2 operands produced by the DiT's own Q/K/V kernels.

SageAttention2's ``sageattn_qk_int8_pv_fp8_cuda`` (the sm_89/sm_120 path:
per-warp INT8 Q/K, FP8 V with ``pv_accum_dtype="fp32+fp16"``) first
preprocesses its BF16 inputs in separate passes. It quantizes Q to INT8 with
one scale per 32 tokens of a head, subtracts the K sequence mean and
quantizes K per 64 tokens, and transposes, pads, permutes and FP8-quantizes V
per channel. These kernels produce the same operands, bit for bit, from the
tensors Qwen-Image 2.1 already holds:

* ``qknorm_complex_rope_int8`` fuses the Q RMSNorm + complex RoPE with the
  per-warp INT8 quantization, so the BF16 Q never reaches memory.
* ``sage_v_fp8`` reads the cached prefix V and the step's target V through
  two pointers and writes SageAttention2's FP8 V layout with its per-channel
  scales, so the ``[prefix; target]`` BF16 V pack never exists.

The quantization arithmetic follows SageAttention 2.2.0's ``csrc/fused``
kernels (``QuantInt8Kernel``, ``TransposePadPermuteKernel``,
``MeanScaleKernel``), which are compiled with ``--use_fast_math``: the
divisions there are the approximate ``div.full.f32``, reproduced here with
inline PTX.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.rope.complex_rope_triton import _fuse_real_sin
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_triton import (
    _qknorm_complex_rope_rows,
    can_use_qknorm_complex_rope,
    token_stride,
)

# SageAttention2 constants for the sm_89/sm_120 per-warp path.
Q_WARP_TOKENS = 32
Q_BLOCK_TOKENS = 128
V_PAD_TOKENS = 64
V_SCALE_MAX = 2.25  # quant_v_scale_max for pv_accum_dtype="fp32+fp16"

# How the fast-math CUDA source divides: "full" (div.full.f32), "rn"
# (IEEE div.rn), or "approx" (div.approx.f32). Kept as a parameter so the
# unit check can confirm which one matches SageAttention2 bit for bit.
DIV_MODE = "full"


@triton.jit
def _fdiv(a, b, MODE: tl.constexpr):
    if MODE == "rn":
        return tl.math.div_rn(a, b)
    elif MODE == "approx":
        return tl.inline_asm_elementwise(
            "div.approx.f32 $0, $1, $2;",
            "=r,r,r",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        return tl.inline_asm_elementwise(
            "div.full.f32 $0, $1, $2;",
            "=r,r,r",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


@triton.jit
def _int8_rn(x):
    # cvt.rni.sat.s8.f32, as SageAttention2's float_to_int8_rn
    return tl.inline_asm_elementwise(
        "cvt.rni.sat.s8.f32 $0, $1;",
        "=r,r",
        [x],
        dtype=tl.int8,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _qknorm_complex_rope_int8_kernel(
    x_ptr,
    weight_ptr,
    rope_ptr,
    out_ptr,
    scale_ptr,
    ROWS: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALES_PER_HEAD: tl.constexpr,
    EPS: tl.constexpr,
    FUSE_REAL_SIN: tl.constexpr,
    DIV_MODE: tl.constexpr,
    row_scale_ptr=None,
    col_scale_ptr=None,
    HAS_SCALE: tl.constexpr = False,
):
    # One program quantizes 32 tokens of one head, like one SageAttention2
    # QuantInt8Kernel block. The rows go through the RMSNorm + RoPE row
    # function four at a time (its bit-exact layout), twice: once for the
    # absolute maximum and once to quantize.
    group = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    column = tl.arange(0, 128)
    amax = 0.0000001
    for chunk in tl.static_range(0, 32, 4):
        token = group * 32 + chunk + tl.arange(0, 4)
        row = (batch * SEQ + token) * HEADS + head
        row = tl.where(token < SEQ, row, ROWS)  # masked rows load zeros
        value = _qknorm_complex_rope_rows(
            x_ptr,
            weight_ptr,
            rope_ptr,
            row,
            ROWS,
            SEQ,
            HEADS,
            TOKEN_STRIDE,
            EPS,
            FUSE_REAL_SIN,
            row_scale_ptr,
            col_scale_ptr,
            HAS_SCALE,
        )
        value = value.to(x_ptr.dtype.element_ty).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.max(tl.abs(value), 1), 0))
    tl.store(
        scale_ptr + (batch * HEADS + head) * SCALES_PER_HEAD + group,
        _fdiv(amax, 127.0, DIV_MODE),
    )
    inverse = _fdiv(127.0, amax, DIV_MODE)
    for chunk in tl.static_range(0, 32, 4):
        token = group * 32 + chunk + tl.arange(0, 4)
        valid = token < SEQ
        row = (batch * SEQ + token) * HEADS + head
        row = tl.where(valid, row, ROWS)
        value = _qknorm_complex_rope_rows(
            x_ptr,
            weight_ptr,
            rope_ptr,
            row,
            ROWS,
            SEQ,
            HEADS,
            TOKEN_STRIDE,
            EPS,
            FUSE_REAL_SIN,
            row_scale_ptr,
            col_scale_ptr,
            HAS_SCALE,
        )
        value = value.to(x_ptr.dtype.element_ty).to(tl.float32)
        out_row = (batch * SEQ + token) * HEADS + head
        tl.store(
            out_ptr + out_row[:, None] * 128 + column[None, :],
            _int8_rn(value * inverse),
            valid[:, None],
        )


def can_use_qknorm_complex_rope_int8(x, weight, rope):
    return can_use_qknorm_complex_rope(x, weight, rope) and x.dtype == torch.bfloat16


def qknorm_complex_rope_int8(
    x, weight, rope, eps, div_mode=DIV_MODE, row_scale=None, col_scale=None
):
    """RMSNorm + complex RoPE of Q, quantized as SageAttention2's per-warp INT8.

    Returns ``(q_int8 [B, S, H, 128], q_scale [B, H, ceil(S / 128) * 4])``,
    the NHD tensors ``per_warp_int8(..., BLKQ=128, WARPQ=32)`` would return
    for the BF16 output of ``qknorm_complex_rope``. With ``row_scale``
    [B * S] and ``col_scale`` [H * 128], x is an unscaled FP8 GEMM product,
    scaled and rounded as it is loaded, bit for bit as the GEMM route's scale
    pass.
    """
    assert can_use_qknorm_complex_rope_int8(x, weight, rope)
    has_scale = row_scale is not None
    assert has_scale == (col_scale is not None)
    batch, seq, heads, _ = x.shape
    groups = triton.cdiv(seq, Q_BLOCK_TOKENS) * (Q_BLOCK_TOKENS // Q_WARP_TOKENS)
    out = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty((batch, heads, groups), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        _qknorm_complex_rope_int8_kernel[(groups, heads, batch)](
            x,
            weight,
            torch.view_as_real(rope),
            out,
            scale,
            batch * seq * heads,
            seq,
            heads,
            token_stride(x),
            groups,
            eps,
            _fuse_real_sin(x.device),
            div_mode,
            row_scale if has_scale else x,
            col_scale if has_scale else x,
            has_scale,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out, scale


@triton.jit
def _load_v(
    prefix_ptr,
    v_ptr,
    token,
    batch,
    head,
    PREFIX: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    V_TOKEN_STRIDE: tl.constexpr,
    row_scale_ptr=None,
    col_scale_ptr=None,
    HAS_SCALE: tl.constexpr = False,
):
    """[TOKENS, 128] of [prefix V; target V] for one head, as float; zero past the end.

    HAS_SCALE: the target V is an unscaled FP8 GEMM product, scaled here by
    row_scale[batch * SEQ + token] * col_scale[head * 128 + channel] and rounded
    to V's dtype, bit for bit as the GEMM route's scale pass.
    """
    column = tl.arange(0, 128)
    in_prefix = token < PREFIX
    in_target = (token >= PREFIX) & (token < PREFIX + SEQ)
    prefix_index = ((batch * PREFIX + token) * HEADS + head) * 128
    target_index = (batch * SEQ + token - PREFIX) * V_TOKEN_STRIDE + head * 128
    prefix = tl.load(
        prefix_ptr + prefix_index[:, None] + column[None, :], in_prefix[:, None], 0.0
    )
    target = tl.load(
        v_ptr + target_index[:, None] + column[None, :], in_target[:, None], 0.0
    )
    if HAS_SCALE:
        token_scale = tl.load(
            row_scale_ptr + batch * SEQ + token - PREFIX, in_target, 0.0
        )
        channel_scale = tl.load(col_scale_ptr + head * 128 + column)
        scale = token_scale[:, None] * channel_scale[None, :]
        target = (
            (target.to(tl.float32) * scale).to(v_ptr.dtype.element_ty).to(tl.float32)
        )
        return tl.where(in_prefix[:, None], prefix.to(tl.float32), target)
    return tl.where(in_prefix[:, None], prefix, target).to(tl.float32)


@triton.jit
def _sage_v_amax_kernel(
    prefix_ptr,
    v_ptr,
    amax_ptr,
    PREFIX: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    V_TOKEN_STRIDE: tl.constexpr,
    TOKENS: tl.constexpr,
    row_scale_ptr=None,
    col_scale_ptr=None,
    HAS_SCALE: tl.constexpr = False,
):
    # Per-channel absolute maximum over every token. Max is exact in any
    # order, so partial maxima combine with float atomics (the values are
    # non-negative) without changing the result.
    token = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    value = _load_v(
        prefix_ptr,
        v_ptr,
        token,
        batch,
        head,
        PREFIX,
        SEQ,
        HEADS,
        V_TOKEN_STRIDE,
        row_scale_ptr,
        col_scale_ptr,
        HAS_SCALE,
    )
    tl.atomic_max(
        amax_ptr + (batch * HEADS + head) * 128 + tl.arange(0, 128),
        tl.max(tl.abs(value), 0),
        sem="relaxed",
    )


@triton.jit
def _sage_v_fp8_kernel(
    prefix_ptr,
    v_ptr,
    amax_ptr,
    out_ptr,
    scale_ptr,
    PREFIX: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    V_TOKEN_STRIDE: tl.constexpr,
    PADDED: tl.constexpr,
    SCALE_MAX: tl.constexpr,
    TOKENS: tl.constexpr,
    DIV_MODE: tl.constexpr,
    row_scale_ptr=None,
    col_scale_ptr=None,
    HAS_SCALE: tl.constexpr = False,
):
    # The transposed, padded, permuted FP8 layout that SageAttention2's
    # TransposePadPermuteKernel + MeanScaleKernel produce, with MeanScaleKernel's
    # per-channel scale amax / scale_max.
    block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    channel = tl.arange(0, 128)
    amax = tl.load(amax_ptr + (batch * HEADS + head) * 128 + channel)
    if block == 0:
        tl.store(
            scale_ptr + (batch * HEADS + head) * 128 + channel,
            _fdiv(amax, SCALE_MAX, DIV_MODE),
        )
    inverse = _fdiv(tl.full((128,), SCALE_MAX, tl.float32), amax, DIV_MODE)
    # Within each 16 tokens, token m lands in column
    # (m // 8) * 2 + (m // 2) % 4 * 4 + m % 2 (the FP8 mma operand order): a bit
    # shuffle whose inverse gives the token for each output column c, so the
    # tile is loaded in output order and stored contiguously along tokens.
    column = block * TOKENS + tl.arange(0, TOKENS)
    c = column % 16
    token = column - c + (c // 2) % 2 * 8 + c // 8 * 4 + (c // 4) % 2 * 2 + c % 2
    value = _load_v(
        prefix_ptr,
        v_ptr,
        token,
        batch,
        head,
        PREFIX,
        SEQ,
        HEADS,
        V_TOKEN_STRIDE,
        row_scale_ptr,
        col_scale_ptr,
        HAS_SCALE,
    )
    quantized = tl.trans((value * inverse[None, :]).to(tl.float8e4nv))
    # out is [batch, 128 channels, heads, PADDED]
    index = ((batch * 128 + channel[:, None]) * HEADS + head) * PADDED + column[None, :]
    tl.store(out_ptr + index, quantized)


def can_use_sage_v_fp8(v_prefix, v):
    return (
        v.is_cuda
        and v.dtype == torch.bfloat16
        and token_stride(v) is not None
        and v_prefix.dtype == v.dtype
        and v_prefix.device == v.device
        and v_prefix.is_contiguous()
        and v_prefix.ndim == 4
        and v_prefix.shape[0] == v.shape[0]
        and v_prefix.shape[2:] == v.shape[2:]
        and v.shape[-1] == 128
    )


def sage_v_fp8(
    v_prefix, v, div_mode=DIV_MODE, tokens=64, row_scale=None, col_scale=None
):
    """SageAttention2's FP8 V for ``cat([v_prefix, v], 1)`` without the cat.

    Returns ``(v_fp8 [B, 128, H, padded], v_scale [B, H, 128])``, what
    ``per_channel_fp8(v, "NHD", scale_max=2.25, smooth_v=False)`` returns.
    With ``row_scale`` [B * S] and ``col_scale`` [H * 128], v is an unscaled
    FP8 GEMM product, scaled and rounded as it is loaded, bit for bit as the
    GEMM route's scale pass.
    """
    assert can_use_sage_v_fp8(v_prefix, v)
    has_scale = row_scale is not None
    assert has_scale == (col_scale is not None)
    scale_args = (row_scale, col_scale, True) if has_scale else (v, v, False)
    batch, seq, heads, dim = v.shape
    prefix = v_prefix.shape[1]
    padded = triton.cdiv(prefix + seq, V_PAD_TOKENS) * V_PAD_TOKENS
    amax = torch.zeros((batch, heads, dim), dtype=torch.float32, device=v.device)
    out = torch.empty(
        (batch, dim, heads, padded), dtype=torch.float8_e4m3fn, device=v.device
    )
    scale = torch.empty((batch, heads, dim), dtype=torch.float32, device=v.device)
    stride = token_stride(v)
    with torch.cuda.device(v.device):
        _sage_v_amax_kernel[(triton.cdiv(prefix + seq, tokens), heads, batch)](
            v_prefix,
            v,
            amax,
            prefix,
            seq,
            heads,
            stride,
            tokens,
            *scale_args,
            num_warps=2,
        )
        _sage_v_fp8_kernel[(padded // tokens, heads, batch)](
            v_prefix,
            v,
            amax,
            out,
            scale,
            prefix,
            seq,
            heads,
            stride,
            padded,
            V_SCALE_MAX,
            tokens,
            div_mode,
            *scale_args,
            num_warps=4,
        )
    return out, scale


def can_use_sage_prequant_attention(
    q, q_weight, k, k_weight, rope, v, k_prefix, v_prefix
):
    """The shapes and layouts sage_prequant_attention reproduces sageattn for.

    Only sm_120, where sageattn dispatches to the per-warp INT8 / FP8
    ``fp32+fp16`` path these kernels mirror.
    """
    from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_kv_triton import (
        can_use_qknorm_complex_rope_k,
    )

    return (
        q.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) == (12, 0)
        and can_use_qknorm_complex_rope_int8(q, q_weight, rope)
        and can_use_qknorm_complex_rope_k(k, k_weight, rope, k_prefix)
        and can_use_sage_v_fp8(v_prefix, v)
        and q.shape == k.shape == v.shape
    )


def sage_prequant_attention(
    q,
    q_weight,
    q_eps,
    k,
    k_weight,
    k_eps,
    rope,
    v,
    k_prefix,
    v_prefix,
    sm_scale,
    row_scale=None,
    qkv_col_scale=None,
):
    """``sageattn(qknorm_rope(q), [k_prefix; qknorm_rope(k)], [v_prefix; v])``.

    Same result bit for bit (NHD, non-causal, sm_120), with the Q quantization
    fused into the Q kernel, no BF16 Q or packed V in memory, and V quantized
    straight from its two sources. K keeps SageAttention2's own mean and
    quantization kernels, which need the whole packed K.

    With ``row_scale`` [B * S] and ``qkv_col_scale`` [3 * H * 128], q, k and v
    are views of an unscaled merged-QKV FP8 GEMM product (H16's deferred
    scale); each kernel applies its section of the scales on load with the scale
    pass's rounding, so the result is bitwise the scaled path's.
    """
    from sageattention import _fused, sm89_compile

    from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_kv_triton import (
        qknorm_complex_rope_k,
    )

    q_cs = k_cs = v_cs = None
    if row_scale is not None:
        width = q.shape[2] * q.shape[3]
        q_cs, k_cs, v_cs = (
            qkv_col_scale[i * width : (i + 1) * width] for i in range(3)
        )
    q_int8, q_scale = qknorm_complex_rope_int8(
        q, q_weight, rope, q_eps, row_scale=row_scale, col_scale=q_cs
    )
    k_full = qknorm_complex_rope_k(
        k, k_weight, rope, k_prefix, k_eps, row_scale=row_scale, col_scale=k_cs
    )
    k_mean = k_full.mean(dim=1)
    k_int8 = torch.empty(k_full.shape, dtype=torch.int8, device=k.device)
    k_scale = torch.empty(
        (k.shape[0], k.shape[2], triton.cdiv(k_full.shape[1], 64)),
        dtype=torch.float32,
        device=k.device,
    )
    _fused.quant_per_block_int8_fuse_sub_mean_cuda(
        k_full, k_mean, k_int8, k_scale, 64, 0
    )
    v_fp8, v_scale = sage_v_fp8(v_prefix, v, row_scale=row_scale, col_scale=v_cs)
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    # tensor_layout NHD (0), non-causal, per-warp granularity (2), no LSE
    sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
        q_int8, k_int8, v_fp8, out, q_scale, k_scale, v_scale, 0, 0, 2, sm_scale, 0
    )
    return out
