# SPDX-License-Identifier: Apache-2.0
"""Fuse 128-wide RMSNorm and complex RoPE with native rounding boundaries."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.rope.complex_rope_triton import _fuse_real_sin
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _qknorm_complex_rope_rows(
    x_ptr,
    weight_ptr,
    rope_ptr,
    row,
    ROWS: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    FUSE_REAL_SIN: tl.constexpr,
    row_scale_ptr=None,
    col_scale_ptr=None,
    HAS_SCALE: tl.constexpr = False,
):
    # A row is one head of one token: 128 contiguous values, with tokens
    # TOKEN_STRIDE elements apart (HEADS * 128 when x is contiguous).
    # HAS_SCALE=False compiles to the unscaled row function unchanged.
    # Four rows / four warps gives each lane four consecutive components.
    # Match aten's vectorized 128-wide FP32 mean: combine four components
    # left-to-right, then reduce 32 lanes with decreasing shuffle offsets.
    # Increasing rows per warp changes this order and is not bit-exact.
    column = tl.arange(0, 128)
    mask = row[:, None] < ROWS
    offset = row // HEADS * TOKEN_STRIDE + row % HEADS * 128
    value = tl.load(x_ptr + offset[:, None] + column[None, :], mask, 0).to(tl.float32)
    if HAS_SCALE:
        # x holds an unscaled FP8 GEMM product: apply the per-token (row_scale,
        # indexed by batch * SEQ + token) and per-channel (col_scale, indexed by
        # head * 128 + column) scales and round to x's dtype, bit for bit as the
        # GEMM route's scale pass (x * (sa * sb)) does.
        token_scale = tl.load(row_scale_ptr + row // HEADS, row < ROWS, 0.0)
        channel = row % HEADS * 128
        channel_scale = tl.load(
            col_scale_ptr + channel[:, None] + column[None, :], mask, 0.0
        )
        value = (
            (value * (token_scale[:, None] * channel_scale))
            .to(x_ptr.dtype.element_ty)
            .to(tl.float32)
        )
    square = tl.reshape(value * value, (4, 32, 2, 2))
    even, odd = tl.split(square)
    a, c = tl.split(even)
    b, d = tl.split(odd)
    variance = tl.sum(((a + b) + c) + d, 1) * (1.0 / 128)
    inv = tl.rsqrt(variance + EPS)
    weight = tl.load(weight_ptr + column).to(tl.float32)
    value = (value * inv[:, None]).to(x_ptr.dtype.element_ty).to(tl.float32)
    value = (value * weight[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)
    real, imag = tl.split(tl.reshape(value, (4, 64, 2)))
    token = row // HEADS % SEQ
    rotation = tl.load(rope_ptr + token[:, None] * 128 + column[None, :], mask, 0)
    cos, sin = tl.split(tl.reshape(rotation, (4, 64, 2)))
    out_real = tl.fma(real, cos, -imag * sin)
    if FUSE_REAL_SIN:
        out_imag = tl.fma(real, sin, imag * cos)
    else:
        out_imag = tl.fma(imag, cos, real * sin)
    return tl.reshape(tl.join(out_real, out_imag), (4, 128))


@triton.jit
def _qknorm_complex_rope_onepass_kernel(
    x_ptr,
    weight_ptr,
    rope_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    FUSE_REAL_SIN: tl.constexpr,
):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    out = _qknorm_complex_rope_rows(
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
    )
    tl.store(
        out_ptr + row[:, None] * 128 + tl.arange(0, 128)[None, :],
        out,
        row[:, None] < ROWS,
    )


def token_stride(x):
    """Elements between consecutive tokens of a [batch, seq, heads, 128] x whose
    heads are dense within a token and whose tokens are evenly spaced (a
    contiguous tensor, or one of the q/k/v views of a merged QKV projection
    output); None for any other layout."""
    if x.ndim != 4 or x.shape[-1] != 128:
        return None
    batch_stride, stride, head_stride, dim_stride = x.stride()
    heads_dense = dim_stride == 1 and (head_stride == 128 or x.shape[2] == 1)
    tokens_even = batch_stride == x.shape[1] * stride or x.shape[0] == 1
    if not (heads_dense and tokens_even and stride >= x.shape[2] * 128):
        return None
    return stride


def can_use_qknorm_complex_rope(x, weight, rope):
    # can_use_rmsnorm_preserve_reduction and can_use_fused_complex_rope,
    # except that x only needs a token_stride, not a contiguous layout.
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.numel() > 0
        and token_stride(x) is not None
        and weight.device == x.device
        and weight.dtype == x.dtype
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
        and rope.dtype == torch.complex64
        and rope.device == x.device
        and rope.shape == (x.shape[1], x.shape[-1] // 2)
        and rope.is_contiguous()
    )


def _fake_qknorm_complex_rope(x, weight, rope, eps):
    return torch.empty_like(x)


@register_custom_op(
    op_name="qknorm_complex_rope",
    mutates_args=[],
    fake_impl=_fake_qknorm_complex_rope,
)
def qknorm_complex_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    rope: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert can_use_qknorm_complex_rope(x, weight, rope)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    with torch.cuda.device(x.device):
        _qknorm_complex_rope_onepass_kernel[(triton.cdiv(x.numel() // 128, 4),)](
            x,
            weight,
            torch.view_as_real(rope),
            out,
            x.numel() // 128,
            x.shape[1],
            x.shape[2],
            token_stride(x),
            eps,
            _fuse_real_sin(x.device),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
