# SPDX-License-Identifier: Apache-2.0
"""Bit-exact fused ``silu(a) * b`` over two same-shape tensors.

For SwiGLU MLPs whose gate/up projections are separate GEMMs (so the
concatenated-input ``silu_and_mul`` kernels don't apply without an extra
full-width ``cat`` pass), this fuses the eager pair

    ``s = F.silu(a)`` (one kernel) ``out = s * b`` (another kernel)

into one pass while reproducing both aten bf16 rounding boundaries:
``silu`` is a single aten op (fp32 opmath, one round), the multiply rounds
once more.  ``tl.sigmoid`` lowers to the same fp32 sigmoid aten uses, which
makes the replication exact (verified ``torch.equal`` on 1M random bf16
values); callers still verify the first call and fall back on mismatch.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32
from sglang.srt.utils.custom_op import register_custom_op

# One program holds a whole row in registers (fp32), so rows are bounded.
_FP8_ROW_MAX = 16384


@triton.jit
def _silu_mul_kernel(
    out_ptr,
    a_ptr,
    b_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = round_bf16_to_fp32(a * tl.sigmoid(a))
    tl.store(out_ptr + offs, s * b, mask=mask)  # store rounds the multiply


@triton.jit
def _scaled_silu_mul_kernel(
    out_ptr,
    a_ptr,
    b_ptr,
    row_scale_ptr,
    a_col_scale_ptr,
    b_col_scale_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    # a and b are unscaled FP8 GEMM products [M, N]; scale and round them to
    # bf16 exactly as the GEMM route's scale pass (x * (sa * sb)) would, then
    # round as the eager silu / multiply chain does.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    row_scale = tl.load(row_scale_ptr + row)
    a_scale = tl.load(a_col_scale_ptr + cols, mask=mask, other=0.0)
    b_scale = tl.load(b_col_scale_ptr + cols, mask=mask, other=0.0)
    a = tl.load(a_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    a = (a * (row_scale * a_scale)).to(out_ptr.dtype.element_ty).to(tl.float32)
    b = (b * (row_scale * b_scale)).to(out_ptr.dtype.element_ty).to(tl.float32)
    s = round_bf16_to_fp32(a * tl.sigmoid(a))
    tl.store(out_ptr + row * N + cols, s * b, mask=mask)


@triton.jit
def _scaled_silu_mul_fp8_kernel(
    q_ptr,
    q_scale_ptr,
    a_ptr,
    b_ptr,
    row_scale_ptr,
    a_col_scale_ptr,
    b_col_scale_ptr,
    N,
    BLOCK_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    # One program holds one whole row: the per-token scale needs the row's
    # absolute maximum before any element is quantized.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    row_scale = tl.load(row_scale_ptr + row)
    a_scale = tl.load(a_col_scale_ptr + cols, mask=mask, other=0.0)
    b_scale = tl.load(b_col_scale_ptr + cols, mask=mask, other=0.0)
    a = tl.load(a_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    a = round_bf16_to_fp32(a * (row_scale * a_scale))
    b = round_bf16_to_fp32(b * (row_scale * b_scale))
    hidden = round_bf16_to_fp32(round_bf16_to_fp32(a * tl.sigmoid(a)) * b)
    # per_token_quant_fp8_warp_kernel's arithmetic (IEEE divisions, no
    # fast-math in the JIT build): scale = amax / 448, x * (1 / scale).
    scale = tl.math.div_rn(tl.max(tl.abs(hidden), 0), FP8_MAX)
    tl.store(q_scale_ptr + row, scale)
    inverse = tl.where(scale == 0.0, 0.0, tl.math.div_rn(1.0, scale))
    quantized = tl.clamp(hidden * inverse, -FP8_MAX, FP8_MAX)
    tl.store(q_ptr + row * N + cols, quantized.to(q_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _packed_silu_mul_kernel(
    out_ptr,
    x_ptr,
    num_rows,
    row_stride,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = (row < num_rows) & (cols < D)
    row_base = row * row_stride
    a = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + row_base + D + cols, mask=mask, other=0.0).to(tl.float32)
    s = round_bf16_to_fp32(a * tl.sigmoid(a))
    tl.store(out_ptr + row * D + cols, s * b, mask=mask)


def can_use_fused_silu_mul(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.dtype is torch.bfloat16
        and b.dtype is torch.bfloat16
        and a.is_cuda
        and b.is_cuda
        and a.device == b.device
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
        and a.numel() > 0
    )


def _fake_silu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a)


@register_custom_op(
    op_name="triton_fused_silu_mul_bitexact",
    mutates_args=[],
    fake_impl=_fake_silu_mul,
)
def fused_silu_mul_bitexact(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``silu(a) * b``, bit-exact vs the eager two-kernel chain."""
    out = torch.empty_like(a)
    numel = a.numel()
    with torch.cuda.device(a.device):
        _silu_mul_kernel[(triton.cdiv(numel, 1024),)](
            out,
            a,
            b,
            numel,
            BLOCK=1024,
        )
    return out


def can_use_fused_scaled_silu_mul(a, b, row_scale, a_col_scale, b_col_scale) -> bool:
    return (
        can_use_fused_silu_mul(a, b)
        and a.dim() == 2
        and all(
            x.is_cuda and x.dtype is torch.float32 and x.is_contiguous()
            for x in (row_scale, a_col_scale, b_col_scale)
        )
        and row_scale.numel() == a.shape[0]
        and a_col_scale.numel() == b_col_scale.numel() == a.shape[1]
    )


def fused_scaled_silu_mul(
    a: torch.Tensor,
    b: torch.Tensor,
    row_scale: torch.Tensor,
    a_col_scale: torch.Tensor,
    b_col_scale: torch.Tensor,
) -> torch.Tensor:
    """``silu(a') * b'`` with ``x' = bf16(x * (row_scale[m] * x_col_scale[n]))``.

    a and b are [M, N] unscaled FP8 GEMM products (the SM120 cuBLASLt route's
    deferred scale). x' is rounded as the route's scale pass rounds it, so the
    result is bitwise equal to fused_silu_mul_bitexact of the scaled GEMMs.
    """
    assert can_use_fused_scaled_silu_mul(a, b, row_scale, a_col_scale, b_col_scale)
    m, n = a.shape
    out = torch.empty_like(a)
    block_n = 1024
    with torch.cuda.device(a.device):
        _scaled_silu_mul_kernel[(m, triton.cdiv(n, block_n))](
            out, a, b, row_scale, a_col_scale, b_col_scale, n, BLOCK_N=block_n
        )
    return out


def can_use_fused_scaled_silu_mul_fp8(
    a, b, row_scale, a_col_scale, b_col_scale
) -> bool:
    return (
        can_use_fused_scaled_silu_mul(a, b, row_scale, a_col_scale, b_col_scale)
        and a.dtype is torch.bfloat16
        and a.shape[1] <= _FP8_ROW_MAX
    )


def fused_scaled_silu_mul_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    row_scale: torch.Tensor,
    a_col_scale: torch.Tensor,
    b_col_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``sglang_per_token_quant_fp8(fused_scaled_silu_mul(...))`` in one pass.

    Returns ``(q [M, N] float8_e4m3fn, q_scale [M, 1] fp32)``, bitwise what the
    bf16 SiLU-mul followed by the per-token FP8 quantization of the next
    linear's input produces, without writing the bf16 intermediate. An
    all-zero row quantizes to zeros with scale 0, as the quantization's warp
    kernel does; its small-batch CTA kernel has no zero-scale guard.
    """
    assert can_use_fused_scaled_silu_mul_fp8(a, b, row_scale, a_col_scale, b_col_scale)
    m, n = a.shape
    q = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=a.device)
    q_scale = torch.empty((m, 1), dtype=torch.float32, device=a.device)
    block_n = triton.next_power_of_2(n)
    with torch.cuda.device(a.device):
        _scaled_silu_mul_fp8_kernel[(m,)](
            q,
            q_scale,
            a,
            b,
            row_scale,
            a_col_scale,
            b_col_scale,
            n,
            BLOCK_N=block_n,
            FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
            num_warps=16,
        )
    return q, q_scale


def fused_packed_silu_mul_bitexact(x: torch.Tensor) -> torch.Tensor:
    """Bit-exact SwiGLU over a contiguous packed ``[..., 2 * D]`` input."""
    if not (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and x.dim() == 3
        and x.stride(-1) == 1
        and x.stride(-2) >= x.shape[-1]
        and x.stride(0) == x.shape[1] * x.stride(1)
        and x.shape[-1] % 2 == 0
        and x.numel() > 0
    ):
        raise RuntimeError("unsupported input for packed fused SiLU-mul")
    hidden = x.shape[-1] // 2
    rows = x.numel() // x.shape[-1]
    row_stride = x.stride(-2)
    out = torch.empty((*x.shape[:-1], hidden), dtype=x.dtype, device=x.device)
    with torch.cuda.device(x.device):
        _packed_silu_mul_kernel[(rows, triton.cdiv(hidden, 1024))](
            out,
            x,
            rows,
            row_stride,
            D=hidden,
            BLOCK=1024,
        )
    return out
