"""FP8 e4m3 GEMM on SM120's block-scaled (MXFP8) tensor-core path with unit scale factors.

On GeForce SM120 (RTX 5090) the block-scaled MMA (QMMA.SF) runs at twice the rate
of the plain FP8 MMA that CUTLASS's OpClassTensorOp FP8 kernels issue. With every
UE8M0 scale factor equal to 2^0 the product is the plain FP8 product, so this
reaches cuBLASLt's rate while keeping a CUTLASS epilogue: the per-token x
per-channel scales are applied in fp32 before the one bf16 rounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Below this many tokens the kernel was not measured; callers keep their route.
MIN_BLOCKSCALED_M = 1024
# UE8M0 encodes 2^(e - 127): 0x7f is 2^0.
_UE8M0_ONE = 0x7F


class Sm120BlockScaledConfig(NamedTuple):
    tile_m: int
    tile_n: int
    tile_k: int
    pingpong: bool
    streamk: bool
    max_swizzle: int


# Measured on an RTX 5090 (170 SMs) at M=4096 for the (N, K) of Qwen-Image 2.1's
# FP8 linears, (12288, 4096), (4096, 4096) and (4096, 12288): pingpong 128x128x128
# without StreamK is the fastest or within 0.5% of it at all three, with or
# without the scales. Only 128x128x128 and 128x64x128 compile (a K=64 tile breaks
# the scale-factor TMA), the auto carveout leaves 2 stages, and pingpong has no
# StreamK. Re-tune when the kernel or the GPU changes.
_DEFAULT_CONFIG = Sm120BlockScaledConfig(128, 128, 128, True, False, 1)


def select_sm120_blockscaled_config(m: int, n: int, k: int) -> Sm120BlockScaledConfig:
    """The tile, schedule and scheduler for an [m, k] x [k, n] GEMM."""
    return _DEFAULT_CONFIG


def unit_sf_bytes(rows: int, k: int) -> int:
    """Bytes of UE8M0 scale factors one [rows, k] operand needs (unit_sf_bytes in the .cuh)."""
    return -(-rows // 128) * 128 * (-(-k // 128) * 4)


_unit_sf: Dict[int, torch.Tensor] = {}


def _unit_scale_factors(device: torch.device, nbytes: int) -> torch.Tensor:
    """A per-device all-2^0 UE8M0 buffer of at least nbytes, grown on demand.

    Grow it before CUDA graph capture (the first eager call of each shape does).
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    buf = _unit_sf.get(index)
    if buf is None or buf.numel() < nbytes:
        buf = torch.full((nbytes,), _UE8M0_ONE, dtype=torch.uint8, device=device)
        _unit_sf[index] = buf
    return buf


def _cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_TEST_LEVEL=0",
        "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]


@cache_once
def _jit_module(
    tile_m: int, tile_n: int, tile_k: int, pingpong: bool, streamk: bool, scaled: bool
) -> Module:
    targs = (
        f"{tile_m}, {tile_n}, {tile_k}, {str(pingpong).lower()}, {str(streamk).lower()}"
    )
    entry = (
        "fp8_unit_blockscaled_scaled_mm_sm120"
        if scaled
        else "fp8_unit_blockscaled_mm_sm120"
    )
    return load_jit(
        entry,
        tile_m,
        tile_n,
        tile_k,
        int(pingpong),
        int(streamk),
        cuda_files=["gemm/fp8_blockscaled/fp8_unit_blockscaled_mm_sm120.cuh"],
        cuda_wrappers=[("mm", f"{entry}<{targs}>")],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=_cuda_flags(),
    )


def _module_and_sf(mat_a: torch.Tensor, w_nk: torch.Tensor, scaled: bool):
    m, k = mat_a.shape
    n = w_nk.shape[0]
    cfg = select_sm120_blockscaled_config(m, n, k)
    module = _jit_module(
        cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.pingpong, cfg.streamk, scaled
    )
    sf = _unit_scale_factors(mat_a.device, unit_sf_bytes(max(m, n), k))
    return module, sf, cfg.max_swizzle


@register_custom_op(
    op_name="fp8_blockscaled_scaled_mm_sm120",
    mutates_args=["out"],
)
def _fp8_blockscaled_scaled_mm_op(
    out: torch.Tensor,
    mat_a: torch.Tensor,
    w_nk: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
) -> None:
    module, sf, swizzle = _module_and_sf(mat_a, w_nk, scaled=True)
    module.mm(out, mat_a, w_nk, sf, scales_a, scales_b, swizzle)


@register_custom_op(
    op_name="fp8_blockscaled_unscaled_mm_sm120",
    mutates_args=["out"],
)
def _fp8_blockscaled_unscaled_mm_op(
    out: torch.Tensor, mat_a: torch.Tensor, w_nk: torch.Tensor
) -> None:
    module, sf, swizzle = _module_and_sf(mat_a, w_nk, scaled=False)
    module.mm(out, mat_a, w_nk, sf, swizzle)


@debug_kernel_api
def fp8_blockscaled_scaled_mm_sm120(
    mat_a: torch.Tensor,
    w_nk: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """bf16 out[M, N] = scales_a[M] * scales_b[N] * (mat_a[M, K] @ w_nk[N, K]^T), e4m3 inputs.

    fp32 accumulation, scales applied in fp32, one bf16 rounding. ``out`` may be a
    row-strided view (16-byte aligned pitch), for example a slice of a wider buffer.
    """
    if out is None:
        out = torch.empty(
            (mat_a.shape[0], w_nk.shape[0]), dtype=torch.bfloat16, device=mat_a.device
        )
    _fp8_blockscaled_scaled_mm_op(
        out, mat_a, w_nk, scales_a.reshape(-1), scales_b.reshape(-1)
    )
    return out


@debug_kernel_api
def fp8_blockscaled_unscaled_mm_sm120(
    mat_a: torch.Tensor, w_nk: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """bf16 out[M, N] = mat_a[M, K] @ w_nk[N, K]^T (e4m3, fp32 accumulation, no scales).

    The caller owns the per-token and per-channel scales; see
    apply_fp8_linear_deferred_scale.
    """
    if out is None:
        out = torch.empty(
            (mat_a.shape[0], w_nk.shape[0]), dtype=torch.bfloat16, device=mat_a.device
        )
    _fp8_blockscaled_unscaled_mm_op(out, mat_a, w_nk)
    return out


def maybe_fp8_blockscaled_scaled_mm_sm120(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """This kernel for apply_fp8_linear's per-token x per-channel GEMM, or None.

    mat_b is the [K, N] column-major view of the stored [N, K] weight.
    """
    w_nk = mat_b.t()
    if (
        mat_a.shape[0] < MIN_BLOCKSCALED_M
        or out_dtype != torch.bfloat16
        or not mat_a.is_contiguous()
        or not w_nk.is_contiguous()
        or scales_a.numel() != mat_a.shape[0]
        or scales_b.numel() != w_nk.shape[0]
        or mat_a.shape[1] % 16 != 0
        or w_nk.shape[0] % 8 != 0
    ):
        return None
    return fp8_blockscaled_scaled_mm_sm120(mat_a, w_nk, scales_a, scales_b)
