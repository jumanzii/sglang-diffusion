from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Below this many tokens the cuBLASLt route was not measured; callers keep their kernel.
MIN_CUBLASLT_M = 1024
# cuBLASLt's fastest FP8 algos on SM120 need a large workspace; 32 MiB was measured.
_WORKSPACE_BYTES = 32 << 20
# Arbitrary: GPU time on the first heuristic algo before timing starts, so an idle GPU
# leaves its idle clock (the ramp took under 1 ms on an RTX 5090).
_TUNE_WARMUP_MS = 50.0
# Arbitrary: every algo is timed once per round, rounds interleaved, and ranked by its
# median; a clock change or a co-tenant burst then skews one sample, not one algo.
_TUNE_ROUNDS = 7
_TUNE_REPS = 3
# Arbitrary: an algo within this fraction of the fastest median counts as a tie, and
# ties go to cuBLASLt's heuristic order, so the pick does not follow timing noise.
_TUNE_TIE_FRACTION = 0.03

_algo_cache: Dict[Tuple[int, int, int, int], int] = {}


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "fp8_cublaslt_unit_scale_gemm",
        cuda_files=["gemm/fp8_cublaslt/fp8_unit_scale_gemm.cuh"],
        cuda_wrappers=[
            ("gemm", "fp8_unit_scale_gemm"),
            ("num_algos", "fp8_unit_scale_gemm_num_algos"),
        ],
        extra_ldflags=["-lcublasLt"],
    )


@cache_once
def _workspace(device_index: int) -> torch.Tensor:
    return torch.empty(
        _WORKSPACE_BYTES, dtype=torch.uint8, device=f"cuda:{device_index}"
    )


def _time_ms(run: Callable[[], None], reps: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _pick_algo(median_ms: List[float]) -> int:
    fastest = min(median_ms)
    return next(
        idx
        for idx, t in enumerate(median_ms)
        if t <= fastest * (1 + _TUNE_TIE_FRACTION)
    )


def _select_algo(module: Module, out, mat_a, w_nk, workspace) -> int:
    """Fastest of cuBLASLt's heuristic algos for this shape, timed once and cached.

    cuBLASLt's first choice is not always the fastest on SM120 (11% slower on
    M=4096, K=12288, N=4096). Under CUDA graph capture, take the first choice.
    """
    key = (mat_a.device.index, mat_a.shape[0], w_nk.shape[0], mat_a.shape[1])
    algo = _algo_cache.get(key)
    if algo is not None:
        return algo
    num_algos = module.num_algos(mat_a, w_nk, workspace)
    if num_algos == 1 or torch.cuda.is_current_stream_capturing():
        return 0

    def run(idx: int) -> Callable[[], None]:
        return lambda: module.gemm(out, mat_a, w_nk, workspace, idx)

    warmed_ms = 0.0
    while warmed_ms < _TUNE_WARMUP_MS:
        warmed_ms += _time_ms(run(0), reps=_TUNE_REPS)
    samples: List[List[float]] = [[] for _ in range(num_algos)]
    for _ in range(_TUNE_ROUNDS):
        for idx in range(num_algos):
            samples[idx].append(_time_ms(run(idx), reps=_TUNE_REPS))
    algo = _pick_algo([statistics.median(times) for times in samples])
    _algo_cache[key] = algo
    return algo


@triton.jit
def _scale_rows_cols_kernel(
    out_ptr, sa_ptr, sb_ptr, N, stride_m, BLOCK_N: tl.constexpr
):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    sa = tl.load(sa_ptr + m)
    sb = tl.load(sb_ptr + offs, mask=mask, other=0.0)
    ptrs = out_ptr + m * stride_m + offs
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(ptrs, (x * (sa * sb)).to(out_ptr.dtype.element_ty), mask=mask)


@register_custom_op(
    op_name="fp8_per_channel_scaled_mm_cublaslt",
    mutates_args=["out"],
)
def _fp8_per_channel_scaled_mm_cublaslt_op(
    out: torch.Tensor,
    mat_a: torch.Tensor,
    w_nk: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
) -> None:
    module = _jit_module()
    workspace = _workspace(mat_a.device.index or 0)
    algo = _select_algo(module, out, mat_a, w_nk, workspace)
    module.gemm(out, mat_a, w_nk, workspace, algo)
    m, n = out.shape
    block_n = 2048
    _scale_rows_cols_kernel[(m, triton.cdiv(n, block_n))](
        out, scales_a, scales_b, n, out.stride(0), BLOCK_N=block_n, num_warps=8
    )


@debug_kernel_api
def fp8_per_channel_scaled_mm_cublaslt(
    mat_a: torch.Tensor,
    w_nk: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
) -> torch.Tensor:
    """bf16 out[M, N] = scales_a[M] * scales_b[N] * (mat_a[M, K] @ w_nk[N, K]^T), e4m3 inputs.

    The GEMM runs in cuBLASLt with unit per-tensor scales and fp32 accumulation into
    bf16; the scales are applied by a second in-place pass. cuBLASLt on SM120 has no
    per-row or per-column FP8 scale mode, so the unscaled product is rounded to bf16
    once more than in a fused-epilogue kernel (relative error about 2.8e-3).
    """
    out = torch.empty(
        (mat_a.shape[0], w_nk.shape[0]), dtype=torch.bfloat16, device=mat_a.device
    )
    _fp8_per_channel_scaled_mm_cublaslt_op(
        out, mat_a, w_nk, scales_a.reshape(-1), scales_b.reshape(-1)
    )
    return out


def maybe_fp8_per_channel_scaled_mm_cublaslt(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """The cuBLASLt route for apply_fp8_linear's per-token x per-channel GEMM, or None.

    mat_b is the [K, N] column-major view of the stored [N, K] weight.
    """
    w_nk = mat_b.t()
    if (
        mat_a.shape[0] < MIN_CUBLASLT_M
        or out_dtype != torch.bfloat16
        or not mat_a.is_contiguous()
        or not w_nk.is_contiguous()
        or not scales_a.is_contiguous()
        or not scales_b.is_contiguous()
        or scales_a.numel() != mat_a.shape[0]
        or scales_b.numel() != w_nk.shape[0]
    ):
        return None
    return fp8_per_channel_scaled_mm_cublaslt(mat_a, w_nk, scales_a, scales_b)
