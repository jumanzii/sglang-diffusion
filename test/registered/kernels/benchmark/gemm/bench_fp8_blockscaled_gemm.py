import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.gemm.fp8_blockscaled_gemm import (
    fp8_blockscaled_scaled_mm_sm120,
    fp8_blockscaled_unscaled_mm_sm120,
)
from sglang.kernels.ops.gemm.fp8_cublaslt_gemm import (
    fp8_per_channel_scaled_mm_cublaslt,
    fp8_unit_scale_gemm_cublaslt,
)
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)

FN_MAP = {
    "blockscaled_scaled": lambda a, w, sa, sb: fp8_blockscaled_scaled_mm_sm120(
        a, w, sa, sb
    ),
    "cublaslt_scaled": lambda a, w, sa, sb: fp8_per_channel_scaled_mm_cublaslt(
        a, w, sa, sb
    ),
    "blockscaled_unscaled": lambda a, w, sa, sb: fp8_blockscaled_unscaled_mm_sm120(
        a, w
    ),
    "cublaslt_unscaled": lambda a, w, sa, sb: fp8_unit_scale_gemm_cublaslt(a, w),
}


# (M, N, K): Qwen-Image 2.1's FP8 linears at 1024x1024 (4096 image tokens).
@marker.parametrize(
    "m,n,k",
    [(4096, 12288, 4096), (4096, 4096, 4096), (4096, 4096, 12288), (1024, 4096, 4096)],
    [(4096, 4096, 4096)],
)
@marker.benchmark("impl", list(FN_MAP))
def benchmark(m: int, n: int, k: int, impl: str):
    if not is_sm120_supported():
        return marker.skip("requires SM120")
    fp8 = torch.float8_e4m3fn
    a = (torch.randn(m, k, device="cuda") * 2).clamp(-448, 448).to(fp8)
    w = (torch.randn(n, k, device="cuda") * 2).clamp(-448, 448).to(fp8)
    sa = torch.rand(m, 1, device="cuda") * 0.01 + 0.001
    sb = torch.rand(1, n, device="cuda") * 0.01 + 0.001
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(a, w, sa, sb),
        graph_clone_args=(0, 1),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
