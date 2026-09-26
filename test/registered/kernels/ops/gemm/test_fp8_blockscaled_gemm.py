import sys

import pytest
import torch

from sglang.kernels.ops.gemm import fp8_blockscaled_gemm as bs
from sglang.kernels.ops.gemm.fp8_blockscaled_gemm import (
    MIN_BLOCKSCALED_M,
    fp8_blockscaled_scaled_mm_sm120,
    fp8_blockscaled_unscaled_mm_sm120,
    maybe_fp8_blockscaled_scaled_mm_sm120,
    select_sm120_blockscaled_config,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b",
    runner_config="1-gpu-small",
)

pytestmark = pytest.mark.skipif(not is_sm120_supported(), reason="requires SM120")

# The (N, K) of Qwen-Image 2.1's FP8 linears: qkv/gate/up, out, down.
SHAPES = [(12288, 4096), (4096, 4096), (4096, 12288)]


def _operands(m, n, k):
    fp8 = torch.float8_e4m3fn
    a = (torch.randn(m, k, device="cuda") * 2).clamp(-448, 448).to(fp8)
    w = (torch.randn(n, k, device="cuda") * 2).clamp(-448, 448).to(fp8)
    scale_a = torch.rand(m, 1, device="cuda") * 0.01 + 0.001
    scale_b = torch.rand(1, n, device="cuda") * 0.01 + 0.001
    return a, w, scale_a, scale_b


def _product(a, w):
    return a.float() @ w.float().t()


# One bf16 rounding of an fp32 result that differs from the reference only in
# summation order: within one bf16 ulp (2^-8 relative) plus a small absolute term.
ONE_ROUNDING = dict(rtol=8e-3, atol=1e-3)


@pytest.mark.parametrize("m", [1024, 1500, 4104])
@pytest.mark.parametrize("n, k", SHAPES)
def test_scaled_matches_reference_and_is_deterministic(m, n, k):
    torch.manual_seed(0)
    a, w, scale_a, scale_b = _operands(m, n, k)
    expected = _product(a, w) * scale_a * scale_b
    out = fp8_blockscaled_scaled_mm_sm120(a, w, scale_a, scale_b)
    again = fp8_blockscaled_scaled_mm_sm120(a, w, scale_a, scale_b)
    assert out.dtype == torch.bfloat16 and out.shape == (m, n)
    torch.testing.assert_close(out.float(), expected, **ONE_ROUNDING)
    assert torch.equal(out, again)


@pytest.mark.parametrize("m", [1024, 1500])
@pytest.mark.parametrize("n, k", SHAPES)
def test_unscaled_matches_reference_and_is_deterministic(m, n, k):
    torch.manual_seed(0)
    a, w, _, _ = _operands(m, n, k)
    out = fp8_blockscaled_unscaled_mm_sm120(a, w)
    again = fp8_blockscaled_unscaled_mm_sm120(a, w)
    torch.testing.assert_close(out.float(), _product(a, w), rtol=8e-3, atol=1.0)
    assert torch.equal(out, again)


def _tuned_configs():
    configs = {select_sm120_blockscaled_config(4096, n, k) for n, k in SHAPES}
    return [
        pytest.param(c, id="_".join(str(int(x)) for x in c)) for c in sorted(configs)
    ]


@pytest.mark.parametrize("cfg", _tuned_configs())
@pytest.mark.parametrize("scaled", [False, True])
def test_every_selected_config_is_correct(cfg, scaled):
    torch.manual_seed(0)
    m, n, k = 1500, 4096, 4096
    a, w, scale_a, scale_b = _operands(m, n, k)
    module = bs._jit_module(
        cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.pingpong, cfg.streamk, scaled
    )
    sf = bs._unit_scale_factors(a.device, bs.unit_sf_bytes(max(m, n), k))
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    if scaled:
        module.mm(
            out, a, w, sf, scale_a.reshape(-1), scale_b.reshape(-1), cfg.max_swizzle
        )
        expected = _product(a, w) * scale_a * scale_b
        torch.testing.assert_close(out.float(), expected, **ONE_ROUNDING)
    else:
        module.mm(out, a, w, sf, cfg.max_swizzle)
        torch.testing.assert_close(out.float(), _product(a, w), rtol=8e-3, atol=1.0)


def test_writes_into_a_row_strided_output_and_reads_a_row_strided_input():
    torch.manual_seed(0)
    m, n, k = 1024, 4096, 4096
    a_wide, w, scale_a, scale_b = _operands(m, n, k + 256)
    a = a_wide[:, 128 : 128 + k]
    w = w[:, :k].contiguous()
    buf = torch.zeros(m, 3 * n, dtype=torch.bfloat16, device="cuda")
    out = buf[:, n : 2 * n]
    result = fp8_blockscaled_scaled_mm_sm120(a, w, scale_a, scale_b, out=out)
    assert result.data_ptr() == out.data_ptr()
    expected = _product(a, w) * scale_a * scale_b
    torch.testing.assert_close(out.float(), expected, **ONE_ROUNDING)
    assert not buf[:, :n].any() and not buf[:, 2 * n :].any()


def test_fp32_accumulation_does_not_overflow():
    # 32 products of 448 * 448 overflow an fp16 accumulator in the first MMA step.
    m = n = k = MIN_BLOCKSCALED_M
    a = torch.ones(m, k, device="cuda")
    a[:, :32] = 448
    a = a.to(torch.float8_e4m3fn)
    w = torch.full((n, k), 448.0, device="cuda").to(torch.float8_e4m3fn)
    out = fp8_blockscaled_unscaled_mm_sm120(a, w)
    torch.testing.assert_close(out.float(), _product(a, w), rtol=8e-3, atol=0.0)


def test_unit_scale_factor_buffer_grows_with_the_shape():
    torch.manual_seed(0)
    bs._unit_sf.clear()
    for m, n, k in [(1024, 4096, 4096), (4096, 4096, 12288)]:
        a, w, scale_a, scale_b = _operands(m, n, k)
        out = fp8_blockscaled_scaled_mm_sm120(a, w, scale_a, scale_b)
        expected = _product(a, w) * scale_a * scale_b
        torch.testing.assert_close(out.float(), expected, **ONE_ROUNDING)
    assert bs._unit_sf[torch.cuda.current_device()].numel() >= bs.unit_sf_bytes(
        4096, 12288
    )


def test_rejects_a_short_unit_scale_factor_buffer():
    torch.manual_seed(0)
    m, n, k = 1024, 4096, 4096
    a, w, _, _ = _operands(m, n, k)
    cfg = select_sm120_blockscaled_config(m, n, k)
    module = bs._jit_module(
        cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.pingpong, cfg.streamk, False
    )
    short = torch.full(
        (bs.unit_sf_bytes(n, k) - 1,), 0x7F, dtype=torch.uint8, device="cuda"
    )
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="unit_sf"):
        module.mm(out, a, w, short, cfg.max_swizzle)


@pytest.mark.parametrize(
    "case",
    ["small_m", "fp16_out", "per_tensor_scale", "non_contiguous_a"],
)
def test_declines_unsupported_calls(case):
    torch.manual_seed(0)
    m = 256 if case == "small_m" else MIN_BLOCKSCALED_M
    a, w, scale_a, scale_b = _operands(m, 4096, 4096)
    out_dtype = torch.float16 if case == "fp16_out" else torch.bfloat16
    if case == "per_tensor_scale":
        scale_a = scale_a[:1, 0]
    if case == "non_contiguous_a":
        a = a.t().contiguous().t()
    assert (
        maybe_fp8_blockscaled_scaled_mm_sm120(a, w.t(), scale_a, scale_b, out_dtype)
        is None
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_apply_fp8_linear_takes_this_route_only_when_enabled(enabled):
    from sglang.kernels.ops.gemm.fp8_cublaslt_gemm import (
        fp8_per_channel_scaled_mm_cublaslt,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

    torch.manual_seed(0)
    m, n, k = 1024, 4096, 4096
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    _, w, _, weight_scale = _operands(m, n, k)
    weight_scale = weight_scale.reshape(n, 1)
    with envs.SGLANG_ENABLE_SM120_FP8_BLOCKSCALED_GEMM.override(enabled):
        out = apply_fp8_linear(x, w.t(), weight_scale, use_per_token_if_dynamic=True)
    qx, x_scale = sglang_per_token_quant_fp8(x)
    if enabled:
        expected = fp8_blockscaled_scaled_mm_sm120(qx, w, x_scale, weight_scale)
    else:
        expected = fp8_per_channel_scaled_mm_cublaslt(qx, w, x_scale, weight_scale)
    assert torch.equal(out, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
