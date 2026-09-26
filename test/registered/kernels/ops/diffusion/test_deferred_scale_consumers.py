# SPDX-License-Identifier: Apache-2.0
"""Consumers of an unscaled FP8 GEMM product that apply its per-token (row) and
per-channel (column) scales on load: the Qwen-Image 2.1 SA2 Q/K/V kernels and
the scaled SiLU-mul. Each must match the scale pass followed by the unscaled
consumer bit for bit."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.activation.silu_mul_bitexact import (
    fused_scaled_silu_mul,
    fused_scaled_silu_mul_fp8,
    fused_silu_mul_bitexact,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

EPS = 1e-6


def _scales(rows, cols, unit):
    if unit:
        return torch.ones(rows, device="cuda"), torch.ones(cols, device="cuda")
    return (
        torch.rand(rows, device="cuda") * 0.01 + 0.001,
        torch.rand(cols, device="cuda") * 0.01 + 0.001,
    )


def _apply(x2d, row_scale, col_scale):
    """What the SM120 cuBLASLt route's scale pass stores: bf16(x * (sa * sb))."""
    return (x2d.float() * (row_scale[:, None] * col_scale[None, :])).to(x2d.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("unit", [True, False])
@pytest.mark.parametrize("m, n", [(4096, 12288), (1500, 4096)])
def test_scaled_silu_mul(unit, m, n):
    torch.manual_seed(0)
    a = (torch.randn(m, n, device="cuda") * 300).to(torch.bfloat16)
    b = (torch.randn(m, n, device="cuda") * 300).to(torch.bfloat16)
    row_scale, a_col = _scales(m, n, unit)
    _, b_col = _scales(m, n, unit)
    out = fused_scaled_silu_mul(a, b, row_scale, a_col, b_col)
    reference = fused_silu_mul_bitexact(
        _apply(a, row_scale, a_col), _apply(b, row_scale, b_col)
    )
    assert torch.equal(out, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("unit", [True, False])
@pytest.mark.parametrize("m, n", [(4096, 12288), (1500, 4096), (3, 100)])
def test_scaled_silu_mul_fp8_matches_per_token_quant(unit, m, n):
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

    torch.manual_seed(0)
    a = (torch.randn(m, n, device="cuda") * 300).to(torch.bfloat16)
    b = (torch.randn(m, n, device="cuda") * 300).to(torch.bfloat16)
    if m >= torch.cuda.get_device_properties(0).multi_processor_count * 16:
        # All-zero row: scale 0, as the warp kernel the reference uses at this
        # size handles it (its small-batch CTA kernel does not guard it).
        a[0] = 0
    a[-1, :4] = 3e4  # large values that clamp to the e4m3 range
    row_scale, a_col = _scales(m, n, unit)
    _, b_col = _scales(m, n, unit)
    q, q_scale = fused_scaled_silu_mul_fp8(a, b, row_scale, a_col, b_col)
    ref_q, ref_scale = sglang_per_token_quant_fp8(
        fused_scaled_silu_mul(a, b, row_scale, a_col, b_col)
    )
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(q_scale, ref_scale)


sage_available = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() == (12, 0)
)


@pytest.mark.skipif(not sage_available, reason="SageAttention2's sm_120 path required")
@pytest.mark.parametrize("unit", [True, False])
@pytest.mark.parametrize("shape", [(1, 4096, 24, 34), (1, 257, 4, 7)])
def test_sage_prequant_attention_with_deferred_scales(unit, shape):
    pytest.importorskip("sageattention")
    from sglang.kernels.ops.diffusion.attention.sageattn_prequant_triton import (
        qknorm_complex_rope_int8,
        sage_prequant_attention,
        sage_v_fp8,
    )
    from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_kv_triton import (
        qknorm_complex_rope_k,
    )

    batch, seq, heads, prefix = shape
    width = heads * 128
    torch.manual_seed(1)
    unscaled = (torch.randn(batch * seq, 3 * width, device="cuda") * 300).to(
        torch.bfloat16
    )
    row_scale, col_scale = _scales(batch * seq, 3 * width, unit)
    scaled = _apply(unscaled, row_scale, col_scale)

    def views(x):
        return x.view(batch, seq, 3, heads, 128).unbind(2)

    q, k, v = views(unscaled)
    qs, ks, vs = views(scaled)
    weight = (torch.rand(128, device="cuda") + 0.5).to(torch.bfloat16)
    angles = torch.rand(seq, 64, device="cuda") * 6.283
    rope = torch.polar(torch.ones_like(angles), angles)
    kp = torch.randn(batch, prefix, heads, 128, device="cuda", dtype=torch.bfloat16)
    vp = torch.randn_like(kp) * 3
    q_cs, k_cs, v_cs = (col_scale[i * width : (i + 1) * width] for i in range(3))

    q_int8, q_scale = qknorm_complex_rope_int8(
        q, weight, rope, EPS, row_scale=row_scale, col_scale=q_cs
    )
    ref_q_int8, ref_q_scale = qknorm_complex_rope_int8(qs, weight, rope, EPS)
    k_full = qknorm_complex_rope_k(
        k, weight, rope, kp, EPS, row_scale=row_scale, col_scale=k_cs
    )
    ref_k_full = qknorm_complex_rope_k(ks, weight, rope, kp, EPS)
    v_fp8, v_scale = sage_v_fp8(vp, v, row_scale=row_scale, col_scale=v_cs)
    ref_v_fp8, ref_v_scale = sage_v_fp8(vp, vs)
    out = sage_prequant_attention(
        q,
        weight,
        EPS,
        k,
        weight,
        EPS,
        rope,
        v,
        kp,
        vp,
        128**-0.5,
        row_scale=row_scale,
        qkv_col_scale=col_scale,
    )
    ref = sage_prequant_attention(
        qs, weight, EPS, ks, weight, EPS, rope, vs, kp, vp, 128**-0.5
    )

    assert torch.equal(q_int8, ref_q_int8) and torch.equal(q_scale, ref_q_scale)
    assert torch.equal(k_full, ref_k_full)
    assert torch.equal(v_fp8.view(torch.uint8), ref_v_fp8.view(torch.uint8))
    assert torch.equal(v_scale, ref_v_scale)
    assert torch.equal(out, ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
