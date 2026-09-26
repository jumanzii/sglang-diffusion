# SPDX-License-Identifier: Apache-2.0
"""SageAttention2 operands from the Qwen-Image 2.1 Q/K/V kernels, against sageattn."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.attention.sageattn_prequant_triton import (
    can_use_sage_prequant_attention,
    qknorm_complex_rope_int8,
    sage_prequant_attention,
    sage_v_fp8,
)
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_kv_triton import (
    qknorm_complex_rope_k,
    qknorm_complex_rope_kv,
)
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_triton import (
    qknorm_complex_rope,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

sageattention = pytest.importorskip("sageattention")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (12, 0),
    reason="SageAttention2's sm_120 path required",
)

EPS = 1e-6


def inputs(batch, seq, heads, prefix, seed=0):
    torch.manual_seed(seed)
    merged = torch.randn(
        batch, seq, 3 * heads * 128, device="cuda", dtype=torch.bfloat16
    )
    merged *= (torch.rand(3 * heads * 128, device="cuda") * 4).to(torch.bfloat16)
    q, k, v = merged.unflatten(-1, (3, heads, 128)).unbind(-3)
    weight = (torch.rand(128, device="cuda") + 0.5).to(torch.bfloat16)
    angles = torch.rand(seq, 64, device="cuda") * 6.283
    rope = torch.polar(torch.ones_like(angles), angles)
    kp = torch.randn(batch, prefix, heads, 128, device="cuda", dtype=torch.bfloat16)
    vp = torch.randn_like(kp) * 3
    return q, k, v, weight, rope, kp, vp


def as_bytes(x):
    return x.view(torch.uint8) if x.dtype == torch.float8_e4m3fn else x


SHAPES = [(1, 4096, 32, 34), (2, 257, 4, 7), (1, 33, 2, 1), (1, 130, 3, 64)]


@pytest.mark.parametrize("shape", SHAPES)
def test_operands_match_sageattention_preprocessing(shape):
    from sageattention.quant import per_channel_fp8, per_warp_int8

    q, k, v, weight, rope, kp, vp = inputs(*shape)
    qn = qknorm_complex_rope(q, weight, rope, EPS)
    kf, vf = qknorm_complex_rope_kv(k, weight, rope, v, kp, vp, EPS)
    q_int8, q_scale, _, _ = per_warp_int8(
        qn,
        kf,
        kf.mean(1, keepdim=True),
        tensor_layout="NHD",
        BLKQ=128,
        WARPQ=32,
        BLKK=64,
    )
    v_fp8, v_scale, _ = per_channel_fp8(
        vf, tensor_layout="NHD", scale_max=2.25, smooth_v=False
    )

    actual_q, actual_q_scale = qknorm_complex_rope_int8(q, weight, rope, EPS)
    assert torch.equal(actual_q, q_int8)
    assert torch.equal(actual_q_scale, q_scale)
    actual_v, actual_v_scale = sage_v_fp8(vp, v)
    assert torch.equal(as_bytes(actual_v), as_bytes(v_fp8))
    assert torch.equal(actual_v_scale, v_scale)
    assert torch.equal(qknorm_complex_rope_k(k, weight, rope, kp, EPS), kf)


@pytest.mark.parametrize("shape", SHAPES)
def test_attention_matches_sageattn(shape):
    q, k, v, weight, rope, kp, vp = inputs(*shape, seed=1)
    assert can_use_sage_prequant_attention(q, weight, k, weight, rope, v, kp, vp)
    qn = qknorm_complex_rope(q, weight, rope, EPS)
    kf, vf = qknorm_complex_rope_kv(k, weight, rope, v, kp, vp, EPS)
    expected = sageattention.sageattn(
        qn, kf, vf, tensor_layout="NHD", sm_scale=128**-0.5
    )
    actual = sage_prequant_attention(
        q, weight, EPS, k, weight, EPS, rope, v, kp, vp, 128**-0.5
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_layout_guards():
    q, k, v, weight, rope, kp, vp = inputs(1, 33, 2, 7)
    assert not can_use_sage_prequant_attention(
        q.float(), weight, k, weight, rope, v, kp, vp
    )
    assert not can_use_sage_prequant_attention(
        q,
        weight,
        k,
        weight,
        rope,
        v,
        kp.transpose(1, 2).contiguous().transpose(1, 2),
        vp,
    )
    assert not can_use_sage_prequant_attention(
        q, weight, k, weight, rope, v[..., :64], kp, vp
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
