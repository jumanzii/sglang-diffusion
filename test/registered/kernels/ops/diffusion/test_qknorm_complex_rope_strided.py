# SPDX-License-Identifier: Apache-2.0
"""The fused Q/K RMSNorm + RoPE kernels on q/k/v views of a merged QKV output."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_kv_triton import (
    can_use_qknorm_complex_rope_kv,
    qknorm_complex_rope_kv,
)
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_triton import (
    can_use_qknorm_complex_rope,
    qknorm_complex_rope,
    token_stride,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)

EPS = 1e-6


def merged_inputs(batch, seq, heads, prefix):
    torch.manual_seed(0)
    merged = torch.randn(
        batch, seq, 3 * heads * 128, device="cuda", dtype=torch.bfloat16
    )
    q, k, v = merged.unflatten(-1, (3, heads, 128)).unbind(-3)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(seq, 64, device="cuda") * 20
    rope = torch.polar(torch.ones_like(angles), angles)
    kp = torch.randn(batch, prefix, heads, 128, device="cuda", dtype=torch.bfloat16)
    return q, k, v, weight, rope, kp, torch.randn_like(kp)


@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 3), (1, 257, 16, 5), (2, 33, 4, 7), (1, 4096, 32, 237)]
)
def test_strided_views_match_contiguous_inputs(shape):
    q, k, v, weight, rope, kp, vp = merged_inputs(*shape)
    assert token_stride(q) == 3 * shape[2] * 128
    for x in (q, k):
        assert can_use_qknorm_complex_rope(x, weight, rope)
        actual = qknorm_complex_rope(x, weight, rope, EPS)
        assert actual.is_contiguous()
        torch.testing.assert_close(
            actual,
            qknorm_complex_rope(x.contiguous(), weight, rope, EPS),
            atol=0,
            rtol=0,
        )
    assert can_use_qknorm_complex_rope_kv(k, weight, rope, v, kp, vp)
    actual = qknorm_complex_rope_kv(k, weight, rope, v, kp, vp, EPS)
    expected = qknorm_complex_rope_kv(
        k.contiguous(), weight, rope, v.contiguous(), kp, vp, EPS
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_token_stride_layout_guards():
    q, k, v, weight, rope, kp, vp = merged_inputs(2, 33, 4, 7)
    assert token_stride(q.contiguous()) == 4 * 128
    assert token_stride(q.transpose(1, 2)) is None  # heads not dense within a token
    assert token_stride(q[:, ::2]) is None  # batch stride is not seq * token stride
    assert token_stride(q[:1, ::2]) == 2 * 3 * 4 * 128  # one sample: any token spacing
    assert token_stride(q[..., :64]) is None
    assert not can_use_qknorm_complex_rope(q.float(), weight.float(), rope)
    assert not can_use_qknorm_complex_rope(q, weight, rope[:-1])
    # the prefix buffers are still read as dense tensors
    strided_prefix = kp.transpose(1, 2).contiguous().transpose(1, 2)
    assert not can_use_qknorm_complex_rope_kv(k, weight, rope, v, strided_prefix, vp)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
