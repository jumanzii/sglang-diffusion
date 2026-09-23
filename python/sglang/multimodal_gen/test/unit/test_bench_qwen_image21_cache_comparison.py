# SPDX-License-Identifier: Apache-2.0
"""Arm definitions and block accounting of the cache comparison benchmark; CPU only."""

import json
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.benchmarks import bench_qwen_image21_cache_comparison as comp
from sglang.multimodal_gen.benchmarks import bench_qwen_image21_dpcache as bench


def test_every_cache_dit_preset_keeps_taylorseer_and_dmd_off():
    """Predeclared presets differ only in the documented knobs."""
    for name, params in comp.CACHE_DIT_PRESETS.items():
        assert params["enable_taylorseer"] is False, name
        assert params["enable_dmd"] is False, name
        assert params["Fn_compute_blocks"] == 1, name
    forced = comp.CACHE_DIT_PRESETS["cachedit-forced-full"]
    assert forced["max_warmup_steps"] == comp.STEPS
    assert forced["residual_diff_threshold"] == 0.0


def test_arm_families_do_not_parse_k_out_of_the_name():
    assert comp.arm_family("native") == "native"
    assert comp.arm_family("dp-K12") == "dpcache"
    assert comp.arm_family("uniform-K12") == "uniform"
    assert comp.arm_family("cachedit-aggressive") == "cache_dit"
    assert comp.arm_family(comp.TEACACHE_ARM) == "teacache_unsupported"
    with pytest.raises(ValueError, match="unknown arm"):
        comp.arm_family("cachedit-made-up")


def test_arm_requests_carry_their_own_switch_only(tmp_path):
    schedule = tmp_path / "K12.json"
    schedule.write_text(json.dumps({"num_full_steps": 12, "full_steps": [0, 1, 2]}))
    request, metadata = comp.arm_request("dp-K12", tmp_path)
    assert set(request) == {"dpcache_schedule"}
    assert metadata["schedule_sha256"] == bench.sha256_file(schedule)
    assert metadata["schedule_method"] == "exact-pair-state-dp"
    request, metadata = comp.arm_request("cachedit-stock", tmp_path)
    assert request["enable_cache_dit"] is True
    assert request["cache_dit_params"] == comp.CACHE_DIT_STOCK
    assert "dpcache_schedule" not in request
    request, metadata = comp.arm_request(comp.TEACACHE_ARM, tmp_path)
    assert request == {"enable_teacache": True}
    assert metadata["unsupported_diagnostic"] is True
    assert comp.arm_request("native", tmp_path) == ({}, None)


def test_uniform_arm_metadata_is_labelled_as_a_control(tmp_path):
    schedule = tmp_path / "uniform-K12.json"
    schedule.write_text(
        json.dumps(
            {
                "num_full_steps": 12,
                "full_steps": [0, 1, 2, 6],
                "schedule_method": "endpoint-matched-uniform-v1",
            }
        )
    )
    _, metadata = comp.arm_request("uniform-K12", tmp_path)
    assert metadata["schedule_method"] == "endpoint-matched-uniform-v1"


def evidence_row(arm, family, blocks_by_step):
    total = sum(blocks_by_step.values())
    return dict(
        kind="evidence",
        status="measured",
        arm=arm,
        family=family,
        total_blocks=total,
        full_block_equivalent=total / comp.LAYERS,
        full_block_steps=sum(1 for n in blocks_by_step.values() if n == comp.LAYERS),
        partial_block_steps=sum(
            1 for n in blocks_by_step.values() if 0 < n < comp.LAYERS
        ),
        zero_block_steps=comp.STEPS - len(blocks_by_step),
        requested_scheduler_steps=comp.STEPS,
        real_blocks_discovered=comp.LAYERS,
        transformer_blocks_attr_len=1 if family == "cache_dit" else comp.LAYERS,
        layer_loads_by_step={},
        denoise_device_ms=1000.0,
        final_feature_dtypes=["torch.bfloat16"],
        transformer_weight_dtype="torch.bfloat16",
        cache_dit_buffer_dtypes=(
            {"b": ["torch.bfloat16", [1]]} if family == "cache_dit" else {}
        ),
        actual_scheduler_steps=comp.STEPS,
        instrumentation_ok=True,
        instrumentation_notes=[],
    )


def test_partial_steps_are_reported_not_flagged_as_errors():
    """Cache-DiT legitimately runs 1..31 blocks on a cached step."""
    cache_dit = {step: (comp.LAYERS if step < 4 else 1) for step in range(comp.STEPS)}
    dpcache = {step: comp.LAYERS for step in range(12)}
    summary = comp.compute_summary(
        [
            evidence_row("cachedit-stock", "cache_dit", cache_dit),
            evidence_row("dp-K12", "dpcache", dpcache),
        ]
    )
    stock = summary["cachedit-stock"]
    assert stock["partial_block_steps"]["mean"] == comp.STEPS - 4
    assert stock["full_block_steps"]["mean"] == 4
    assert stock["zero_block_steps"]["mean"] == 0
    assert stock["full_block_equivalent"]["mean"] == (4 * 32 + 36) / 32
    assert stock["transformer_blocks_attr_len"] == [1]  # wrapped, blocks still counted
    assert stock["real_blocks_discovered"] == [comp.LAYERS]
    dp = summary["dp-K12"]
    assert dp["full_block_steps"]["mean"] == 12
    assert dp["partial_block_steps"]["mean"] == 0
    assert dp["zero_block_steps"]["mean"] == comp.STEPS - 12
    assert dp["full_block_equivalent"]["mean"] == 12


def test_comparator_heldout_prompts_are_new():
    corpus = comp.build_corpus("rev")
    heldout = {p["prompt"] for p in corpus["splits"]["heldout"]}
    seen = {
        prompt
        for group in (
            bench.CALIBRATION_PROMPTS,
            bench.VALIDATION_PROMPTS,
            bench.HELDOUT_PROMPTS,
        )
        for _, prompt in group
    }
    assert len(heldout) == 10 and not heldout & seen
    assert len(corpus["splits"]["heldout"]) == 20
    assert {p["prompt"] for p in corpus["splits"]["controls"]} <= seen


def capture(**overrides):
    base = dict(
        real_blocks_discovered=comp.LAYERS,
        actual_scheduler_steps=comp.STEPS,
        final_feature_dtypes=["torch.bfloat16"],
        transformer_weight_dtype="torch.bfloat16",
        layers_by_step={"0": list(range(comp.LAYERS))},
        blocks_by_step={str(s): comp.LAYERS for s in range(comp.STEPS)},
        cache_dit_buffer_dtypes={},
        expected_full_steps=[],
    )
    base.update(overrides)
    return base


def counts(total=comp.STEPS * comp.LAYERS, full=comp.STEPS, partial=0):
    return dict(total_blocks=total, full_block_steps=full, partial_block_steps=partial)


def test_audit_accepts_a_healthy_native_capture():
    assert comp.audit_evidence("native", "native", capture(), counts()) == []


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"real_blocks_discovered": 1}, "real blocks"),
        ({"actual_scheduler_steps": 39}, "scheduler.step ran 39"),
        ({"final_feature_dtypes": ["torch.float32"]}, "final features"),
        ({"transformer_weight_dtype": "torch.float16"}, "weights"),
        ({"layers_by_step": {"0": [0, 0, 1]}}, "duplicate layer ids"),
    ],
)
def test_audit_rejects_untrustworthy_instrumentation(overrides, expected):
    notes = comp.audit_evidence("native", "native", capture(**overrides), counts())
    assert any(expected in note for note in notes), notes


def test_audit_requires_all_full_arms_to_run_every_block():
    notes = comp.audit_evidence(
        "cache_dit", "cachedit-forced-full", capture(), counts(total=1279)
    )
    assert any("all-full arm ran 1279" in n for n in notes)


def test_audit_requires_a_bf16_cache_buffer_on_caching_runs():
    """An empty audit must not read as a passed audit."""
    notes = comp.audit_evidence("cache_dit", "cachedit-stock", capture(), counts())
    assert any("no Cache-DiT tensor buffer" in n for n in notes)
    ok = comp.audit_evidence(
        "cache_dit",
        "cachedit-stock",
        capture(cache_dit_buffer_dtypes={"c.r": ["torch.bfloat16", [1, 2]]}),
        counts(),
    )
    assert ok == []
    fp32 = comp.audit_evidence(
        "cache_dit",
        "cachedit-stock",
        capture(cache_dit_buffer_dtypes={"c.r": ["torch.float32", [1, 2]]}),
        counts(),
    )
    assert any("cached tensors" in n for n in fp32)


def test_audit_holds_dpcache_to_exactly_its_schedule():
    full_steps = [0, 1, 2, 6]
    blocks = {str(s): comp.LAYERS for s in full_steps}
    healthy = comp.audit_evidence(
        "dpcache",
        "dp-K12",
        capture(
            blocks_by_step=blocks,
            expected_full_steps=full_steps,
            layers_by_step={str(s): list(range(comp.LAYERS)) for s in full_steps},
        ),
        counts(total=4 * comp.LAYERS, full=4),
    )
    assert healthy == []
    wrong_steps = comp.audit_evidence(
        "dpcache",
        "dp-K12",
        capture(
            blocks_by_step={str(s): comp.LAYERS for s in [0, 1, 2, 7]},
            expected_full_steps=full_steps,
        ),
        counts(total=4 * comp.LAYERS, full=4),
    )
    assert any("differ from the schedule" in n for n in wrong_steps)
    partial = comp.audit_evidence(
        "dpcache",
        "dp-K12",
        capture(blocks_by_step={**blocks, "7": 1}, expected_full_steps=full_steps),
        counts(total=4 * comp.LAYERS + 1, full=4, partial=1),
    )
    assert any("partial steps" in n for n in partial)


def plan_with_regimes():
    steps = []
    for kind in ("timing", "service-timing"):
        for pair in ("p0", "p1"):
            for arm in ("native", "cachedit-stock"):
                steps.append(dict(pair_id=pair, arm=arm, kind=kind, repeat=0))
    for index, step in enumerate(steps):
        step["order_index"] = index
    return dict(
        steps=steps, corpus_sha256="c" * 64, split="validation", phase="validation"
    )


def test_timing_regimes_are_reported_separately():
    plan = plan_with_regimes()
    walls = {
        ("timing", "native"): 13.0,
        ("timing", "cachedit-stock"): 12.0,
        ("service-timing", "native"): 13.0,
        ("service-timing", "cachedit-stock"): 9.0,
    }
    rows = [
        dict(
            order_index=s["order_index"],
            status="measured",
            client_wall_seconds=walls[(s["kind"], s["arm"])],
        )
        for s in plan["steps"]
    ]
    mixed = comp.paired_timing(plan, rows, "timing")
    service = comp.paired_timing(plan, rows, "service-timing")
    assert mixed["cachedit-stock"]["paired_speedup"]["mean"] == pytest.approx(13 / 12)
    assert service["cachedit-stock"]["paired_speedup"]["mean"] == pytest.approx(13 / 9)
    assert mixed["cachedit-stock"]["complete"] and service["cachedit-stock"]["complete"]


def test_report_reads_only_fields_the_capture_emits():
    """compute_summary must not reference a key the worker never writes."""
    row = evidence_row("native", "native", {s: comp.LAYERS for s in range(comp.STEPS)})
    emitted = set(row)
    assert "scheduler_steps" not in emitted
    summary = comp.compute_summary([row])["native"]
    assert summary["requested_scheduler_steps"] == [comp.STEPS]
    assert summary["actual_scheduler_steps"] == [comp.STEPS]
    assert summary["instrumentation_ok"]


def test_timing_subset_must_use_distinct_prompts():
    pairs = [dict(pair_id=f"p{i}", prompt=f"prompt {i % 3}") for i in range(6)]
    args = SimpleNamespace(timing_pair_indices=[0, 1, 2], timing_pairs=0)
    assert [p["pair_id"] for p in comp.select_timing_pairs(pairs, args)] == [
        "p0",
        "p1",
        "p2",
    ]
    repeated = SimpleNamespace(timing_pair_indices=[0, 3], timing_pairs=0)
    with pytest.raises(ValueError, match="distinct prompts"):
        comp.select_timing_pairs(pairs, repeated)


def test_timing_regimes_carry_their_label_and_sample_shape():
    plan = plan_with_regimes()
    rows = [
        dict(
            order_index=s["order_index"],
            status="measured",
            client_wall_seconds=13.0 if s["arm"] == "native" else 6.5,
        )
        for s in plan["steps"]
    ]
    service = comp.paired_timing(plan, rows, "service-timing")
    assert service["cachedit-stock"]["regime"] == "service-timing"
    assert service["cachedit-stock"]["distinct_pairs"] == 2
    assert service["cachedit-stock"]["ratio_of_means"] == pytest.approx(2.0)
    assert service["native"]["ratio_of_means"] is None
