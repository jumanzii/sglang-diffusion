# SPDX-License-Identifier: Apache-2.0
"""DPCache predictor, calibration cost, planner and request lifecycle; CPU only."""

import itertools
import json
import math
from types import SimpleNamespace

import msgspec
import pytest
import torch

from sglang.multimodal_gen.configs.sample.qwenimage21 import QwenImage21SamplingParams
from sglang.multimodal_gen.runtime.cache.dpcache import (
    DPCacheRequestSignature,
    DPCacheState,
    build_schedule_artifact,
    check_schedule_matches,
    checkpoint_identity,
    config_digest,
    pact_costs,
    pact_step_errors,
    plan_schedule,
    predict_feature,
    schedule_cost,
    uniform_full_steps,
    validate_schedule,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    qwen_image21 as stage_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    QwenImage21DenoisingStage,
)


def linear_features(num_steps, shape=(2, 3), dtype=torch.bfloat16):
    # quarter-integer slopes keep every intermediate exactly representable
    base = torch.arange(6, dtype=torch.float32).reshape(shape) - 2
    slope = torch.tensor([0.25, -0.5, 1.0, 0.75, -1.25, 2.0]).reshape(shape)
    return [(base + slope * step).to(dtype) for step in range(num_steps)]


def collapsed_plan(costs, num_full_steps, lead):
    """Reference-style planner: one predecessor per (budget, current key)."""
    num_steps = costs.shape[0]
    best = {lead - 1: (0.0, [*range(lead)])}
    for _ in range(lead, num_full_steps):
        level = {}
        for j, (value, keys) in sorted(best.items()):
            for k in range(j + 1, num_steps):
                candidate = value + costs[keys[-2], j, k].item()
                if candidate < level.get(k, (math.inf,))[0]:
                    level[k] = (candidate, keys + [k])
        best = level
    return min(
        (value + costs[keys[-2], j, num_steps].item(), keys)
        for j, (value, keys) in best.items()
    )


def brute_force(costs, num_full_steps, lead, force_last_full=False):
    num_steps = costs.shape[0]
    best = (math.inf, None)
    for rest in itertools.combinations(range(lead, num_steps), num_full_steps - lead):
        keys = [*range(lead), *rest]
        if force_last_full and keys[-1] != num_steps - 1:
            continue
        best = min(best, (schedule_cost(costs, keys, num_steps), keys))
    return best


def random_costs(num_steps, generator):
    errors = torch.full((num_steps,) * 3, math.nan, dtype=torch.float64)
    for i, j in itertools.combinations(range(num_steps), 2):
        for t in range(j + 1, num_steps):
            errors[i, j, t] = torch.rand((), generator=generator, dtype=torch.float64)
    return pact_costs(errors)


def test_predictor_extrapolates_linear_features_over_irregular_gaps():
    features = linear_features(12)
    for i, j, t in [(0, 1, 2), (1, 4, 5), (2, 5, 11), (0, 3, 9), (4, 7, 8)]:
        actual = predict_feature(features[i], features[j], i, j, t)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, features[t], atol=0, rtol=0)


def test_predictor_stays_in_feature_dtype_and_order():
    previous = torch.tensor([1.0, 3.0], dtype=torch.bfloat16)
    last = torch.tensor([1.5, 2.0], dtype=torch.bfloat16)
    actual = predict_feature(previous, last, 3, 6, 10)
    slope = (last - previous) / 3
    expected = last + slope * 4
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError):
        predict_feature(previous, last, 6, 3, 10)
    with pytest.raises(ValueError):
        predict_feature(previous, last, 3, 6, 6)


def test_step_errors_vanish_on_linear_trajectories():
    errors = pact_step_errors(linear_features(7))
    for i, j, t in itertools.product(range(7), repeat=3):
        if i < j < t:
            assert errors[i, j, t].item() == 0.0
        else:
            assert math.isnan(errors[i, j, t].item())


def test_step_errors_are_mean_abs_of_the_runtime_prediction():
    torch.manual_seed(0)
    features = [torch.randn(4, 5).bfloat16() for _ in range(5)]
    errors = pact_step_errors(features)
    predicted = predict_feature(features[1], features[3], 1, 3, 4)
    expected = (predicted.float() - features[4].float()).abs().double().mean()
    assert errors[1, 3, 4].item() == pytest.approx(expected.item(), rel=1e-12)


def test_costs_score_only_predicted_steps():
    generator = torch.Generator().manual_seed(1)
    num_steps = 6
    errors = torch.full((num_steps,) * 3, math.nan, dtype=torch.float64)
    for i, j in itertools.combinations(range(num_steps), 2):
        for t in range(j + 1, num_steps):
            errors[i, j, t] = torch.rand((), generator=generator, dtype=torch.float64)
    costs = pact_costs(errors)
    for i, j in itertools.combinations(range(num_steps), 2):
        assert costs[i, j, j + 1].item() == 0.0  # adjacent keys predict nothing
        for k in range(j + 2, num_steps + 1):
            expected = sum(errors[i, j, t].item() for t in range(j + 1, k))
            assert costs[i, j, k].item() == pytest.approx(expected, rel=1e-12)
        # the terminal sentinel T scores steps through T-1 and nothing at T
        assert costs[i, j, num_steps].item() == pytest.approx(
            errors[i, j, j + 1 :].sum().item(), rel=1e-12
        )
    assert math.isinf(costs[3, 2, 4].item())


@pytest.mark.parametrize("num_steps", [5, 7, 9])
@pytest.mark.parametrize("lead", [2, 3])
def test_exact_planner_matches_exhaustive_enumeration(num_steps, lead):
    generator = torch.Generator().manual_seed(num_steps * 10 + lead)
    for _ in range(4):
        costs = random_costs(num_steps, generator)
        for budget in range(lead, num_steps + 1):
            mandatory = tuple(range(lead))
            keys, value = plan_schedule(costs, budget, mandatory)
            expected_value, _ = brute_force(costs, budget, lead)
            assert len(keys) == budget and keys[:lead] == list(mandatory)
            assert value == pytest.approx(expected_value, abs=1e-12)
            assert schedule_cost(costs, keys, num_steps) == pytest.approx(
                value, abs=1e-12
            )
            if budget > lead:
                forced, forced_value = plan_schedule(
                    costs, budget, mandatory, force_last_full=True
                )
                assert forced[-1] == num_steps - 1
                assert forced_value == pytest.approx(
                    brute_force(costs, budget, lead, force_last_full=True)[0],
                    abs=1e-12,
                )


def test_predecessor_collapsed_dp_is_not_optimal_but_exact_dp_is():
    generator = torch.Generator().manual_seed(7)
    for _ in range(200):
        costs = random_costs(7, generator)
        collapsed, _ = collapsed_plan(costs, 5, 3)
        exact_keys, exact = plan_schedule(costs, 5)
        optimum, _ = brute_force(costs, 5, 3)
        assert exact == pytest.approx(optimum, abs=1e-12)
        if collapsed > optimum + 1e-9:
            break
    else:
        pytest.fail("no counterexample for the predecessor-collapsed planner")


@pytest.mark.parametrize(
    "budget, last_key, expected",
    [
        (12, 38, [0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38]),
        (
            20,
            39,
            [0, 1, 2, 4, 6, 9, 11, 13, 15, 17, 19, 22, 24, 26, 28, 30, 32, 35, 37, 39],
        ),
        (4, 39, [0, 1, 2, 39]),
    ],
)
def test_uniform_keys_match_the_prefix_budget_and_last_key(budget, last_key, expected):
    """The placement control differs from its DP schedule only in the interior."""
    keys = uniform_full_steps(40, budget, last_key)
    assert keys == expected
    assert keys[:3] == [0, 1, 2] and keys[-1] == last_key and len(keys) == budget
    assert keys == sorted(set(keys))


def test_uniform_keys_round_half_up_and_reject_infeasible_budgets():
    # r * span / interior lands exactly on .5 for r=1 of this span
    assert uniform_full_steps(40, 5, 12)[3] == 7
    for budget, last_key in ((3, 39), (41, 39), (12, 40), (12, 2)):
        with pytest.raises((ValueError, TypeError)):
            uniform_full_steps(40, budget, last_key)
    with pytest.raises(TypeError):
        uniform_full_steps(40, 12.0, 38)


def test_planner_is_deterministic_on_ties():
    costs = pact_costs(torch.zeros(6, 6, 6, dtype=torch.float64))
    assert plan_schedule(costs, 4) == plan_schedule(costs, 4) == ([0, 1, 2, 3], 0.0)


def test_planner_rejects_infeasible_budgets_and_bad_mandatory_steps():
    costs = random_costs(6, torch.Generator().manual_seed(0))
    for budget in (2, 7, -1):
        with pytest.raises(ValueError):
            plan_schedule(costs, budget)
    with pytest.raises(ValueError):
        plan_schedule(costs, 3, force_last_full=True)
    with pytest.raises(TypeError):
        plan_schedule(costs, 4.0)
    for mandatory in [(0,), (1, 2), (0, 2), ()]:
        with pytest.raises(ValueError):
            plan_schedule(costs, 4, mandatory)
    assert plan_schedule(costs, 6) == ([0, 1, 2, 3, 4, 5], 0.0)
    unreachable = costs.clone()
    unreachable[:, :, 6] = math.inf
    with pytest.raises(ValueError, match="no finite schedule"):
        plan_schedule(unreachable, 4)


def brute_force_gap(costs, num_full_steps, lead, max_gap):
    num_steps = costs.shape[0]
    best = math.inf
    for rest in itertools.combinations(range(lead, num_steps), num_full_steps - lead):
        keys = [*range(lead), *rest, num_steps]
        if all(b - a <= max_gap for a, b in zip(keys, keys[1:])):
            best = min(best, schedule_cost(costs, keys[:-1], num_steps))
    return best


@pytest.mark.parametrize("max_gap", [2, 3, 4])
def test_gap_bounded_planner_matches_exhaustive_enumeration(max_gap):
    """Every key gap, the terminal one included, must respect max_gap."""
    generator = torch.Generator().manual_seed(max_gap)
    num_steps = 9
    for _ in range(3):
        raw = torch.full((num_steps,) * 3, math.nan, dtype=torch.float64)
        for i, j in itertools.combinations(range(num_steps), 2):
            if j - i <= max_gap:
                for t in range(j + 1, min(num_steps, j + max_gap)):
                    raw[i, j, t] = torch.rand(
                        (), generator=generator, dtype=torch.float64
                    )
        costs = pact_costs(raw, max_gap=max_gap)
        for budget in range(3, num_steps + 1):
            expected = brute_force_gap(costs, budget, 3, max_gap)
            if math.isinf(expected):
                with pytest.raises(ValueError, match="no finite schedule"):
                    plan_schedule(costs, budget, max_gap=max_gap)
                continue
            keys, value = plan_schedule(costs, budget, max_gap=max_gap)
            gaps = [b - a for a, b in zip(keys, keys[1:] + [num_steps])]
            assert max(gaps) <= max_gap
            assert value == pytest.approx(expected, abs=1e-12)


def test_gap_bound_applies_to_the_terminal_segment_even_with_unmasked_costs():
    costs = pact_costs(torch.zeros(6, 6, 6, dtype=torch.float64))
    with pytest.raises(ValueError, match="no finite schedule"):
        plan_schedule(costs, 3, max_gap=2)
    keys, _ = plan_schedule(costs, 4, max_gap=2)
    assert 6 - keys[-1] <= 2 and max(b - a for a, b in zip(keys, keys[1:])) <= 2


def test_gap_bounded_errors_score_only_usable_triples():
    features = [
        torch.randn(3, generator=torch.Generator().manual_seed(s)).bfloat16()
        for s in range(8)
    ]
    errors = pact_step_errors(features, max_gap=3)
    for i, j, t in itertools.product(range(8), repeat=3):
        usable = i < j < t and j - i <= 3 and t < j + 3
        assert math.isfinite(errors[i, j, t].item()) == usable
    costs = pact_costs(errors, max_gap=3)
    assert math.isinf(costs[0, 4, 5].item())  # anchor gap 4 > 3
    assert math.isinf(costs[1, 2, 6].item())  # key gap 4 > 3
    assert math.isfinite(costs[4, 5, 8].item())  # terminal gap 3


def test_pact_costs_reject_nonfinite_errors():
    errors = torch.zeros(4, 4, 4, dtype=torch.float64)
    errors[0, 1, 3] = math.inf
    with pytest.raises(ValueError, match="nonfinite"):
        pact_costs(errors)


def test_state_uses_only_full_step_anchors():
    features = linear_features(8)
    state = DPCacheState([0, 1, 2, 5], num_steps=8)
    for step in (0, 1, 2):
        state.record(step, features[step])
    torch.testing.assert_close(state.predict(3), features[3], atol=0, rtol=0)
    torch.testing.assert_close(state.predict(4), features[4], atol=0, rtol=0)
    # a full step whose feature departs from the line resets the anchors to (2, 5)
    kink = features[5] + 1
    state.record(5, kink)
    expected = predict_feature(features[2], kink, 2, 5, 6)
    torch.testing.assert_close(state.predict(6), expected, atol=0, rtol=0)
    assert (state.num_full, state.num_predicted) == (4, 3)


def test_state_stores_a_detached_copy():
    feature = torch.ones(3, dtype=torch.bfloat16)
    state = DPCacheState([0, 1], num_steps=3)
    state.record(0, feature)
    feature.fill_(5)
    state.record(1, torch.full((3,), 2.0, dtype=torch.bfloat16))
    torch.testing.assert_close(state.predict(2), torch.full((3,), 3.0).bfloat16())


def test_state_rejects_out_of_order_unscheduled_and_non_bf16_steps():
    zero = torch.zeros(1, dtype=torch.bfloat16)
    state = DPCacheState([0, 1, 3], num_steps=4)
    with pytest.raises(RuntimeError, match="BF16 features only"):
        state.record(0, torch.zeros(1))
    with pytest.raises(RuntimeError, match="not a scheduled full step"):
        state.record(2, zero)
    state.record(0, zero)
    with pytest.raises(RuntimeError, match="fewer than two"):
        DPCacheState([0, 2], num_steps=3).predict(1)
    with pytest.raises(RuntimeError, match="must increase"):
        state.record(0, zero)
    with pytest.raises(RuntimeError, match="is a scheduled full step"):
        state.predict(1)
    with pytest.raises(ValueError, match="outside"):
        state.predict(4)


def test_states_are_isolated_per_branch():
    positive = DPCacheState([0, 1], num_steps=3)
    negative = DPCacheState([0, 1], num_steps=3)
    for step in (0, 1):
        positive.record(step, torch.full((2,), float(step), dtype=torch.bfloat16))
        negative.record(step, torch.full((2,), -2.0 * step, dtype=torch.bfloat16))
    assert positive.predict(2).tolist() == [2.0, 2.0]
    assert negative.predict(2).tolist() == [-4.0, -4.0]


SIGNATURE = DPCacheRequestSignature(
    pipeline="SimpleNamespace",
    checkpoint="790c92633540aa0cb11d9abf19eb46d861714758",
    num_inference_steps=6,
    height=64,
    width=64,
    guidance_scale=1.0,
    do_classifier_free_guidance=False,
    quality="lossless",
    attention_backend="torch_sdpa",
    dtype="bfloat16",
    scheduler="SimpleNamespace",
    scheduler_config_sha256=config_digest({"shift": 1.0}),
    timesteps=(1000.0, 800.0, 600.0, 400.0, 200.0, 100.0),
    sigmas=(1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0),
)


def artifact(full_steps=(0, 1, 2, 4), max_gap=None, **overrides):
    result = build_schedule_artifact(
        signature=SIGNATURE,
        full_steps=list(full_steps),
        mandatory=(0, 1, 2),
        calibrated_cost=0.5,
        calibration={"manifest_sha256": "0" * 64, "num_samples": 1},
        source_commit="deadbeef",
        max_gap=max_gap,
    )
    request = overrides.pop("request", {})
    result.update(overrides)
    result["request"].update(request)
    return result


def test_schedule_artifact_roundtrip_and_request_match():
    schedule = json.loads(json.dumps(artifact()))
    assert validate_schedule(schedule) == (0, 1, 2, 4)
    assert check_schedule_matches(schedule, SIGNATURE) == (0, 1, 2, 4)
    assert QwenImage21SamplingParams(dpcache_schedule=schedule).dpcache_schedule
    assert "dpcache_schedule" in QwenImage21SamplingParams.image_request_extra_fields()


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "sglang-dpcache-schedule-v1"},
        {"full_steps": [0, 1, 4, 2], "num_full_steps": 4},
        {"full_steps": [0, 2, 3, 4], "num_full_steps": 4},
        {"full_steps": [0, 1, 2, 6], "num_full_steps": 4},
        {"full_steps": [0, 1, 2, 2], "num_full_steps": 4},
        {"full_steps": [0, True, 2, 4], "num_full_steps": 4},
        {"num_full_steps": 5},
        {"mandatory_full_steps": [1, 2]},
        {"mandatory_full_steps": [0, True, 2]},
        {"predictor": {"name": "taylor", "order": 2, "arithmetic": "fp32"}},
        {"objective": "endpoint-inclusive"},
        {"force_last_full": True},
        {"max_gap": 1},
        {"max_gap": 0},
        {"source_commit": ""},
        {"calibration": {"num_samples": 1}},
        {"calibrated_cost": float("nan")},
        {"request": {"timesteps": [1.0]}},
        {"request": {"sigmas": [1.0, 0.8, 0.6, 0.4, 0.2, float("inf"), 0.0]}},
        {"request": {"height": 0}},
        {"request": {"dtype": "float32"}},
        {"request": {"extra": 1}},
    ],
)
def test_malformed_schedules_are_rejected(overrides):
    with pytest.raises(ValueError):
        validate_schedule(artifact(**overrides))
    with pytest.raises(ValueError):
        QwenImage21SamplingParams(dpcache_schedule=artifact(**overrides))


def test_schedule_gaps_include_the_terminal_segment():
    assert validate_schedule(artifact(full_steps=(0, 1, 2, 4), max_gap=2))
    with pytest.raises(ValueError, match="max_gap"):
        validate_schedule(artifact(full_steps=(0, 1, 2, 3), max_gap=2))


def test_schedule_without_required_fields_is_rejected():
    schedule = artifact()
    del schedule["calibration"]
    with pytest.raises(ValueError, match="missing"):
        validate_schedule(schedule)
    with pytest.raises(TypeError):
        validate_schedule([0, 1, 2])


@pytest.mark.parametrize(
    "field, value",
    [
        ("height", 128),
        ("num_inference_steps", 7),
        ("guidance_scale", 4.0),
        ("do_classifier_free_guidance", True),
        ("quality", "high"),
        ("attention_backend", "fa"),
        ("dtype", "float16"),
        ("checkpoint", "other-revision"),
        ("scheduler", "OtherScheduler"),
        ("scheduler_config_sha256", config_digest({"shift": 3.0})),
        ("sigmas", (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01)),
        ("timesteps", (1000.0, 800.0, 600.0, 400.0, 200.0, 99.0)),
    ],
)
def test_mismatched_request_is_rejected(field, value):
    signature = msgspec.structs.replace(SIGNATURE, **{field: value})
    with pytest.raises(ValueError, match="does not match"):
        check_schedule_matches(artifact(), signature)


def test_checkpoint_identity_uses_snapshot_revision(tmp_path):
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    assert checkpoint_identity(str(snapshot)) == "abc123"
    assert checkpoint_identity("Qwen/Qwen-Image-2.1") == "Qwen/Qwen-Image-2.1"


@pytest.fixture(autouse=True)
def single_gpu(monkeypatch):
    monkeypatch.setattr(stage_module, "get_sp_world_size", lambda: 1)
    monkeypatch.setattr(stage_module, "get_tp_world_size", lambda: 1)


class _FakeLoRA(BaseLayerWithLoRA):
    def __init__(self, disable_lora):
        torch.nn.Module.__init__(self)
        self.merged, self.disable_lora = False, disable_lora


def make_stage(dtype=torch.bfloat16, lora=None, **server_overrides):
    stage = object.__new__(QwenImage21DenoisingStage)
    transformer = torch.nn.Module()
    transformer.proj_out = torch.nn.Linear(1, 1, dtype=dtype)
    if lora is not None:
        transformer.lora = _FakeLoRA(disable_lora=not lora)
    stage.transformer = transformer
    stage._cache_dit_enabled = False
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(),
        model_path="790c92633540aa0cb11d9abf19eb46d861714758",
        attention_backend="torch_sdpa",
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        enable_cfg_parallel=False,
    )
    server_args.__dict__.update(server_overrides)
    return stage, server_args


def make_batch(schedule, **overrides):
    fields = dict(
        dpcache_schedule=schedule,
        enable_cache_dit=False,
        enable_teacache=False,
        enable_spectrum=False,
        skip_softmax_params=None,
        attention_backend_override=None,
        do_classifier_free_guidance=False,
        guidance_scale=1.0,
        quality="lossless",
        condition_image=None,
        prompt="a cat",
        num_outputs_per_prompt=1,
    )
    fields.update(overrides)
    return SimpleNamespace(
        sampling_params=SimpleNamespace(**fields),
        timesteps=torch.tensor(SIGNATURE.timesteps),
        scheduler=SimpleNamespace(
            sigmas=torch.tensor(SIGNATURE.sigmas), config={"shift": 1.0}
        ),
        height=64,
        width=64,
        is_warmup=False,
        extra={"qwen21_positive": {"layouts": []}},
        **fields,
    )


def test_stage_attaches_fresh_branch_state_and_cleans_up_on_error(monkeypatch):
    stage, server_args = make_stage()
    batch = make_batch(artifact())
    batch.extra["qwen21_negative"] = {"layouts": []}
    seen = {}

    def failing_forward(self, batch, server_args):
        seen.update(
            {k: batch.extra[k]["dpcache_state"] for k in stage_module.DPCACHE_BRANCHES}
        )
        raise RuntimeError("boom")

    monkeypatch.setattr(DenoisingStage, "forward", failing_forward)
    with pytest.raises(RuntimeError, match="boom"):
        stage.forward(batch, server_args)
    assert seen["qwen21_positive"] is not seen["qwen21_negative"]
    assert all(s.full_steps == frozenset({0, 1, 2, 4}) for s in seen.values())
    assert all("dpcache_state" not in batch.extra[k] for k in seen)


def test_stage_without_schedule_is_untouched(monkeypatch):
    stage, server_args = make_stage(lora=True)
    batch = make_batch(None, do_classifier_free_guidance=True)
    calls = []
    monkeypatch.setattr(
        DenoisingStage,
        "forward",
        lambda self, batch, server_args: calls.append(dict(batch.extra)) or batch,
    )
    assert stage.forward(batch, server_args) is batch
    assert calls == [{"qwen21_positive": {"layouts": []}}]


def test_stage_requires_the_positive_branch(monkeypatch):
    """An enabled schedule must never silently run uncached."""
    stage, server_args = make_stage()
    batch = make_batch(artifact())
    batch.extra = {}
    monkeypatch.setattr(DenoisingStage, "forward", lambda *a: pytest.fail("ran"))
    with pytest.raises(RuntimeError, match="branch kwargs"):
        stage.forward(batch, server_args)


@pytest.mark.parametrize(
    "stage_overrides, batch_overrides",
    [
        ({"enable_torch_compile": True}, {}),
        ({"enable_breakable_cuda_graph": True}, {}),
        ({"enable_cfg_parallel": True}, {}),
        ({"dtype": torch.float32}, {}),
        ({"lora": True}, {}),
        ({}, {"enable_teacache": True}),
        ({}, {"enable_spectrum": True}),
        ({}, {"enable_cache_dit": True}),
        ({}, {"skip_softmax_params": {"threshold": 0.1}}),
        ({}, {"attention_backend_override": "fa"}),
        ({}, {"do_classifier_free_guidance": True}),
        ({}, {"quality": "high"}),
        ({}, {"condition_image": "ref.png"}),
        ({}, {"prompt": ["a", "b"]}),
        ({}, {"num_outputs_per_prompt": 2}),
    ],
)
def test_stage_rejects_unsupported_compositions(
    monkeypatch, stage_overrides, batch_overrides
):
    stage, server_args = make_stage(**stage_overrides)
    batch = make_batch(artifact(), **batch_overrides)
    monkeypatch.setattr(DenoisingStage, "forward", lambda *a: pytest.fail("ran"))
    with pytest.raises(ValueError, match="cannot be combined"):
        stage.forward(batch, server_args)


def test_stage_accepts_loaded_but_disabled_lora(monkeypatch):
    stage, server_args = make_stage(lora=False)
    batch = make_batch(artifact())
    monkeypatch.setattr(DenoisingStage, "forward", lambda self, b, s: b)
    assert stage.forward(batch, server_args) is batch


def test_stale_cache_dit_mount_is_unmounted_instead_of_rejected(monkeypatch):
    """A DPCache request after a Cache-DiT request must not fail on the old mount."""
    stage, server_args = make_stage()
    stage._cache_dit_enabled = True
    unmounted = []
    monkeypatch.setattr(
        QwenImage21DenoisingStage,
        "_unmount_cache_dit",
        lambda self: (
            unmounted.append(True) or setattr(self, "_cache_dit_enabled", False)
        ),
        raising=False,
    )
    monkeypatch.setattr(DenoisingStage, "forward", lambda self, b, s: b)
    batch = make_batch(artifact())
    assert stage.forward(batch, server_args) is batch
    assert unmounted == [True]

    stage, server_args = make_stage()
    stage._cache_dit_enabled = True
    batch = make_batch(artifact(), enable_cache_dit=True)
    monkeypatch.setattr(DenoisingStage, "forward", lambda *a: pytest.fail("ran"))
    with pytest.raises(ValueError, match="Cache-DiT"):
        stage.forward(batch, server_args)


def test_stage_rejects_mismatched_schedule(monkeypatch):
    stage, server_args = make_stage()
    batch = make_batch(artifact())
    batch.scheduler.config = {"shift": 2.0}
    monkeypatch.setattr(DenoisingStage, "forward", lambda *a: pytest.fail("ran"))
    with pytest.raises(ValueError, match="does not match"):
        stage.forward(batch, server_args)
