# SPDX-License-Identifier: Apache-2.0
"""Evidence-integrity rules of the Qwen-Image 2.1 DPCache benchmark; CPU only."""

import json
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.benchmarks import bench_qwen_image21_dpcache as bench


def make_plan(arms=("native", "K20"), pairs=("p0", "p1"), timing_repeats=1):
    steps = []
    for pair in pairs:
        for arm in arms:
            steps.append(dict(pair_id=pair, arm=arm, kind="evidence", repeat=0))
    for repeat in range(timing_repeats):
        for pair in pairs:
            for arm in arms:
                steps.append(dict(pair_id=pair, arm=arm, kind="timing", repeat=repeat))
    for index, step in enumerate(steps):
        step["order_index"] = index
    return dict(steps=steps, corpus_sha256="c" * 64, split="validation", phase="sweep")


def score(plan, index, lpips, status="measured"):
    step = plan["steps"][index]
    entry = dict(
        order_index=index,
        arm=step["arm"],
        kind=step["kind"],
        repeat=step["repeat"],
        pair_id=step["pair_id"],
        status=status,
    )
    if status == "measured":
        entry.update(
            latents_equal=False,
            samples_equal=False,
            png_equal=False,
            lpips_white=lpips,
            lpips_rgb_dropped_alpha=lpips,
            alpha_mean_abs=0.0,
            alpha_max_abs=0.0,
            category="c",
            stem=f"s{index}",
        )
    return entry


def test_failed_capture_keeps_an_arm_from_passing_on_a_favorable_subset():
    """A planned capture that failed must not drop out of the gate denominator."""
    plan = make_plan()
    # p0/K20 is excellent; p1/K20 failed and would have been the bad one
    scores = [score(plan, 1, 0.01), score(plan, 3, None, status="failed")]
    quality = bench.arm_quality(plan, scores)
    assert quality["K20"]["missing_or_failed"] == [3]
    assert not quality["K20"]["complete"]
    assert not quality["K20"]["gate_provisional"]
    assert not quality["K20"]["gate_strict"]


def test_capture_absent_from_scores_is_incomplete():
    plan = make_plan()
    quality = bench.arm_quality(plan, [score(plan, 1, 0.01)])
    assert quality["K20"]["missing_or_failed"] == [3]
    assert not quality["K20"]["gate_provisional"]


def test_complete_arm_is_gated_on_mean_and_max():
    plan = make_plan()
    quality = bench.arm_quality(plan, [score(plan, 1, 0.05), score(plan, 3, 0.29)])
    assert quality["K20"]["gate_provisional"]
    assert not quality["K20"]["gate_strict"]
    quality = bench.arm_quality(plan, [score(plan, 1, 0.01), score(plan, 3, 0.31)])
    assert not quality["K20"]["gate_provisional"]


def test_duplicate_scores_or_rows_are_rejected():
    plan = make_plan()
    with pytest.raises(ValueError, match="duplicate score"):
        bench.arm_quality(plan, [score(plan, 1, 0.01), score(plan, 1, 0.01)])
    with pytest.raises(ValueError, match="duplicate result rows"):
        bench.rows_by_step([dict(order_index=4), dict(order_index=4)])


def test_missing_or_failed_reference_is_an_error():
    plan = make_plan()
    rows = [
        dict(order_index=0, status="measured"),
        dict(order_index=2, status="failed"),
    ]
    with pytest.raises(ValueError, match="no measured native reference for p1"):
        bench.resolve_references(plan, rows, {"p0", "p1"})
    assert bench.resolve_references(plan, rows, {"p0"})["p0"]["order_index"] == 0


def test_scores_are_bound_to_the_exact_results(tmp_path):
    run, ref = tmp_path / "run", tmp_path / "ref"
    for path in (run, ref):
        path.mkdir()
        (path / "plan.json").write_text("{}")
        (path / "results.jsonl").write_text("{}\n")
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            dict(
                reference_run=str(ref),
                run_digests=bench.run_digests(run),
                reference_digests=bench.run_digests(ref),
                scores=[],
            )
        )
    )
    bench.load_bound_scores(run, scores)
    (run / "results.jsonl").write_text('{"rerun": 1}\n')
    with pytest.raises(ValueError, match="stale"):
        bench.load_bound_scores(run, scores)
    (run / "results.jsonl").write_text("{}\n")
    (ref / "results.jsonl").write_text('{"rerun": 1}\n')
    with pytest.raises(ValueError, match="reference run changed"):
        bench.load_bound_scores(run, scores)


def test_mismatched_reference_request_is_rejected():
    plan = make_plan()
    env = [dict(kind="environment", runner_config={"model_path": "m"})]
    row = dict(stem="a", prompt="cat", seed=1)
    bench.check_comparable(env, env, plan, plan, [(row, dict(row))])
    with pytest.raises(ValueError, match="prompt/seed"):
        bench.check_comparable(env, env, plan, plan, [(row, dict(row, seed=2))])
    other = [dict(kind="environment", runner_config={"model_path": "other"})]
    with pytest.raises(ValueError, match="runner configs"):
        bench.check_comparable(env, other, plan, plan, [])
    with pytest.raises(ValueError, match="corpora"):
        bench.check_comparable(env, env, plan, dict(plan, corpus_sha256="d" * 64), [])


def timing_rows(plan, walls, skip=()):
    rows = []
    for step in plan["steps"]:
        if step["kind"] == "timing" and step["order_index"] not in skip:
            rows.append(
                dict(
                    order_index=step["order_index"],
                    status="measured",
                    client_wall_seconds=walls[step["arm"]],
                )
            )
    return rows


def test_selection_ignores_arms_with_incomplete_timing():
    plan = make_plan(arms=("native", "K12", "K20"))
    quality = {
        arm: dict(gate_provisional=True, gate_strict=arm == "K20")
        for arm in ("K12", "K20")
    }
    walls = {"native": 14.0, "K12": 5.0, "K20": 7.0}
    timing = bench.timing_summary(plan, timing_rows(plan, walls))
    assert bench.select_candidates(quality, timing) == {
        "provisional": "K12",
        "strict": "K20",
    }
    k12_timing = [
        s["order_index"]
        for s in plan["steps"]
        if s["kind"] == "timing" and s["arm"] == "K12"
    ][:1]
    timing = bench.timing_summary(plan, timing_rows(plan, walls, skip=k12_timing))
    assert not timing["K12"]["complete"]
    assert bench.select_candidates(quality, timing)["provisional"] == "K20"


def test_missing_native_timing_makes_the_paired_candidate_ineligible():
    """A speedup must never rest on the subset of blocks whose native survived."""
    plan = make_plan(arms=("native", "K12"))
    quality = {"K12": dict(gate_provisional=True, gate_strict=True)}
    walls = {"native": 14.0, "K12": 5.0}
    native_timing = [
        s["order_index"]
        for s in plan["steps"]
        if s["kind"] == "timing" and s["arm"] == "native"
    ][:1]
    timing = bench.timing_summary(plan, timing_rows(plan, walls, skip=native_timing))
    assert timing["K12"]["measured"] == timing["K12"]["planned"]
    assert timing["K12"]["paired_count"] == 1
    assert not timing["K12"]["complete"]
    assert bench.select_candidates(quality, timing) == {
        "provisional": None,
        "strict": None,
    }


def test_heldout_plan_requires_the_frozen_selection(tmp_path):
    schedule = tmp_path / "K20.json"
    schedule.write_text(json.dumps({"num_full_steps": 20}))
    corpus = bench.build_corpus("rev")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            dict(
                corpus_sha256=corpus["corpus_sha256"],
                arms={"K20": bench.schedule_arm(schedule)},
            )
        )
    )
    args = SimpleNamespace(
        phase="sweep",
        split="heldout",
        arms=["K20"],
        schedule_dir=str(tmp_path),
        selection=str(selection),
        evidence=True,
        timing_repeats=1,
        timing_pairs=2,
        order_seed=0,
        source_dir=str(tmp_path),
        model_path=str(tmp_path),
    )
    plan = bench.build_plan(args, corpus)
    assert plan["selection"]["sha256"] == bench.sha256_file(selection)
    with pytest.raises(ValueError, match="exactly the frozen selection"):
        bench.build_plan(
            SimpleNamespace(**{**vars(args), "arms": ["K20", "K12"]}), corpus
        )
    schedule.write_text(json.dumps({"num_full_steps": 21}))
    with pytest.raises(ValueError, match="changed after selection"):
        bench.build_plan(args, corpus)
