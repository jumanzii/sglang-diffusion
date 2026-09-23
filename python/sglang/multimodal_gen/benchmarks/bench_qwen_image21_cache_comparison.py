# SPDX-License-Identifier: Apache-2.0
"""Compare the existing Qwen-Image 2.1 caching strategies on one GPU.

Arms are the strategies SGLang already ships: native, DPCache (calibrated DP
schedules and endpoint-matched uniform controls that share its runtime), and
Cache-DiT presets. `enable_teacache` is carried only as an unsupported/no-op
diagnostic: Qwen-Image 2.1 has no TeaCache adapter, so it must reproduce native
output and the full block count.

Run by path, like the DPCache benchmark it extends:
  corpus    freeze the comparator corpus (reused controls + fresh held-out)
  schedules build endpoint-matched uniform schedules from frozen DP artifacts
  launch    freeze a plan and run it under the GPU flock (controls, validation,
            heldout, service)
  report    quality, timing and real block accounting per arm
  select    freeze at most two tuned Cache-DiT configs from validation only
Scoring and contact sheets reuse `bench_qwen_image21_dpcache.py` unchanged.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from functools import partial
from pathlib import Path

from sglang.multimodal_gen.benchmarks import bench_qwen_image21_dpcache as bench

HARNESS = Path(__file__).resolve()
CAPTURE_ENV = "DPCACHE_COMPARATOR_CAPTURE"
MODEL_MODULE = "sglang.multimodal_gen.runtime.models.dits.qwen_image21"
STEPS, LAYERS = bench.STEPS, 32
MANDATORY = bench.MANDATORY

# Frozen Cache-DiT knob sets. Stock mirrors envs.py; the other three are the
# predeclared presets, TaylorSeer and DMD off everywhere.
CACHE_DIT_STOCK = {
    "Fn_compute_blocks": 1,
    "Bn_compute_blocks": 0,
    "max_warmup_steps": 4,
    "residual_diff_threshold": 0.24,
    "max_continuous_cached_steps": 3,
    "enable_taylorseer": False,
    "enable_dmd": False,
}
CACHE_DIT_PRESETS = {
    "cachedit-stock": CACHE_DIT_STOCK,
    "cachedit-conservative": {
        **CACHE_DIT_STOCK,
        "Bn_compute_blocks": 1,
        "max_warmup_steps": 10,
        "residual_diff_threshold": 0.12,
        "max_continuous_cached_steps": 1,
    },
    "cachedit-short-cache": {
        **CACHE_DIT_STOCK,
        "residual_diff_threshold": 0.12,
        "max_continuous_cached_steps": 1,
    },
    "cachedit-aggressive": {
        **CACHE_DIT_STOCK,
        "residual_diff_threshold": 0.40,
    },
    # control: every step warms up, so every block must run
    "cachedit-forced-full": {
        **CACHE_DIT_STOCK,
        "max_warmup_steps": STEPS,
        "residual_diff_threshold": 0.0,
    },
}
SCHEDULE_ARMS = {
    "dp-K12": "K12",
    "dp-K20": "K20",
    "dp-K40": "K40",
    "uniform-K12": "uniform-K12",
    "uniform-K20": "uniform-K20",
    "uniform-K40": "uniform-K40",
}
TEACACHE_ARM = "teacache-stock-unsupported"


def arm_family(name):
    if name == "native":
        return "native"
    if name in SCHEDULE_ARMS:
        return "uniform" if name.startswith("uniform") else "dpcache"
    if name in CACHE_DIT_PRESETS:
        return "cache_dit"
    if name == TEACACHE_ARM:
        return "teacache_unsupported"
    raise ValueError(f"unknown arm {name}")


def arm_request(name, schedule_dir):
    """Per-arm request parameters; every other request setting is identical."""
    if name == "native":
        return {}, None
    if name in SCHEDULE_ARMS:
        path = Path(schedule_dir) / f"{SCHEDULE_ARMS[name]}.json"
        schedule = json.loads(path.read_text())
        return {"dpcache_schedule": schedule}, dict(
            schedule=str(path.resolve()),
            schedule_sha256=bench.sha256_file(path),
            num_full_steps=schedule["num_full_steps"],
            schedule_method=schedule.get("schedule_method", "exact-pair-state-dp"),
            full_steps=schedule["full_steps"],
        )
    if name in CACHE_DIT_PRESETS:
        params = dict(CACHE_DIT_PRESETS[name])
        return {"enable_cache_dit": True, "cache_dit_params": params}, dict(
            cache_dit_params=params,
            config_sha256=bench.sha256_bytes(bench.canonical(params)),
        )
    if name == TEACACHE_ARM:
        # diagnostic only: Qwen-Image 2.1 has no TeaCache adapter
        return {"enable_teacache": True}, dict(unsupported_diagnostic=True)
    raise ValueError(f"unknown arm {name}")


# ---------------------------------------------------------------------------
# Corpus: three reused control prompts plus fresh held-out prompts
# ---------------------------------------------------------------------------

CONTROL_PROMPTS = bench.CALIBRATION_PROMPTS[:3]
HELDOUT_PROMPTS = [
    (
        "typography",
        'A vintage enamel shop sign reading "BLUE WHALE LAUNDRY, EST 1931" above a narrow doorway',
    ),
    (
        "counting-spatial",
        "Six wooden chairs around a round table, three with red cushions and three with blue cushions",
    ),
    (
        "faces-hands",
        "A close-up of a potter's hands shaping a clay bowl on a spinning wheel, studio light",
    ),
    (
        "faces",
        "A smiling barista with freckles and short dreadlocks handing over a paper cup, shallow depth of field",
    ),
    (
        "complex-scene",
        "A crowded train platform at rush hour with departure boards, luggage and a busker playing violin",
    ),
    (
        "illustration",
        "A retro travel poster of a lighthouse on a cliff, bold geometric shapes and three flat colors",
    ),
    (
        "color",
        "A macro photograph of oil and water droplets refracting magenta, teal and amber light",
    ),
    (
        "transparency",
        "A clear glass perfume bottle with a faceted stopper on a mirrored tray, soft window light",
    ),
    (
        "transparency",
        "A stack of translucent acrylic sheets in mint, rose and amber on a white studio background",
    ),
    (
        "natural-scene",
        "A misty pine forest at dawn with a deer drinking from a shallow stream",
    ),
]
HELDOUT_SEEDS = (7001, 7002)
CONTROL_SEED = 31


def build_corpus(model_revision):
    controls = [
        dict(
            pair_id=f"controls:{bench.sha256_bytes(prompt.encode())[:12]}:{CONTROL_SEED}",
            split="controls",
            category=category,
            prompt=prompt,
            seed=CONTROL_SEED,
        )
        for category, prompt in CONTROL_PROMPTS
    ]
    heldout = [
        dict(
            pair_id=f"comparator-heldout:{bench.sha256_bytes(prompt.encode())[:12]}:{seed}",
            split="heldout",
            category=category,
            prompt=prompt,
            seed=seed,
        )
        for seed in HELDOUT_SEEDS
        for category, prompt in HELDOUT_PROMPTS
    ]
    dpcache_prompts = {
        p
        for group in (
            bench.CALIBRATION_PROMPTS,
            bench.VALIDATION_PROMPTS,
            bench.HELDOUT_PROMPTS,
        )
        for _, p in group
    }
    fresh = {p["prompt"] for p in heldout}
    if fresh & dpcache_prompts:
        raise ValueError("held-out prompts must be new for this comparison")
    corpus = dict(
        schema="dpcache-comparator-corpus-v1",
        settings=dict(
            model_revision=model_revision,
            width=bench.WIDTH,
            height=bench.HEIGHT,
            steps=STEPS,
            guidance_scale=1.0,
            negative_prompt=None,
            num_outputs_per_prompt=1,
            generator_device="cpu",
        ),
        note="controls reuse the first three DPCache calibration prompts (already "
        "seen, exactness only); held-out prompts are new for this comparison",
        splits=dict(controls=controls, heldout=heldout),
    )
    corpus["corpus_sha256"] = bench.sha256_bytes(bench.canonical(corpus))
    return corpus


# ---------------------------------------------------------------------------
# Uniform schedules derived from the frozen DP artifacts
# ---------------------------------------------------------------------------


def diagnostic_pact_cost(pact_errors, keys, max_gap):
    """PACT cost of an explicit key set; a diagnostic, never a minimized value."""
    import numpy as np
    import torch

    from sglang.multimodal_gen.runtime.cache import dpcache

    mean = torch.from_numpy(np.load(pact_errors)["mean"])
    costs = dpcache.pact_costs(mean, max_gap=max_gap)
    return dpcache.schedule_cost(costs, keys, STEPS)


def cmd_schedules(args):
    from sglang.multimodal_gen.runtime.cache import dpcache

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for budget, source in ((12, "K12"), (20, "K20"), (40, "K20")):
        reference = json.loads((Path(args.schedule_dir) / f"{source}.json").read_text())
        last_key = reference["full_steps"][-1] if budget != 40 else STEPS - 1
        keys = dpcache.uniform_full_steps(STEPS, budget, last_key, MANDATORY)
        signature = dpcache.DPCacheRequestSignature(
            **{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in reference["request"].items()
            }
        )
        calibration = dict(
            reference["calibration"],
            note="uniform placement control; keys are NOT DP-optimized. Only the "
            "interior spacing differs from the named DP schedule; prefix, budget "
            "and last key match it.",
            uniform_reference_schedule=f"{source}.json",
            uniform_generator=dpcache.UNIFORM_GENERATOR,
        )
        cost = diagnostic_pact_cost(args.pact_errors, keys, reference["max_gap"])
        calibration["diagnostic_pact_cost_of_uniform_keys"] = cost
        calibration["calibrated_cost_semantics"] = (
            "diagnostic PACT cost of these uniform keys on the existing "
            "calibration features; it was NOT minimized over key sets"
        )
        artifact = dpcache.build_schedule_artifact(
            signature=signature,
            full_steps=keys,
            mandatory=MANDATORY,
            calibrated_cost=cost,
            calibration=calibration,
            source_commit=args.source_commit,
            max_gap=reference["max_gap"],
            schedule_method=dpcache.UNIFORM_GENERATOR,
        )
        name = f"uniform-K{budget}"
        bench.write_json(out / f"{name}.json", artifact)
        written[name] = dict(
            full_steps=keys,
            diagnostic_pact_cost=cost,
            sha256=bench.sha256_file(out / f"{name}.json"),
        )
    print(json.dumps(written, indent=1))
    return 0


# ---------------------------------------------------------------------------
# Worker hooks: count the real blocks, survive the Cache-DiT wrapper
# ---------------------------------------------------------------------------


def _install_worker_hooks(config):
    import importlib

    import torch

    from sglang.multimodal_gen.runtime.managers.memory_managers import (
        layerwise_offload,
    )

    stage_cls = importlib.import_module(bench.STAGE_MODULE).QwenImage21DenoisingStage
    block_cls = importlib.import_module(MODEL_MODULE).QwenImage21TransformerBlock
    if getattr(stage_cls.forward, "_comparator_bench", False):
        return
    native_forward = stage_cls.forward
    manager_cls = layerwise_offload.LayerwiseOffloadManager
    native_prefetch = manager_cls.prefetch_layer
    active = {}
    # record what Cache-DiT actually stores; reading buffers after the request
    # can show an empty dict because the library clears them
    context_cls = importlib.import_module(
        "cache_dit.caching.cache_contexts.cache_context"
    ).CachedContext
    native_set_buffer = context_cls.set_buffer

    def set_buffer(self, name, buffer):
        record = active.get("record")
        if record is not None and isinstance(buffer, torch.Tensor):
            record["cache_buffer_dtypes"][f"{self.name}.{name}"] = [
                str(buffer.dtype),
                list(buffer.shape),
            ]
        return native_set_buffer(self, name, buffer)

    def prefetch_layer(self, layer_idx, non_blocking=True):
        record = active.get("record")
        if record is None:
            return native_prefetch(self, layer_idx, non_blocking)

        def resident():
            return layer_idx in self._gpu_layers or layer_idx in self._courier_inflight

        before = resident()
        result = native_prefetch(self, layer_idx, non_blocking)
        if not before and resident():
            record["layer_loads_by_step"][bench._current_step()] += 1
        return result

    def count_block(record, layer_id, module, args):
        step = bench._current_step()
        record["blocks_by_step"][step] += 1
        record["layers_by_step"].setdefault(step, []).append(layer_id)

    def cache_dit_buffer_dtypes(transformer):
        """Dtypes of the tensors Cache-DiT actually keeps, not just the weights."""
        audit = {}
        for module in transformer.modules():
            manager = getattr(module, "context_manager", None)
            contexts = getattr(manager, "_cached_context_manager", None)
            if not contexts:
                continue
            for context in contexts.values():
                for key, value in getattr(context, "buffers", {}).items():
                    if isinstance(value, torch.Tensor):
                        audit[f"{context.name}.{key}"] = [
                            str(value.dtype),
                            list(value.shape),
                        ]
        return audit

    def forward(self, batch, server_args):
        flag = Path(config["flag"])
        if not flag.exists():
            return native_forward(self, batch, server_args)
        request = json.loads(flag.read_text())
        record = {
            "blocks_by_step": defaultdict(int),
            "layer_loads_by_step": defaultdict(int),
            "layers_by_step": {},
            "cache_buffer_dtypes": {},
        }
        blocks = [m for m in self.transformer.modules() if isinstance(m, block_cls)]
        handles = [
            block.register_forward_pre_hook(
                partial(count_block, record, block._layer_id)
            )
            for block in blocks
        ]
        feature_dtypes = []
        handles.append(
            self.transformer.norm_out.register_forward_pre_hook(
                lambda module, args: feature_dtypes.append(str(args[0].dtype))
            )
        )
        artifact = {
            "stem": request["stem"],
            "real_blocks_discovered": len(blocks),
            "block_class": block_cls.__name__,
            "transformer_blocks_attr_len": len(self.transformer.transformer_blocks),
            "requested_scheduler_steps": len(batch.timesteps),
        }
        scheduler_calls = {"count": 0}
        scheduler = batch.scheduler
        native_scheduler_step = scheduler.step

        # wraps() keeps the signature the stage inspects to filter step kwargs
        @functools.wraps(native_scheduler_step)
        def counted_step(*args, **kwargs):
            scheduler_calls["count"] += 1
            return native_scheduler_step(*args, **kwargs)

        scheduler.step = counted_step
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        active["record"] = record
        try:
            batch = native_forward(self, batch, server_args)
        finally:
            active.pop("record", None)
            for handle in handles:
                handle.remove()
            del scheduler.step
        end.record()
        end.synchronize()
        artifact["denoise_device_ms"] = start.elapsed_time(end)
        artifact["blocks_by_step"] = dict(record["blocks_by_step"])
        artifact["layer_loads_by_step"] = dict(record["layer_loads_by_step"])
        artifact["layers_by_step"] = {
            str(step): sorted(ids) for step, ids in record["layers_by_step"].items()
        }
        artifact["final_feature_dtypes"] = sorted(set(feature_dtypes))
        artifact["transformer_weight_dtype"] = str(
            self.transformer.proj_out.weight.dtype
        )
        artifact["actual_scheduler_steps"] = scheduler_calls["count"]
        artifact["cache_dit_buffer_dtypes"] = dict(record["cache_buffer_dtypes"])
        artifact["cache_dit_buffer_dtypes_after_request"] = cache_dit_buffer_dtypes(
            self.transformer
        )
        directory = Path(config["dir"])
        torch.save(
            batch.latents.detach().cpu().contiguous(),
            directory / f"{request['stem']}-latents.pt",
        )
        bench.write_json(directory / f"{request['stem']}-capture.json", artifact)
        return batch

    forward._comparator_bench = True
    stage_cls.forward = forward
    manager_cls.prefetch_layer = prefetch_layer
    context_cls.set_buffer = set_buffer


if os.environ.get(CAPTURE_ENV):
    _install_worker_hooks(json.loads(os.environ[CAPTURE_ENV]))


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

CONTROL_ARMS = ("native", "dp-K40", "uniform-K40", "cachedit-forced-full", TEACACHE_ARM)


def select_timing_pairs(pairs, args):
    """Deterministic timing subset: explicit indices, spread over the manifest."""
    if args.timing_pair_indices:
        chosen = [pairs[i] for i in args.timing_pair_indices]
    else:
        chosen = pairs[: args.timing_pairs] if args.timing_pairs else list(pairs)
    prompts = [p["prompt"] for p in chosen]
    if len(set(prompts)) != len(prompts):
        raise ValueError("timing pairs must use distinct prompts")
    return chosen


def build_plan(args, corpus):
    plan = bench.PlanBuilder()
    arms, split = {}, args.split
    pairs = corpus["splits"][split]
    order = random.Random(args.order_seed)

    def register(name):
        request, metadata = arm_request(name, args.schedule_dir)
        arms[name] = dict(
            family=arm_family(name), request=request, metadata=metadata or {}
        )

    if args.phase == "controls":
        # every mode, then native again, then a failing request, then native
        for name in CONTROL_ARMS:
            register(name)
        register("dp-K12")
        for pair in pairs:
            plan.add(pair, "native", "evidence")
            for name in CONTROL_ARMS[1:]:
                plan.add(pair, name, "evidence")
                plan.add(pair, "native", "evidence", repeat=1)
        plan.add(pairs[0], "dp-K12", "expect-failure")
        plan.add(pairs[0], "native", "evidence", repeat=2)
    elif args.phase in ("validation", "heldout"):
        # native is the paired reference of every arm, so it is always present
        for name in ("native", *[a for a in args.arms if a != "native"]):
            register(name)
        for name in arms:
            plan.add(pairs[0], name, "warmup")
        if args.evidence:
            for pair in pairs:
                for name in arms:
                    plan.add(pair, name, "evidence")
        timing_pairs = select_timing_pairs(pairs, args)
        for repeat in range(args.timing_repeats):
            for pair in timing_pairs:
                block = list(arms)
                order.shuffle(block)
                for name in block:
                    plan.add(pair, name, "timing", repeat=repeat)
        # grouped, same-config repeats: a deployed service pays a mount once
        for name in arms:
            plan.add(timing_pairs[0], name, "warmup", repeat=1)
            for repeat in range(args.service_repeats):
                for pair in timing_pairs:
                    plan.add(pair, name, "service-timing", repeat=repeat)
    elif args.phase == "service":
        # grouped, same-config repeats: isolates Cache-DiT mount/unmount churn
        for name in ("native", *[a for a in args.arms if a != "native"]):
            register(name)
        timing_pairs = select_timing_pairs(pairs, args)
        for name in arms:
            plan.add(timing_pairs[0], name, "warmup")
            for repeat in range(args.service_repeats or args.timing_repeats):
                for pair in timing_pairs:
                    plan.add(pair, name, "service-timing", repeat=repeat)
    else:
        raise ValueError(f"unknown phase {args.phase}")
    used = {s["pair_id"] for s in plan.steps}
    return dict(
        schema="dpcache-comparator-plan-v1",
        phase=args.phase,
        split=split,
        source_dir=str(Path(args.source_dir).resolve()),
        model_path=str(Path(args.model_path).resolve()),
        corpus=str(Path(args.corpus).resolve()),
        corpus_sha256=corpus["corpus_sha256"],
        pairs=[
            p
            for group in corpus["splits"].values()
            for p in group
            if p["pair_id"] in used
        ],
        arms=arms,
        steps=plan.steps,
        order_seed=args.order_seed,
        harness_sha256=bench.sha256_file(HARNESS),
        dpcache_harness_sha256=bench.sha256_file(bench.HARNESS),
        timing_regimes=dict(
            primary="service-timing (grouped same-config repeats)",
            secondary="timing (randomized mixed-mode; mount overhead only)",
            mixed_repeats=args.timing_repeats,
            service_repeats=args.service_repeats,
        ),
        timing_pair_indices=list(args.timing_pair_indices or []),
        timing_pair_ids=[
            p["pair_id"]
            for p in select_timing_pairs(corpus["splits"][args.split], args)
        ]
        if args.phase != "controls"
        else [],
        selection=None
        if args.selection is None
        else dict(
            path=str(Path(args.selection).resolve()),
            sha256=bench.sha256_file(args.selection),
        ),
    )


def check_selection(args, corpus):
    """Held-out runs may only use the frozen arms, configs and corpus."""
    selection = json.loads(Path(args.selection).read_text())
    if set(args.arms) != set(selection["final_arms"]):
        raise SystemExit(
            f"held-out arms {sorted(args.arms)} differ from the frozen selection "
            f"{sorted(selection['final_arms'])}"
        )
    if corpus["corpus_sha256"] != selection["expected_heldout_corpus_sha256"]:
        raise SystemExit("held-out corpus differs from the one the selection froze")
    if (
        bench.run_digests(selection["validation_run"])
        != selection["validation_digests"]
    ):
        raise SystemExit("the validation run changed after the selection was frozen")
    if bench.sha256_file(selection["scores"]) != selection["scores_sha256"]:
        raise SystemExit("the validation scores changed after the selection was frozen")
    for name in args.arms:
        _, metadata = arm_request(name, args.schedule_dir)
        if (metadata or {}) != (selection["arms"][name]["metadata"] or {}):
            raise SystemExit(f"arm {name} config differs from the frozen selection")
    return selection


def cmd_launch(args):
    corpus = bench.load_corpus(args.corpus)
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; runs are never overwritten")
    if args.phase == "heldout":
        check_selection(args, corpus)
    plan = build_plan(args, corpus)
    handle = bench.acquire_gpu(args.gpu_lock, args.lock_wait)
    try:
        out.mkdir(parents=True)
        bench.write_json(out / "plan.json", plan)
        env = dict(
            os.environ, PYTHONPATH=plan["source_dir"], PYTHONDONTWRITEBYTECODE="1"
        )
        if args.gpu_lock:
            env[bench.LOCK_ENV] = str(Path(args.gpu_lock).resolve())
        with open(out / "worker.log", "ab") as log:
            rc = subprocess.call(
                [args.python, str(HARNESS), "worker", str(out / "plan.json")],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(out),
            )
    finally:
        if handle is not None:
            handle.close()
    print(f"{out}: rc={rc}", flush=True)
    return rc


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def cmd_worker(args):
    plan_path = Path(args.plan).resolve()
    out = plan_path.parent
    plan = json.loads(plan_path.read_text())
    capture_dir = out / "capture"
    capture_dir.mkdir(exist_ok=True)
    flag = out / "capture.flag"

    def emit(row):
        with (out / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row, default=str) + "\n")

    lock = bench.check_lock_held()
    inherited = sorted(k for k in os.environ if k.startswith("SGLANG_CACHE_DIT"))
    if inherited:
        raise SystemExit(
            f"inherited Cache-DiT env would change the stock knobs: {inherited}"
        )
    os.environ[CAPTURE_ENV] = json.dumps({"flag": str(flag), "dir": str(capture_dir)})
    _install_worker_hooks(json.loads(os.environ[CAPTURE_ENV]))
    smi = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm",
            "--format=csv,noheader",
            "-l",
            "1",
        ],
        stdout=open(out / "nvidia-smi.csv", "w"),
        stderr=subprocess.DEVNULL,
    )
    runner, failures = None, 0
    try:
        import cache_dit

        common = dict(
            phase=plan["phase"],
            split=plan["split"],
            corpus_sha256=plan["corpus_sha256"],
            harness_sha256=bench.sha256_file(HARNESS),
            gpu_lock=lock,
            model_path=plan["model_path"],
            cache_dit_version=cache_dit.__version__,
            **{
                "source_" + k: v
                for k, v in bench.source_identity(plan["source_dir"]).items()
            },
        )
        emit(
            dict(
                common,
                kind="environment",
                **bench.environment_record(),
                runner_config=bench.runner_config(plan["model_path"]),
                arms={name: arm["metadata"] for name, arm in plan["arms"].items()},
            )
        )
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )

        setup_start = time.perf_counter()
        runner = DiffGenerator.from_pretrained(
            **bench.runner_config(plan["model_path"])
        )
        emit(
            dict(common, kind="setup", setup_seconds=time.perf_counter() - setup_start)
        )
        pairs = {p["pair_id"]: p for p in plan["pairs"]}
        for step in plan["steps"]:
            pair, arm = pairs[step["pair_id"]], step["arm"]
            stem = f"{step['order_index']:04d}-{arm}-{step['kind']}-{step['repeat']}"
            row = dict(
                common,
                **step,
                prompt=pair["prompt"],
                seed=pair["seed"],
                pair_split=pair["split"],
                category=pair["category"],
                stem=stem,
                family=plan["arms"][arm]["family"],
                arm_metadata=plan["arms"][arm]["metadata"],
            )
            captured = step["kind"] == "evidence"
            try:
                kwargs = dict(
                    bench.request_kwargs(pair), **plan["arms"][arm]["request"]
                )
                if step["kind"] == "expect-failure":
                    # a schedule that cannot match this request must be refused
                    kwargs["dpcache_schedule"] = dict(
                        kwargs["dpcache_schedule"],
                        request=dict(kwargs["dpcache_schedule"]["request"], height=512),
                    )
                if captured:
                    bench.write_json(flag, dict(stem=stem))
                try:
                    result, wall = bench.run_request(runner, kwargs)
                finally:
                    flag.unlink(missing_ok=True)
                if step["kind"] == "expect-failure":
                    raise AssertionError("a mismatched schedule was accepted")
                row.update(
                    status="measured",
                    client_wall_seconds=wall,
                    peak_memory_mb=result.peak_memory_mb,
                    metrics=json.loads(json.dumps(result.metrics, default=str)),
                )
                if captured:
                    row.update(
                        save_evidence(
                            out,
                            capture_dir,
                            stem,
                            result,
                            arm,
                            row["family"],
                            plan["arms"][arm]["metadata"].get("full_steps"),
                        )
                    )
            except Exception as exc:
                if step["kind"] == "expect-failure" and not isinstance(
                    exc, AssertionError
                ):
                    row.update(
                        status="failed-as-expected",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    failures += 1
                    traceback.print_exc()
                    row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            emit(row)
    except Exception as exc:
        failures += 1
        traceback.print_exc()
        emit(dict(kind="fatal", status="failed", reason=f"{type(exc).__name__}: {exc}"))
    finally:
        if runner is not None:
            runner.shutdown()
        smi.terminate()
    return 1 if failures else 0


ALL_FULL_ARMS = (
    "native",
    "dp-K40",
    "uniform-K40",
    "cachedit-forced-full",
    TEACACHE_ARM,
)


def audit_evidence(family, arm, capture, counts):
    """Fail closed: an arm whose instrumentation cannot be trusted is ineligible."""
    notes = []
    if capture["real_blocks_discovered"] != LAYERS:
        notes.append(
            f"discovered {capture['real_blocks_discovered']} real blocks, expected {LAYERS}"
        )
    if capture["actual_scheduler_steps"] != STEPS:
        notes.append(
            f"scheduler.step ran {capture['actual_scheduler_steps']} times, expected {STEPS}"
        )
    if capture["final_feature_dtypes"] != ["torch.bfloat16"]:
        notes.append(f"final features {capture['final_feature_dtypes']}")
    if capture["transformer_weight_dtype"] != "torch.bfloat16":
        notes.append(f"weights {capture['transformer_weight_dtype']}")
    for step, ids in capture["layers_by_step"].items():
        if len(set(ids)) != len(ids):
            notes.append(f"duplicate layer ids at step {step}")
            break
    if arm in ALL_FULL_ARMS and counts["total_blocks"] != STEPS * LAYERS:
        notes.append(
            f"all-full arm ran {counts['total_blocks']} blocks, expected {STEPS * LAYERS}"
        )
    if family in ("dpcache", "uniform"):
        budget = len(capture["expected_full_steps"])
        if counts["total_blocks"] != budget * LAYERS:
            notes.append(f"{counts['total_blocks']} blocks, expected {budget * LAYERS}")
        if counts["full_block_steps"] != budget or counts["partial_block_steps"]:
            notes.append(
                f"{counts['full_block_steps']} full and {counts['partial_block_steps']} "
                f"partial steps, expected {budget} full and none partial"
            )
        observed = sorted(int(s) for s, n in capture["blocks_by_step"].items() if n)
        if observed != sorted(capture["expected_full_steps"]):
            notes.append("executed steps differ from the schedule")
    if family == "cache_dit" and arm != "cachedit-forced-full":
        dtypes = {d[0] for d in capture["cache_dit_buffer_dtypes"].values()}
        if not dtypes:
            notes.append("no Cache-DiT tensor buffer was observed while caching")
        elif dtypes != {"torch.bfloat16"}:
            notes.append(f"Cache-DiT cached tensors {sorted(dtypes)}")
    return notes


def save_evidence(out, capture_dir, stem, result, arm, family, expected_full_steps):
    import torch
    from PIL import Image

    samples = result.samples.detach().cpu().contiguous()
    torch.save(samples, out / f"{stem}-samples.pt")
    image = Image.fromarray(result.frames[0])
    image.save(out / f"{stem}.png")
    capture = json.loads((capture_dir / f"{stem}-capture.json").read_text())
    capture["expected_full_steps"] = expected_full_steps or []
    blocks = {int(k): v for k, v in capture["blocks_by_step"].items()}
    steps_with_blocks = {s: n for s, n in blocks.items() if 0 <= s < STEPS}
    total_blocks = sum(steps_with_blocks.values())
    counts = dict(
        total_blocks=total_blocks,
        full_block_steps=sum(1 for n in steps_with_blocks.values() if n == LAYERS),
        partial_block_steps=sum(
            1 for n in steps_with_blocks.values() if 0 < n < LAYERS
        ),
        zero_block_steps=STEPS - len(steps_with_blocks),
        full_block_equivalent=total_blocks / LAYERS,
    )
    notes = audit_evidence(family, arm, capture, counts)
    return dict(
        samples=f"{stem}-samples.pt",
        samples_shape=list(samples.shape),
        samples_dtype=str(samples.dtype),
        samples_sha256=bench.sha256_bytes(samples.view(torch.uint8).numpy().tobytes()),
        png=f"{stem}.png",
        png_mode=image.mode,
        latents=f"capture/{stem}-latents.pt",
        instrumentation_ok=not notes,
        instrumentation_notes=notes,
        **counts,
        **{k: v for k, v in capture.items() if k != "stem"},
    )


# ---------------------------------------------------------------------------
# Report and selection
# ---------------------------------------------------------------------------


def paired_timing(plan, rows, kind):
    """Paired wall time for one timing regime; incomplete arms are flagged.

    ``timing`` is randomized mixed-mode request latency, so a Cache-DiT arm
    pays a mount on nearly every request; ``service-timing`` repeats one config
    in a row, which is what a steady single-config service sees.
    """
    renamed = [dict(s, kind="timing") for s in plan["steps"] if s["kind"] == kind]
    indexed = {s["order_index"] for s in renamed}
    summary = bench.timing_summary(
        dict(plan, steps=renamed),
        [r for r in rows if r.get("order_index") in indexed],
    )
    pairs = {s["pair_id"] for s in renamed}
    native = summary.get("native", {}).get("client_wall_seconds")
    for arm, entry in summary.items():
        entry["regime"] = kind
        entry["distinct_pairs"] = len(pairs)
        walls = entry["client_wall_seconds"]
        entry["ratio_of_means"] = (
            None
            if arm == "native" or not native or not walls
            else native["mean"] / walls["mean"]
        )
    return summary


def compute_summary(rows):
    """Real block accounting per arm; partial steps are legitimate for Cache-DiT."""
    per_arm = defaultdict(list)
    for row in rows:
        if row.get("kind") == "evidence" and row.get("status") == "measured":
            per_arm[row["arm"]].append(row)
    summary = {}
    for arm, entries in sorted(per_arm.items()):
        blocks = [e["total_blocks"] for e in entries]
        notes = sorted({n for e in entries for n in e["instrumentation_notes"]})
        summary[arm] = dict(
            family=entries[0]["family"],
            requests=len(entries),
            instrumentation_ok=all(e["instrumentation_ok"] for e in entries),
            instrumentation_notes=notes,
            actual_scheduler_steps=sorted(
                {e["actual_scheduler_steps"] for e in entries}
            ),
            cache_dit_buffer_dtypes_observed=sorted(
                {d[0] for e in entries for d in e["cache_dit_buffer_dtypes"].values()}
            ),
            total_blocks=bench.stats(blocks),
            full_block_equivalent=bench.stats(
                e["full_block_equivalent"] for e in entries
            ),
            full_block_steps=bench.stats(e["full_block_steps"] for e in entries),
            partial_block_steps=bench.stats(e["partial_block_steps"] for e in entries),
            zero_block_steps=bench.stats(e["zero_block_steps"] for e in entries),
            requested_scheduler_steps=sorted(
                {e["requested_scheduler_steps"] for e in entries}
            ),
            real_blocks_discovered=sorted(
                {e["real_blocks_discovered"] for e in entries}
            ),
            transformer_blocks_attr_len=sorted(
                {e["transformer_blocks_attr_len"] for e in entries}
            ),
            layer_loads=bench.stats(
                sum(e["layer_loads_by_step"].values()) for e in entries
            ),
            denoise_device_seconds=bench.stats(
                e["denoise_device_ms"] / 1000.0 for e in entries
            ),
            final_feature_dtypes=sorted(
                {d for e in entries for d in e["final_feature_dtypes"]}
            ),
            weight_dtypes=sorted({e["transformer_weight_dtype"] for e in entries}),
        )
    return summary


def summarize(run, scores_path=None):
    run = Path(run).resolve()
    plan = json.loads((run / "plan.json").read_text())
    rows = bench.read_rows(run)
    summary = dict(
        run=str(run),
        phase=plan["phase"],
        split=plan["split"],
        **bench.run_digests(run),
        arms={
            name: arm["metadata"] | {"family": arm["family"]}
            for name, arm in plan["arms"].items()
        },
        failures=[r.get("reason") for r in rows if r.get("status") == "failed"],
        expected_failures=[
            r.get("reason") for r in rows if r.get("status") == "failed-as-expected"
        ],
        setup_seconds=[r["setup_seconds"] for r in rows if r.get("kind") == "setup"],
        timing_mixed_mode=paired_timing(plan, rows, "timing"),
        timing_service=paired_timing(plan, rows, "service-timing"),
        compute=compute_summary(rows),
    )
    if scores_path is not None:
        scores = bench.load_bound_scores(run, scores_path)
        summary["quality_vs_native"] = bench.arm_quality(
            plan, scores["scores"], Path(scores["reference_run"]) == run
        )
    return summary


def cmd_report(args):
    summary = summarize(args.run, args.scores)
    text = json.dumps(summary, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


def cmd_select(args):
    """Freeze at most two Cache-DiT configs, on validation evidence only.

    Ranking uses the grouped same-config (service) timing regime, declared
    before the run; the randomized mixed-mode times are reported beside it.
    """
    run = Path(args.run).resolve()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a selection is frozen once")
    plan = json.loads((run / "plan.json").read_text())
    if plan["phase"] != "validation" or plan["split"] != "validation":
        raise SystemExit("a selection may only come from a validation sweep")
    heldout_corpus = bench.load_corpus(args.heldout_corpus)
    summary = summarize(run, args.scores)
    quality, service = summary["quality_vs_native"], summary["timing_service"]
    eligible = {
        arm: summary["compute"][arm]["instrumentation_ok"] for arm in summary["compute"]
    }
    tuned = {}
    for gate in bench.GATES:
        candidates = [
            arm
            for arm, q in quality.items()
            if plan["arms"][arm]["family"] == "cache_dit"
            and arm != "cachedit-forced-full"
            and q[f"gate_{gate}"]
            and eligible.get(arm)
            and service.get(arm, {}).get("complete")
            and service[arm]["paired_speedup"]
        ]
        candidates.sort(key=lambda arm: -service[arm]["paired_speedup"]["mean"])
        tuned[gate] = candidates[0] if candidates else None
    selected = [a for a in dict.fromkeys(tuned.values()) if a][:2]
    # stock is eligible as a tuned alias; it is always reported either way
    final_arms = [
        "native",
        "dp-K12",
        "dp-K20",
        "uniform-K12",
        "uniform-K20",
        "cachedit-stock",
        *[a for a in selected if a != "cachedit-stock"],
    ]
    arms = {}
    for name in final_arms:
        _, metadata = arm_request(name, args.schedule_dir)
        validated = plan["arms"][name]["metadata"] or {}
        if (metadata or {}) != validated:
            raise SystemExit(
                f"arm {name} on disk differs from the config validated in {run}; "
                "a selection may not relabel a different configuration"
            )
        arms[name] = dict(family=arm_family(name), metadata=metadata or {})
    bench.write_json(
        out,
        dict(
            schema="dpcache-comparator-selection-v2",
            validation_corpus_sha256=plan["corpus_sha256"],
            expected_heldout_corpus_sha256=heldout_corpus["corpus_sha256"],
            heldout_corpus=str(Path(args.heldout_corpus).resolve()),
            validation_run=str(run),
            validation_digests=bench.run_digests(run),
            scores=str(Path(args.scores).resolve()),
            scores_sha256=bench.sha256_file(args.scores),
            gates=bench.GATES,
            timing_regime_for_ranking="service-timing (grouped same-config repeats)",
            rule="Cache-DiT arms only (forced-full excluded, stock eligible as an "
            "alias): arms whose every planned capture and timing block is measured "
            "and whose instrumentation audit passed, passing each white-composite "
            "LPIPS gate, ranked by mean paired grouped speedup; at most two distinct "
            "configs advance. DPCache and uniform arms are frozen from the earlier "
            "study and are not tuned here.",
            tuned_by_gate=tuned,
            selected_tuned=selected,
            final_arms=final_arms,
            arms=arms,
            validation_quality={
                a: {
                    k: q.get(k)
                    for k in (
                        "lpips_white",
                        "complete",
                        "gate_provisional",
                        "gate_strict",
                    )
                }
                for a, q in quality.items()
            },
            validation_timing_service={
                a: t["paired_speedup"] for a, t in service.items()
            },
            validation_timing_mixed_mode={
                a: t["paired_speedup"] for a, t in summary["timing_mixed_mode"].items()
            },
            validation_compute={
                a: dict(
                    full_block_equivalent=c["full_block_equivalent"],
                    instrumentation_ok=c["instrumentation_ok"],
                    instrumentation_notes=c["instrumentation_notes"],
                )
                for a, c in summary["compute"].items()
            },
        ),
    )
    os.chmod(out, 0o444)
    print(json.dumps(dict(tuned_by_gate=tuned, final_arms=final_arms), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("corpus")
    p.add_argument("--out", required=True)
    p.add_argument("--model-revision", required=True)
    p = sub.add_parser("schedules")
    p.add_argument("--schedule-dir", required=True, help="frozen DP artifacts")
    p.add_argument("--out", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument(
        "--pact-errors", required=True, help="pact-errors.npz from calibration"
    )
    p = sub.add_parser("launch")
    p.add_argument("phase", choices=["controls", "validation", "heldout", "service"])
    p.add_argument("--corpus", required=True)
    p.add_argument("--split", default="controls")
    p.add_argument("--out", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--schedule-dir", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--gpu-lock")
    p.add_argument("--lock-wait", type=float, default=4 * 3600)
    p.add_argument("--arms", nargs="*", default=[])
    p.add_argument("--no-evidence", dest="evidence", action="store_false")
    p.add_argument("--timing-repeats", type=int, default=2)
    p.add_argument("--service-repeats", type=int, default=2)
    p.add_argument("--timing-pairs", type=int, default=4)
    p.add_argument(
        "--timing-pair-indices",
        type=int,
        nargs="*",
        default=[],
        help="explicit corpus indices for the timing subset",
    )
    p.add_argument("--order-seed", type=int, default=20260923)
    p.add_argument("--selection")
    p = sub.add_parser("worker")
    p.add_argument("plan")
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p.add_argument("--scores")
    p.add_argument("--out")
    p = sub.add_parser("select")
    p.add_argument("--run", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--schedule-dir", required=True)
    p.add_argument("--heldout-corpus", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "corpus":
        path = Path(args.out)
        if path.exists():
            raise SystemExit(f"{path} already frozen")
        corpus = build_corpus(args.model_revision)
        bench.write_json(path, corpus)
        print(corpus["corpus_sha256"])
        return 0
    commands = {
        "schedules": cmd_schedules,
        "launch": cmd_launch,
        "worker": cmd_worker,
        "report": cmd_report,
        "select": cmd_select,
    }
    return commands[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
