# SPDX-License-Identifier: Apache-2.0
"""Reproducible DPCache benchmark for Qwen-Image 2.1 on one GPU.

Run this file by path (``python .../bench_qwen_image21_dpcache.py <command>``)
so that the CPU-only ``score`` command never imports sglang, and so that the
spawned GPU worker re-imports this file and installs the evidence hooks.

Commands:
  corpus         freeze prompt/seed splits (calibration / validation / heldout)
  launch PHASE   freeze a request plan, then run it in one DiffGenerator process
                 tree under an optional GPU flock (phases: calib, baseline,
                 parity, sweep, overhead)
  calibrate      PACT errors from captured final features, exact DP schedules
  score          tensor exactness plus LPIPS / alpha metrics against native
  report         timing and quality summary of one run
  contact-sheet  side-by-side native vs DPCache images

Evidence and calibration requests copy tensors to the host (synchronizing) and
are never timing samples; warmup and timing requests run with every hook inert.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from functools import partial
from pathlib import Path

HARNESS = Path(__file__).resolve()
CAPTURE_ENV = "DPCACHE_BENCH_CAPTURE"
LOCK_ENV = "DPCACHE_BENCH_LOCK"
STAGE_MODULE = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages."
    "model_specific_stages.qwen_image21"
)
STEPS, WIDTH, HEIGHT = 40, 1024, 1024
MANDATORY = (0, 1, 2)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def read_rows(run):
    return [json.loads(line) for line in (Path(run) / "results.jsonl").open()]


# ---------------------------------------------------------------------------
# Corpus: frozen before any output
# ---------------------------------------------------------------------------

CALIBRATION_PROMPTS = [
    (
        "text",
        'A vintage travel poster with the words "VISIT MARS" in bold letters over a red desert landscape',
    ),
    (
        "people",
        "A young woman with curly hair reading a newspaper on a subway train, candid photo",
    ),
    (
        "objects-spatial",
        "A stack of three books with a pair of glasses on top, next to a steaming mug, on a desk under a lamp",
    ),
    (
        "scene",
        "A futuristic city street at night with flying cars, holographic billboards and wet reflective pavement",
    ),
    (
        "illustration",
        "A children's book illustration of a fox and a rabbit sharing an umbrella in the rain, soft watercolor",
    ),
    ("counting", "Five red apples arranged in a row on a white kitchen counter"),
    (
        "colors",
        "A field of tulips in bright red, yellow, purple and orange under a clear blue sky",
    ),
    (
        "text",
        'A hand-painted wooden sign that says "FRESH BREAD" hanging outside a bakery',
    ),
    (
        "people",
        "An old man playing chess in a park with a young girl, warm afternoon light",
    ),
    (
        "complex-scene",
        "A busy harbor at dawn with fishing boats, seagulls, stacked crates and workers unloading nets",
    ),
]
VALIDATION_PROMPTS = [
    (
        "text",
        'A bookstore window display with a large banner reading "SUMMER SALE 50% OFF"',
    ),
    (
        "people",
        "A chef in a white uniform carefully plating a dessert in a professional kitchen",
    ),
    (
        "counting",
        "Three yellow rubber ducks and two blue toy boats floating in a bathtub",
    ),
    (
        "illustration",
        "A flat vector illustration of a mountain landscape with a rising sun, minimal pastel colors",
    ),
    (
        "colors",
        "A stained glass window depicting a peacock with vivid blue, green and gold panes",
    ),
    (
        "complex-scene",
        "A cozy living room with a fireplace, a sleeping dog on a rug and snow falling outside the window",
    ),
    (
        "objects",
        "A glass jar of colorful marbles on a windowsill with sunlight casting refracted colors",
    ),
    (
        "spatial",
        "A red bicycle leaning against a green door, with a potted plant to its right",
    ),
]
HELDOUT_PROMPTS = [
    (
        "text",
        'A storefront with a red neon sign that reads "OPEN 24 HOURS" above a glass door on a rainy night street',
    ),
    (
        "text",
        'A chalkboard cafe menu with handwritten lines "Espresso 3.50", "Latte 4.25" and "Tea 2.75"',
    ),
    (
        "people",
        "A portrait photograph of an elderly fisherman with a weathered face and gray beard wearing a yellow raincoat",
    ),
    (
        "people",
        "Two children flying a red kite on a grassy hill at sunset, seen from behind",
    ),
    (
        "objects-counting",
        "A still life of a blue ceramic teapot, three green apples and a brass key on a wooden table",
    ),
    (
        "spatial",
        "A black cat sitting to the left of a white dog with a red ball between them on a checkered floor",
    ),
    (
        "complex-scene",
        "A bustling medieval market square with stalls of fruit and fabric, stone towers in the background",
    ),
    (
        "scene",
        "An aerial view of a winding river through an autumn forest with a small wooden bridge and a cabin",
    ),
    (
        "illustration",
        "An anime-style illustration of a girl with a red scarf standing on a rooftop under a starry night sky",
    ),
    (
        "counting-colors",
        "Four coffee cups in different colors, red, blue, green and yellow, lined up on a wooden shelf",
    ),
]
SPLIT_SEEDS = {"calibration": (31,), "validation": (505,), "heldout": (1001, 2002)}


def build_corpus(model_revision):
    splits = {}
    for split, prompts in (
        ("calibration", CALIBRATION_PROMPTS),
        ("validation", VALIDATION_PROMPTS),
        ("heldout", HELDOUT_PROMPTS),
    ):
        splits[split] = [
            dict(
                pair_id=f"{split}:{sha256_bytes(prompt.encode())[:12]}:{seed}",
                split=split,
                category=category,
                prompt=prompt,
                seed=seed,
            )
            for seed in SPLIT_SEEDS[split]
            for category, prompt in prompts
        ]
    texts = [{p["prompt"] for p in pairs} for pairs in splits.values()]
    if any(a & b for i, a in enumerate(texts) for b in texts[i + 1 :]):
        raise ValueError("splits share a prompt")
    corpus = dict(
        schema="dpcache-corpus-v1",
        settings=dict(
            model_revision=model_revision,
            width=WIDTH,
            height=HEIGHT,
            steps=STEPS,
            guidance_scale=1.0,
            negative_prompt=None,
            num_outputs_per_prompt=1,
            generator_device="cpu",
        ),
        note="prompt-disjoint splits; heldout prompts 1-8 reuse the older "
        "MANIFEST-pilot-v1 evaluation prompts (no DPCache tuning ever saw them)",
        splits=splits,
    )
    corpus["corpus_sha256"] = sha256_bytes(canonical(corpus))
    return corpus


def load_corpus(path):
    corpus = json.loads(Path(path).read_text())
    body = {k: v for k, v in corpus.items() if k != "corpus_sha256"}
    if sha256_bytes(canonical(body)) != corpus["corpus_sha256"]:
        raise ValueError(f"{path} was modified after freezing")
    return corpus


# ---------------------------------------------------------------------------
# GPU lock
# ---------------------------------------------------------------------------


def gpu_compute_jobs():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def acquire_gpu(lock_path, wait_seconds):
    """Hold ``lock_path`` (if given) and require an idle GPU; never kills anything."""
    handle = None
    if lock_path:
        handle = open(lock_path, "a")
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise SystemExit(f"timed out waiting for {lock_path}")
                time.sleep(5)
    jobs = gpu_compute_jobs()
    if jobs:
        raise SystemExit(f"unrelated GPU compute jobs present, not starting: {jobs}")
    return handle


def check_lock_held():
    path = os.environ.get(LOCK_ENV)
    if not path:
        return None
    with open(path, "a") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return path
    raise SystemExit(f"{path} is not held by the launcher")


# ---------------------------------------------------------------------------
# Worker-side hooks, installed in the spawned GPU worker process
# ---------------------------------------------------------------------------


def _current_step():
    from sglang.multimodal_gen.runtime.managers import forward_context

    context = forward_context._forward_context
    if context is None or context.current_timestep is None:
        return -1
    return int(context.current_timestep)


def _install_worker_hooks(config):
    import importlib

    import torch

    from sglang.multimodal_gen.runtime.managers.memory_managers import (
        layerwise_offload,
    )

    stage_cls = importlib.import_module(STAGE_MODULE).QwenImage21DenoisingStage
    if getattr(stage_cls.forward, "_dpcache_bench", False):
        return
    native_forward = stage_cls.forward
    manager_cls = layerwise_offload.LayerwiseOffloadManager
    native_prefetch = manager_cls.prefetch_layer
    active = {}

    def prefetch_layer(self, layer_idx, non_blocking=True):
        record = active.get("record")
        if record is None:
            return native_prefetch(self, layer_idx, non_blocking)

        def resident():
            return layer_idx in self._gpu_layers or layer_idx in self._courier_inflight

        before = resident()
        result = native_prefetch(self, layer_idx, non_blocking)
        if not before and resident():
            record["layer_loads_by_step"][_current_step()] += 1
        return result

    def count_block(record, module, args):
        record["blocks_by_step"][_current_step()] += 1

    def forward(self, batch, server_args):
        flag = Path(config["flag"])
        if not flag.exists():
            return native_forward(self, batch, server_args)
        request = json.loads(flag.read_text())
        record = {
            "blocks_by_step": defaultdict(int),
            "layer_loads_by_step": defaultdict(int),
        }
        handles = [
            block.register_forward_pre_hook(partial(count_block, record))
            for block in self.transformer.transformer_blocks
        ]
        features = []
        if request["mode"] == "calibration":
            # final block feature, streamed to host at every step
            handles.append(
                self.transformer.norm_out.register_forward_pre_hook(
                    lambda module, args: features.append(args[0].detach().to("cpu"))
                )
            )
        artifact = {"stem": request["stem"], "mode": request["mode"]}
        if request.get("signature"):
            import msgspec

            artifact["signature"] = msgspec.to_builtins(
                self.dpcache_signature(batch, server_args)
            )
        artifact["timesteps"] = batch.timesteps.float().cpu().tolist()
        artifact["sigmas"] = batch.scheduler.sigmas.float().cpu().tolist()
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
        end.record()
        end.synchronize()
        artifact["denoise_device_ms"] = start.elapsed_time(end)
        artifact["blocks_by_step"] = dict(record["blocks_by_step"])
        artifact["layer_loads_by_step"] = dict(record["layer_loads_by_step"])
        directory = Path(config["dir"])
        torch.save(
            batch.latents.detach().cpu().contiguous(),
            directory / f"{request['stem']}-latents.pt",
        )
        if features:
            from safetensors.torch import save_file

            stacked = torch.stack(features).contiguous()
            save_file(
                {"features": stacked},
                directory / f"{request['stem']}-features.safetensors",
            )
            artifact["features_shape"] = list(stacked.shape)
            artifact["features_dtype"] = str(stacked.dtype)
        write_json(directory / f"{request['stem']}-capture.json", artifact)
        return batch

    forward._dpcache_bench = True
    stage_cls.forward = forward
    manager_cls.prefetch_layer = prefetch_layer


if os.environ.get(CAPTURE_ENV):
    _install_worker_hooks(json.loads(os.environ[CAPTURE_ENV]))


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def schedule_arm(path):
    schedule = json.loads(Path(path).read_text())
    return dict(
        schedule=str(Path(path).resolve()),
        schedule_sha256=sha256_file(path),
        num_full_steps=schedule["num_full_steps"],
    )


class PlanBuilder:
    def __init__(self):
        self.steps = []

    def add(self, pair, arm, kind, repeat=0, signature=False):
        self.steps.append(
            dict(
                order_index=len(self.steps),
                pair_id=pair["pair_id"],
                arm=arm,
                kind=kind,
                repeat=repeat,
                capture_signature=signature,
            )
        )


def build_plan(args, corpus):
    """Deterministic request order, frozen to plan.json before any GPU work."""
    splits = corpus["splits"]
    regression = splits["calibration"][:3]
    plan = PlanBuilder()
    arms = {"native": None}
    order = random.Random(args.order_seed)
    if args.phase == "calib":
        arms["all-full"] = "all-full-from-captured-signature"
        plan.add(splits["calibration"][0], "native", "warmup")
        for pair in splits["calibration"]:
            plan.add(pair, "native", "calibration", signature=True)
        for pair in regression:
            plan.add(pair, "native", "evidence", repeat=1)
            plan.add(pair, "all-full", "evidence")
        plan.add(regression[0], "native", "evidence", repeat=2)
    elif args.phase == "baseline":
        for repeat in range(2):
            for pair in regression:
                plan.add(pair, "native", "evidence", repeat=repeat)
    elif args.phase == "parity":
        # disabled, all-full, a request that fails inside denoising, then recovery
        arms["all-full"] = "all-full-from-captured-signature"
        arms["mismatched"] = "all-full-with-wrong-height"
        for pair in regression:
            plan.add(pair, "native", "evidence", signature=True)
            plan.add(pair, "all-full", "evidence")
        plan.add(regression[1], "mismatched", "expect-failure")
        plan.add(regression[0], "native", "evidence", repeat=1)
        plan.add(regression[0], "all-full", "evidence", repeat=1)
    elif args.phase == "sweep":
        pairs = splits[args.split]
        if args.split == "heldout":
            selection = load_selection(args.selection, corpus)
            if set(args.arms) != set(selection["arms"]):
                raise ValueError("held-out arms must be exactly the frozen selection")
        for name in args.arms:
            arms[name] = schedule_arm(Path(args.schedule_dir) / f"{name}.json")
            if args.split == "heldout" and arms[name] != selection["arms"][name]:
                raise ValueError(f"{name} differs from the frozen selection")
        plan.add(pairs[0], "native", "warmup")
        for name in args.arms:
            plan.add(pairs[0], name, "warmup")
        if args.evidence:
            for pair in pairs:
                plan.add(pair, "native", "evidence")
                for name in args.arms:
                    plan.add(pair, name, "evidence")
        timing_pairs = pairs[: args.timing_pairs] if args.timing_pairs else pairs
        for repeat in range(args.timing_repeats):
            for pair in timing_pairs:
                block = list(arms)
                order.shuffle(block)
                for name in block:
                    plan.add(pair, name, "timing", repeat=repeat)
    elif args.phase == "overhead":
        pairs = splits["validation"][: args.timing_pairs or 3]
        plan.add(pairs[0], "native", "warmup")
        for repeat in range(args.timing_repeats):
            for pair in pairs:
                plan.add(pair, "native", "timing", repeat=repeat)
    else:
        raise ValueError(f"unknown phase {args.phase}")
    used = {s["pair_id"] for s in plan.steps}
    return dict(
        schema="dpcache-plan-v2",
        phase=args.phase,
        split=args.split,
        source_dir=str(Path(args.source_dir).resolve()),
        model_path=str(Path(args.model_path).resolve()),
        corpus_sha256=corpus["corpus_sha256"],
        pairs=[p for pairs in splits.values() for p in pairs if p["pair_id"] in used],
        arms=arms,
        steps=plan.steps,
        order_seed=args.order_seed,
        harness_sha256=sha256_file(HARNESS),
        selection=None
        if args.selection is None
        else dict(
            path=str(Path(args.selection).resolve()), sha256=sha256_file(args.selection)
        ),
    )


def cmd_launch(args):
    corpus = load_corpus(args.corpus)
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; runs are never overwritten")
    plan = build_plan(args, corpus)
    handle = acquire_gpu(args.gpu_lock, args.lock_wait)
    try:
        out.mkdir(parents=True)
        write_json(out / "plan.json", plan)
        env = dict(
            os.environ, PYTHONPATH=plan["source_dir"], PYTHONDONTWRITEBYTECODE="1"
        )
        if args.gpu_lock:
            env[LOCK_ENV] = str(Path(args.gpu_lock).resolve())
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
# GPU worker
# ---------------------------------------------------------------------------


def runner_config(model_path):
    """Native single-GPU recipe, identical for every arm and for the baseline."""
    return dict(
        model_path=str(model_path),
        model_id="Qwen-Image-2.1",
        num_gpus=1,
        performance_mode="manual",
        dit_layerwise_offload=True,
        attention_backend="torch_sdpa",
        enable_attention_backend_autotune=False,
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        warmup_mode="off",
    )


def request_kwargs(pair):
    return dict(
        prompt=pair["prompt"],
        seed=pair["seed"],
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        guidance_scale=1.0,
        generator_device="cpu",
        num_outputs_per_prompt=1,
        enable_cache_dit=False,
        save_output=False,
        return_file_paths_only=False,
    )


def source_identity(source_dir):
    """Import location plus git state (or a REVISION file for exported trees)."""
    import sglang

    imported = Path(sglang.__file__).resolve()
    if not str(imported).startswith(str(Path(source_dir).resolve())):
        raise RuntimeError(f"imported sglang from {imported}, expected {source_dir}")
    root = Path(source_dir).resolve().parent
    if (root / "REVISION").exists():
        return dict(
            kind="exported",
            revision=(root / "REVISION").read_text().strip(),
            path=str(root),
        )

    def git(*a):
        return subprocess.check_output(["git", "-C", str(root), *a])

    diff = git("diff", "HEAD")
    untracked = git("ls-files", "--others", "--exclude-standard").decode().split()
    digest = hashlib.sha256(diff)
    for name in sorted(untracked):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    return dict(
        kind="git",
        revision=git("rev-parse", "HEAD").decode().strip(),
        path=str(root),
        working_tree_sha256=digest.hexdigest(),
        untracked=sorted(untracked),
        diff_files=git("diff", "HEAD", "--name-only").decode().split(),
    )


def environment_record():
    import importlib.metadata as md

    versions = {}
    for name in (
        "torch",
        "diffusers",
        "transformers",
        "sglang-kernel",
        "flashinfer-python",
        "safetensors",
        "msgspec",
    ):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return dict(
        python=sys.version.split()[0],
        packages=versions,
        gpu=gpu,
        hostname=os.uname().nodename,
    )


def derived_schedule(name, capture_dir):
    """All-full (K=T) schedules built from a signature this process captured."""
    from sglang.multimodal_gen.runtime.cache.dpcache import (
        DPCacheRequestSignature,
        build_schedule_artifact,
    )

    for path in sorted(Path(capture_dir).glob("*-capture.json")):
        capture = json.loads(path.read_text())
        if "signature" in capture:
            break
    else:
        raise RuntimeError("no captured signature yet")
    signature = DPCacheRequestSignature(
        **{
            k: tuple(v) if isinstance(v, list) else v
            for k, v in capture["signature"].items()
        }
    )
    schedule = build_schedule_artifact(
        signature=signature,
        full_steps=list(range(STEPS)),
        mandatory=MANDATORY,
        calibrated_cost=0.0,
        calibration=dict(
            manifest_sha256="0" * 64, num_samples=0, kind="all-full; no calibration"
        ),
        source_commit="benchmark-derived",
    )
    if name == "mismatched":
        schedule["request"]["height"] = HEIGHT // 2
    return schedule


def run_request(runner, kwargs):
    start = time.perf_counter()
    result = runner.generate(sampling_params_kwargs=kwargs)
    wall = time.perf_counter() - start
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError("expected exactly one output")
        result = result[0]
    if result is None or result.samples is None:
        raise RuntimeError("no output tensor")
    return result, wall


def save_evidence(out, capture_dir, stem, result):
    import torch
    from PIL import Image

    samples = result.samples.detach().cpu().contiguous()
    torch.save(samples, out / f"{stem}-samples.pt")
    image = Image.fromarray(result.frames[0])
    image.save(out / f"{stem}.png")
    capture = json.loads((capture_dir / f"{stem}-capture.json").read_text())
    blocks = capture["blocks_by_step"]
    return dict(
        samples=f"{stem}-samples.pt",
        samples_shape=list(samples.shape),
        samples_dtype=str(samples.dtype),
        samples_sha256=sha256_bytes(samples.view(torch.uint8).numpy().tobytes()),
        png=f"{stem}.png",
        png_mode=image.mode,
        latents=f"capture/{stem}-latents.pt",
        full_steps_observed=sorted(int(s) for s, n in blocks.items() if n),
        **{k: v for k, v in capture.items() if k not in ("stem", "mode")},
    )


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

    lock = check_lock_held()
    inherited = sorted(k for k in os.environ if k.startswith("SGLANG_CACHE_DIT"))
    if inherited:
        raise SystemExit(f"inherited Cache-DiT env would change defaults: {inherited}")
    # spawned workers re-import this file and install the hooks at import
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
        common = dict(
            phase=plan["phase"],
            corpus_sha256=plan["corpus_sha256"],
            harness_sha256=sha256_file(HARNESS),
            gpu_lock=lock,
            model_path=plan["model_path"],
            **{
                "source_" + k: v for k, v in source_identity(plan["source_dir"]).items()
            },
        )
        emit(
            dict(
                common,
                kind="environment",
                **environment_record(),
                runner_config=runner_config(plan["model_path"]),
            )
        )
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )

        setup_start = time.perf_counter()
        runner = DiffGenerator.from_pretrained(**runner_config(plan["model_path"]))
        emit(
            dict(common, kind="setup", setup_seconds=time.perf_counter() - setup_start)
        )
        pairs = {p["pair_id"]: p for p in plan["pairs"]}
        schedules = {}
        for name, arm in plan["arms"].items():
            if isinstance(arm, dict):
                if sha256_file(arm["schedule"]) != arm["schedule_sha256"]:
                    raise RuntimeError(f"schedule {name} changed after planning")
                schedules[name] = json.loads(Path(arm["schedule"]).read_text())
        for step in plan["steps"]:
            pair, arm = pairs[step["pair_id"]], step["arm"]
            stem = f"{step['order_index']:04d}-{arm}-{step['kind']}-{step['repeat']}"
            row = dict(
                common,
                **step,
                prompt=pair["prompt"],
                seed=pair["seed"],
                split=pair["split"],
                category=pair["category"],
                stem=stem,
            )
            captured = step["kind"] in ("evidence", "calibration")
            try:
                kwargs = request_kwargs(pair)
                if plan["arms"][arm] is not None:
                    if arm not in schedules:
                        schedules[arm] = derived_schedule(arm, capture_dir)
                        write_json(out / f"{arm}-schedule.json", schedules[arm])
                    kwargs["dpcache_schedule"] = schedules[arm]
                    row["schedule_sha256"] = sha256_bytes(canonical(schedules[arm]))
                    row["num_full_steps"] = schedules[arm]["num_full_steps"]
                if captured:
                    write_json(
                        flag,
                        dict(
                            mode=step["kind"],
                            stem=stem,
                            signature=step["capture_signature"],
                        ),
                    )
                try:
                    result, wall = run_request(runner, kwargs)
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
                    row.update(save_evidence(out, capture_dir, stem, result))
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


# ---------------------------------------------------------------------------
# Calibration and planning
# ---------------------------------------------------------------------------


def fp32(values):
    import torch

    return torch.tensor(values, dtype=torch.float32).tolist()


def captured_signature(rows):
    signatures = {
        canonical(r["signature"])
        for r in rows
        if r.get("status") == "measured" and isinstance(r.get("signature"), dict)
    }
    if len(signatures) != 1:
        raise ValueError(
            f"expected one distinct captured signature, got {len(signatures)}"
        )
    return json.loads(signatures.pop())


def runner_configs(rows):
    configs = {
        canonical(r["runner_config"]) for r in rows if r.get("kind") == "environment"
    }
    if len(configs) != 1:
        raise ValueError(f"expected one runner config, got {len(configs)}")
    return json.loads(configs.pop())


def check_calibration_inputs(corpus, calib_plan, calib_rows, signature_rows):
    """Refuse features that do not belong to the request the schedules will serve.

    Returns the signature fields the calibration run did not capture; those are
    bound only through the identical runner config and request settings.
    """
    if calib_plan["corpus_sha256"] != corpus["corpus_sha256"]:
        raise ValueError("calibration run used a different corpus")
    captures = [r for r in calib_rows if r.get("kind") == "calibration"]
    ids = [r["pair_id"] for r in captures]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate calibration captures")
    if set(ids) != {p["pair_id"] for p in corpus["splits"]["calibration"]}:
        raise ValueError(
            "calibration captures do not cover exactly the calibration split"
        )
    if any(r["status"] != "measured" for r in captures):
        raise ValueError("a calibration capture failed")
    if runner_configs(calib_rows) != runner_configs(signature_rows):
        raise ValueError("calibration and signature runs used different runner configs")
    signature = captured_signature(signature_rows)
    uncaptured = set()
    for row in captures:
        captured = row["signature"]
        uncaptured |= set(signature) - set(captured)
        for key in set(captured) & set(signature):
            same = (
                fp32(captured[key]) == fp32(signature[key])
                if key in ("timesteps", "sigmas")
                else captured[key] == signature[key]
            )
            if not same:
                raise ValueError(f"{row['stem']}: signature field {key} differs")
        if row["features_dtype"] != "torch.bfloat16":
            raise ValueError(f"{row['stem']}: features are {row['features_dtype']}")
    if len({tuple(r["features_shape"]) for r in captures}) != 1:
        raise ValueError("calibration features differ in shape")
    return signature, sorted(uncaptured)


def cmd_calibrate(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file

    from sglang.multimodal_gen.runtime.cache import dpcache

    corpus = load_corpus(args.corpus)
    run, out = Path(args.run), Path(args.out)
    signature, uncaptured = check_calibration_inputs(
        corpus,
        json.loads((run / "plan.json").read_text()),
        read_rows(run),
        read_rows(args.signature_run),
    )
    captures = [r for r in read_rows(run) if r.get("kind") == "calibration"]
    if args.check_only:
        print(json.dumps(dict(ok=True, uncaptured_signature_fields=uncaptured)))
        return 0
    handle = acquire_gpu(args.gpu_lock, args.lock_wait)
    try:
        out.mkdir(parents=True, exist_ok=False)
        total, per_sample = None, {}
        torch.cuda.synchronize()
        gpu_start = time.perf_counter()
        for row in captures:
            path = run / "capture" / f"{row['stem']}-features.safetensors"
            features = load_file(path)["features"]
            if features.shape[0] != STEPS or features.dtype != torch.bfloat16:
                raise SystemExit(f"bad features {features.shape} {features.dtype}")
            # one sample's T features on the inference device at a time
            errors = dpcache.pact_step_errors(
                list(features.cuda()), max_gap=args.max_gap
            )
            per_sample[row["pair_id"]] = errors
            total = errors if total is None else total + errors
        torch.cuda.synchronize()
        gpu_seconds = time.perf_counter() - gpu_start
    finally:
        if handle is not None:
            handle.close()
    mean_errors = total / len(captures)
    np.savez(
        out / "pact-errors.npz",
        mean=mean_errors.numpy(),
        **{f"sample_{i}": v.numpy() for i, v in enumerate(per_sample.values())},
    )
    cpu_start = time.perf_counter()
    costs = dpcache.pact_costs(mean_errors, max_gap=args.max_gap)
    signature_struct = dpcache.DPCacheRequestSignature(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in signature.items()}
    )
    calibration = dict(
        manifest_sha256=corpus["corpus_sha256"],
        split="calibration",
        pair_ids=sorted(per_sample),
        num_samples=len(per_sample),
        feature="final transformer-block feature (norm_out input), BF16",
        error_arithmetic="BF16 prediction on CUDA; FP32 elementwise error, FP64 sum; "
        "mean over samples",
        calibration_run=str(run.resolve()),
        calibration_results_sha256=sha256_file(run / "results.jsonl"),
        signature_run=str(Path(args.signature_run).resolve()),
        signature_fields_not_captured_in_calibration=uncaptured,
        endpoint_convention="terminal sentinel T scores steps through T-1 only; "
        "the paper's proxy also scores the endpoint",
    )
    summary = dict(gpu_error_seconds=gpu_seconds, max_gap=args.max_gap, schedules={})
    for budget in args.budgets:
        name = f"K{budget}"
        try:
            keys, value = dpcache.plan_schedule(costs, budget, max_gap=args.max_gap)
        except ValueError as exc:
            summary["schedules"][name] = f"infeasible: {exc}"
            continue
        artifact = dpcache.build_schedule_artifact(
            signature=signature_struct,
            full_steps=keys,
            mandatory=MANDATORY,
            calibrated_cost=value,
            calibration=calibration,
            source_commit=args.source_commit,
            max_gap=args.max_gap,
        )
        write_json(out / f"{name}.json", artifact)
        summary["schedules"][name] = dict(
            full_steps=keys,
            predicted_steps=[s for s in range(STEPS) if s not in keys],
            calibrated_cost=value,
            sha256=sha256_file(out / f"{name}.json"),
        )
    summary["cpu_planning_seconds"] = time.perf_counter() - cpu_start
    write_json(out / "calibration-summary.json", summary)
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------------------
# Evidence identities: every planned request is accounted for
# ---------------------------------------------------------------------------

CAPTURED_KINDS = ("evidence", "calibration")


def rows_by_step(rows):
    """Planned-request rows keyed by their immutable plan order_index."""
    indexed = {}
    for row in rows:
        if "order_index" not in row:
            continue
        if row["order_index"] in indexed:
            raise ValueError(
                f"duplicate result rows for plan step {row['order_index']}"
            )
        indexed[row["order_index"]] = row
    return indexed


def reference_steps(plan):
    """The first planned native capture of each pair is its reference."""
    references = {}
    for step in plan["steps"]:
        if step["arm"] == "native" and step["kind"] in CAPTURED_KINDS:
            references.setdefault(step["pair_id"], step)
    return references


def candidate_steps(plan, same_run_reference):
    references = {s["order_index"] for s in reference_steps(plan).values()}
    return [
        s
        for s in plan["steps"]
        if s["kind"] in CAPTURED_KINDS
        and not (same_run_reference and s["order_index"] in references)
    ]


def resolve_references(ref_plan, ref_rows, pair_ids):
    """Measured reference row per pair; missing or failed references are errors."""
    steps = reference_steps(ref_plan)
    indexed = rows_by_step(ref_rows)
    resolved = {}
    for pair_id in sorted(pair_ids):
        step = steps.get(pair_id)
        row = indexed.get(step["order_index"]) if step else None
        if row is None or row.get("status") != "measured":
            raise ValueError(f"no measured native reference for {pair_id}")
        resolved[pair_id] = row
    return resolved


def check_comparable(run_rows, ref_rows, run_plan, ref_plan, pairs):
    """Same corpus, runner recipe and request per pair; sources may differ."""
    if run_plan["corpus_sha256"] != ref_plan["corpus_sha256"]:
        raise ValueError("runs use different corpora")
    if runner_configs(run_rows) != runner_configs(ref_rows):
        raise ValueError("runs use different runner configs")
    for row, ref in pairs:
        if (row["prompt"], row["seed"]) != (ref["prompt"], ref["seed"]):
            raise ValueError(f"{row['stem']}: prompt/seed differ from reference")


def run_digests(run):
    run = Path(run)
    return dict(
        plan_sha256=sha256_file(run / "plan.json"),
        results_sha256=sha256_file(run / "results.jsonl"),
    )


def load_bound_scores(run, scores_path):
    """Scores are only usable for the exact plan and results they were computed on."""
    scores = json.loads(Path(scores_path).read_text())
    if scores.get("run_digests") != run_digests(run):
        raise ValueError(f"{scores_path} is stale or belongs to another run")
    if scores.get("reference_digests") != run_digests(scores["reference_run"]):
        raise ValueError(f"{scores_path}: reference run changed after scoring")
    return scores


# ---------------------------------------------------------------------------
# Scoring (CPU only; needs the `lpips` package, never imports sglang)
# ---------------------------------------------------------------------------


def load_rgba(path):
    import numpy as np
    from PIL import Image

    image = Image.open(path)
    return np.asarray(image.convert("RGBA")), image.mode


def white_composite(rgba):
    import numpy as np

    alpha = rgba[..., 3:4].astype(np.float64) / 255.0
    return rgba[..., :3].astype(np.float64) * alpha + 255.0 * (1.0 - alpha)


def compare_images(model, run, row, ref_run, ref):
    import numpy as np
    import torch

    def to_lpips(rgb):
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
        return tensor.float() / 127.5 - 1.0

    def equal(a, b):
        return bool(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))

    load = partial(torch.load, weights_only=True)
    a, mode_a = load_rgba(run / row["png"])
    b, mode_b = load_rgba(ref_run / ref["png"])
    with torch.no_grad():
        white = model(to_lpips(white_composite(a)), to_lpips(white_composite(b))).item()
        dropped = model(to_lpips(a[..., :3]), to_lpips(b[..., :3])).item()
    alpha = np.abs(a[..., 3].astype(np.int32) - b[..., 3].astype(np.int32)) / 255.0
    mse = ((white_composite(a) - white_composite(b)) ** 2).mean()
    return dict(
        latents_equal=equal(load(run / row["latents"]), load(ref_run / ref["latents"])),
        samples_equal=equal(load(run / row["samples"]), load(ref_run / ref["samples"])),
        png_equal=bool(np.array_equal(a, b)),
        png_modes=[mode_a, mode_b],
        lpips_white=white,
        lpips_rgb_dropped_alpha=dropped,
        alpha_mean_abs=float(alpha.mean()),
        alpha_max_abs=float(alpha.max()),
        alpha_all_opaque=[
            bool((a[..., 3] == 255).all()),
            bool((b[..., 3] == 255).all()),
        ],
        psnr_white=float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse),
    )


def cmd_score(args):
    import lpips
    import torch

    torch.set_num_threads(args.threads)
    model = lpips.LPIPS(net="alex", verbose=False).eval()
    weights = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        weights.update(name.encode() + tensor.numpy().tobytes())
    run = Path(args.run).resolve()
    ref_run = Path(args.reference_run or args.run).resolve()
    plan = json.loads((run / "plan.json").read_text())
    ref_plan = json.loads((ref_run / "plan.json").read_text())
    rows, ref_rows = read_rows(run), read_rows(ref_run)
    indexed = rows_by_step(rows)
    candidates = candidate_steps(plan, same_run_reference=ref_run == run)
    references = resolve_references(
        ref_plan, ref_rows, {s["pair_id"] for s in candidates}
    )
    measured = [
        (indexed[s["order_index"]], references[s["pair_id"]])
        for s in candidates
        if indexed.get(s["order_index"], {}).get("status") == "measured"
    ]
    check_comparable(rows, ref_rows, plan, ref_plan, measured)
    scores = []
    for step in candidates:
        row = indexed.get(step["order_index"])
        entry = dict(
            order_index=step["order_index"],
            arm=step["arm"],
            kind=step["kind"],
            repeat=step["repeat"],
            pair_id=step["pair_id"],
            status="missing" if row is None else row["status"],
        )
        if row is not None and row["status"] == "measured":
            ref = references[step["pair_id"]]
            entry.update(
                stem=row["stem"],
                reference_stem=ref["stem"],
                category=row["category"],
                num_full_steps=row.get("num_full_steps"),
                full_steps_observed=row.get("full_steps_observed"),
                **compare_images(model, run, row, ref_run, ref),
            )
        scores.append(entry)
        print(
            json.dumps(
                {
                    k: entry.get(k)
                    for k in (
                        "order_index",
                        "arm",
                        "status",
                        "latents_equal",
                        "lpips_white",
                    )
                }
            ),
            flush=True,
        )
    write_json(
        Path(args.out) if args.out else run / "scores.json",
        dict(
            schema="dpcache-scores-v2",
            run=str(run),
            reference_run=str(ref_run),
            run_digests=run_digests(run),
            reference_digests=run_digests(ref_run),
            harness_sha256=sha256_file(HARNESS),
            metric=f"LPIPS alex (lpips package), CPU FP32, {args.threads} thread(s); "
            "evaluation arithmetic only",
            lpips_weights_sha256=weights.hexdigest(),
            pixel_policy="white-composited RGBA (provisional gate); alpha-dropped RGB "
            "and alpha error reported separately",
            scores=scores,
        ),
    )


# ---------------------------------------------------------------------------
# Report, quality gates and selection
# ---------------------------------------------------------------------------

GATES = {"provisional": (0.18, 0.30), "strict": (0.05, 0.10)}


def stats(values):
    values = list(values)
    if not values:
        return None
    return dict(
        n=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        min=min(values),
        max=max(values),
    )


def arm_quality(plan, scores, same_run_reference=True):
    """Per-arm quality over EVERY planned capture; incomplete arms never pass."""
    expected = defaultdict(set)
    for step in candidate_steps(plan, same_run_reference):
        expected[step["arm"]].add(step["order_index"])
    seen = {}
    for entry in scores:
        if entry["order_index"] in seen:
            raise ValueError(f"duplicate score for plan step {entry['order_index']}")
        seen[entry["order_index"]] = entry
    quality = {}
    for arm, indices in sorted(expected.items()):
        entries = [seen[i] for i in sorted(indices) if i in seen]
        good = [e for e in entries if e["status"] == "measured"]
        missing = sorted(indices - {e["order_index"] for e in good})
        result = dict(
            planned=len(indices),
            measured=len(good),
            missing_or_failed=missing,
            complete=not missing,
        )
        if good:
            result.update(
                exact_latents=sum(e["latents_equal"] for e in good),
                exact_samples=sum(e["samples_equal"] for e in good),
                exact_png=sum(e["png_equal"] for e in good),
                lpips_white=stats(e["lpips_white"] for e in good),
                lpips_rgb_dropped_alpha=stats(
                    e["lpips_rgb_dropped_alpha"] for e in good
                ),
                alpha_mean_abs=stats(e["alpha_mean_abs"] for e in good),
                alpha_max_abs=max(e["alpha_max_abs"] for e in good),
                worst=sorted(
                    (
                        (round(e["lpips_white"], 4), e["category"], e["stem"])
                        for e in good
                    ),
                    reverse=True,
                )[:3],
            )
        for gate, (mean_bound, max_bound) in GATES.items():
            result[f"gate_{gate}"] = bool(
                result["complete"]
                and good
                and result["lpips_white"]["mean"] <= mean_bound
                and result["lpips_white"]["max"] <= max_bound
            )
        quality[arm] = result
    return quality


def timing_summary(plan, rows):
    """Paired timing over planned (pair, repeat) blocks; incomplete arms are flagged."""
    indexed = rows_by_step(rows)
    planned, blocks = defaultdict(int), defaultdict(dict)
    missing = defaultdict(list)
    for step in plan["steps"]:
        if step["kind"] != "timing":
            continue
        planned[step["arm"]] += 1
        row = indexed.get(step["order_index"])
        if row is None or row.get("status") != "measured":
            missing[step["arm"]].append(step["order_index"])
            continue
        blocks[(step["pair_id"], step["repeat"])][step["arm"]] = row[
            "client_wall_seconds"
        ]
    planned_blocks = defaultdict(set)
    for step in plan["steps"]:
        if step["kind"] == "timing":
            planned_blocks[step["arm"]].add((step["pair_id"], step["repeat"]))
    result = {}
    for arm in planned:
        walls = [b[arm] for b in blocks.values() if arm in b]
        paired = [b for b in blocks.values() if arm in b and "native" in b]
        # a candidate is complete only if every planned block has BOTH its own
        # and the native measurement, so a speedup never rests on a subset
        complete = not missing[arm] and (
            arm == "native" or len(paired) == len(planned_blocks[arm])
        )
        result[arm] = dict(
            planned=planned[arm],
            measured=len(walls),
            paired_count=None if arm == "native" else len(paired),
            complete=complete,
            client_wall_seconds=stats(walls),
            paired_speedup=None
            if arm == "native"
            else stats(b["native"] / b[arm] for b in paired),
            paired_saving_seconds=None
            if arm == "native"
            else stats(b["native"] - b[arm] for b in paired),
        )
    return result


def evidence_summary(rows):
    device, offload = defaultdict(list), {}
    for r in rows:
        if r.get("kind") != "evidence" or r.get("status") != "measured":
            continue
        device[r["arm"]].append(r["denoise_device_ms"] / 1000.0)
        full = set(r["full_steps_observed"])
        loads = {int(k): v for k, v in r["layer_loads_by_step"].items()}
        entry = offload.setdefault(
            r["arm"],
            dict(
                requests=0,
                predicted_steps=0,
                layer_loads_on_predicted_steps=0,
                layer_loads_total=0,
                partial_block_steps=0,
                schedule_mismatches=0,
            ),
        )
        entry["requests"] += 1
        entry["predicted_steps"] += STEPS - len(full)
        entry["layer_loads_on_predicted_steps"] += sum(
            v for s, v in loads.items() if 0 <= s < STEPS and s not in full
        )
        entry["layer_loads_total"] += sum(loads.values())
        entry["partial_block_steps"] += sum(
            1 for n in r["blocks_by_step"].values() if n % 32
        )
        if r.get("num_full_steps") is not None and len(full) != r["num_full_steps"]:
            entry["schedule_mismatches"] += 1
    return {a: stats(v) for a, v in device.items()}, offload


def summarize(run, scores_path=None):
    run = Path(run).resolve()
    plan = json.loads((run / "plan.json").read_text())
    rows = read_rows(run)
    device, offload = evidence_summary(rows)
    summary = dict(
        run=str(run),
        **run_digests(run),
        failures=[r.get("reason") for r in rows if r.get("status") == "failed"],
        expected_failures=[
            r.get("reason") for r in rows if r.get("status") == "failed-as-expected"
        ],
        setup_seconds=[r["setup_seconds"] for r in rows if r.get("kind") == "setup"],
        timing=timing_summary(plan, rows),
        evidence_denoise_device_seconds=device,
        offload=offload,
    )
    if scores_path is not None:
        scores = load_bound_scores(run, scores_path)
        summary["quality_vs_native"] = arm_quality(
            plan, scores["scores"], Path(scores["reference_run"]) == run
        )
    return summary


def select_candidates(quality, timing):
    """Declared rule: fastest complete arm passing each gate; ties go to larger K."""
    selection = {}
    for gate in GATES:
        passing = [
            arm
            for arm, q in quality.items()
            if arm != "native"
            and q[f"gate_{gate}"]
            and timing.get(arm, {}).get("complete")
            and timing.get(arm, {}).get("paired_speedup")
        ]
        passing.sort(
            key=lambda arm: (
                -timing[arm]["paired_speedup"]["mean"],
                -int(arm.lstrip("K")),
            )
        )
        selection[gate] = passing[0] if passing else None
    return selection


def cmd_report(args):
    summary = summarize(args.run, args.scores)
    text = json.dumps(summary, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


def cmd_select(args):
    run = Path(args.run).resolve()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a selection is frozen once")
    plan = json.loads((run / "plan.json").read_text())
    if plan["split"] != "validation" or plan["phase"] != "sweep":
        raise SystemExit("selection must come from a validation sweep")
    summary = summarize(run, args.scores)
    chosen = select_candidates(summary["quality_vs_native"], summary["timing"])
    arms = {name: plan["arms"][name] for name in dict.fromkeys(chosen.values()) if name}
    write_json(
        out,
        dict(
            schema="dpcache-selection-v1",
            corpus_sha256=plan["corpus_sha256"],
            validation_run=str(run),
            validation_digests=run_digests(run),
            scores=str(Path(args.scores).resolve()),
            scores_sha256=sha256_file(args.scores),
            gates=GATES,
            rule="validation split only: for each gate, the complete arm (every planned "
            "capture and timing request measured) passing white-composite LPIPS mean and max "
            "with the highest mean paired speedup over native; ties go to larger K",
            selected=chosen,
            arms=arms,
            validation_quality={
                a: {
                    k: q[k]
                    for k in (
                        "lpips_white",
                        "complete",
                        "gate_provisional",
                        "gate_strict",
                    )
                    if k in q
                }
                for a, q in summary["quality_vs_native"].items()
            },
            validation_timing={
                a: t["paired_speedup"] for a, t in summary["timing"].items()
            },
        ),
    )
    os.chmod(out, 0o444)
    print(json.dumps(chosen))


def load_selection(path, corpus):
    selection = json.loads(Path(path).read_text())
    if selection["corpus_sha256"] != corpus["corpus_sha256"]:
        raise ValueError("selection was made on another corpus")
    for name, arm in selection["arms"].items():
        if sha256_file(arm["schedule"]) != arm["schedule_sha256"]:
            raise ValueError(f"selected schedule {name} changed after selection")
    return selection


def cmd_contact_sheet(args):
    from PIL import Image, ImageDraw

    run = Path(args.run)
    rows = [
        r
        for r in read_rows(run)
        if r.get("kind") == "evidence" and r.get("status") == "measured"
    ]
    scores = {
        s["stem"]: s
        for s in load_bound_scores(run, args.scores)["scores"]
        if s["status"] == "measured"
    }
    arms = ["native", *args.arms]
    pair_ids = list(dict.fromkeys(r["pair_id"] for r in rows))
    if args.worst_by:
        # worst LPIPS of the given arm first, so failures are never cropped out
        lpips = {
            s["pair_id"]: s["lpips_white"]
            for s in scores.values()
            if s["arm"] == args.worst_by
        }
        pair_ids.sort(key=lambda pair_id: -lpips[pair_id])
    pair_ids = pair_ids[: args.pairs]
    tile, caption = args.tile, 34
    sheet = Image.new(
        "RGB", (tile * len(arms), (tile + caption) * len(pair_ids)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for y, pair_id in enumerate(pair_ids):
        for x, arm in enumerate(arms):
            row = next(
                (r for r in rows if r["pair_id"] == pair_id and r["arm"] == arm), None
            )
            if row is None:
                continue
            rgba = Image.open(run / row["png"]).convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
            top = y * (tile + caption)
            sheet.paste(
                image.resize((tile, tile), Image.Resampling.LANCZOS), (x * tile, top)
            )
            score = scores.get(row["stem"])
            label = arm if score is None else f"{arm}  LPIPS {score['lpips_white']:.3f}"
            if x == 0:
                label = f"native  [{row['category']}]"
            draw.text((x * tile + 4, top + tile + 4), label, fill="black")
            draw.text(
                (x * tile + 4, top + tile + 18),
                row["prompt"][: tile // 7] if x == 0 else "",
                fill="gray",
            )
    sheet.save(args.out)
    print(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("corpus")
    p.add_argument("--out", required=True)
    p.add_argument("--model-revision", required=True)
    p = sub.add_parser("launch")
    p.add_argument(
        "phase", choices=["calib", "baseline", "parity", "sweep", "overhead"]
    )
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument(
        "--source-dir",
        required=True,
        help="python/ directory of the SGLang tree under test (PYTHONPATH)",
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--gpu-lock", help="flock file shared with other GPU users")
    p.add_argument("--lock-wait", type=float, default=4 * 3600)
    p.add_argument("--split", choices=["validation", "heldout"], default="validation")
    p.add_argument("--arms", nargs="*", default=[])
    p.add_argument("--schedule-dir")
    p.add_argument("--no-evidence", dest="evidence", action="store_false")
    p.add_argument("--selection", help="frozen selection (required for heldout)")
    p.add_argument("--timing-repeats", type=int, default=1)
    p.add_argument("--timing-pairs", type=int, default=0, help="0 means every pair")
    p.add_argument("--order-seed", type=int, default=20260922)
    p = sub.add_parser("worker")
    p.add_argument("plan")
    p = sub.add_parser("calibrate")
    p.add_argument("--corpus", required=True)
    p.add_argument("--run", required=True, help="run with calibration feature captures")
    p.add_argument(
        "--signature-run",
        required=True,
        help="run whose captured request signature the schedules serve",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--budgets", type=int, nargs="+", required=True)
    p.add_argument("--max-gap", type=int, default=None)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--gpu-lock")
    p.add_argument("--lock-wait", type=float, default=4 * 3600)
    p.add_argument("--check-only", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("--run", required=True)
    p.add_argument("--reference-run")
    p.add_argument("--out")
    p.add_argument("--threads", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p.add_argument("--scores")
    p.add_argument("--out")
    p = sub.add_parser("select")
    p.add_argument("--run", required=True, help="validation sweep run")
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("contact-sheet")
    p.add_argument("--run", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--tile", type=int, default=320)
    p.add_argument("--worst-by", help="order rows by this arm's LPIPS, worst first")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "corpus":
        path = Path(args.out)
        if path.exists():
            raise SystemExit(f"{path} already frozen")
        corpus = build_corpus(args.model_revision)
        write_json(path, corpus)
        print(corpus["corpus_sha256"])
        return 0
    commands = {
        "launch": cmd_launch,
        "worker": cmd_worker,
        "calibrate": cmd_calibrate,
        "score": cmd_score,
        "report": cmd_report,
        "select": cmd_select,
        "contact-sheet": cmd_contact_sheet,
    }
    return commands[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
