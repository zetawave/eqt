"""Run the pending Galaxy experiments stage by stage, with results under results/local/galaxy-<date>/.

Stages are explicit so that long or storage-heavy steps (the 22 GB target acquisition) run only when
requested. Every A/B alternates order across repetitions. Nothing here roots the phone, changes thermal
limits, stops other apps or clears user data.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from device import adb, snapshot
from runner import REMOTE, ROOT, Runner

TOOLS = ROOT / "tools"
TARGET = f"{REMOTE}/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
VARIANTS = ["android_armv8.0_1", "android_armv8.2_2", "android_armv8.6_1", "android_armv9.0_1"]


def tool(*args):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(path)


def big_core_mask(serial):
    """CPU-id mask of the fastest frequency domains (Exynos 2400: X4 + A720 clusters), from sysfs."""
    text = adb("shell", "for c in /sys/devices/system/cpu/cpu[0-9]*; do echo ${c##*cpu} "
               "$(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0); done", serial=serial).stdout
    cores = {int(cpu): int(freq) for cpu, freq in (line.split() for line in text.splitlines() if line.strip())}
    slowest = min(cores.values())
    return hex(sum(1 << cpu for cpu, freq in cores.items() if freq > slowest)), cores


def preflight(out, serial):
    state = snapshot(serial)
    meminfo = dict(line.split(":", 1) for line in (state["meminfo"] or "").splitlines() if ":" in line)
    available_mib = int(meminfo.get("MemAvailable", "0 kB").split()[0]) // 1024
    free_kib = int(adb("shell", "df", "-k", "/data/local/tmp", serial=serial).stdout.splitlines()[-1].split()[3])
    mask, cores = big_core_mask(serial)
    # Target streamed config: ~2447 MiB shared weights + expert cache + context/workspace + 512 MiB reserve.
    write(out / "preflight.json", dict(device=state, mem_available_mib=available_mib, data_free_gib=free_kib / 2**20,
                                       cpu_max_khz=cores, big_core_mask=mask,
                                       target_storage_ok=free_kib * 1024 > 22134528992 + 2**30,
                                       target_ram_hint="MemAvailable >= ~4200 MiB for a 1024 MiB expert cache"))


def simd(out, serial):
    runner = Runner("android", serial)
    runner.stage(["eqt-simd-test"])
    run = adb("shell", f"{REMOTE}/eqt-simd-test", serial=serial, check=False)
    write(out / "simd.json", dict(returncode=run.returncode, output=json.loads(run.stdout.strip().splitlines()[-1])))


def variants(out, serial, repetitions):
    for variant in VARIANTS:
        tool(TOOLS / "verify_runtime.py", "--target", "android", "--cpu-variant", variant,
             "--output", out / f"variants/correctness-{variant}.json")
    for rep in range(repetitions):
        for variant in (VARIANTS if rep % 2 == 0 else VARIANTS[::-1]):
            tool(TOOLS / "benchmark.py", "--target", "android", "--cpu-variant", variant, "--repetitions", 1,
                 "--request", "configs/throughput-request.json", "--output", out / f"variants/{variant}/rep-{rep}")


def threadpool(out, serial, repetitions):
    mask, _ = big_core_mask(serial)
    base = json.loads((ROOT / "configs/throughput-request.json").read_text(encoding="utf-8"))
    arms = {"disposable": dict(threadpool=False), "persistent": dict(threadpool=True),
            "persistent_poll0": dict(threadpool=True, poll=0), "persistent_bigcores": dict(threadpool=True, cpu_mask=mask)}
    for rep in range(repetitions):
        for name in (list(arms) if rep % 2 == 0 else list(arms)[::-1]):
            request = json.loads(json.dumps(base))
            request["load"].update(arms[name])
            path = out / f"threadpool/{name}.request.json"
            write(path, request)
            tool(TOOLS / "benchmark.py", "--target", "android", "--repetitions", 1, "--request", path,
                 "--output", out / f"threadpool/{name}/rep-{rep}")


def target_runs(out, serial, budgets):
    base = json.loads((ROOT / "configs/target-stream-request.json").read_text(encoding="utf-8"))
    for budget in budgets:
        request = json.loads(json.dumps(base))
        request["load"].update(expert_cache_mib=budget, memory_budget_mib=2448 + budget + 1024)
        path = out / f"target/budget-{budget}.request.json"
        write(path, request)
        tool(TOOLS / "benchmark.py", "--target", "android", "--remote-model", TARGET, "--repetitions", 2,
             "--request", path, "--output", out / f"target/budget-{budget}")


def traces(out, budget):
    plan = ROOT / "results/2026-10-06/target-expert-plan.json"
    for name, workload in [("calibration", "workloads/routing-calibration.jsonl"), ("evaluation", "workloads/quality.jsonl")]:
        tool(TOOLS / "collect_traces.py", "--target", "android", "--model", TARGET, "--workload", workload,
             "--max-tokens", 128, "--output", out / f"traces/{name}")
    tool(TOOLS / "simulate_cache.py", "--calibrate", out / "traces/calibration/trace.jsonl",
         "--evaluate", out / "traces/evaluation/trace.jsonl", "--experts", 256, "--experts-used", 8,
         "--plan", plan, "--budget-mib", 512, 768, 1024, 1536, 2048, "--output", out / "traces/replay.json")
    replay = json.loads((out / "traces/replay.json").read_text(encoding="utf-8"))
    weights = next(r["native_expert_layer_weights"] for r in replay["results"] if r["budget_mib"] == budget)
    base = json.loads((ROOT / "configs/target-stream-request.json").read_text(encoding="utf-8"))
    base["load"].update(expert_cache_mib=budget, memory_budget_mib=2448 + budget + 1024)
    optimized = json.loads(json.dumps(base))
    optimized["load"]["expert_layer_weights"] = weights
    write(out / "traces/uniform.request.json", base)
    write(out / "traces/optimized.request.json", optimized)
    for rep in range(3):
        for name in (["uniform", "optimized"] if rep % 2 == 0 else ["optimized", "uniform"]):
            tool(TOOLS / "benchmark.py", "--target", "android", "--remote-model", TARGET, "--repetitions", 1,
                 "--request", out / f"traces/{name}.request.json", "--output", out / f"traces/ab-{name}/rep-{rep}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", choices=["preflight", "simd", "variants", "threadpool", "streaming",
                                                       "acquire-target", "target", "traces"])
    parser.add_argument("--serial")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[768, 1024, 1536])
    parser.add_argument("--trace-budget", type=int, default=1024)
    parser.add_argument("--output", default=f"results/local/galaxy-{datetime.date.today().isoformat()}")
    args = parser.parse_args()
    if args.serial:
        os.environ["ANDROID_SERIAL"] = args.serial  # inherited by adb in every sub-tool
    out = ROOT / args.output
    for stage in args.stages:
        if stage == "preflight":
            preflight(out, args.serial)
        elif stage == "simd":
            simd(out, args.serial)
        elif stage == "variants":
            variants(out, args.serial, args.repetitions)
        elif stage == "threadpool":
            threadpool(out, args.serial, args.repetitions)
        elif stage == "streaming":
            tool(TOOLS / "verify_streaming.py", "--target", "android", "--direct-io",
                 "--output", out / "streaming-correctness.json")
        elif stage == "acquire-target":
            tool(TOOLS / "acquire_model.py", "configs/target-model.json", "--adb-stream", REMOTE)
        elif stage == "target":
            target_runs(out, args.serial, args.budgets)
        elif stage == "traces":
            traces(out, args.trace_budget)


if __name__ == "__main__":
    main()
