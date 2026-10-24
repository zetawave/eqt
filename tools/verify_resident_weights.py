"""Focused exactness and lifetime checks for opt-in resident-weight allocators; fixture timings are not benchmarks."""

import argparse
import copy
import json
from pathlib import Path

from benchmark import source_hash
from runner import Runner
from verify_streaming import build_pack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["host", "linux", "android"], default="host")
    parser.add_argument("--fixture", action="append")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.target)
    report = dict(target=args.target, source_sha256=source_hash(), fixtures=[],
                  criterion="Every pre-sampling logit SHA-256 and output token equals the default allocator")
    for fixture in args.fixture or ["models/eqt-tiny-qwen4exp.gguf", "models/eqt-tiny-qwen35moe.gguf"]:
        local = Path(fixture).resolve()
        work = output / local.stem
        work.mkdir(exist_ok=True)
        hashes = runner.stage(["eqt-bench", "eqt-check", "eqt-pack"], [local])
        model = str(local) if runner.local else local.name
        load = dict(context=2048, batch=16, threads=2, memory_budget_mib=1024, mmap=False,
                    expert_streaming=True, expert_slots=True, expert_cache_mib=1, io_threads=4,
                    prefetch=True, speculative_pool=True, lookahead_layers=2, read_priority=True,
                    prefetch_inflight=2)
        if args.target != "host":
            pack, _ = build_pack(runner, model, work)
            load.update(expert_pack=str(pack), expert_pack_split=2, direct_io=True)
        request = dict(load=load, messages=[dict(role="user", content="Explain virtual memory in English.")],
                       max_tokens=12, temperature=0, seed=42, thinking=False, ignore_eos=False,
                       diagnostic_logits_hash=True, diagnostic_residency=True)
        runs = []
        reference = None
        modes = ["default", "anonymous"] + (["ahwb"] if args.target == "android" else [])
        for mode in modes:
            req = copy.deepcopy(request)
            req["load"]["resident_weights"] = mode
            directory = work / mode
            directory.mkdir(exist_ok=True)
            process, result = runner.run("eqt-bench", model, req, directory)
            (directory / "stderr.txt").write_text(process.stderr, encoding="utf-8")
            if reference is None:
                reference = result
            passed = (bool(result["diagnostic_logits_sha256"]) and
                      result["diagnostic_logits_sha256"] == reference["diagnostic_logits_sha256"] and
                      result["token_ids"] == reference["token_ids"])
            if mode != "default":
                stats = result["resident_weights_after_decode"]
                passed &= stats["allocated_bytes"] > 0 and stats["buffers"] > 0
                if args.target != "host":
                    passed &= stats["resident_fraction"] is not None and stats["sampled_pages"] > 0
            runs.append(dict(mode=mode, passed=passed, result=result))
            print(f"{local.name} {mode}: {'PASS' if passed else 'FAIL'}", flush=True)
            if not passed:
                raise AssertionError("Resident allocator changed outputs or failed to expose its allocation")
        lifecycle = copy.deepcopy(request)
        lifecycle.pop("diagnostic_logits_hash")
        lifecycle["load"]["resident_weights"] = modes[-1]
        directory = work / "lifecycle"
        directory.mkdir(exist_ok=True)
        lifecycle_result = runner.run("eqt-check", model, lifecycle, directory)[1]
        rejected = []
        for label, overrides in [("invalid_mode", dict(resident_weights="invalid")),
                                 ("unsupported_repack", dict(resident_weights="anonymous", repack=True)),
                                 ("unsupported_nonstreaming", dict(resident_weights="anonymous", expert_streaming=False))]:
            req = copy.deepcopy(request)
            req["load"].update(overrides)
            directory = work / label
            directory.mkdir(exist_ok=True)
            process, _ = runner.run("eqt-bench", model, req, directory, check=False)
            if process.returncode == 0:
                raise AssertionError("Invalid resident-weight configuration accepted: " + label)
            rejected.append(label)
        report["fixtures"].append(dict(model=fixture, artifacts=hashes, runs=runs,
                                       lifecycle=lifecycle_result, rejected=rejected))
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
