"""Bounded known-route prefetch screening, not production inference or a universal scheduling bound.

Records one plain greedy request, then compares plain/one-layer/two-layer/two-layer/plain runs.
Every sampled logit vector is hashed before sampling. The native router must match every recorded
route, including prefill. Reads use the existing slot pool, cache quotas and demand-first queues.
"""

import argparse
import copy
from collections import Counter
import json
from pathlib import Path

from benchmark import source_hash
from device import adb
from runner import REMOTE, Runner, sha256
from sweep_target import cool_down, temperatures, thermal


def run_screen(runner, model, request, output, cool_to=None, cool_timeout=60, failure_checks=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if "sequence" in request or request.get("temperature", 0) != 0:
        raise ValueError("Oracle screening needs one greedy request")
    request = copy.deepcopy(request)
    request.update(diagnostic_logits_hash=True, ignore_eos=False)
    load = request["load"]
    if not all(load.get(key) for key in ("expert_streaming", "expert_slots", "prefetch", "speculative_pool")):
        raise ValueError("Explicit expert streaming, slots, prefetch and speculative pool are required")
    if load.get("lookahead_layers") != 2:
        raise ValueError("Use the same two-layer slot budget in every arm")
    load.pop("expert_trace", None)
    load.pop("diagnostic_oracle_routes", None)
    hashes = runner.stage(["eqt-bench"])
    trace = output / "routes.jsonl"
    target_trace = str(trace) if runner.local else "routes-oracle.jsonl" if runner.run_as else f"{REMOTE}/routes-oracle.jsonl"
    report = dict(kind="diagnostic_known_route_oracle", source_sha256=source_hash(), artifact_sha256=hashes,
                  request=request, model_path=str(model), runs=[], useful_compute_wall_ms=None,
                  exposed_io_wait_wall_ms=None, energy_j_per_token=None,
                  limitations=["Short natural-EOS screening, not sustained speed.",
                               "Future routes and zero decode predictor cost; not an inference implementation.",
                               "Only this one/two-layer whole-expert schedule is evaluated, not a universal upper bound.",
                               "Hashing and route checks add diagnostic overhead to timing.",
                               "Physical byte claims require an expert pack; ordinary GGUF counters are logical payload."])

    def state():
        return None if runner.local else dict(thermal=thermal(runner.serial), temperatures=temperatures(runner.serial))

    def execute(name, req, check=True):
        if cool_to is not None and not runner.local:
            cool_down(runner.serial, cool_to, cool_timeout)
        before = state()
        work = output / name
        work.mkdir(exist_ok=True)
        process, result = runner.run("eqt-bench", model, req, work, check=False)
        (work / "stderr.txt").write_text(process.stderr, encoding="utf-8")
        if check and (process.returncode or result is None):
            raise RuntimeError(f"{name}: {process.stderr[-2000:]}")
        if not check:
            return process, result
        digests = result.get("diagnostic_logits_sha256")
        if not isinstance(digests, list) or not digests or not all(len(d) == 64 for d in digests):
            raise RuntimeError(f"{name}: missing per-position logit hashes")
        report["runs"].append(dict(name=name, before=before, after=state(), result=result))
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {result['generated_tokens']} tokens, {result['decode_tokens_per_second']:.3f} tokens/s", flush=True)
        return result

    recording = copy.deepcopy(request)
    recording["load"]["expert_trace"] = target_trace
    reference = execute("record", recording)
    if not runner.local:
        if runner.run_as:
            trace.write_text(runner.app("cat files/bench/routes-oracle.jsonl").stdout, encoding="utf-8")
        else:
            adb("pull", target_trace, str(trace), serial=runner.serial)
    report["route_sha256"] = sha256(trace)
    decode_rows = Counter()
    for line in trace.read_text(encoding="utf-8").splitlines():
        route = json.loads(line)
        if route.get("decode"):
            decode_rows[route["layer"]] += route["tokens"]
    forward_tokens = next(iter(decode_rows.values())) if decode_rows and len(set(decode_rows.values())) == 1 else None
    report["decode_forward_tokens"] = forward_tokens
    for name, depth in [("plain_a", 0), ("oracle1", 1), ("oracle2_a", 2), ("oracle2_b", 2), ("plain_b", 0)]:
        req = copy.deepcopy(request)
        if depth:
            req["load"].update(diagnostic_oracle_routes=target_trace, diagnostic_oracle_layers=depth)
        result = execute(name, req)
        matched = (result["token_ids"] == reference["token_ids"] and
                   result["diagnostic_logits_sha256"] == reference["diagnostic_logits_sha256"] and
                   result["termination"] == reference["termination"])
        if depth:
            matched &= result["expert_cache"]["diagnostic_oracle"]["all_routes_matched"]
        report["runs"][-1]["bitwise_logits_and_tokens_equal"] = matched
        if not matched:
            raise RuntimeError(f"{name}: native outputs or routes disagreed")
    plains = [r["result"] for r in report["runs"] if r["name"].startswith("plain")]
    reference_bytes = max(r["expert_cache"]["decode_io_bytes"] for r in plains)
    for run in report["runs"]:
        result = run["result"]
        cache = result["expert_cache"]
        if sum(v for bank in cache["decode_io_bytes_by_bank"].values() for v in bank.values()) != cache["decode_io_bytes"]:
            raise RuntimeError(f"{run['name']}: drained bank counters do not sum to total reads")
        run["decode_MB_per_emitted_token"] = cache["decode_io_bytes"] / max(1, result["generated_tokens"]) / 1e6
        run["decode_MB_per_evaluated_token"] = cache["decode_io_bytes"] / forward_tokens / 1e6 if forward_tokens else None
        if run["name"].startswith("oracle"):
            run["physical_bytes_within_plain_budget"] = (cache["decode_io_bytes"] <= reference_bytes) if load.get("expert_pack") else None
            run["logical_bytes_within_plain_budget"] = (cache["decode_io_bytes"] <= reference_bytes) if not load.get("expert_pack") else None
    if failure_checks:
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        failures = []
        for kind in ["wrong_expert", "truncated", "invalid_future", "second_request"]:
            modified = copy.deepcopy(records)
            if kind == "wrong_expert":
                next(r for r in modified if r.get("decode"))["experts"][0] = -1
            elif kind == "truncated":
                modified.pop()
            elif kind == "invalid_future":
                [r for r in modified if r.get("decode")][1]["layer"] = 1000000
            bad = output / "bad-routes.jsonl"
            bad.write_text("".join(json.dumps(r) + "\n" for r in modified), encoding="utf-8")
            bad_target = str(bad) if runner.local else f"{REMOTE}/bad-routes.jsonl"
            if not runner.local:
                adb("push", str(bad), bad_target, serial=runner.serial)
                adb("shell", "chmod", "644", bad_target, serial=runner.serial)
            req = copy.deepcopy(request)
            req["load"].update(diagnostic_oracle_routes=bad_target)
            if kind == "second_request":
                req = dict(load=req["load"], sequence=[request, request])
            process, _ = execute("reject_" + kind, req, check=False)
            passed = process.returncode != 0 and "oracle" in process.stderr.lower()
            failures.append(dict(case=kind, rejected=passed))
            if not passed:
                raise RuntimeError(f"Invalid oracle input was not rejected: {kind}")
        report["failure_checks"] = failures
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["host", "linux", "android"], default="host")
    parser.add_argument("--model", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-as")
    parser.add_argument("--serial")
    parser.add_argument("--load", default="{}")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--cool-to", type=float)
    parser.add_argument("--cool-timeout", type=int, default=60)
    parser.add_argument("--failure-checks", action="store_true")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    request["load"].update(json.loads(args.load))
    request["max_tokens"] = args.max_tokens
    run_screen(Runner(args.target, args.serial, run_as=args.run_as), args.model, request, args.output,
               args.cool_to, args.cool_timeout, args.failure_checks)


if __name__ == "__main__":
    main()
