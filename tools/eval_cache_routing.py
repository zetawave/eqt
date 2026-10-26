"""Quality and I/O screening of opt-in cache-aware routing, which is not exact, against native routing.

Every arm decodes the same teacher-forced answers (configs/cache-routing-eval.json) from an empty expert cache, all
on one load. Native routing is the reference. Per answer position the tool compares the full softmax (KL divergence
from native), the greedy prediction (top-1 agreement) and the forced token's log-likelihood (perplexity). Position 0
comes from the native prompt evaluation in every arm and is excluded. Expert misses per decoded token come from the
store's counters; bytes use the phone's 1.875 MB expert, since host reads are logical GGUF payload.

`--fixture-check` instead verifies the mechanism on the random-weight qwen4exp fixture: substitutions happen, a
repeated request decodes identical logits, the selection does not depend on prefetch/pool/top-k-hook settings, and
invalid settings are rejected. Fixture outputs say nothing about quality.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from benchmark import source_hash
from runner import Runner
from sweep_target import temperatures, thermal

ROOT = Path(__file__).resolve().parents[1]
EXPERT_MB = 1.875  # one Flash-Next expert, all three banks
HOST_LOAD = dict(context=4096, batch=128, threads=4, memory_budget_mib=8192, mmap=False, expert_streaming=True,
                 expert_cache_mib=1792, io_threads=8, direct_io=False, prefetch=True, speculative_pool=True,
                 lookahead_layers=2, lookahead2_count=5, read_priority=True, prefetch_inflight=2,
                 lookahead_router_q8=True, coalesce_kib=2048, coalesce_gap=1, prefill_prefetch_mib=0,
                 prefill_prefetch_min_coverage=0.9, expert_slots=True)
FIXTURE = "models/eqt-tiny-qwen4exp.gguf"
FIXTURE_LOAD = dict(context=1024, batch=128, threads=2, expert_streaming=True, expert_cache_mib=1, io_threads=2,
                    expert_slots=True, prefetch=True, speculative_pool=True, lookahead_layers=2, read_priority=True)


def parse_arms(spec):
    arms = []
    for item in spec.split(","):
        if item == "exact":
            arms.append(("exact", None))
        else:
            keep, max_rank = (int(v) for v in item.split(":"))
            arms.append((f"keep{keep}_rank{max_rank}", dict(keep=keep, max_rank=max_rank)))
    if arms[0][0] != "exact":
        raise ValueError("The first arm must be exact (the reference)")
    return arms


def log_softmax(rows):
    rows = rows.astype(np.float64)
    shifted = rows - rows.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def compare(reference_path, candidate_path, forced, vocab, chunk=32):
    reference = np.memmap(reference_path, dtype=np.float32, mode="r").reshape(-1, vocab)
    candidate = np.memmap(candidate_path, dtype=np.float32, mode="r").reshape(-1, vocab)
    if reference.shape != candidate.shape or reference.shape[0] != len(forced):
        raise RuntimeError(f"Logit rows disagree: {reference.shape} {candidate.shape} {len(forced)}")
    kl, agree, nll_ref, nll_cand = [], [], [], []
    for start in range(1, reference.shape[0], chunk):
        ref, cand = log_softmax(reference[start:start + chunk]), log_softmax(candidate[start:start + chunk])
        kl.extend((np.exp(ref) * (ref - cand)).sum(axis=1))
        agree.extend(ref.argmax(axis=1) == cand.argmax(axis=1))
        tokens = np.asarray(forced[start:start + chunk])
        rows = np.arange(len(tokens))
        nll_ref.extend(-ref[rows, tokens])
        nll_cand.extend(-cand[rows, tokens])
    return dict(kl=kl, agree=agree, nll_reference=nll_ref, nll_candidate=nll_cand)


def screen(runner, model, load, arms, cases, output):
    output.mkdir(parents=True, exist_ok=True)
    logits = output / "logits"
    logits.mkdir(exist_ok=True)
    sequence = []
    for arm, routing in arms:
        for case in cases:
            request = dict(messages=case["messages"], forced_text=case["forced_text"], max_tokens=1024, temperature=0,
                           seed=42, thinking=False, prompt_cache=False, diagnostic_release_expert_cache=True,
                           diagnostic_logits_path=str(logits / f"{arm}-{case['id']}.f32"))
            if routing:
                request["cache_routing"] = routing
            sequence.append(request)
    hashes = runner.stage(["eqt-bench"])
    process, result = runner.run("eqt-bench", model, dict(load=load, sequence=sequence), output)
    (output / "stderr.txt").write_text(process.stderr, encoding="utf-8")
    results = result["results"]
    report = dict(kind="cache_routing_quality_screen", source_sha256=source_hash(), artifact_sha256=hashes,
                  model=str(model), load=load, cases=[c["id"] for c in cases], arms={},
                  limitations=["Teacher-forced synthetic answers on a host PC; numerics match the phone's CPU path "
                               "only in kind, not bitwise.",
                               "Routing changes accumulate in recurrent/KV state; long answers are not covered.",
                               "Perplexity of authored text is a sensitivity measure, not a task-quality evaluation.",
                               "Misses per token are host cache counters at the phone's budget, not phone I/O."])
    index = 0
    per_arm = {}
    for arm, routing in arms:
        rows = []
        for case in cases:
            entry = results[index]
            index += 1
            if entry["termination"] != "forced_end" or entry["forced_tokens"] != entry["generated_tokens"]:
                raise RuntimeError(f"{arm}/{case['id']}: answer did not follow the forced text")
            rows.append((case, entry))
        per_arm[arm] = rows
    vocab = results[0]["vocabulary_size"]
    for arm, routing in arms:
        metrics = dict(kl=[], agree=[], nll_reference=[], nll_candidate=[])
        decoded = misses = demanded = substituted = 0
        cases_out = {}
        for (case, entry), (_, reference) in zip(per_arm[arm], per_arm["exact"]):
            forced = entry["token_ids"]
            if forced != reference["token_ids"]:
                raise RuntimeError(f"{arm}/{case['id']}: forced tokens differ from the reference arm")
            values = compare(logits / f"exact-{case['id']}.f32", logits / f"{arm}-{case['id']}.f32", forced, vocab)
            for key in metrics:
                metrics[key].extend(values[key])
            cache = entry["expert_cache"]
            steps = len(forced) - 1
            decoded += steps
            misses += cache["decode_misses"]
            demanded += cache["decode_demanded_experts"]
            routed = cache.get("cache_routing") or {}
            substituted += routed.get("substituted_experts", 0)
            cases_out[case["id"]] = dict(positions=steps, mean_kl=float(np.mean(values["kl"])),
                                         top1_agreement=float(np.mean(values["agree"])),
                                         perplexity=float(np.exp(np.mean(values["nll_candidate"]))),
                                         decode_misses_per_token=cache["decode_misses"] / steps)
        kl = np.asarray(metrics["kl"])
        ppl = float(np.exp(np.mean(metrics["nll_candidate"])))
        ppl_reference = float(np.exp(np.mean(metrics["nll_reference"])))
        report["arms"][arm] = dict(
            cache_routing=routing, positions=len(kl), mean_kl=float(kl.mean()), median_kl=float(np.median(kl)),
            p95_kl=float(np.percentile(kl, 95)), max_kl=float(kl.max()), top1_agreement=float(np.mean(metrics["agree"])),
            perplexity=ppl, perplexity_change=ppl / ppl_reference - 1, decode_misses_per_token=misses / decoded,
            demanded_per_token=demanded / decoded, substituted_per_token=substituted / decoded,
            phone_expert_MB_per_token=misses / decoded * EXPERT_MB, cases=cases_out)
        a = report["arms"][arm]
        print(f"{arm:16s} KL mean {a['mean_kl']:.2e} p95 {a['p95_kl']:.2e} max {a['max_kl']:.2e} | top-1 "
              f"{a['top1_agreement']:.3f} | PPL {a['perplexity']:.4f} ({a['perplexity_change']:+.2%}) | "
              f"misses/token {a['decode_misses_per_token']:.1f} (~{a['phone_expert_MB_per_token']:.0f} MB) | "
              f"substituted/token {a['substituted_per_token']:.1f}", flush=True)
    exact_mb = report["arms"]["exact"]["phone_expert_MB_per_token"]
    for arm in report["arms"].values():
        arm["miss_bytes_change"] = arm["phone_expert_MB_per_token"] / exact_mb - 1 if exact_mb else None
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def fixture_check(runner, output):
    output.mkdir(parents=True, exist_ok=True)
    model = ROOT / FIXTURE
    base = dict(messages=[{"role": "user", "content": "Count from one to twenty, in words."}], max_tokens=24,
                temperature=0, seed=42, thinking=False, prompt_cache=False, diagnostic_release_expert_cache=True,
                diagnostic_logits_hash=True)
    routed = dict(base, cache_routing=dict(keep=1, max_rank=12))
    runner.stage(["eqt-bench"])
    report = dict(kind="cache_routing_fixture_check", source_sha256=source_hash(), checks=[])

    def run(name, load, sequence, expect_failure=None):
        work = output / name
        work.mkdir(exist_ok=True)
        process, result = runner.run("eqt-bench", model, dict(load=load, sequence=sequence), work, check=False)
        if expect_failure:
            passed = process.returncode != 0 and expect_failure in process.stderr
            report["checks"].append(dict(name=name, rejected=passed, stderr=process.stderr[-300:]))
            if not passed:
                raise RuntimeError(f"{name}: invalid setting was not rejected: {process.stderr[-500:]}")
            return None
        if process.returncode or result is None:
            raise RuntimeError(f"{name}: {process.stderr[-2000:]}")
        return result["results"]

    first = run("repeat", FIXTURE_LOAD, [routed, routed, base])
    hashes = first[0]["diagnostic_logits_sha256"]
    substituted = first[0]["expert_cache"]["cache_routing"]["substituted_experts"]
    checks = dict(substitutions=substituted > 0, repeat_equal=first[1]["diagnostic_logits_sha256"] == hashes,
                  native_restored=first[2]["expert_cache"]["cache_routing"] is None,
                  differs_from_native=first[2]["diagnostic_logits_sha256"] != hashes)
    variants = {"no_prefetch": dict(prefetch=False, speculative_pool=False, lookahead_layers=1, read_priority=False),
                "topk_in_graph": dict(topk_in_graph=True), "io_threads_1": dict(io_threads=1)}
    for name, change in variants.items():
        results = run(name, dict(FIXTURE_LOAD, **change), [routed])
        checks[f"equal_{name}"] = results[0]["diagnostic_logits_sha256"] == hashes
    report["checks"].append(dict(name="mechanism", substituted_experts=substituted, **checks))
    if not all(checks.values()):
        raise RuntimeError(f"Fixture mechanism check failed: {checks}")
    invalid = {"keep_zero": dict(keep=0, max_rank=12), "keep_all": dict(keep=4, max_rank=12),
               "rank_too_small": dict(keep=1, max_rank=4), "rank_beyond_experts": dict(keep=1, max_rank=17)}
    for name, routing in invalid.items():
        run("reject_" + name, FIXTURE_LOAD, [dict(base, cache_routing=routing)], "cache_routing")
    run("reject_lookup", dict(FIXTURE_LOAD, rollback_slots=4), [dict(routed, prompt_lookup=3, diagnostic_logits_hash=False)],
        "cache_routing")
    run("reject_forced_sampling", FIXTURE_LOAD, [dict(base, forced_text="one two", temperature=0.7,
                                                      diagnostic_logits_hash=False)], "forced_text")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"][0], indent=1))
    return report


def speed(runner, model, load, routing, request, tokens, output):
    """Exact/routed/routed/exact on one load, each from an empty cache: expert traffic and wait counters per token
    beside tokens/s. Routed answers differ from exact ones, so per-token counters compare different text."""
    output.mkdir(parents=True, exist_ok=True)
    request = {k: v for k, v in request.items() if k != "load"}
    request.update(max_tokens=tokens, temperature=0, ignore_eos=False, prompt_cache=False,
                   diagnostic_release_expert_cache=True)
    arms = [("exact", None), ("routed", routing), ("routed", routing), ("exact", None)]
    sequence = [dict(request, **({"cache_routing": r} if r else {})) for _, r in arms]
    hashes = runner.stage(["eqt-bench"])
    state = (lambda: None) if runner.local else (lambda: dict(thermal=thermal(runner.serial),
                                                              temperatures=temperatures(runner.serial)))
    before = state()
    process, result = runner.run("eqt-bench", model, dict(load=load, sequence=sequence), output)
    (output / "stderr.txt").write_text(process.stderr, encoding="utf-8")
    rows = []
    for (arm, routed), entry in zip(arms, result["results"]):
        cache = entry["expert_cache"]
        steps = max(1, entry["generated_tokens"] - 1)
        rows.append(dict(arm=arm, cache_routing=routed, tokens=entry["generated_tokens"], termination=entry["termination"],
                         decode_tokens_per_second=entry["decode_tokens_per_second"],
                         io_MB_per_token=cache["decode_io_bytes"] / steps / 1e6,
                         demand_MB_per_token=sum(cache["decode_io_bytes_by_bank"]["demand"].values()) / steps / 1e6,
                         misses_per_token=cache["decode_misses"] / steps,
                         substituted_per_token=(cache.get("cache_routing") or {}).get("substituted_experts", 0) / steps,
                         compute_wait_thread_ms_per_token=cache["compute_wait_thread_ms"] / steps,
                         io_busy_share=cache["decode_io_busy_ms"] / entry["decode_ms"],
                         major_faults=(entry.get("decode_process_cpu") or {}).get("major_faults"),
                         prefill_ms=entry["prefill_ms"], text=entry["text"]))
        r = rows[-1]
        print(f"{arm:7s} {r['decode_tokens_per_second']:.2f} tok/s | {r['io_MB_per_token']:.0f} MB/token "
              f"(demand {r['demand_MB_per_token']:.0f}) | misses {r['misses_per_token']:.1f} | substituted "
              f"{r['substituted_per_token']:.1f} | wait {r['compute_wait_thread_ms_per_token']:.0f} thread-ms | "
              f"faults {r['major_faults']}", flush=True)
    report = dict(kind="cache_routing_speed_screen", source_sha256=source_hash(), artifact_sha256=hashes,
                  model=str(model), load=load, request=request, before=before, after=state(), runs=rows,
                  limitations=["One short ABBA sequence: drift and thermal state are not controlled.",
                               "Routed and exact runs decode different text; counters are per decoded token."])
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["host", "linux", "android"], default="host")
    parser.add_argument("--run-as")
    parser.add_argument("--serial")
    parser.add_argument("--speed", help="keep:max_rank of a speed screen against exact routing")
    parser.add_argument("--request", default="configs/flash-next-oracle-request.json", help="speed screen request")
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--model", default="models/flash-next/Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf")
    parser.add_argument("--cases", default="configs/cache-routing-eval.json")
    parser.add_argument("--arms", default="exact,3:16,3:24,2:32")
    parser.add_argument("--load", default="{}", help="JSON overrides of the host load options")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture-check", action="store_true")
    args = parser.parse_args()
    runner = Runner(args.target, args.serial, run_as=args.run_as)
    output = Path(args.output).resolve()
    if args.fixture_check:
        fixture_check(runner, output)
        return
    if args.speed:
        request = json.loads((ROOT / args.request).read_text(encoding="utf-8"))
        load = dict(request["load"], **json.loads(args.load))
        keep, max_rank = (int(v) for v in args.speed.split(":"))
        model = args.model if args.target == "android" else (ROOT / args.model).resolve()
        speed(runner, model, load, dict(keep=keep, max_rank=max_rank), request, args.tokens, output)
        return
    if args.target == "android":
        raise ValueError("The quality screen reads logit files locally; run it on the host")
    cases = json.loads((ROOT / args.cases).read_text(encoding="utf-8"))["cases"]
    load = dict(copy.deepcopy(HOST_LOAD), **json.loads(args.load))
    screen(runner, (ROOT / args.model).resolve(), load, parse_arms(args.arms), cases, output)


if __name__ == "__main__":
    main()
