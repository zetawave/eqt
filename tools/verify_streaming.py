"""Verify that bounded expert streaming reproduces the resident path exactly.

Each streamed configuration must match a resident run with the same prefill chunk bit for bit: identical
first-token logits and identical greedy token IDs. Budgets are chosen to force eviction and overflow.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

from benchmark import source_hash
from runner import ROOT, Runner, sha256

FIXTURES = ["models/eqt-tiny-qwen35moe.gguf", "models/eqt-tiny-qwen35moe-ff96.gguf"]
MTP_FIXTURE = "models/eqt-tiny-qwen35moe-mtp.gguf"
MTP_CASES = [
    ("plain_resident", {}),
    ("plain_streamed", dict(expert_streaming=True, expert_cache_mib=1, io_threads=2)),
    ("mtp_k2_resident", dict(mtp=True, draft_max=2)),
    ("mtp_k4_streamed_prefetch", dict(mtp=True, draft_max=4, expert_streaming=True, expert_cache_mib=1,
                                      io_threads=2, prefetch=True)),
    ("mtp_k2_streamed_slots", dict(mtp=True, draft_max=2, expert_streaming=True, expert_cache_mib=1,
                                   io_threads=2, prefetch=True, expert_slots=True)),
]
CASES = [
    dict(name="evict_every_layer_io1", batch=128, load=dict(expert_cache_mib=1, io_threads=1)),
    dict(name="evict_every_layer_io4_chunk16", batch=16, load=dict(expert_cache_mib=1, io_threads=4)),
    dict(name="fully_cached", batch=128, load=dict(expert_cache_mib=64, io_threads=4)),
    dict(name="prefetch_lookahead", batch=128, load=dict(expert_cache_mib=2, io_threads=2, prefetch=True)),
    dict(name="prefetch_extra_candidates", batch=128,
         load=dict(expert_cache_mib=2, io_threads=2, prefetch=True, prefetch_extra=6)),
    dict(name="coalesced_reads_evicting", batch=128,
         load=dict(expert_cache_mib=1, io_threads=4, coalesce_kib=512, coalesce_gap=2)),
    dict(name="coalesced_reads_prefetch_chunk16", batch=16,
         load=dict(expert_cache_mib=2, io_threads=4, prefetch=True, coalesce_kib=4096, coalesce_gap=8)),
    dict(name="whole_layer_prefill_prefetch", batch=128,
         load=dict(expert_cache_mib=2, io_threads=4, prefill_prefetch_mib=8, coalesce_kib=512, coalesce_gap=2)),
    dict(name="slots_evicting", batch=128, load=dict(expert_cache_mib=1, io_threads=4, expert_slots=True)),
    dict(name="bank_layout_evicting_coalesced", batch=128,
         load=dict(expert_cache_mib=1, io_threads=4, coalesce_kib=512, coalesce_gap=2, expert_slots=False)),
    dict(name="slots_coalesced_prefetch_layer", batch=16,
         load=dict(expert_cache_mib=2, io_threads=4, prefetch=True, coalesce_kib=512, coalesce_gap=2,
                   prefill_prefetch_mib=8, expert_slots=True)),
    dict(name="nonuniform_layer_weights", batch=128,
         load=dict(expert_cache_mib=2, io_threads=2, expert_layer_weights=[4, 1, 1, 1, 1, 1, 1, 4])),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["host", "linux", "android"], default="host")
    parser.add_argument("--serial")
    parser.add_argument("--cpu-variant", help="Android only: force one GGML CPU backend variant")
    parser.add_argument("--direct-io", action="store_true", help="Also run a direct-I/O case (Android/Linux only)")
    parser.add_argument("--output", default="results/local/streaming-correctness.json")
    args = parser.parse_args()
    runner = Runner(args.target, args.serial, args.cpu_variant)
    base = json.loads((ROOT / "configs/smoke-request.json").read_text(encoding="utf-8"))
    base.update(max_tokens=24, capture_logits=True, temperature=0, ignore_eos=True)
    cases = CASES + ([dict(name="direct_io", batch=128, load=dict(expert_cache_mib=1, io_threads=4, direct_io=True)),
                      dict(name="direct_io_coalesced", batch=128,
                           load=dict(expert_cache_mib=1, io_threads=4, direct_io=True, coalesce_kib=256, coalesce_gap=2)),
                      dict(name="direct_io_bank_layout", batch=128,
                           load=dict(expert_cache_mib=1, io_threads=4, direct_io=True, coalesce_kib=256,
                                     prefill_prefetch_mib=8, expert_slots=False)),
                      dict(name="direct_io_slots", batch=128,
                           load=dict(expert_cache_mib=1, io_threads=4, direct_io=True, expert_slots=True)),
                      dict(name="direct_io_slots_coalesced", batch=128,
                           load=dict(expert_cache_mib=1, io_threads=4, direct_io=True, coalesce_kib=512, coalesce_gap=2,
                                     prefill_prefetch_mib=8, expert_slots=True)),
                      dict(name="direct_io_prefill_prefetch", batch=128,
                           load=dict(expert_cache_mib=1, io_threads=4, direct_io=True, coalesce_kib=256,
                                     prefill_prefetch_mib=8))]
                     if args.direct_io else [])
    report = dict(target=args.target, source_sha256=source_hash(), cpu_variant=args.cpu_variant,
                  criterion="bitwise first-token logits and identical greedy token IDs versus resident, same chunk",
                  fixtures=[])
    (ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eqt-stream-", dir=ROOT / ".cache") as workdir:
        for fixture in FIXTURES:
            # EQT_FIXTURE_DIR lets a container read fixtures from a filesystem that supports O_DIRECT.
            local = (Path(os.environ.get("EQT_FIXTURE_DIR", ROOT / "models")) / Path(fixture).name).resolve()
            hashes = runner.stage(["eqt-bench", "eqt-check"], [local])
            model = local if runner.local else local.name
            references, outcomes = {}, []
            for case in cases:
                if case["batch"] not in references:
                    request = json.loads(json.dumps(base))
                    request["load"].update(batch=case["batch"], mmap=True)
                    references[case["batch"]] = runner.run("eqt-bench", model, request, workdir)[1]
                reference = references[case["batch"]]
                request = json.loads(json.dumps(base))
                request["load"].update(batch=case["batch"], expert_streaming=True, **case["load"])
                result = runner.run("eqt-bench", model, request, workdir)[1]
                delta = max(abs(a - b) for a, b in zip(reference["first_logits"], result["first_logits"]))
                outcome = dict(case=case["name"], load=case["load"], batch=case["batch"],
                               bitwise_logits=reference["first_logits"] == result["first_logits"],
                               max_abs_logit_delta=delta, token_ids_equal=reference["token_ids"] == result["token_ids"],
                               expert_cache=result["expert_cache"],
                               layer_quotas=result["configuration"]["expert_streaming"]["layer_quotas"])
                outcome["passed"] = outcome["bitwise_logits"] and outcome["token_ids_equal"]
                outcomes.append(outcome)
                print(fixture, case["name"], "PASS" if outcome["passed"] else "FAIL", outcome["expert_cache"]["misses"], "misses", flush=True)
                if not outcome["passed"]:
                    raise AssertionError(outcome)
            lifecycle_request = json.loads(json.dumps(base))
            lifecycle_request["load"].update(expert_streaming=True, expert_cache_mib=1, io_threads=2, prefetch=True)
            lifecycle = runner.run("eqt-check", model, lifecycle_request, workdir)[1]
            rejected = json.loads(json.dumps(base))
            rejected["load"].update(expert_cache_mib=1)
            run, _ = runner.run("eqt-bench", model, rejected, workdir, check=False)
            if run.returncode != 1 or "require expert_streaming" not in run.stderr + run.stdout:
                raise AssertionError("Expert options without streaming were not rejected")
            report["fixtures"].append(dict(model=fixture, model_sha256=hashes[local.name],
                                           binary_sha256=hashes[runner.tool("eqt-bench").name],
                                           outcomes=outcomes, lifecycle=lifecycle,
                                           rejects_cache_options_without_streaming=True))
        # MTP self-speculation must reproduce plain greedy decoding. The random head rejects nearly every
        # draft, which exercises the recurrent-state rollback on every step.
        local = (Path(os.environ.get("EQT_FIXTURE_DIR", ROOT / "models")) / Path(MTP_FIXTURE).name).resolve()
        hashes = runner.stage(["eqt-bench"], [local])
        model = local if runner.local else local.name
        mtp_outcomes, reference = [], None
        for name, load in MTP_CASES:
            request = json.loads(json.dumps(base))
            request["load"].update(load)
            result = runner.run("eqt-bench", model, request, workdir)[1]
            reference = reference or result
            outcome = dict(case=name, load=load, token_ids_equal=result["token_ids"] == reference["token_ids"],
                           speculative=result["speculative"])
            outcome["passed"] = outcome["token_ids_equal"] and (result["speculative"] is not None) == load.get("mtp", False)
            mtp_outcomes.append(outcome)
            print(MTP_FIXTURE, name, "PASS" if outcome["passed"] else "FAIL", flush=True)
            if not outcome["passed"]:
                raise AssertionError(outcome)
        report["mtp"] = dict(model=MTP_FIXTURE, model_sha256=hashes[local.name], outcomes=mtp_outcomes,
                             criterion="identical greedy token IDs with and without MTP self-speculation")
    report["scope"] = ("Random-weight fixtures with the target's qwen35moe graph, tokenizer and routed quant types. "
                       "Exactness of the streamed and speculative paths only; no routing locality, acceptance, "
                       "speed or quality evidence.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
