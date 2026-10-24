"""Short Android comparison of ordinary/explicit anonymous and AHWB resident-weight allocators.

Natural EOS is preserved. Explicit anonymous uses identical chunking; default keeps GGML's layout.
Output hashes must match. This is
a screening, not a sustained-speed claim. A current run finishes at its token cap if the soft
wall-time budget expires; no further arm starts. Models must already exist on the device.
"""

import argparse
import copy
import json
from pathlib import Path
import time

from benchmark import source_hash
from runner import Runner
from sweep_target import temperatures, thermal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request", default="configs/flash-next-oracle-request.json")
    parser.add_argument("--run-as", default="org.equity.app")
    parser.add_argument("--serial")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--baseline", choices=["anonymous", "default"], default="anonymous")
    parser.add_argument("--pairs", type=int, choices=[1, 2], default=2)
    parser.add_argument("--budget-seconds", type=int, default=720)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.tokens <= 1024 or not 1 <= args.budget_seconds <= 1800:
        parser.error("Tokens must be in 1..1024 and the soft budget in 1..1800 seconds")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runner = Runner("android", args.serial, run_as=args.run_as)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if "sequence" in request or request.get("temperature", 0) != 0:
        parser.error("Use one greedy text request")
    request.update(max_tokens=args.tokens, ignore_eos=False, diagnostic_logits_hash=True, diagnostic_residency=True)
    report = dict(kind="resident_weight_allocator_screening", source_sha256=source_hash(),
                  artifact_sha256=runner.stage(["eqt-bench"]), request=request, runs=[],
                  complete=False, energy_j_per_token=None)
    start = time.monotonic()
    reference = None
    arms = [("ahwb_a", "ahwb"), (args.baseline + "_a", args.baseline),
            (args.baseline + "_b", args.baseline), ("ahwb_b", "ahwb")]
    if args.pairs == 1:
        arms = [(args.baseline + "_a", args.baseline), ("ahwb_a", "ahwb")]
    for name, mode in arms:
        if time.monotonic() - start > args.budget_seconds:
            break
        req = copy.deepcopy(request)
        req["load"]["resident_weights"] = mode
        before = dict(thermal=thermal(args.serial), temperatures=temperatures(args.serial))
        print("Starting " + name, flush=True)
        work = output / name
        work.mkdir(exist_ok=True)
        process, result = runner.run("eqt-bench", args.model, req, work, check=False)
        (work / "stderr.txt").write_text(process.stderr, encoding="utf-8")
        entry = dict(name=name, mode=mode, before=before,
                     after=dict(thermal=thermal(args.serial), temperatures=temperatures(args.serial)))
        if process.returncode or result is None:
            entry.update(error=process.stderr[-3000:], returncode=process.returncode)
        else:
            if reference is None:
                reference = result
            equal = (bool(result["diagnostic_logits_sha256"]) and
                     result["token_ids"] == reference["token_ids"] and
                     result["diagnostic_logits_sha256"] == reference["diagnostic_logits_sha256"] and
                     result["termination"] == reference["termination"])
            entry.update(result=result, bitwise_logits_and_tokens_equal=equal)
            print(f"{name}: {result['decode_tokens_per_second']:.3f} tokens/s, equal={equal}", flush=True)
        report["runs"].append(entry)
        report["wall_seconds"] = time.monotonic() - start
        report["complete"] = len(report["runs"]) == len(arms) and all(r.get("bitwise_logits_and_tokens_equal") for r in report["runs"])
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if "error" in entry or not entry["bitwise_logits_and_tokens_equal"]:
            raise RuntimeError("Resident-weight comparison failed: " + name)


if __name__ == "__main__":
    main()
