"""Screen verification or compare real greedy prompt lookup, normally in five short runs.

Acceptance 1 measures the verifier with perfect proposals; it is an optimistic bound, not speculative inference.
Use --lookup for actual greedy generation. The final plain run exposes drift. No cooldown or sustained test is
performed. Android models must already exist.
"""

import argparse
import copy
import json
from pathlib import Path
import tempfile

from benchmark import source_hash
from runner import ROOT, Runner
from sweep_target import temperatures, thermal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["host", "linux", "android"], default="android")
    parser.add_argument("--model", required=True, help="Local path, or an existing Android model path")
    parser.add_argument("--request", default="configs/verification-request.json")
    parser.add_argument("--load", type=json.loads, default={}, help="JSON overrides for the request's load options")
    parser.add_argument("--run-as")
    parser.add_argument("--serial")
    parser.add_argument("--cpu-variant")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 4, 2], help="Verification rows including the anchor (2..8)")
    parser.add_argument("--acceptance", type=float, default=1.0, help="Conditional acceptance at each reached draft depth")
    parser.add_argument("--lookup", action="store_true", help="Run real greedy prompt lookup instead of teacher-forced replay")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 8 <= args.tokens <= 256 or any(not 2 <= n <= 8 for n in args.rows):
        parser.error("Use 8..256 tokens and verification rows in 2..8")
    if not 0 <= args.acceptance <= 1 or not isinstance(args.load, dict):
        parser.error("Acceptance must be in [0, 1] and --load must be an object")
    base = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if any(key in base for key in ("sequence", "verify_replay", "prompt_lookup", "session", "images", "steering")):
        parser.error("Use a single plain text request without session or steering")
    base.setdefault("load", {}).update(args.load)
    base["load"].update(mtp=False, rollback_slots=0)
    base.update(max_tokens=args.tokens, temperature=0, ignore_eos=True)
    if base["load"].get("batch", 512) < max(args.rows):
        parser.error("The load batch must fit every verification block")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.target, args.serial, args.cpu_variant, run_as=args.run_as)
    hashes = runner.stage(["eqt-bench"])
    report = dict(source_sha256=source_hash(), artifacts=hashes, target=args.target, model=args.model,
                  base_request=base, mode="prompt_lookup" if args.lookup else "verify_replay",
                  requested_acceptance=None if args.lookup else args.acceptance,
                  scope=("Short real greedy prompt-lookup comparison; no general-quality or sustained-speed claim."
                         if args.lookup else "Short verifier screening with zero draft cost; no inference speedup or sustained-speed claim."),
                  rows=[])
    reference = None
    mode = "lookup" if args.lookup else "verify"
    arms = [("plain_before", None)] + [(f"{mode}_{n}", n) for n in args.rows] + [("plain_after", None)]
    with tempfile.TemporaryDirectory(prefix="eqt-verification-", dir=ROOT / ".cache") as workdir:
        for name, width in arms:
            request = copy.deepcopy(base)
            if width is not None:
                request["load"]["rollback_slots"] = width
                if args.lookup:
                    request["prompt_lookup"] = width - 1
                else:
                    request["verify_replay"] = dict(tokens=reference, acceptance=args.acceptance, draft_max=width - 1)
            before = thermal(args.serial) if not runner.local else None
            celsius = temperatures(args.serial) if not runner.local else None
            _, result = runner.run("eqt-bench", args.model, request, workdir)
            after = thermal(args.serial) if not runner.local else None
            (output / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            if reference is None:
                reference = result["token_ids"]
                if len(reference) != args.tokens:
                    raise RuntimeError("Reference did not emit the requested number of tokens")
            same = result["token_ids"] == reference
            cache = result.get("expert_cache") or {}
            replay = result.get("verify_replay") or result.get("prompt_lookup") or {}
            count = result["generated_tokens"]
            row = dict(arm=name, verification_rows=width, decode_tokens_per_second=result["decode_tokens_per_second"],
                       decode_ms=result["decode_ms"], tokens=count, token_ids_equal=same,
                       target_agreement=replay.get("target_agreement"), tokens_per_step=replay.get("tokens_per_step"),
                       step_ms=result["decode_ms"] / replay["verification_steps"] if replay else None,
                       drafted_tokens=replay.get("drafted_tokens"), acceptance_rate=replay.get("acceptance_rate"),
                       thermal_before=before, thermal_after=after, celsius_before=celsius,
                       read_bytes_per_token=cache["decode_io_bytes"] / count if "decode_io_bytes" in cache else None,
                       memory_after=result["memory_after"])
            report["rows"].append(row)
            (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row), flush=True)
            if not same:
                raise RuntimeError(f"{name} did not follow the reference text; raw results and summary were retained")
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
