"""Verify numerical equivalence and state lifecycle on a real model."""

import argparse
import json
import math
from pathlib import Path
import tempfile

from benchmark import ROOT, source_hash
from runner import Runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["host", "linux", "android"], default="host")
    parser.add_argument("--serial")
    parser.add_argument("--cpu-variant", help="Android only: force one GGML CPU backend variant")
    parser.add_argument("--output", default="results/local/correctness.json")
    parser.add_argument("--model", default="models/Qwen3-0.6B-Q8_0.gguf")
    args = parser.parse_args()
    runner = Runner(args.target, args.serial, args.cpu_variant)
    request = json.loads((ROOT / "configs/smoke-request.json").read_text(encoding="utf-8"))
    request.update(max_tokens=24, capture_logits=True)
    lifecycle_request = json.loads(json.dumps(request))
    local = Path(args.model).resolve()
    hashes = runner.stage(["eqt-bench", "eqt-check"], [local])
    model = local if runner.local else local.name
    outcomes, reference, configuration = [], None, None
    (ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eqt-verify-", dir=ROOT / ".cache") as directory:
        for mapped, batch in [(True, 128), (False, 128), (True, 16)]:
            request["load"].update(mmap=mapped, batch=batch)
            result = runner.run("eqt-bench", model, request, directory)[1]
            configuration = configuration or result["configuration"]
            logits = result["first_logits"]
            if reference is None:
                reference = result
            assert len(logits) == len(reference["first_logits"])
            assert all(math.isfinite(value) for value in logits)
            delta = max(abs(a - b) for a, b in zip(logits, reference["first_logits"]))
            token_match = result["token_ids"] == reference["token_ids"]
            # F32 reductions can reorder across GEMM batch shapes; this is an absolute logit tolerance.
            passed = token_match and delta <= 0.002
            outcomes.append(dict(mmap=mapped, batch=batch, max_abs_logit_delta=delta, token_ids_equal=token_match, passed=passed))
            if not passed:
                raise AssertionError(outcomes[-1])
        overflow = json.loads(json.dumps(request))
        overflow["load"]["context"] = 128
        overflow["max_tokens"] = 2048
        rejected, _ = runner.run("eqt-bench", model, overflow, directory, check=False)
        assert rejected.returncode == 1 and "exceeds context" in rejected.stderr + rejected.stdout
        outcomes.append(dict(case="context_overflow", passed=True))
        lifecycle = runner.run("eqt-check", model, lifecycle_request, directory)[1]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tool = runner.tool("eqt-bench").name
    output.write_text(json.dumps(dict(target=args.target, model_sha256=hashes[local.name], binary_sha256=hashes[tool],
                                      lifecycle_binary_sha256=hashes[runner.tool("eqt-check").name],
                                      source_sha256=source_hash(), cpu_variant=configuration.get("cpu_variant"),
                                      threadpool=configuration.get("threadpool"), lifecycle=lifecycle,
                                      tolerance_absolute=0.002, outcomes=outcomes,
                                      scope=f"{local.name}: mapped/allocated, prefill chunks and state reset; not expert streaming"), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
