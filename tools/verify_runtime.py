"""Verify numerical equivalence and state lifecycle on a real model."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

from benchmark import ROOT, sha256, source_hash
from device import adb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["host", "android"], default="host")
    parser.add_argument("--output", default="results/local/correctness.json")
    parser.add_argument("--model", default="models/Qwen3-0.6B-Q8_0.gguf")
    args = parser.parse_args()
    request = json.loads((ROOT / "configs/smoke-request.json").read_text(encoding="utf-8"))
    request.update(max_tokens=24, capture_logits=True)
    lifecycle_request = json.dumps(request)
    outcomes = []
    reference = None
    binary = ROOT / "build" / args.target / "bin" / ("Release/eqt-bench.exe" if args.target == "host" else "eqt-bench")
    checker = binary.with_name("eqt-check.exe" if args.target == "host" else "eqt-check")
    model = Path(args.model).resolve()
    model_hash, binary_hash, checker_hash = sha256(model), sha256(binary), sha256(checker)
    remote = "/data/local/tmp/eqt"
    if args.target == "android":
        adb("shell", "mkdir", "-p", remote)
        for local, name, expected in [(binary, "eqt-bench", binary_hash), (checker, "eqt-check", checker_hash),
                                      (model, "model.gguf", model_hash)]:
            adb("push", "--sync", str(local), remote + "/" + name)
            actual = adb("shell", "sha256sum", remote + "/" + name).stdout.split()[0]
            if actual != expected:
                raise RuntimeError(f"Device artifact hash mismatch: {name}")
        # ADB push can reset the executable bit, including after a previous successful run.
        adb("shell", "chmod", "700", remote + "/eqt-bench", remote + "/eqt-check")
    (ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eqt-verify-", dir=ROOT / ".cache") as directory:
        directory = Path(directory)
        for mapped, batch in [(True, 128), (False, 128), (True, 16)]:
            request["load"].update(mmap=mapped, batch=batch)
            source, target = directory / "request.json", directory / "result.json"
            source.write_text(json.dumps(request), encoding="utf-8")
            if args.target == "android":
                adb("push", str(source), remote + "/verify-request.json")
                run = adb("shell", remote + "/eqt-bench", remote + "/model.gguf", remote + "/verify-request.json", remote + "/verify-result.json")
                adb("pull", remote + "/verify-result.json", str(target))
            else:
                run = subprocess.run([str(binary), str(model), str(source), str(target)], capture_output=True, check=True)
            result = json.loads(target.read_text(encoding="utf-8"))
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
        request["load"]["context"] = 128
        request["max_tokens"] = 2048
        source.write_text(json.dumps(request), encoding="utf-8")
        if args.target == "host":
            rejected = subprocess.run([str(binary), str(model), str(source), str(target)], capture_output=True)
            assert rejected.returncode == 1 and b"exceeds context" in rejected.stderr
        else:
            adb("push", str(source), remote + "/verify-request.json")
            rejected = adb("shell", remote + "/eqt-bench", remote + "/model.gguf", remote + "/verify-request.json",
                           remote + "/verify-result.json", check=False)
            assert rejected.returncode == 1 and "exceeds context" in rejected.stderr + rejected.stdout
        outcomes.append(dict(case="context_overflow", passed=True))
        source.write_text(lifecycle_request, encoding="utf-8")
        if args.target == "host":
            subprocess.run([str(checker), str(model), str(source), str(target)], capture_output=True, check=True)
        else:
            adb("push", str(source), remote + "/verify-request.json")
            adb("shell", remote + "/eqt-check", remote + "/model.gguf", remote + "/verify-request.json",
                remote + "/verify-result.json")
            adb("pull", remote + "/verify-result.json", str(target))
        lifecycle = json.loads(target.read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(target=args.target, model_sha256=model_hash, binary_sha256=binary_hash,
                                      source_sha256=source_hash(),
                                      lifecycle_binary_sha256=checker_hash, lifecycle=lifecycle,
                                      tolerance_absolute=0.002, outcomes=outcomes,
                                      scope=f"{model.name}: mapped/allocated, prefill chunks and state reset; not expert streaming"), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
