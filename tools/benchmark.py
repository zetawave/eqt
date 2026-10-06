"""Run the identical native CLI request on the host or stock Android, and retain provenance."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

from device import adb, snapshot

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hash():
    digest = hashlib.sha256()
    for base in ["native", "android/app/src", "tools", "scripts"]:
        for path in sorted((ROOT / base).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                digest.update(path.relative_to(ROOT).as_posix().encode())
                digest.update(path.read_bytes())
    for name in ["CMakeLists.txt", "dependencies.json", "android/app/build.gradle.kts"]:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["host", "android"], default="android")
    parser.add_argument("--model", default="models/Qwen3-0.6B-Q8_0.gguf")
    parser.add_argument("--request", default="configs/smoke-request.json")
    parser.add_argument("--output", default="results/local/smoke")
    parser.add_argument("--serial")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--duration-seconds", type=int, default=0)
    args = parser.parse_args()
    if args.repetitions < 1 or args.duration_seconds < 0:
        parser.error("Invalid repetition count or duration")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = Path(args.model).resolve()
    request = Path(args.request).resolve()
    binary = ROOT / "build" / args.target / "bin" / ("Release/eqt-bench.exe" if args.target == "host" else "eqt-bench")
    provenance = {
        "source_sha256": source_hash(), "binary_sha256": sha256(binary), "model_sha256": sha256(model),
        "workload_sha256": sha256(request), "workload": json.loads(request.read_text(encoding="utf-8")),
        "dependencies": json.loads((ROOT / "dependencies.json").read_text()), "surface": "cli",
        "target": args.target, "page_cache": "uncontrolled", "profiling": False,
    }
    remote = "/data/local/tmp/eqt"
    if args.target == "android":
        adb("shell", "mkdir", "-p", remote, serial=args.serial)
        for source, name in [(binary, "eqt-bench"), (model, "model.gguf"), (request, "request.json")]:
            adb("push", "--sync", str(source), remote + "/" + name, serial=args.serial)
        adb("shell", "chmod", "700", remote + "/eqt-bench", serial=args.serial)
    start = time.monotonic()
    index = 0
    while index < args.repetitions or time.monotonic() - start < args.duration_seconds:
        prefix = output / f"run-{index:03d}"
        before = snapshot(args.serial) if args.target == "android" else None
        utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if args.target == "android":
            run = adb("shell", remote + "/eqt-bench", remote + "/model.gguf", remote + "/request.json",
                      remote + "/result.json", serial=args.serial, check=False)
            if run.returncode == 0:
                adb("pull", remote + "/result.json", str(prefix.with_suffix(".raw.json")), serial=args.serial)
        else:
            run = subprocess.run([str(binary), str(model), str(request), str(prefix.with_suffix(".raw.json"))],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
        prefix.with_suffix(".log").write_text(run.stdout + "\n" + run.stderr, encoding="utf-8")
        if run.returncode:
            raise RuntimeError(f"Inference failed ({run.returncode}); see {prefix}.log")
        result = json.loads(prefix.with_suffix(".raw.json").read_text(encoding="utf-8"))
        result["provenance"] = dict(provenance, started_utc=utc, repetition=index,
                                    device_before=before, device_after=snapshot(args.serial) if before else None)
        prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f'{index}: {result["generated_tokens"]} tokens, {result["decode_tokens_per_second"]:.2f} tok/s, TTFT {result["ttft_ms"]} ms', flush=True)
        index += 1


if __name__ == "__main__":
    main()
