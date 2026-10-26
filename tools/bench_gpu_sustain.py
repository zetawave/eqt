"""Sustained CPU/GPU weight-read screening on the Galaxy (eqt-gpu-bench sustain).

Gate for a GPU dense-weight path: after the CPU's Q5_K products have run long enough to reach the throttled state
decoding runs in, does the Xclipse GPU keep its Q5_K and raw read rates? Each phase prints a 5 s window with CPU
clocks and thermal-zone temperatures read by the benchmark itself (no polling over adb during the run).

The binary and GGML modules come from the app's Vulkan build (scripts/android-app.ps1 or Gradle with
-Pequity.vulkan=true); they run from /data/local/tmp as the shell user, without a model. USB power, no cooldown
inside the run, no thermal controls.
"""

import argparse
import hashlib
import json
from pathlib import Path

from device import adb
from sweep_target import temperatures, thermal

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "android/app/build/intermediates/cxx/Release"
REMOTE = "/data/local/tmp/eqt-gpu"


def artifacts():
    found = sorted(BUILD.glob("*/obj/arm64-v8a/eqt-gpu-bench"))
    if not found:
        raise FileNotFoundError("Build eqt-gpu-bench in the app's Vulkan tree first")
    folder = found[-1].parent
    files = [folder / "eqt-gpu-bench", folder / "libc++_shared.so"] + sorted(folder.glob("libggml*.so"))
    return [f for f in files if f.exists()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial")
    parser.add_argument("--seconds", type=float, default=60, help="length of each main phase")
    parser.add_argument("--gpu-first", action="store_true", help="start from a cool GPU instead of a throttled CPU")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = artifacts()
    adb("shell", "mkdir", "-p", REMOTE, serial=args.serial)
    hashes = {}
    for path in files:
        adb("push", "--sync", str(path), f"{REMOTE}/{path.name}", serial=args.serial)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    adb("shell", "chmod", "755", f"{REMOTE}/eqt-gpu-bench", serial=args.serial)
    before = dict(thermal=thermal(args.serial), temperatures=temperatures(args.serial))
    order = " gpu_first" if args.gpu_first else ""
    run = adb("shell", f"cd {REMOTE} && LD_LIBRARY_PATH={REMOTE} ./eqt-gpu-bench sustain {args.seconds:g}{order}",
              serial=args.serial, check=False)
    after = dict(thermal=thermal(args.serial), temperatures=temperatures(args.serial))
    (output / "stdout.json").write_text(run.stdout, encoding="utf-8")
    (output / "stderr.txt").write_text(run.stderr, encoding="utf-8")
    if run.returncode:
        raise RuntimeError(f"eqt-gpu-bench failed ({run.returncode}): {run.stderr[-2000:]}")
    bench = json.loads(run.stdout)
    samples = [r for r in bench["results"] if r.get("test") == "sustain"]
    phases = {}
    for sample in samples:
        phases.setdefault(sample["phase"], []).append(sample["gb_s"])
    report = dict(kind="gpu_cpu_sustained_read_screen", artifact_sha256=hashes, device=bench.get("device"),
                  before=before, after=after, seconds_per_phase=args.seconds, gpu_first=args.gpu_first, bench=bench,
                  energy_j=None,
                  limitations=["Synthetic Q5_K 2048x4096 products and a 256 MiB row sum, not decoding.",
                               "One sequence per session; phases inherit the thermal state of the previous ones.",
                               "GGML's Vulkan shaders; a custom shader could read faster than gpu_q5_k but not "
                               "than gpu_read."])
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for sample in samples:
        print(f"{sample['t_s']:6.1f}s {sample['phase']:9s} {sample['gb_s']:5.1f} GB/s  cpu MHz {sample['cpu_mhz']}")


if __name__ == "__main__":
    main()
