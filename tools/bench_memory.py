"""Run pinned read probes on stock Android; achieved payload rates are not a hardware bandwidth ceiling.

The default alternates sums and forced loads over 256/512 MiB with dynamic 1 MiB work distribution.
No model is loaded. Dense-kernel comparison is a separate optional run, with inherited CPU affinity;
its legacy raw_read_gb_s field is explicitly excluded. USB energy and unavailable counters stay null.
"""

import argparse
import json
from pathlib import Path

from benchmark import source_hash
from runner import Runner
from sweep_target import temperatures, thermal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", default="4,5,6,7,8,9")
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--passes", type=int, default=8)
    parser.add_argument("--serial")
    parser.add_argument("--run-as", default="org.equity.app")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cpus = [int(v) for v in args.cpus.split(",")]
    if not cpus or len(cpus) > 16 or len(set(cpus)) != len(cpus) or any(c < 0 or c > 63 for c in cpus):
        parser.error("Need 1..16 distinct CPU indices in 0..63")
    if not 1 <= args.seconds <= 300 or not 1 <= args.passes <= 64:
        parser.error("Seconds must be in 1..300; passes in 1..64")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runner = Runner("android", args.serial, run_as=args.run_as)
    hashes = runner.stage(["eqt-memory-bench"] + (["eqt-kernel-bench"] if args.dense else []))
    report = dict(kind="achieved_cpu_read_bandwidth", source_sha256=source_hash(), artifact_sha256=hashes,
                  memory_controller_clock=None, dram_controller_bytes=None, energy_j_per_token=None, runs=[],
                  limitations=["No hardware bandwidth ceiling follows from a software microbenchmark.",
                               "Payload bytes include cache-served reads; buffers exceed the expected CPU caches.",
                               "Static thermal/power controls; USB powered; no concurrent inference.",
                               "CPU frequencies sampled after a window can show sleeping worker clocks.",
                               "Dense comparison uses synthetic shapes; old raw_read_gb_s is not a valid ceiling."])

    def state():
        return dict(thermal=thermal(args.serial), temperatures=temperatures(args.serial))

    arms = [("sum256", 256, "sum-dynamic"), ("load256", 256, "load-dynamic"),
            ("load512", 512, "load-dynamic"), ("sum512", 512, "sum-dynamic"),
            ("load256_repeat", 256, "load-dynamic")]
    if args.dense:
        arms.insert(2, ("dense", 0, ""))
    for name, mib, mode in arms:
        if name == "dense":
            mask = sum(1 << cpu for cpu in cpus)
            command = f"taskset {mask:x} ./eqt-kernel-bench {len(cpus)} --sustain={args.seconds}"
        else:
            command = f"./eqt-memory-bench {','.join(map(str, cpus))} {mib} {args.seconds} {args.passes} {mode}"
        print(f"Starting {name}", flush=True)
        before = state()
        process = runner.app("cd files/bench && LD_LIBRARY_PATH=. " + command, check=False)
        (output / f"{name}.stderr.txt").write_text(process.stderr, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(process.stderr[-2000:] + process.stdout[-1000:])
        result = json.loads(process.stdout)
        if name != "dense" and not result["checksum_and_affinity_valid"]:
            raise RuntimeError("Read checksum or affinity check failed")
        report["runs"].append(dict(name=name, command=command, before=before, after=state(), result=result))
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if name != "dense":
            samples = result["samples"]
            rate = sum(s["bytes"] for s in samples) / sum(s["seconds"] for s in samples) / 1e9
            print(f"{name}: {rate:.2f} GB/s", flush=True)


if __name__ == "__main__":
    main()
