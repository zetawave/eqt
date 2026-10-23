"""Repeat natural-EOS requests on one loaded model and report decode and wall-time throughput.

Energy comes from the stock battery service (`dumpsys battery` charge counter and voltage); /sys battery
nodes are not readable by the shell user. It is the whole phone's drain, so an idle baseline measured just
before the run is subtracted. Run unplugged; nothing is changed on the device (no thermal or power settings).

  python tools/sustained_run.py --serial S --model /data/local/tmp/eqt/M.gguf --tokens 3600 \
      --load '{"expert_streaming": true, ...}' --output results/local/sustained/M.json
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import time

from device import adb
from runner import ROOT, Runner
from sweep_target import temperatures, thermal
from benchmark import source_hash


def battery(serial):
    text = adb("shell", "dumpsys battery", serial=serial).stdout
    def value(key):
        match = re.search(rf"^\s*{re.escape(key)}: (-?\d+)", text, re.M)
        return int(match.group(1)) if match else None
    powered = any(re.search(rf"^\s*{kind} powered: true", text, re.M) for kind in ("AC", "USB", "Wireless"))
    return dict(charge_uah=value("Charge counter"), voltage_mv=value("voltage"), level=value("level"), powered=powered,
                time=time.monotonic())


def joules(before, after):
    """Charge drawn between two readings at their mean voltage."""
    if any(reading.get(key) is None for reading in (before, after) for key in ("charge_uah", "voltage_mv")):
        return None
    mean_volts = (before["voltage_mv"] + after["voltage_mv"]) / 2000
    return (before["charge_uah"] - after["charge_uah"]) * 1e-6 * 3600 * mean_volts


def sustained_request(base, load, tokens, duration):
    request = copy.deepcopy(base)
    steps = request.get("sequence", [{k: v for k, v in request.items() if k != "load"}])
    if not isinstance(steps, list) or not steps:
        raise ValueError("A nonempty request sequence is required")
    for step in steps:
        if step.get("ignore_eos", False):
            raise ValueError("Sustained workloads must respect EOS")
        step["ignore_eos"] = False
        if tokens is not None:
            step["max_tokens"] = tokens
        if step.get("max_tokens", 0) <= 0:
            raise ValueError("Each request needs a positive max_tokens safety limit")
    return dict(load={**request.get("load", {}), **load}, sequence=steps, sustained_seconds=duration)


def summarize(result):
    runs = result["results"]
    if not runs:
        raise ValueError("No completed requests")
    tokens = sum(r["generated_tokens"] for r in runs)
    decode_ms = sum(r["decode_ms"] for r in runs)
    wall = result["sequence_wall_seconds"]
    # Concatenate active decode time only. Final EOS work advances the timeline but emits no token.
    bins, elapsed = {}, 0.0
    for run in runs:
        start = elapsed
        for latency in run["token_latency_ms"]:
            elapsed += latency
            index = max(0, int((elapsed - 1e-9) // 60000))
            bins[index] = bins.get(index, 0) + 1
        elapsed = start + run["decode_ms"]
    complete = int(decode_ms // 60000)
    minutes = [bins.get(i, 0) / 60 for i in range(complete)]
    def counter_sum(key):
        values = [(r.get("expert_cache") or {}).get(key) for r in runs]
        return sum(values) if all(v is not None for v in values) else None
    io_bytes, busy_ms = counter_sum("decode_io_bytes"), counter_sum("decode_io_busy_ms")
    hits, prefetched, demanded = (counter_sum(key) for key in ("decode_hits", "decode_prefetch_hits", "decode_demanded_experts"))
    bank_rows = [(r.get("expert_cache") or {}).get("decode_io_bytes_by_bank") for r in runs]
    banks = None
    if all(row is not None for row in bank_rows):
        banks = {group: {bank: sum(row[group][bank] for row in bank_rows) for bank in ("gate", "up", "down", "other")}
                 for group in ("demand", "speculative")}
    prefill_ms = sum(r["prefill_ms"] for r in runs)
    return dict(tokens=tokens, request_count=len(runs), decode_ms=decode_ms, prefill_ms=prefill_ms,
                decode_tokens_per_second=tokens * 1000 / decode_ms if decode_ms > 0 else None,
                tokens_per_second_by_minute=minutes, minute_scope="concatenated decode time; excludes prefill and request gaps",
                sequence_wall_seconds=wall, end_to_end_tokens_per_second=tokens / wall if wall > 0 else None,
                other_wall_ms=max(0, wall * 1000 - decode_ms - prefill_ms),
                terminations={reason: sum(r["termination"] == reason for r in runs)
                              for reason in sorted({r["termination"] for r in runs})},
                expert_io_bytes_per_token=io_bytes / tokens if io_bytes is not None and tokens else None,
                expert_io_busy_gb_s=io_bytes / busy_ms / 1e6 if io_bytes is not None and busy_ms else None,
                expert_io_bytes_by_bank=banks,
                decode_demand_cache_hit_fraction=(hits - prefetched) / demanded
                    if hits is not None and prefetched is not None and demanded else None,
                useful_compute_ms_per_token=None, exposed_io_wait_ms_per_token=None,
                requests=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial")
    parser.add_argument("--target", choices=["host", "android"], default="android")
    parser.add_argument("--run-as", metavar="PACKAGE", help="Run as this debug app (models in its files/models/; pass app-relative model paths)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--load", required=True, help="JSON load options")
    parser.add_argument("--request", default="configs/flash-next-sustained-request.json")
    parser.add_argument("--tokens", type=int, help="Override each request's output safety limit")
    parser.add_argument("--duration-seconds", type=float, default=900,
                        help="Repeat natural-EOS requests on one loaded model; finish the active request at the deadline")
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--powered", action="store_true",
                        help="Allow a charging phone: speed and thermal behavior only, energy reported as null")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not math.isfinite(args.duration_seconds) or not 0 < args.duration_seconds <= 86400:
        parser.error("duration-seconds must be in (0, 86400]")
    if args.idle_seconds < 1 or (args.tokens is not None and args.tokens < 1):
        parser.error("Idle duration and token limit must be positive")
    base = json.loads(Path(args.request).read_text(encoding="utf-8"))
    request = sustained_request(base, json.loads(args.load), args.tokens, args.duration_seconds)
    runner = Runner(args.target, args.serial, run_as=args.run_as)
    source_before = source_hash()
    idle_start = battery(args.serial) if args.target == "android" else None
    if idle_start and idle_start["powered"] and not args.powered:
        raise SystemExit("The phone is powered; unplug it so the battery counter measures the run, or pass --powered.")
    hashes = runner.stage(["eqt-bench"])
    idle_start = battery(args.serial) if args.target == "android" else None
    if idle_start and idle_start["powered"] and not args.powered:
        raise SystemExit("The phone became powered; unplug it or pass --powered.")
    # While charging, the charge counter measures the charger too: no energy figures.
    measure_energy = idle_start is not None and not idle_start["powered"]
    idle_watts = None
    if measure_energy:
        deadline = time.monotonic() + args.idle_seconds
        while (remaining := deadline - time.monotonic()) > 0:
            time.sleep(min(remaining, 30))
        idle_end = battery(args.serial)
        idle_joules = joules(idle_start, idle_end)
        measure_energy = not idle_end["powered"]
        idle_watts = idle_joules / (idle_end["time"] - idle_start["time"]) if idle_joules is not None and measure_energy else None

    def state():
        return dict(battery=battery(args.serial), thermal=thermal(args.serial), celsius=temperatures(args.serial)) if args.target == "android" else None
    start = state()
    wall_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="eqt-sustained-", dir=ROOT / ".cache") as workdir:
        process, result = runner.run("eqt-bench", args.model, request, workdir, check=False)
    seconds = time.monotonic() - wall_start
    end = state()
    if process.returncode or result is None:
        raise SystemExit(f"eqt-bench failed: {process.stderr[-800:]}")
    measure_energy = measure_energy and not end["battery"]["powered"]
    run_joules = joules(start["battery"], end["battery"]) if measure_energy else None
    energy_seconds = end["battery"]["time"] - start["battery"]["time"] if start else None
    net_joules = run_joules - idle_watts * energy_seconds if run_joules is not None and idle_watts is not None else None
    report = summarize(result)
    report.update(model=args.model, load=request["load"], configuration=result.get("configuration"), wall_seconds=seconds,
                  idle_watts=idle_watts, run_joules=run_joules, net_joules=net_joules,
                  net_joules_per_token=net_joules / report["tokens"] if net_joules is not None and report["tokens"] else None,
                  device_before=start, device_after=end,
                  provenance=dict(source_sha256=source_before, source_sha256_after=source_hash(), artifacts=hashes, target=args.target,
                                  request=request, request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()),
                  energy_scope="whole phone over runner interval including load; idle baseline subtracted; null while powered")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("tokens", "decode_tokens_per_second", "tokens_per_second_by_minute",
                                             "end_to_end_tokens_per_second", "net_joules_per_token", "terminations")}, indent=1))


if __name__ == "__main__":
    main()
