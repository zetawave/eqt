"""Sustained decode on the phone, on battery: tokens/s over minutes and energy per token.

Energy comes from the stock battery service (`dumpsys battery` charge counter and voltage); /sys battery
nodes are not readable by the shell user. It is the whole phone's drain, so an idle baseline measured just
before the run is subtracted. Run unplugged; nothing is changed on the device (no thermal or power settings).

  python tools/sustained_run.py --serial S --model /data/local/tmp/eqt/M.gguf --tokens 3600 \
      --load '{"expert_streaming": true, ...}' --output results/local/sustained/M.json
"""

import argparse
import json
from pathlib import Path
import re
import tempfile
import time

from device import adb
from runner import ROOT, Runner
from sweep_target import temperatures, thermal


def battery(serial):
    text = adb("shell", "dumpsys battery", serial=serial).stdout
    value = lambda key: int(re.search(rf"^\s*{key}: (-?\d+)", text, re.M).group(1))
    powered = any(re.search(rf"^\s*{kind} powered: true", text, re.M) for kind in ("AC", "USB", "Wireless"))
    return dict(charge_uah=value("Charge counter"), voltage_mv=value("voltage"), level=value("level"), powered=powered,
                time=time.monotonic())


def joules(before, after):
    """Charge drawn between two readings at their mean voltage."""
    mean_volts = (before["voltage_mv"] + after["voltage_mv"]) / 2000
    return (before["charge_uah"] - after["charge_uah"]) * 1e-6 * 3600 * mean_volts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial")
    parser.add_argument("--model", required=True)
    parser.add_argument("--load", required=True, help="JSON load options")
    parser.add_argument("--request", default="configs/target-stream-request.json")
    parser.add_argument("--tokens", type=int, default=3600)
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runner = Runner("android", args.serial)
    runner.stage(["eqt-bench"])
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    request.update(max_tokens=args.tokens, temperature=0, ignore_eos=True)
    request["load"].update(json.loads(args.load))
    request["load"]["context"] = max(request["load"].get("context", 2048), args.tokens + 256)

    idle_start = battery(args.serial)
    if idle_start["powered"]:
        raise SystemExit("The phone is powered; unplug it so the battery counter measures the run.")
    time.sleep(args.idle_seconds)
    idle_end = battery(args.serial)
    idle_watts = joules(idle_start, idle_end) / (idle_end["time"] - idle_start["time"])

    start = dict(battery=battery(args.serial), thermal=thermal(args.serial), celsius=temperatures(args.serial))
    with tempfile.TemporaryDirectory(prefix="eqt-sustained-", dir=ROOT / ".cache") as workdir:
        process, result = runner.run("eqt-bench", args.model, request, workdir, check=False)
    end = dict(battery=battery(args.serial), thermal=thermal(args.serial), celsius=temperatures(args.serial))
    if result is None:
        raise SystemExit(f"eqt-bench failed: {process.stderr[-800:]}")

    seconds = end["battery"]["time"] - start["battery"]["time"]
    run_joules = joules(start["battery"], end["battery"])
    net_joules = run_joules - idle_watts * seconds
    latencies = result["token_latency_ms"]
    # Decode speed per minute shows throttling; latencies are per generated token.
    minutes, elapsed, count = [], 0.0, 0
    for latency in latencies:
        elapsed += latency
        count += 1
        if elapsed >= 60000:
            minutes.append(round(count / (elapsed / 1000), 2))
            elapsed, count = 0.0, 0
    report = dict(
        model=args.model, load=request["load"], tokens=result["generated_tokens"],
        decode_tokens_per_second=result["decode_tokens_per_second"], tokens_per_second_by_minute=minutes,
        prefill_ms=result["prefill_ms"], decode_ms=result["decode_ms"], wall_seconds=seconds,
        idle_watts=idle_watts, run_joules=run_joules, net_joules=net_joules,
        net_joules_per_token=net_joules / max(result["generated_tokens"], 1),
        battery_level=[start["battery"]["level"], end["battery"]["level"]],
        thermal=[start["thermal"], end["thermal"]], celsius=[start["celsius"], end["celsius"]],
        expert_cache=result["expert_cache"], decode_process_cpu=result.get("decode_process_cpu"),
        memory_after=result["memory_after"], text_head=result["text"][:400],
        energy_scope="whole phone, idle baseline subtracted; battery service charge counter")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("tokens", "decode_tokens_per_second", "tokens_per_second_by_minute",
                                             "idle_watts", "net_joules_per_token", "battery_level", "thermal")}, indent=1))


if __name__ == "__main__":
    main()
