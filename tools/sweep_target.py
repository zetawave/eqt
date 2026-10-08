"""Sweep streamed-target load options with alternating order; identical prompts give identical routing.

Each arm overrides `load` fields of the base request. Greedy decoding with a fixed output length makes
every arm demand the same expert sequence, so differences isolate I/O, cache and scheduling options.
"""

import argparse
import json
from pathlib import Path
import re
import tempfile
import time

from device import adb
from runner import REMOTE, ROOT, Runner

TARGET = f"{REMOTE}/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


def thermal(serial):
    text = adb("shell", "dumpsys thermalservice | grep -m1 'Thermal Status'", serial=serial, check=False).stdout
    return text.strip().split(":")[-1].strip() if text else None


def temperatures(serial):
    """Live HAL temperatures (Celsius) by sensor name; the thermal service's cached list can be stale."""
    text = adb("shell", "dumpsys thermalservice | grep -A12 'Current temperatures from HAL'", serial=serial,
               check=False).stdout or ""
    return {name: float(value) for value, name in re.findall(r"mValue=([0-9.]+), mType=\d+, mName=(\w+)", text)}


def cool_down(serial, ceiling, timeout_s):
    """Wait (read-only, no thermal overrides) until the skin sensor, which drives the stock throttling
    status on this phone, is at or below `ceiling`; return the readings."""
    deadline = time.monotonic() + timeout_s
    while (value := temperatures(serial)).get("SKIN", 0) > ceiling and time.monotonic() < deadline:
        time.sleep(15)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", required=True, help='JSON object: {"name": {load overrides}, ...}')
    parser.add_argument("--request", default="configs/target-stream-request.json")
    parser.add_argument("--quality-item", help="Replace the request messages with this workloads/quality-v1 item "
                                                "(generated long-context items included); context grows to fit")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--serial")
    parser.add_argument("--model", default=TARGET, help="Staged device path of the target artifact")
    parser.add_argument("--cpu-variant")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cool-to", type=float, help="Before each run, wait until the skin temperature is at most this (C)")
    parser.add_argument("--cool-timeout", type=float, default=600, help="Maximum wait per run for --cool-to, seconds")
    args = parser.parse_args()
    arms = json.loads(args.arms)
    runner = Runner("android", args.serial, args.cpu_variant)
    runner.stage(["eqt-bench"])
    base = json.loads(Path(args.request).read_text(encoding="utf-8"))
    base.update(max_tokens=args.max_tokens, temperature=0, ignore_eos=True)
    if args.quality_item:
        from eval_quality import SYSTEM, generate, load_items
        item = load_items(ROOT / "workloads/quality-v1.jsonl", [args.quality_item])[0]
        messages = generate(item)[0] if "generator" in item else (
            item.get("messages") or [{"role": "user", "content": item["prompt"]}])
        base["messages"] = [{"role": "system", "content": SYSTEM}] + messages
        base["load"]["context"] = max(base["load"].get("context", 2048), item.get("context", 0))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="eqt-sweep-", dir=ROOT / ".cache") as workdir:
        for rep in range(args.repetitions):
            for name in (list(arms) if rep % 2 == 0 else list(arms)[::-1]):
                request = json.loads(json.dumps(base))
                request["load"].update(arms[name])
                celsius = cool_down(args.serial, args.cool_to, args.cool_timeout) if args.cool_to else temperatures(args.serial)
                before = thermal(args.serial)
                run, result = runner.run("eqt-bench", args.model, request, workdir, check=False)
                if run.returncode or result is None:
                    rows.append(dict(arm=name, repetition=rep, error=(run.stderr or run.stdout)[-1500:]))
                    print(name, rep, "FAILED", rows[-1]["error"][-300:], flush=True)
                    continue
                (output / f"{name}-rep{rep}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                e, mb, ma = result["expert_cache"], result["memory_before"], result["memory_after"]
                row = dict(arm=name, repetition=rep, load=arms[name], thermal_before=before, thermal_after=thermal(args.serial),
                           celsius_before={k: celsius.get(k) for k in ("SKIN", "AP")}, decode_process_cpu=result.get("decode_process_cpu"),
                           tokens=result["token_ids"], load_ms=result["configuration"]["load_ms"],
                           prefill_ms=result["prefill_ms"], decode_ms=result["decode_ms"],
                           decode_tokens_per_second=result["decode_tokens_per_second"],
                           p50_ms=result["token_latency_p50_ms"], p95_ms=result["token_latency_p95_ms"],
                           decode_io_wait_ms=e["decode_io_wait_ms"], io_wait_ms=e["io_wait_ms"],
                           decode_misses=e["decode_misses"], decode_demanded=e["decode_demanded_experts"],
                           decode_read_bytes=e["decode_read_bytes"], demand_read_bytes=e["demand_read_bytes"],
                           physical_read_bytes=ma["storage_read_bytes"] - mb["storage_read_bytes"],
                           peak_resident_bytes=e["peak_resident_bytes"], rss_bytes=ma["rss_bytes"],
                           peak_rss_bytes=ma["peak_rss_bytes"], speculative=result.get("speculative"))
                rows.append(row)
                decode_s = row["decode_ms"] / 1000
                print(f'{name:16s} rep{rep}: decode {row["decode_tokens_per_second"]:.2f} tok/s, p50 {row["p50_ms"]:.0f} ms, '
                      f'decode io wait {row["decode_io_wait_ms"] / 1000:.1f}/{decode_s:.1f} s, '
                      f'{row["decode_read_bytes"] / 1e9 / max(row["decode_io_wait_ms"] / 1000, 1e-9):.2f} GB/s while waiting, '
                      f'prefill {row["prefill_ms"] / 1000:.1f} s, physical {row["physical_read_bytes"] / 1e9:.1f} GB, '
                      f'thermal {row["thermal_before"]}->{row["thermal_after"]}, skin {row["celsius_before"]["SKIN"]} C'
                      + (f', accept {row["speculative"]["acceptance_rate"]:.2f}, {row["speculative"]["tokens_per_step"]:.2f} tok/step'
                         if row["speculative"] else ""), flush=True)
    sequences = {json.dumps(r["tokens"]) for r in rows if "tokens" in r}
    (output / "summary.json").write_text(json.dumps(dict(arms=arms, base_request=base, rows=rows,
                                                         identical_token_sequences=len(sequences) == 1,
                                                         page_cache="uncontrolled; buffered arms can hit page cache"),
                                                    indent=2) + "\n", encoding="utf-8")
    print("identical token sequences:", len(sequences) == 1)


if __name__ == "__main__":
    main()
