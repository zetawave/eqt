"""Collect exact expert-routing traces by running the streamed engine over a versioned workload.

Each prompt runs in its own process with `expert_trace` enabled; traces are concatenated with request
markers. Use different workloads for calibration and evaluation of cache policies.
"""

import argparse
import json
from pathlib import Path
import tempfile

from device import adb
from runner import REMOTE, ROOT, Runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["host", "linux", "android"], default="android")
    parser.add_argument("--serial")
    parser.add_argument("--model", required=True, help="Local path (host/linux) or staged device path (android)")
    parser.add_argument("--request", default="configs/target-stream-request.json", help="Base request with load options")
    parser.add_argument("--workload", required=True, help="JSONL with id and prompt fields")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", required=True, help="Directory for trace.jsonl and summary.json")
    parser.add_argument("--load", default="{}", help="JSON object merged into the request load options")
    args = parser.parse_args()
    runner = Runner(args.target, args.serial)
    runner.stage(["eqt-bench"])
    base = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    combined, summary = [], []
    with tempfile.TemporaryDirectory(prefix="eqt-trace-", dir=ROOT / ".cache") as workdir:
        for line in Path(args.workload).read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            request = json.loads(json.dumps(base))
            request.update(max_tokens=args.max_tokens, temperature=0, ignore_eos=False)
            turns = item.get("messages") or [{"role": "user", "content": item["prompt"]}]
            request["messages"] = [request["messages"][0]] + turns
            trace = (f"{REMOTE}/trace.jsonl" if args.target == "android" else str(Path(workdir) / "trace.jsonl"))
            request["load"].update(json.loads(args.load))
            request["load"].update(expert_streaming=True, expert_trace=trace)
            run, result = runner.run("eqt-bench", args.model, request, workdir)
            local_trace = Path(workdir) / "trace.jsonl"
            if args.target == "android":
                adb("pull", trace, str(local_trace), serial=args.serial)
            combined.append(json.dumps({"request": item["id"]}))
            combined.extend(l for l in local_trace.read_text(encoding="utf-8").splitlines() if '"layer"' in l)
            stats = result["expert_cache"]
            summary.append(dict(id=item["id"], text=result["text"], token_ids=result["token_ids"], prompt_tokens=result["prompt_tokens"],
                                generated_tokens=result["generated_tokens"], termination=result["termination"],
                                decode_tokens_per_second=result["decode_tokens_per_second"],
                                prefill_tokens_per_second=result["prefill_tokens_per_second"],
                                ttft_ms=result["ttft_ms"], expert_cache=stats,
                                memory_after=result["memory_after"]))
            print(f'{item["id"]}: {result["generated_tokens"]} tokens, {result["decode_tokens_per_second"]:.2f} tok/s, '
                  f'hits {stats["hits"]} misses {stats["misses"]}', flush=True)
    (output / "trace.jsonl").write_text("\n".join(combined) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(dict(workload=args.workload, request=base, runs=summary,
                                                         configuration=result["configuration"]), indent=2) + "\n",
                                         encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
