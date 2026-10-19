"""Decode-only expert-cache replay over exact routing traces, per-layer policies.

`simulate_cache.py` scores every routed call; for a model whose prompts sweep nearly all experts (Qwen3.8-Flash-
Next: 512 experts, a prompt batch uses most of each layer) only decoding steps decide sustained speed. Here
prompt calls update the cache with the native semantics (the batch's experts are resident during the call, then
the layer is trimmed to its quota) and only single-token calls are counted. Policies run per layer with uniform
quotas, as the native store does:

- lru: the native policy;
- lfu: evict the least frequently used entry (decayed counts, ties by recency);
- pinned: the `pin` most frequent experts per layer in the calibration trace never leave; LRU for the rest;
- belady: evict the entry used farthest in the future (an offline bound, not implementable).

Reports decode hit rates and requested bytes per decode token, not speed: wall time depends on storage and
overlap and is measured on the device.
"""

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path


def load(path):
    calls = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "layer" in record:
            calls.append((record["layer"], record["tokens"], sorted(set(record["experts"]))))
        elif "request" in record:
            calls.append(None)  # request boundary: the native cache stays warm across requests
    return [c for c in calls if c is not None]


def replay(calls, quota, policy, pinned=None, decay=0.98):
    caches, counts, hits, demanded, clock = {}, {}, 0, 0, 0
    future = {}
    if policy == "belady":
        nxt, last = [None] * len(calls), {}
        for i in range(len(calls) - 1, -1, -1):
            layer, _, selected = calls[i]
            nxt[i] = {e: last.get((layer, e), float("inf")) for e in selected}
            for e in selected:
                last[layer, e] = i
    for i, (layer, tokens, selected) in enumerate(calls):
        clock += 1
        cache = caches.setdefault(layer, OrderedDict())
        count = counts.setdefault(layer, {})
        keep = pinned.get(layer, set()) if pinned else set()
        decode = tokens == 1
        for e in selected:
            if decode:
                demanded += 1
                hits += e in cache or e in keep
            if policy == "lfu":
                # Exponentially decayed use count per call, updated lazily: (count, call of last update).
                value, seen = count.get(e, (0.0, clock))
                count[e] = (value * decay ** (clock - seen) + 1, clock)
            if e not in keep:
                cache[e] = clock
                cache.move_to_end(e)
            if policy == "belady":
                future[layer, e] = nxt[i][e]
        room = max(quota - len(keep), 0)
        while len(cache) > room:
            if policy in ("lru", "pinned"):
                victim = next(iter(cache))
            elif policy == "lfu":
                victim = min(cache, key=lambda k: (count[k][0] * decay ** (clock - count[k][1]), cache[k]))
            else:
                victim = max(cache, key=lambda k: future.get((layer, k), float("inf")))
            del cache[victim]
    return hits, demanded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", required=True)
    parser.add_argument("--evaluate", required=True)
    parser.add_argument("--entry-bytes", type=int, required=True, help="Resident bytes of one expert (slot size)")
    parser.add_argument("--expert-bytes", type=int, required=True, help="Bytes read per missing expert")
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--budget-mib", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    calibration, evaluation = load(args.calibrate), load(args.evaluate)
    frequency = {}
    for layer, tokens, selected in calibration:
        if tokens == 1:
            frequency.setdefault(layer, Counter()).update(selected)
    rows = []
    for budget in args.budget_mib:
        quota = budget * 1024 * 1024 // args.layers // args.entry_bytes
        row = dict(budget_mib=budget, quota_per_layer=quota)
        for policy in ("lru", "lfu", "belady"):
            hits, demanded = replay(evaluation, quota, policy)
            row[policy] = hits / demanded
        for share in (0.25, 0.5):
            pin = int(quota * share)
            pinned = {layer: {e for e, _ in counter.most_common(pin)} for layer, counter in frequency.items()}
            hits, demanded = replay(evaluation, quota, "pinned", pinned)
            row[f"pinned_{int(share * 100)}pct"] = hits / demanded
        row["decode_demanded_per_token"] = demanded / sum(1 for c in evaluation if c[1] == 1) * args.layers
        row["lru_miss_bytes_per_token"] = (1 - row["lru"]) * row["decode_demanded_per_token"] * args.expert_bytes
        rows.append(row)
        print(json.dumps(row))
    # Speculative verification of k drafts reads the union of k + 1 consecutive tokens' experts per layer.
    sequences, current = {}, None
    for layer, tokens, selected in evaluation:
        if tokens == 1:
            sequences.setdefault(layer, []).append(set(selected))
    union = {}
    for width in (2, 3, 4):
        sizes = [len(set().union(*seq[i:i + width])) for seq in sequences.values() for i in range(len(seq) - width + 1)]
        union[f"tokens_{width}"] = sum(sizes) / len(sizes)
    rows.append(dict(mean_distinct_experts_per_layer_over_consecutive_decode_tokens=union))
    print(json.dumps(union))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(dict(kind="decode_expert_cache_replay", calibration=args.calibrate,
                                                 evaluation=args.evaluate, entry_bytes=args.entry_bytes,
                                                 expert_bytes=args.expert_bytes, results=rows,
                                                 scope=__doc__.split("\n\n")[0]), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
