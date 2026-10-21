"""Decode-only expert-cache replay over exact routing traces, per-layer policies.

`simulate_cache.py` scores every routed call; for a model whose prompts sweep nearly all experts (Qwen3.8-Flash-
Next: 512 experts, a prompt batch uses most of each layer) only decoding steps decide sustained speed. Here
prompt calls update the cache with the native semantics (the batch's experts are resident during the call, then
the layer is trimmed to its quota) and only single-token calls are counted. Policies run per layer with uniform
quotas, as the native store does:

- lru: the native policy;
- lfu: evict the least frequently used entry (decayed counts, ties by recency);
- pinned: the `pin` most frequent experts per layer in the calibration trace never leave; LRU for the rest;
- belady: evict the entry used farthest in the future (an offline bound, not implementable);
- rank: evict the entry the router ranked worst at this layer's latest token (traces with `ranking`);
- rank_p: evict the entry least likely to be selected at this layer's next token, estimated from its latest rank
  and the tokens since that rank on the calibration trace.

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
            ranking = record.get("ranking") if record["tokens"] == 1 else None
            calls.append((record["layer"], record["tokens"], sorted(set(record["experts"])), ranking))
        elif "request" in record:
            calls.append(None)  # request boundary: the native cache stays warm across requests
    return [c for c in calls if c is not None]


UNRANKED, MAX_AGE = 64, 8


def reuse_table(calls):
    """P(selected at a layer's next decode token | rank at its latest ranked token, tokens since), calibration."""
    seen, chosen = Counter(), Counter()
    last = {}  # layer -> {expert: (rank, call index of that layer)}
    steps = Counter()
    for layer, tokens, selected, ranking in calls:
        if tokens != 1 or ranking is None:
            last.pop(layer, None)
            continue
        steps[layer] += 1
        state = last.setdefault(layer, {})
        chosen_now = set(selected)
        for expert, (rank, step) in state.items():
            key = (rank, min(steps[layer] - step, MAX_AGE))
            seen[key] += 1
            chosen[key] += expert in chosen_now
        for rank, expert in enumerate(ranking):
            state[expert] = (rank, steps[layer])
    return {key: chosen[key] / seen[key] for key in seen}


def replay(calls, quota, policy, pinned=None, decay=0.98, table=None):
    caches, counts, hits, demanded, clock = {}, {}, 0, 0, 0
    future = {}
    ranks, steps = {}, Counter()  # latest rank per (layer, expert) with the layer's decode step
    if policy == "belady":
        nxt, last = [None] * len(calls), {}
        for i in range(len(calls) - 1, -1, -1):
            layer, _, selected, _ = calls[i]
            nxt[i] = {e: last.get((layer, e), float("inf")) for e in selected}
            for e in selected:
                last[layer, e] = i
    for i, (layer, tokens, selected, ranking) in enumerate(calls):
        clock += 1
        if ranking is not None:
            steps[layer] += 1
            for rank, e in enumerate(ranking):
                ranks[layer, e] = (rank, steps[layer])
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
            elif policy == "rank":
                victim = max(cache, key=lambda k: (current_rank(ranks, steps, layer, k), -cache[k]))
            elif policy == "rank_p":
                victim = min(cache, key=lambda k: (next_use(table, ranks, steps, layer, k), cache[k]))
            else:
                victim = max(cache, key=lambda k: future.get((layer, k), float("inf")))
            del cache[victim]
    return hits, demanded


def current_rank(ranks, steps, layer, expert):
    """An expert's rank at the layer's latest decode token; beyond the recorded ranks when it was not among them."""
    rank, step = ranks.get((layer, expert), (UNRANKED, -1))
    return rank if step == steps[layer] else UNRANKED


def next_use(table, ranks, steps, layer, expert):
    """Calibrated probability that the layer's next decode token selects the expert, from its latest rank."""
    rank, step = ranks.get((layer, expert), (UNRANKED, -MAX_AGE))
    return table.get((rank, min(steps[layer] - step + 1, MAX_AGE)), 0.0)


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
    ranked = any(c[3] is not None for c in evaluation) and any(c[3] is not None for c in calibration)
    table = reuse_table(calibration) if ranked else None
    frequency = {}
    for layer, tokens, selected, _ in calibration:
        if tokens == 1:
            frequency.setdefault(layer, Counter()).update(selected)
    rows = []
    for budget in args.budget_mib:
        quota = budget * 1024 * 1024 // args.layers // args.entry_bytes
        row = dict(budget_mib=budget, quota_per_layer=quota)
        for policy in ("lru", "lfu", "belady") + (("rank", "rank_p") if ranked else ()):
            hits, demanded = replay(evaluation, quota, policy, table=table)
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
    for layer, tokens, selected, _ in evaluation:
        if tokens == 1:
            sequences.setdefault(layer, []).append(set(selected))
    union = {}
    for width in (2, 3, 4):
        sizes = [len(set().union(*seq[i:i + width])) for seq in sequences.values() for i in range(len(seq) - width + 1)]
        union[f"tokens_{width}"] = sum(sizes) / len(sizes)
    rows.append(dict(mean_distinct_experts_per_layer_over_consecutive_decode_tokens=union))
    print(json.dumps(union))
    if table:
        # P(selected at the next decode token | rank at this one), the reuse the rank policies exploit.
        curve = {rank: round(table.get((rank, 1), 0.0), 3) for rank in (0, 5, 9, 10, 12, 15, 20, 30, 40, 63)}
        rows.append(dict(next_token_selection_by_rank=curve))
        print(json.dumps(curve))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(dict(kind="decode_expert_cache_replay", calibration=args.calibrate,
                                                 evaluation=args.evaluate, entry_bytes=args.entry_bytes,
                                                 expert_bytes=args.expert_bytes, results=rows,
                                                 scope=__doc__.split("\n\n")[0]), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
