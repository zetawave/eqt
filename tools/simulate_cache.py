"""Replay exact routing traces against bounded expert-cache policies.

Traces come from the native `expert_trace` option: one JSON line per routed layer evaluation. Every
policy keeps exact routing; only residency differs. Layer-partitioned quotas can be optimized on a
calibration trace from per-layer LRU hit curves (Mattson stack distances) and must be evaluated on a
separate trace. Replay predicts hit rates and requested bytes, not OS reclaim, I/O latency or speed.
"""

import argparse
import heapq
import json
from pathlib import Path
import random
from collections import OrderedDict


def load_trace(path):
    calls = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "layer" in record:
            calls.append((record["layer"], record["tokens"], sorted(set(record["experts"]))))
    if not calls:
        raise ValueError(f"No routed calls in {path}")
    return calls


def decode_tokens(calls):
    first = min(layer for layer, _, _ in calls)
    return sum(tokens for layer, tokens, _ in calls if layer == first)


def layer_hit_curves(calls, experts):
    """hits[layer][q] = accesses that hit a per-layer LRU of q entries with native call semantics.

    Native residency is checked against the cache at call start and trimmed to the quota after the
    call, so an access hits exactly when its recency rank at call start is below q.
    """
    stacks, counts = {}, {}
    for layer, _, selected in calls:
        stack = stacks.setdefault(layer, [])
        histogram = counts.setdefault(layer, [0] * (experts + 2))
        for expert in selected:
            if expert in stack:
                histogram[stack.index(expert) + 1] += 1
        for expert in selected:
            if expert in stack:
                stack.remove(expert)
            stack.insert(0, expert)  # selected is sorted, so the largest id ends most recent, as in replay
    result = {}
    for layer, histogram in counts.items():
        cumulative, total = [], 0
        for q in range(experts + 1):
            total += histogram[q]
            cumulative.append(total)
        result[layer] = cumulative
    return result


def optimize_quotas(curves, entry_bytes, budget, minimum):
    """Greedy marginal-hits-per-byte slot allocation; exact for concave hit curves."""
    quotas = {layer: minimum for layer in curves}
    spent = sum(minimum * entry_bytes[layer] for layer in curves)
    if spent > budget:
        raise ValueError("Budget cannot hold the minimum quota in every layer")
    heap = []

    def push(layer):
        q = quotas[layer]
        if q + 1 < len(curves[layer]):
            gain = (curves[layer][q + 1] - curves[layer][q]) / entry_bytes[layer]
            heapq.heappush(heap, (-gain, layer, q))

    for layer in curves:
        push(layer)
    while heap:
        _, layer, q = heapq.heappop(heap)
        if quotas[layer] != q or spent + entry_bytes[layer] > budget:
            continue
        quotas[layer] += 1
        spent += entry_bytes[layer]
        push(layer)
    return quotas


class Replay:
    def __init__(self, entry_bytes):
        self.entry_bytes = entry_bytes
        self.hits = self.misses = self.read_bytes = 0

    def account(self, layer, hit):
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            self.read_bytes += self.entry_bytes[layer]


def replay_layer_lru(calls, entry_bytes, quotas):
    replay, caches = Replay(entry_bytes), {}
    for layer, _, selected in calls:
        cache = caches.setdefault(layer, OrderedDict())
        for expert in selected:
            replay.account(layer, expert in cache)
            cache[expert] = True
            cache.move_to_end(expert)
        # Overflow beyond the quota lasts only for this call; native trims before the next layer runs.
        while len(cache) > quotas.get(layer, 0):
            cache.popitem(last=False)
    return replay


def replay_global(calls, entry_bytes, budget, policy, seed=0):
    """One shared cache. As in the native store, a call may transiently exceed the budget with its own
    demand set; the cache is trimmed back to the budget once the call has completed."""
    replay, cache, used = Replay(entry_bytes), OrderedDict(), 0
    rng = random.Random(seed)
    if policy == "belady":
        upcoming, last = [None] * len(calls), {}
        for index in range(len(calls) - 1, -1, -1):
            layer, _, selected = calls[index]
            upcoming[index] = {e: last.get((layer, e), float("inf")) for e in selected}
            for e in selected:
                last[layer, e] = index
        next_use, heap = {}, []
    for index, (layer, _, selected) in enumerate(calls):
        for expert in selected:
            key = (layer, expert)
            replay.account(layer, key in cache)
            if key not in cache:
                used += entry_bytes[layer]
            cache[key] = True
            cache.move_to_end(key)
            if policy == "belady":
                next_use[key] = upcoming[index][expert]
                heapq.heappush(heap, (-next_use[key], key))
        while used > budget and cache:
            if policy == "lru":
                victim = next(iter(cache))
            elif policy == "random":
                victim = rng.choice(list(cache))
            else:
                distance, victim = heapq.heappop(heap)
                if victim not in cache or next_use.get(victim) != -distance:
                    continue
            used -= entry_bytes[victim[0]]
            del cache[victim]
    return replay


def summarize(name, replay, tokens):
    total = replay.hits + replay.misses
    return dict(policy=name, hits=replay.hits, misses=replay.misses, hit_rate=replay.hits / total if total else None,
                read_bytes=replay.read_bytes, read_bytes_per_token=replay.read_bytes / tokens if tokens else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", required=True, help="Trace used only to derive optimized quotas")
    parser.add_argument("--evaluate", required=True, help="Held-out trace for reported results")
    parser.add_argument("--budget-mib", type=float, nargs="+", required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--experts-used", type=int, required=True)
    parser.add_argument("--plan", help="expert_read_plan JSON giving bytes_per_complete_expert per layer")
    parser.add_argument("--entry-bytes", type=int, help="Uniform entry size when no plan is available")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    calibration, evaluation = load_trace(args.calibrate), load_trace(args.evaluate)
    if Path(args.calibrate).resolve() == Path(args.evaluate).resolve():
        parser.error("Calibration and evaluation traces must differ")
    layers = sorted({layer for layer, _, _ in calibration + evaluation})
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        entry_bytes = {p["layer"]: p["bytes_per_complete_expert"] for p in plan["layer_plans"]}
    elif args.entry_bytes:
        entry_bytes = {layer: args.entry_bytes for layer in layers}
    else:
        parser.error("--plan or --entry-bytes is required")
    curves = layer_hit_curves(calibration, args.experts)
    tokens = decode_tokens(evaluation)
    results = []
    for budget_mib in args.budget_mib:
        budget = int(budget_mib * 1024 * 1024)
        uniform = {layer: min(args.experts, int(budget / len(layers) / entry_bytes[layer])) for layer in layers}
        optimized = optimize_quotas(curves, entry_bytes, budget, minimum=0)
        rows = [summarize("global_lru", replay_global(evaluation, entry_bytes, budget, "lru"), tokens),
                summarize("global_random", replay_global(evaluation, entry_bytes, budget, "random"), tokens),
                summarize("layer_lru_uniform", replay_layer_lru(evaluation, entry_bytes, uniform), tokens),
                summarize("layer_lru_optimized", replay_layer_lru(evaluation, entry_bytes, optimized), tokens),
                summarize("global_belady_bound", replay_global(evaluation, entry_bytes, budget, "belady"), tokens)]
        weights = [optimized.get(layer, 0) * entry_bytes.get(layer, 0) for layer in range(max(layers) + 1)]
        results.append(dict(budget_mib=budget_mib, policies=rows, optimized_quotas=optimized,
                            native_expert_layer_weights=weights))
        print(budget_mib, " ".join(f'{r["policy"]}={r["hit_rate"]:.3f}' for r in rows))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(schema_version=1, kind="expert_cache_replay", calibration=args.calibrate,
                                      evaluation=args.evaluate, decode_tokens=tokens, results=results,
                                      scope="Exact-routing replay; hit rates and requested bytes only. Belady is a "
                                            "demand-only offline bound, not an implementable policy."), indent=2) + "\n",
                      encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
