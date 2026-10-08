import json
import random
import tempfile
import unittest
from pathlib import Path

from simulate_cache import (layer_hit_curves, load_trace, optimize_quotas, replay_global, replay_layer_lru)


def synthetic(layers, experts, used, tokens, reuse, seed=1):
    """Each layer keeps `reuse` of its previous selection per token; the rest is uniform random."""
    rng, previous, calls = random.Random(seed), {}, []
    for _ in range(tokens):
        for layer in range(layers):
            kept = rng.sample(previous.get(layer, []), min(reuse, len(previous.get(layer, []))))
            fresh = [e for e in rng.sample(range(experts), experts) if e not in kept][:used - len(kept)]
            previous[layer] = kept + fresh
            calls.append((layer, 1, sorted(previous[layer])))
    return calls


class SimulateCacheTest(unittest.TestCase):
    def test_trace_parsing_skips_request_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text('{"request":1}\n{"layer":0,"tokens":2,"experts":[3,1,3,2]}\n', encoding="utf-8")
            self.assertEqual(load_trace(path), [(0, 2, [1, 2, 3])])

    def test_cyclic_access_defeats_global_lru_but_not_layer_partitioning(self):
        # Every token touches 4 layers x 2 experts with perfect per-layer reuse. A global LRU holding
        # 6 of the 8 entries evicts each layer's pair just before its next use (the capacity cliff);
        # the same 6 slots partitioned per layer keep three layers fully resident.
        calls = [(layer, 1, [0, 1]) for _ in range(50) for layer in range(4)]
        sizes = {layer: 1 for layer in range(4)}
        global_lru = replay_global(calls, sizes, 6, "lru")
        layered = replay_layer_lru(calls, sizes, {0: 2, 1: 2, 2: 2, 3: 0})
        self.assertEqual(global_lru.hits, 0)
        self.assertEqual(layered.hits, 3 * 2 * 49)

    def test_belady_bounds_lru(self):
        calls = synthetic(layers=6, experts=32, used=4, tokens=200, reuse=2)
        sizes = {layer: 10 for layer in range(6)}
        for budget in (80, 240, 600):
            belady = replay_global(calls, sizes, budget, "belady")
            lru = replay_global(calls, sizes, budget, "lru")
            self.assertLessEqual(belady.misses, lru.misses)
            self.assertEqual(belady.hits + belady.misses, lru.hits + lru.misses)

    def test_full_quota_has_only_compulsory_misses(self):
        calls = synthetic(layers=3, experts=16, used=4, tokens=100, reuse=1)
        replay = replay_layer_lru(calls, {0: 1, 1: 1, 2: 1}, {0: 16, 1: 16, 2: 16})
        distinct = len({(layer, e) for layer, _, selected in calls for e in selected})
        self.assertEqual(replay.misses, distinct)

    def test_hit_curve_matches_replay_and_allocation_respects_budget(self):
        calls = synthetic(layers=4, experts=32, used=4, tokens=150, reuse=3)
        curves = layer_hit_curves(calls, 32)
        for q in (4, 9, 32):
            replay = replay_layer_lru(calls, {l: 1 for l in range(4)}, {l: q for l in range(4)})
            self.assertEqual(replay.hits, sum(curves[l][q] for l in range(4)))
        sizes = {0: 3, 1: 5, 2: 3, 3: 5}
        quotas = optimize_quotas(curves, sizes, budget=100, minimum=4)
        self.assertLessEqual(sum(quotas[l] * sizes[l] for l in quotas), 100)
        self.assertTrue(all(q >= 4 for q in quotas.values()))

    def test_optimized_quotas_favor_layers_with_reuse(self):
        stable = [(0, 1, [0, 1, 2, 3]) for _ in range(100)]
        rng = random.Random(3)
        noisy = [(1, 1, sorted(rng.sample(range(64), 4))) for _ in range(100)]
        calls = [c for pair in zip(stable, noisy) for c in pair]
        quotas = optimize_quotas(layer_hit_curves(calls, 64), {0: 1, 1: 1}, budget=8, minimum=0)
        self.assertGreaterEqual(quotas[0], 4)


if __name__ == "__main__":
    unittest.main()
