import unittest

from sustained_run import joules, summarize, sustained_request


class SustainedRunTests(unittest.TestCase):
    def test_request_preserves_context_sampling_and_eos(self):
        base = dict(load=dict(context=8192), max_tokens=512, temperature=0.6, seed=7, ignore_eos=False)
        request = sustained_request(base, dict(expert_cache_mib=3072), 1024, 900)
        self.assertEqual(request['load']['context'], 8192)
        self.assertEqual(request['sequence'][0]['temperature'], 0.6)
        self.assertFalse(request['sequence'][0]['ignore_eos'])
        self.assertEqual(base['max_tokens'], 512)
        for invalid in [dict(base, ignore_eos=True), dict(base, max_tokens=0), dict(sequence=[])]:
            with self.assertRaises(ValueError):
                sustained_request(invalid, {}, None, 900)

    def test_minute_accounting_includes_eos_tail_and_excludes_prefill(self):
        runs = [dict(generated_tokens=2, decode_ms=70000, prefill_ms=10000,
                     token_latency_ms=[10000, 45000], termination='eos', expert_cache=None),
                dict(generated_tokens=1, decode_ms=60000, prefill_ms=20000,
                     token_latency_ms=[55000], termination='length', expert_cache=None)]
        report = summarize(dict(results=runs, sequence_wall_seconds=180))
        self.assertEqual(report['tokens_per_second_by_minute'], [2 / 60, 0])
        self.assertEqual(report['decode_tokens_per_second'], 3 / 130)
        self.assertEqual(report['end_to_end_tokens_per_second'], 3 / 180)
        self.assertEqual(report['other_wall_ms'], 20000)
        self.assertIsNone(report['expert_io_bytes_per_token'])
        self.assertEqual(report['terminations'], dict(eos=1, length=1))

    def test_missing_energy_and_counter_aggregation(self):
        self.assertIsNone(joules(dict(charge_uah=None, voltage_mv=4000), dict(charge_uah=20, voltage_mv=4000)))
        run = dict(generated_tokens=2, decode_ms=1000, prefill_ms=0, token_latency_ms=[100, 400],
                   termination='eos', expert_cache=dict(decode_io_bytes=4000000, decode_io_busy_ms=500))
        report = summarize(dict(results=[run, run], sequence_wall_seconds=2))
        self.assertEqual(report['expert_io_bytes_per_token'], 2000000)
        self.assertEqual(report['expert_io_busy_gb_s'], 0.008)

    def test_decode_cache_fraction_excludes_speculative_hits(self):
        counters = dict(decode_io_bytes=100, decode_io_busy_ms=10, decode_hits=8,
                        decode_prefetch_hits=3, decode_demanded_experts=10,
                        decode_io_bytes_by_bank=dict(demand=dict(gate=10, up=10, down=20, other=0),
                                                     speculative=dict(gate=15, up=15, down=30, other=0)))
        run = dict(generated_tokens=1, decode_ms=1000, prefill_ms=0, token_latency_ms=[100],
                   termination='eos', expert_cache=counters)
        report = summarize(dict(results=[run, run], sequence_wall_seconds=2))
        self.assertEqual(report['decode_demand_cache_hit_fraction'], 0.5)
        self.assertEqual(report['expert_io_bytes_by_bank']['speculative']['down'], 60)


if __name__ == '__main__':
    unittest.main()
