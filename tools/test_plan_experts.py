import copy
import unittest

from plan_experts import aligned_range, plan


class ExpertPlanTests(unittest.TestCase):
    def fixture(self):
        return dict(metadata={"general.architecture": "qwen35moe", "qwen35moe.block_count": 1,
                              "qwen35moe.expert_count": 3, "qwen35moe.expert_used_count": 2,
                              "qwen35moe.embedding_length": 256, "qwen35moe.expert_feed_forward_length": 256},
                    data_offset_bytes=96, file_size_bytes=96 + 3 * 110592,
                    tensors=[dict(name=f"blk.0.ffn_{name}_exps.weight", shape=[256, 256, 3], type_id=12,
                                  offset=index * 110592, storage_span_bytes=110592)
                             for index, name in enumerate(["gate", "up", "down"])])

    def test_exact_ranges_alignment_and_last_expert(self):
        layout = self.fixture()
        result = plan(layout)
        self.assertEqual(result["all_miss_payload_bytes_per_token"], 221184)
        self.assertEqual(result["all_miss_independent_4k_read_bytes_per_token"], 245760)
        last = result["layer_plans"][0]["banks"][2]
        self.assertEqual(last["first_expert_offset_bytes"] + 3 * last["expert_stride_bytes"], layout["file_size_bytes"])
        self.assertEqual(last["aligned_tail_overrun_bytes"], 4000)
        self.assertEqual(aligned_range(4096, 4096), (4096, 4096))
        self.assertEqual(aligned_range(4095, 2), (0, 8192))

    def test_reject_incomplete_and_incompatible_banks(self):
        layout = self.fixture()
        mutations = [lambda d: d["tensors"].pop(),
                     lambda d: d["tensors"].append(copy.deepcopy(d["tensors"][0])),
                     lambda d: d["tensors"][0].update(type_id=999),
                     lambda d: d["tensors"][0].update(shape=[256, 256, 2]),
                     lambda d: d["tensors"][0].update(storage_span_bytes=110560),
                     lambda d: d.update(file_size_bytes=200000)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(layout)
                mutate(changed)
                with self.assertRaises(ValueError):
                    plan(changed)


if __name__ == "__main__":
    unittest.main()
