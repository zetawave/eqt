"""Boundary tests for the offline GGUF header inspector."""

import io
import struct
import unittest

from inspect_gguf import inspect


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def fixture(offset=0):
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    header += string("blk.0.ffn_up_exps.weight") + struct.pack("<IQQQI Q", 3, 32, 32, 4, 0, offset)
    return header + bytes((-len(header)) % 32) + bytes(32 * 32 * 4 * 4)


class HeaderTests(unittest.TestCase):
    def test_tensor_span(self):
        data = fixture()
        result = inspect(io.BytesIO(data), len(data))
        self.assertEqual(result["tensor_count"], 1)
        self.assertEqual(result["storage_spans_bytes"]["routed_experts"], 16384)

    def test_truncation_and_invalid_span(self):
        data = fixture()
        for size in [0, 7, 23, 55]:
            with self.assertRaises(ValueError):
                inspect(io.BytesIO(data[:size]), len(data))
        with self.assertRaises(ValueError):
            inspect(io.BytesIO(fixture(offset=1)), len(data))
        with self.assertRaises(ValueError):
            inspect(io.BytesIO(fixture(offset=1 << 40)), len(data))


if __name__ == "__main__":
    unittest.main()
