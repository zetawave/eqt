"""Exercise the production native import parser with altered GGUF bytes and manifests."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from inspect_gguf import inspect

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "build/host/bin/Release/eqt-steering-check.exe"
VECTOR = ROOT / "models/steering/iq2m-refusal.gguf"
BLOCKS = ROOT / "models/steering/iq2m-refusal-instruction.gguf"


@unittest.skipUnless(BACKEND.is_file() and VECTOR.is_file(), "Build the checker and generate the local IQ2_M vector")
class SteeringImportTests(unittest.TestCase):
    def run_import(self, data, manifest, succeeds=False):
        with tempfile.TemporaryDirectory() as directory:
            vector, sidecar = Path(directory) / "vector.gguf", Path(directory) / "metadata.json"
            vector.write_bytes(data)
            sidecar.write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run([str(BACKEND), "--inspect", str(vector), str(sidecar)], capture_output=True,
                                    text=True, timeout=20)
            self.assertEqual(result.returncode == 0, succeeds, result.stderr)
            return json.loads(result.stdout) if succeeds else result.stderr

    def test_valid_and_reject_modified_or_invalid_content(self):
        data = VECTOR.read_bytes()
        manifest = json.loads(VECTOR.with_suffix(".gguf.json").read_text(encoding="utf-8"))
        info = self.run_import(data, manifest, succeeds=True)
        self.assertEqual(info["model_sha256"], manifest["model_sha256"])
        self.assertEqual(info["layer_index"], 20)
        self.assertGreater(info["l2_norm"], 0)
        self.assertIn("SHA-256", self.run_import(data[:-1] + bytes([data[-1] ^ 1]), manifest))
        with VECTOR.open("rb") as stream:
            offset = inspect(stream, len(data))["data_offset_bytes"]
        for payload in (bytes(manifest["embedding_size"] * 4),
                        struct.pack("<f", float("nan")) + data[offset+4:]):
            altered = data[:offset] + payload
            metadata = dict(manifest, vector_sha256=hashlib.sha256(altered).hexdigest())
            self.run_import(altered, metadata)
        for key, value in (("embedding_size", 7), ("tensor_name", "direction.21"),
                           ("architecture", "another_arch"), ("layer_index", 0), ("model_sha256", "invalid")):
            metadata = copy.deepcopy(manifest)
            metadata[key] = value
            self.run_import(data, metadata)
        malformed = b"NOPE" + data[4:]
        self.run_import(malformed, dict(manifest, vector_sha256=hashlib.sha256(malformed).hexdigest()))

    @unittest.skipUnless(BLOCKS.is_file(), "Generate the local per-block IQ2_M vector")
    def test_per_block_package(self):
        data = BLOCKS.read_bytes()
        manifest = json.loads(BLOCKS.with_suffix(".gguf.json").read_text(encoding="utf-8"))
        info = self.run_import(data, manifest, succeeds=True)
        self.assertEqual((info["layer_count"], info["layer_index"]), (39, 20))
        self.assertAlmostEqual(info["l2_norm"], manifest["block_l2_norms"][19], places=4)
        for key, value in (("layer_count", 38), ("layer_count", 1), ("layer_index", 40)):
            self.run_import(data, dict(manifest, **{key: value}))
        with BLOCKS.open("rb") as stream:
            layout = inspect(stream, len(data))
        start = layout["data_offset_bytes"] + layout["tensors"][5]["offset"]
        width = 4 * manifest["embedding_size"]
        zeroed = data[:start] + bytes(width) + data[start + width:]
        self.assertIn("zero", self.run_import(zeroed, dict(manifest, vector_sha256=hashlib.sha256(zeroed).hexdigest())))


if __name__ == "__main__":
    unittest.main()
