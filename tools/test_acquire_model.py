"""Ensure header acquisition cannot silently turn into a complete weight download."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from acquire_model import acquire


class Response(io.BytesIO):
    def __init__(self, data, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": "bytes 0-15/1024"}
        self.read_calls = 0

    def read(self, size=-1):
        self.read_calls += 1
        return super().read(size)


class HeaderAcquisitionTests(unittest.TestCase):
    def run_case(self, response, succeeds=False):
        data = b"GGUF" + bytes(12)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(repository="test/model", revision="pinned", files=[
                dict(name="model.gguf", size_bytes=1024, sha256="unread-full-artifact",
                     header=dict(size_bytes=16, sha256=hashlib.sha256(data).hexdigest()))])))
            with patch("acquire_model.urllib.request.urlopen", return_value=response) as open_url:
                if succeeds:
                    acquire(manifest, root, header_only=True)
                else:
                    with self.assertRaises(ValueError):
                        acquire(manifest, root, header_only=True)
                self.assertEqual(open_url.call_args.args[0].get_header("Range"), "bytes=0-15")
            self.assertEqual((root / "model.gguf.header.bin").exists(), succeeds)
            self.assertFalse((root / "model.gguf").exists())
            self.assertEqual(list(root.glob("*.part")), [])

    def test_verified_header(self):
        self.run_case(Response(b"GGUF" + bytes(12)), succeeds=True)

    def test_refuse_ignored_range_before_reading_body(self):
        response = Response(b"unused", status=200)
        self.run_case(response)
        self.assertEqual(response.read_calls, 0)

    def test_reject_truncation_and_hash_mismatch(self):
        for data in [b"GGUF", b"GGUF" + bytes([1]) * 12]:
            with self.subTest(data=data):
                self.run_case(Response(data))


if __name__ == "__main__":
    unittest.main()
