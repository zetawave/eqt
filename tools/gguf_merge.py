"""Plan and verify llama.cpp's gguf-split merge byte for byte, from the shard headers alone.

gguf_merge writes the first shard's metadata (split.count removed and re-added last with value 0, since
gguf_set_val replaces a key by appending it), then every shard's tensor infos in shard order with offsets
recomputed at the default 32-byte alignment, zero padding to that alignment, and finally each tensor's bytes
padded to 32, in the same order. Each shard stores its tensors the same way, so the merged data is the
shards' data sections concatenated. A downloader can therefore write the merged header first and append the
shards' data sections as they stream, with no second copy; this tool is the reference for that layout.

`plan` fetches only the shard headers (ranged requests) and checks the merged header against the pinned
manifest and, if given, a local merged file. `--verify-data` also hashes each shard as header + its range of
the local merged file and compares the result with the shard's pinned SHA-256 (reads the whole file).
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

ALIGNMENT = 32
HEADER_PROBE = 1 << 20
# ggml type -> (bytes per block, elements per block), from ggml.c type traits.
TYPE_SIZES = {0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32), 6: (22, 32), 7: (24, 32), 8: (34, 32), 9: (36, 32),
              10: (84, 256), 11: (110, 256), 12: (144, 256), 13: (176, 256), 14: (210, 256), 15: (292, 256),
              16: (66, 256), 17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (18, 32), 21: (110, 256),
              22: (82, 256), 23: (136, 256), 24: (1, 1), 25: (2, 1), 26: (4, 1), 27: (8, 1), 28: (8, 1),
              29: (56, 256), 30: (2, 1)}
SCALARS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Truncated(Exception):
    pass


class Cursor:
    def __init__(self, data):
        self.data, self.at = data, 0

    def take(self, size):
        if self.at + size > len(self.data):
            raise Truncated()
        chunk = self.data[self.at:self.at + size]
        self.at += size
        return chunk

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self):
        return self.take(self.number("Q")).decode("utf-8")

    def skip_value(self, kind, depth=0):
        if kind in SCALARS:
            self.take(SCALARS[kind])
        elif kind == 8:
            self.take(self.number("Q"))
        elif kind == 9 and depth == 0:
            element, count = self.number("I"), self.number("Q")
            if element in SCALARS:
                self.take(SCALARS[element] * count)
            else:
                for _ in range(count):
                    self.skip_value(element, depth + 1)
        else:
            raise ValueError(f"Unsupported metadata type {kind}")


def parse_header(data):
    """Raw KV entries, tensor infos and the data offset of a GGUF v3 header; raises Truncated if incomplete."""
    c = Cursor(data)
    if c.take(4) != b"GGUF" or c.number("I") != 3:
        raise ValueError("Expected GGUF v3")
    n_tensors, n_kv = c.number("Q"), c.number("Q")
    entries, alignment = [], ALIGNMENT
    for _ in range(n_kv):
        start = c.at
        key, kind = c.string(), c.number("I")
        value_start = c.at
        c.skip_value(kind)
        if key == "general.alignment":
            alignment = struct.unpack_from("<I", data, value_start)[0]
        entries.append((key, data[start:c.at]))
    tensors = []
    for _ in range(n_tensors):
        start = c.at
        name, dims = c.string(), c.number("I")
        shape = [c.number("Q") for _ in range(dims)]
        kind, offset = c.number("I"), c.number("Q")
        tensors.append(dict(name=name, shape=shape, type_id=kind, offset=offset, info=data[start:c.at - 8]))
    if alignment != ALIGNMENT:
        raise ValueError("gguf_merge pads tensors to 32 bytes; other alignments are not reproduced")
    data_offset = (c.at + alignment - 1) // alignment * alignment
    if data_offset > len(data):
        raise Truncated()
    return dict(entries=entries, tensors=tensors, data_offset=data_offset, header=data[:data_offset])


def tensor_bytes(tensor):
    size, block = TYPE_SIZES[tensor["type_id"]]
    elements = 1
    for dim in tensor["shape"]:
        elements *= dim
    if tensor["shape"][0] % block:
        raise ValueError(f"Row of {tensor['name']} is not a whole number of blocks")
    return elements // block * size


def padded(size):
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def merged_header(shards):
    """The merged file's header bytes and the data size each shard contributes."""
    first = shards[0]["entries"]
    if not any(key == "split.count" for key, _ in first):
        raise ValueError("The first shard has no split.count")
    entries = [raw for key, raw in first if key != "split.count"]
    entries.append(struct.pack("<Q", 11) + b"split.count" + struct.pack("<IH", 2, 0))
    tensors = [t for shard in shards for t in shard["tensors"]]
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(entries)))
    for raw in entries:
        out += raw
    offset, contributions = 0, []
    for shard in shards:
        start = offset
        for tensor in shard["tensors"]:
            out += tensor["info"] + struct.pack("<Q", offset)
            offset += padded(tensor_bytes(tensor))
        contributions.append(offset - start)
    out += b"\0" * (-len(out) % ALIGNMENT)
    return bytes(out), contributions


def check_shard_layout(shard, size):
    """A shard's data section must hold exactly its padded tensors in info order."""
    expected = 0
    for tensor in shard["tensors"]:
        if tensor["offset"] != expected:
            raise ValueError(f"Shard tensor {tensor['name']} is not stored in info order")
        expected += padded(tensor_bytes(tensor))
    if shard["data_offset"] + expected != size:
        raise ValueError("Shard size differs from its header plus padded tensors")
    return expected


def fetch_header(url, size):
    probe = min(size, HEADER_PROBE)
    while True:
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{probe - 1}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes 0-{probe - 1}/{size}":
                raise ValueError("Server did not honor the header range")
            data = response.read()
        if len(data) != probe:
            raise IOError("Header range ended early")
        try:
            return parse_header(data)
        except Truncated:
            if probe == size:
                raise ValueError("Shard header extends past the shard")
            probe = min(size, probe * 4)


def plan(manifest_path, cache, local=None, verify_data=False):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entry = manifest["files"][0]
    parts = entry["merged_from"]
    base = f'https://huggingface.co/{manifest["repository"]}/resolve/{manifest["revision"]}/'
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    shards = []
    for part in parts:
        stored = cache / (Path(part["name"]).name + ".header.bin")
        if stored.exists():
            shard = parse_header(stored.read_bytes())
        else:
            shard = fetch_header(base + part["name"], part["size_bytes"])
            stored.write_bytes(shard["header"])
        check_shard_layout(shard, part["size_bytes"])
        shards.append(shard)
    header, contributions = merged_header(shards)
    report = dict(merged_header_bytes=len(header), merged_header_sha256=hashlib.sha256(header).hexdigest(),
                  shard_headers=[dict(name=p["name"], header_bytes=s["data_offset"], tensors=len(s["tensors"]),
                                      data_bytes=c) for p, s, c in zip(parts, shards, contributions)],
                  merged_size_bytes=len(header) + sum(contributions))
    report["matches_manifest"] = (report["merged_header_sha256"] == entry["header"]["sha256"] and
                                  len(header) == entry["header"]["size_bytes"] and
                                  report["merged_size_bytes"] == entry["size_bytes"])
    if local:
        with Path(local).open("rb") as stream:
            report["matches_local_header"] = stream.read(len(header)) == header
            if verify_data:
                verified = []
                for part, shard, size in zip(parts, shards, contributions):
                    digest = hashlib.sha256(shard["header"])
                    left = size
                    while left:
                        block = stream.read(min(left, 16 << 20))
                        if not block:
                            raise IOError("Local merged file is shorter than planned")
                        digest.update(block)
                        left -= len(block)
                    verified.append(digest.hexdigest() == part["sha256"])
                report["shards_rebuilt_from_local_match_pinned_sha256"] = verified
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--cache", default=".cache/research/shard-headers")
    parser.add_argument("--local", help="Local merged file to compare against")
    parser.add_argument("--verify-data", action="store_true")
    args = parser.parse_args()
    print(json.dumps(plan(args.manifest, args.cache, args.local, args.verify_data), indent=2))
