"""Inspect GGUF v3 metadata and tensor storage spans without loading weights."""

import argparse
import collections
import json
from pathlib import Path
import struct


class Reader:
    def __init__(self, stream):
        self.stream = stream

    def read(self, size):
        if size < 0 or size > 32 * 1024 * 1024:
            raise ValueError("Invalid or excessive metadata size")
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("Truncated GGUF header")
        return data

    def number(self, format):
        return struct.unpack("<" + format, self.read(struct.calcsize("<" + format)))[0]

    def string(self):
        return self.read(self.number("Q")).decode("utf-8")

    def value(self, kind, keep=True, depth=0):
        if depth > 2:
            raise ValueError("Excessive array nesting")
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if kind in formats:
            return self.number(formats[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            element, count = self.number("I"), self.number("Q")
            if count > 2_000_000:
                raise ValueError("Excessive metadata array")
            values = [] if keep else None
            for _ in range(count):
                value = self.value(element, keep, depth + 1)
                if keep:
                    values.append(value)
            return values
        raise ValueError(f"Unsupported metadata type {kind}")


def inspect(stream, file_size):
    r = Reader(stream)
    if r.read(4) != b"GGUF" or r.number("I") != 3:
        raise ValueError("Expected GGUF v3")
    tensor_count, metadata_count = r.number("Q"), r.number("Q")
    if tensor_count > 1_000_000 or metadata_count > 100_000:
        raise ValueError("Excessive GGUF entries")
    metadata = {}
    for _ in range(metadata_count):
        key, kind = r.string(), r.number("I")
        keep = not key.startswith("tokenizer.")
        value = r.value(kind, keep)
        if keep:
            metadata[key] = value
    tensors = []
    for _ in range(tensor_count):
        name, rank = r.string(), r.number("I")
        if rank < 1 or rank > 4:
            raise ValueError("Invalid tensor rank")
        shape = [r.number("Q") for _ in range(rank)]
        tensors.append(dict(name=name, shape=shape, type_id=r.number("I"), offset=r.number("Q")))
    alignment = metadata.get("general.alignment", 32)
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("Invalid alignment")
    data_offset = (stream.tell() + alignment - 1) // alignment * alignment
    ordered = sorted(tensors, key=lambda t: t["offset"])
    totals = collections.Counter()
    for index, tensor in enumerate(ordered):
        end = ordered[index + 1]["offset"] if index + 1 < len(ordered) else file_size - data_offset
        span = end - tensor["offset"]
        if span <= 0 or tensor["offset"] % alignment or end > file_size - data_offset:
            raise ValueError("Invalid tensor span")
        tensor["storage_span_bytes"] = span
        group = "routed_experts" if "_exps." in tensor["name"] else "other_weights"
        totals[group] += span
    return dict(format="GGUF v3", file_size_bytes=file_size, data_offset_bytes=data_offset,
                metadata=metadata, tensor_count=tensor_count, storage_spans_bytes=dict(totals),
                note="Spans include alignment padding; routed experts selected by _exps. naming; MTP included if present.", tensors=tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--full-size", type=int, help="Full artifact size when inspecting a downloaded header prefix")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.path)
    with path.open("rb") as stream:
        result = inspect(stream, args.full_size or path.stat().st_size)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
