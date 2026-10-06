"""Derive exact Qwen3.5 MoE expert byte ranges from a GGUF header, without reading weights."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from inspect_gguf import inspect

# ggml-common.h and ggml.h at the revision in dependencies.json: 256-value K blocks.
QUANT_BLOCKS = {12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210)}


def aligned_range(offset, size, alignment=4096):
    start = offset // alignment * alignment
    end = (offset + size + alignment - 1) // alignment * alignment
    return start, end - start


def plan(layout):
    metadata = layout["metadata"]
    if metadata.get("general.architecture") != "qwen35moe":
        raise ValueError("Only separate gate/up/down Qwen3.5 MoE tensors are supported")
    keys = ["block_count", "expert_count", "expert_used_count", "embedding_length", "expert_feed_forward_length"]
    counts = [metadata.get("qwen35moe." + key) for key in keys]
    if any(type(value) is not int or value <= 0 for value in counts):
        raise ValueError("Expected positive scalar architecture dimensions")
    layers, experts, selected, width, hidden = counts
    if selected > experts:
        raise ValueError("Selected expert count exceeds bank")
    banks = {}
    payload_total = 0
    aligned_total = 0
    for tensor in layout["tensors"]:
        if "_exps." not in tensor["name"]:
            continue
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor["name"])
        if not match:
            raise ValueError("Unsupported routed tensor layout: " + tensor["name"])
        layer, component = int(match[1]), match[2]
        if layer >= layers or (layer, component) in banks:
            raise ValueError("Unexpected or duplicate expert bank")
        shape = [hidden, width, experts] if component == "down" else [width, hidden, experts]
        if tensor["shape"] != shape or tensor["type_id"] not in QUANT_BLOCKS:
            raise ValueError("Unsupported expert shape or quantization: " + tensor["name"])
        quant, block, block_bytes = QUANT_BLOCKS[tensor["type_id"]]
        if shape[0] % block:
            raise ValueError("Expert row is not quantization-block aligned")
        stride = shape[0] // block * block_bytes * shape[1]
        payload = stride * experts
        padding = tensor["storage_span_bytes"] - payload
        if padding < 0 or padding >= metadata.get("general.alignment", 32):
            raise ValueError("Tensor payload disagrees with its GGUF storage span")
        offset = layout["data_offset_bytes"] + tensor["offset"]
        if offset < layout["data_offset_bytes"] or offset + payload > layout["file_size_bytes"]:
            raise ValueError("Expert bank exceeds file")
        aligned_sizes = [aligned_range(offset + expert * stride, stride)[1] for expert in range(experts)]
        tail_start, tail_size = aligned_range(offset + (experts - 1) * stride, stride)
        # This artifact has equal aligned read lengths for every expert in each bank.
        if min(aligned_sizes) != max(aligned_sizes):
            raise ValueError("Variable aligned lengths require routing-dependent traffic accounting")
        banks[layer, component] = dict(component=component, tensor_name=tensor["name"], quantization=quant,
                                      first_expert_offset_bytes=offset, expert_stride_bytes=stride,
                                      payload_bytes=payload, tensor_padding_bytes=padding,
                                      aligned_tail_overrun_bytes=max(0, tail_start + tail_size - layout["file_size_bytes"]),
                                      independent_aligned_read_bytes=aligned_sizes[0])
        payload_total += payload
        aligned_total += selected * aligned_sizes[0]
    expected = {(layer, component) for layer in range(layers) for component in ["gate", "up", "down"]}
    if banks.keys() != expected:
        raise ValueError("Missing expert banks")
    entries = [dict(layer=layer, bytes_per_complete_expert=sum(banks[layer, c]["expert_stride_bytes"] for c in ["gate", "up", "down"]),
                    banks=[banks[layer, c] for c in ["gate", "up", "down"]]) for layer in range(layers)]
    return dict(schema_version=1, kind="expert_read_plan", layers=layers, experts_per_layer=experts,
                selected_per_layer=selected, routed_payload_bytes=payload_total,
                all_miss_payload_bytes_per_token=sum(e["bytes_per_complete_expert"] for e in entries) * selected,
                all_miss_independent_4k_read_bytes_per_token=aligned_total,
                independent_reads_per_token=layers * selected * 3,
                expert_offset_formula="first_expert_offset_bytes + expert_id * expert_stride_bytes",
                alignment_bytes=4096, layer_plans=entries,
                scope="Header-derived byte ranges only; no weights, routing trace, cache, inference or I/O measured. Independent reads assume no coalescing or reuse; 4 KiB is a planning alignment, not a probed filesystem contract.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("header")
    parser.add_argument("--manifest", default="configs/target-model.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if len(manifest["files"]) != 1:
        parser.error("A single-file manifest is required")
    header = Path(args.header)
    expected = manifest["files"][0].get("header")
    if expected and header.stat().st_size != expected["size_bytes"]:
        raise ValueError("Header size does not match the pinned manifest")
    with header.open("rb") as stream:
        header_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected and header_hash != expected["sha256"]:
        raise ValueError("Header prefix does not match the pinned manifest")
    with header.open("rb") as stream:
        result = plan(inspect(stream, manifest["files"][0]["size_bytes"]))
    result["provenance"] = dict(manifest=manifest, header_sha256=header_hash,
                                header_bytes=header.stat().st_size, full_artifact_hash_verified=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
