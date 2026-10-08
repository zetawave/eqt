"""Build a tiny random-weight qwen35moe GGUF from the pinned target header.

The fixture keeps the target's architecture, tokenizer, chat template and routed-bank quantization
types while shrinking every dimension. Weights are random but valid encodings, so the file exercises
the real hybrid graph, routing and expert streaming paths. Its text output is meaningless.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from inspect_gguf import Reader

F32, F16, Q4_0, Q8_0, Q4_K, Q5_K = 0, 1, 2, 8, 12, 13
ALIGNMENT = 32
SCALAR_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
DROPPED_PREFIXES = ("quantize.", "general.base_model", "general.repo_url", "general.quantized_by")


def read_raw_metadata(stream):
    """Return header counts and (key, type, raw value bytes) without decoding token arrays."""
    reader = Reader(stream)
    if reader.read(4) != b"GGUF" or reader.number("I") != 3:
        raise ValueError("Expected GGUF v3")
    reader.number("Q")
    entries = []
    for _ in range(reader.number("Q")):
        key, kind = reader.string(), reader.number("I")
        start = stream.tell()
        reader.value(kind, keep=False)
        end = stream.tell()
        stream.seek(start)
        entries.append((key, kind, stream.read(end - start)))
    return entries


def encode_string(text):
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def encode_value(kind, value):
    if kind == 8:
        return encode_string(value)
    return struct.pack("<" + SCALAR_FORMATS[kind], value)


def f16_bytes(values):
    return np.asarray(values, dtype=np.float16).tobytes()


def random_q8_0(rng, rows, width, scale):
    blocks = rows * width // 32
    d = np.full((blocks, 1), scale / 64, dtype=np.float16).view(np.uint8)
    q = rng.integers(-127, 128, size=(blocks, 32), dtype=np.int8).view(np.uint8)
    return np.concatenate([d, q], axis=1).tobytes()


def random_q4_0(rng, rows, width, scale):
    blocks = rows * width // 32
    d = np.full((blocks, 1), scale / 8, dtype=np.float16).view(np.uint8)
    q = rng.integers(0, 256, size=(blocks, 16), dtype=np.uint8)
    return np.concatenate([d, q], axis=1).tobytes()


def random_k(rng, rows, width, kind):
    """Random but finite Q4_K/Q5_K super-blocks (ggml-common.h layout: d, dmin, scales[12], [qh], qs)."""
    blocks = rows * width // 256
    d = np.full((blocks, 1), 2e-4 if kind == Q4_K else 1e-4, dtype=np.float16).view(np.uint8)
    dmin = np.full((blocks, 1), 1e-4, dtype=np.float16).view(np.uint8)
    scales = rng.integers(0, 256, size=(blocks, 12), dtype=np.uint8)
    qh = rng.integers(0, 256, size=(blocks, 32), dtype=np.uint8) if kind == Q5_K else np.zeros((blocks, 0), np.uint8)
    qs = rng.integers(0, 256, size=(blocks, 128), dtype=np.uint8)
    return np.concatenate([d.reshape(blocks, 2), dmin.reshape(blocks, 2), scales, qh, qs], axis=1).tobytes()


def build(config, rng):
    e, ff, experts, layers = config["embedding"], config["expert_ff"], config["experts"], config["layers"]
    heads, kv_heads, head_dim, vocab = config["heads"], config["kv_heads"], config["head_dim"], config["vocab"]
    state, groups, dt_rank, inner = config["ssm_state"], config["ssm_groups"], config["ssm_dt_rank"], config["ssm_inner"]
    qkv = 2 * groups * state + inner
    down_type = Q5_K if ff % 256 == 0 else Q8_0
    tensors = []

    def f32(name, shape, values):
        tensors.append((name, shape, F32, np.asarray(values, dtype=np.float32).reshape(-1).tobytes()))

    def norm(name, width):
        f32(name, [width], 1.0 + 0.02 * rng.standard_normal(width))

    def dense(name, width, rows, scale=0.05):
        tensors.append((name, [width, rows], Q8_0, random_q8_0(rng, rows, width, scale)))

    tensors.append(("token_embd.weight", [e, vocab], Q8_0, random_q8_0(rng, vocab, e, 1.0)))
    norm("output_norm.weight", e)
    tensors.append(("output.weight", [e, vocab], Q4_0, random_q4_0(rng, vocab, e, 0.05)))
    # The optional MTP block follows the trunk as one more full-attention MoE layer, as blk.40 does in
    # the target: no own embedding or head, plus nextn projection and norms.
    for layer in range(layers + config["mtp_layers"]):
        p = f"blk.{layer}."
        norm(p + "attn_norm.weight", e)
        norm(p + "post_attention_norm.weight", e)
        if layer < layers and (layer + 1) % config["full_attention_interval"]:
            dense(p + "attn_qkv.weight", e, qkv)
            dense(p + "attn_gate.weight", e, inner)
            f32(p + "ssm_a", [dt_rank], -rng.uniform(0.5, 2.0, dt_rank))
            f32(p + "ssm_alpha.weight", [e, dt_rank], 0.05 * rng.standard_normal(e * dt_rank))
            f32(p + "ssm_beta.weight", [e, dt_rank], 0.05 * rng.standard_normal(e * dt_rank))
            f32(p + "ssm_conv1d.weight", [config["conv_kernel"], qkv], 0.1 * rng.standard_normal(config["conv_kernel"] * qkv))
            f32(p + "ssm_dt.bias", [dt_rank], 0.1 * rng.standard_normal(dt_rank))
            norm(p + "ssm_norm.weight", inner // dt_rank)
            dense(p + "ssm_out.weight", inner, e)
        else:
            dense(p + "attn_q.weight", e, heads * head_dim * 2)
            dense(p + "attn_k.weight", e, kv_heads * head_dim)
            dense(p + "attn_v.weight", e, kv_heads * head_dim)
            dense(p + "attn_output.weight", heads * head_dim, e)
            norm(p + "attn_q_norm.weight", head_dim)
            norm(p + "attn_k_norm.weight", head_dim)
        f32(p + "ffn_gate_inp.weight", [e, experts], 0.2 * rng.standard_normal(e * experts))
        f32(p + "ffn_gate_inp_shexp.weight", [e], 0.05 * rng.standard_normal(e))
        for bank in ("gate", "up"):
            tensors.append((p + f"ffn_{bank}_exps.weight", [e, ff, experts], Q4_K, random_k(rng, ff * experts, e, Q4_K)))
        down = random_k(rng, e * experts, ff, Q5_K) if down_type == Q5_K else random_q8_0(rng, e * experts, ff, 0.05)
        tensors.append((p + "ffn_down_exps.weight", [ff, e, experts], down_type, down))
        dense(p + "ffn_gate_shexp.weight", e, config["shared_ff"])
        dense(p + "ffn_up_shexp.weight", e, config["shared_ff"])
        dense(p + "ffn_down_shexp.weight", config["shared_ff"], e)
        if layer >= layers:
            dense(p + "nextn.eh_proj.weight", 2 * e, e)
            norm(p + "nextn.enorm.weight", e)
            norm(p + "nextn.hnorm.weight", e)
            norm(p + "nextn.shared_head_norm.weight", e)
    return tensors


def write(path, metadata, tensors):
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, kind, raw in metadata:
        header += encode_string(key) + struct.pack("<I", kind) + raw
    offset = 0
    for name, shape, kind, data in tensors:
        header += encode_string(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
        header += struct.pack("<IQ", kind, offset)
        offset += (len(data) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    header += b"\0" * (-len(header) % ALIGNMENT)
    with path.open("wb") as stream:
        stream.write(header)
        for _, _, _, data in tensors:
            stream.write(data + b"\0" * (-len(data) % ALIGNMENT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", default=".cache/research/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf.header.bin")
    parser.add_argument("--manifest", default="configs/target-model.json")
    parser.add_argument("--output", default="models/eqt-tiny-qwen35moe.gguf")
    parser.add_argument("--expert-ff", type=int, default=256, help="Use a non-multiple of 256 to test unaligned Q8_0 down banks")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--mtp", action="store_true", help="Append one random MTP (nextn) block like the target's blk.40")
    args = parser.parse_args()
    header = Path(args.header)
    expected = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["files"][0]["header"]
    digest = hashlib.sha256(header.read_bytes()).hexdigest()
    if digest != expected["sha256"]:
        raise ValueError("Target header does not match the pinned manifest")
    if args.expert_ff % 32:
        parser.error("--expert-ff must be a multiple of 32")
    with header.open("rb") as stream:
        source = read_raw_metadata(stream)
    config = dict(embedding=256, expert_ff=args.expert_ff, shared_ff=256, experts=16, experts_used=4, layers=8,
                  heads=4, kv_heads=2, head_dim=128, ssm_state=32, ssm_groups=2, ssm_dt_rank=4, ssm_inner=128,
                  conv_kernel=4, full_attention_interval=4, mtp_layers=1 if args.mtp else 0)
    vocab = None
    overrides = {"block_count": config["layers"] + config["mtp_layers"], "embedding_length": config["embedding"],
                 "attention.head_count": config["heads"], "attention.head_count_kv": config["kv_heads"],
                 "attention.key_length": config["head_dim"], "attention.value_length": config["head_dim"],
                 "expert_count": config["experts"], "expert_used_count": config["experts_used"],
                 "expert_feed_forward_length": config["expert_ff"],
                 "expert_shared_feed_forward_length": config["shared_ff"], "context_length": 4096,
                 "ssm.state_size": config["ssm_state"], "ssm.group_count": config["ssm_groups"],
                 "ssm.time_step_rank": config["ssm_dt_rank"], "ssm.inner_size": config["ssm_inner"],
                 "ssm.conv_kernel": config["conv_kernel"],
                 "full_attention_interval": config["full_attention_interval"]}
    metadata = []
    for key, kind, raw in source:
        if key.startswith(DROPPED_PREFIXES):
            continue
        short = key.removeprefix("qwen35moe.")
        if short in overrides:
            raw = encode_value(kind, overrides.pop(short))
        elif key in ("general.name", "general.basename"):
            raw = encode_string("eqt-tiny-random-qwen35moe")
        elif key == "tokenizer.ggml.tokens":
            vocab = struct.unpack_from("<Q", raw, 4)[0]
        metadata.append((key, kind, raw))
    if config["mtp_layers"]:
        metadata.append(("qwen35moe.nextn_predict_layers", 4, struct.pack("<I", config["mtp_layers"])))
    if overrides or not vocab:
        raise ValueError(f"Target header lacks expected keys: {sorted(overrides)}")
    config["vocab"] = vocab
    tensors = build(config, np.random.default_rng(args.seed))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, metadata, tensors)
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps(dict(output=str(output), bytes=output.stat().st_size, sha256=digest, seed=args.seed,
                          header_sha256=expected["sha256"], config=config), indent=2))


if __name__ == "__main__":
    main()
