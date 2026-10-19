"""Build a tiny random-weight qwen4exp GGUF from the pinned Qwen3.8-Flash-Next header.

The fixture keeps the target's architecture, tokenizer, chat template and stored quantization types (IQ2_XS
gate/up and IQ4_NL down expert banks, an IQ4_NL n-gram table padded like the converter's) while shrinking
every dimension. It has hyper-connection mixers, Gated DeltaNet layers, sparse-attention layers whose budget
is smaller than a short prompt, and one PLE layer. Weights are random but valid encodings, so the file
exercises the real graph, routing, lookahead and expert streaming paths. Its text output is meaningless.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from make_moe_fixture import (F32, Q4_0, Q5_K, Q8_0, encode_string, encode_value, random_k, random_q4_0,
                              random_q8_0, read_raw_metadata, write)

IQ2_XS, IQ4_NL, BF16 = 17, 20, 30
ARRAY, UINT32, UINT64 = 9, 4, 10
DROPPED_PREFIXES = ("quantize.", "general.base_model", "general.repo_url", "general.quantized_by", "split.")


def random_iq2_xs(rng, rows, width):
    """block_iq2_xs: FP16 d, 32 uint16 grid indices with sign bits, 8 bytes of 4-bit scales."""
    blocks = rows * width // 256
    d = np.full((blocks, 1), 2e-3, dtype=np.float16).view(np.uint8).reshape(blocks, 2)
    qs = rng.integers(0, 256, size=(blocks, 64), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(blocks, 8), dtype=np.uint8)
    return np.concatenate([d, qs, scales], axis=1).tobytes()


def random_iq4_nl(rng, rows, width, scale):
    """block_iq4_nl: FP16 d and 32 4-bit indices into the non-linear table."""
    blocks = rows * width // 32
    d = np.full((blocks, 1), scale / 127, dtype=np.float16).view(np.uint8).reshape(blocks, 2)
    qs = rng.integers(0, 256, size=(blocks, 16), dtype=np.uint8)
    return np.concatenate([d, qs], axis=1).tobytes()


def bf16_bytes(values):
    bits = np.asarray(values, dtype=np.float32).reshape(-1).view(np.uint32)
    return (bits >> 16).astype(np.uint16).tobytes()


def encode_array(element_kind, values):
    fmt = {UINT32: "I", UINT64: "Q"}[element_kind]
    return struct.pack("<IQ", element_kind, len(values)) + struct.pack(f"<{len(values)}{fmt}", *values)


def build(config, rng):
    e, ff, experts, layers, hc = config["embedding"], config["expert_ff"], config["experts"], config["layers"], config["hc"]
    heads, kv_heads, head_dim, vocab = config["heads"], config["kv_heads"], config["head_dim"], config["vocab"]
    state, groups, dt_rank = config["ssm_state"], config["ssm_groups"], config["ssm_dt_rank"]
    rank, idx_heads, idx_dim = config["hc_rank"], config["indexer_heads"], config["indexer_dim"]
    value_dim = state * dt_rank
    qkv = 2 * groups * state + value_dim
    hc_dim = hc * e
    tensors = []

    def f32(name, shape, values):
        tensors.append((name, shape, F32, np.asarray(values, dtype=np.float32).reshape(-1).tobytes()))

    def norm(name, width):
        f32(name, [width], 1.0 + 0.02 * rng.standard_normal(width))

    def q8(name, width, rows, scale=0.05):
        tensors.append((name, [width, rows], Q8_0, random_q8_0(rng, rows, width, scale)))

    def q5k(name, width, rows):
        tensors.append((name, [width, rows], Q5_K, random_k(rng, rows, width, Q5_K)))

    def mixer(prefix):
        # Stored as one [n_embd * streams] vector; the model views it as [n_embd, streams].
        norm(prefix + "_norm.weight", hc_dim)
        q8(prefix + "_down.weight", hc_dim, rank, 0.02)
        q8(prefix + "_up.weight", rank, hc_dim, 0.2)

    tensors.append(("token_embd.weight", [e, vocab], Q8_0, random_q8_0(rng, vocab, e, 1.0)))
    tensors.append(("output.weight", [e, vocab], Q4_0, random_q4_0(rng, vocab, e, 0.05)))
    mixer("output_hc")
    table_rows = config["ple_rows"]
    tensors.append(("per_layer_token_embd.weight", [config["ple_dim"], table_rows], IQ4_NL,
                    random_iq4_nl(rng, table_rows, config["ple_dim"], 0.5)))
    for layer in range(layers):
        p = f"blk.{layer}."
        for module in ("attn", "ffn"):
            mixer(p + "hc_" + module)
            f32(p + f"hc_{module}_inject.weight", [hc_dim, hc], 0.05 * rng.standard_normal(hc_dim * hc))
        if (layer + 1) % config["full_attention_interval"]:
            q5k(p + "attn_qkv.weight", e, qkv)
            q5k(p + "attn_gate.weight", e, value_dim)
            f32(p + "ssm_a", [dt_rank], -rng.uniform(0.5, 2.0, dt_rank))
            f32(p + "ssm_alpha.weight", [e, dt_rank], 0.05 * rng.standard_normal(e * dt_rank))
            f32(p + "ssm_beta.weight", [e, dt_rank], 0.05 * rng.standard_normal(e * dt_rank))
            f32(p + "ssm_conv1d.weight", [config["conv_kernel"], qkv], 0.1 * rng.standard_normal(config["conv_kernel"] * qkv))
            f32(p + "ssm_dt.bias", [dt_rank], 0.1 * rng.standard_normal(dt_rank))
            norm(p + "ssm_norm.weight", state)
            q8(p + "ssm_out.weight", value_dim, e)
        else:
            q5k(p + "attn_q.weight", e, heads * head_dim * 2)
            q8(p + "attn_k.weight", e, kv_heads * head_dim)
            q8(p + "attn_v.weight", e, kv_heads * head_dim)
            q8(p + "attn_output.weight", heads * head_dim, e)
            norm(p + "attn_q_norm.weight", head_dim)
            norm(p + "attn_k_norm.weight", head_dim)
            tensors.append((p + "indexer.q_proj.weight", [e, idx_heads * idx_dim], BF16,
                            bf16_bytes(0.1 * rng.standard_normal(e * idx_heads * idx_dim))))
            tensors.append((p + "indexer.k_proj.weight", [e, idx_dim], BF16, bf16_bytes(0.1 * rng.standard_normal(e * idx_dim))))
            norm(p + "indexer.q_norm.weight", idx_dim)
            norm(p + "indexer.k_norm.weight", idx_dim)
        if layer in config["ple_layers"]:
            f32(p + "ple_conv1d.weight", [config["ple_conv_kernel"], hc_dim],
                0.1 * rng.standard_normal(config["ple_conv_kernel"] * hc_dim))
            q8(p + "ple_key.weight", e, hc_dim)
            q8(p + "ple_value.weight", e, e)
            for which in ("conv", "key", "query"):
                norm(p + f"ple_norm_{which}.weight", hc_dim)
        f32(p + "ffn_gate_inp.weight", [e, experts], 0.2 * rng.standard_normal(e * experts))
        f32(p + "ffn_gate_inp_shexp.weight", [e], 0.05 * rng.standard_normal(e))
        for bank in ("gate", "up"):
            tensors.append((p + f"ffn_{bank}_exps.weight", [e, ff, experts], IQ2_XS, random_iq2_xs(rng, ff * experts, e)))
        tensors.append((p + "ffn_down_exps.weight", [ff, e, experts], IQ4_NL, random_iq4_nl(rng, e * experts, ff, 0.05)))
        q5k(p + "ffn_gate_shexp.weight", e, config["shared_ff"])
        q5k(p + "ffn_up_shexp.weight", e, config["shared_ff"])
        q8(p + "ffn_down_shexp.weight", config["shared_ff"], e)
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", default=".cache/research/Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf.header.bin")
    parser.add_argument("--manifest", default="configs/flash-next-model.json")
    parser.add_argument("--output", default="models/eqt-tiny-qwen4exp.gguf")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--indexer-top-k", type=int, default=16,
                        help="Sparse-attention budget in cells; the default is shorter than the test prompts")
    args = parser.parse_args()
    if args.indexer_top_k < 4 or args.indexer_top_k % 4:
        parser.error("--indexer-top-k must be a positive multiple of the compress ratio 4")
    header = Path(args.header)
    expected = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["files"][0]["header"]
    if hashlib.sha256(header.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError("Target header does not match the pinned manifest")
    with header.open("rb") as stream:
        source = read_raw_metadata(stream)
    config = dict(embedding=256, expert_ff=64, shared_ff=64, experts=16, experts_used=4, layers=8, heads=4, kv_heads=2,
                  head_dim=64, ssm_state=32, ssm_groups=2, ssm_dt_rank=4, conv_kernel=4, full_attention_interval=4,
                  hc=4, hc_rank=64, indexer_heads=2, indexer_dim=64, ple_layers=[1], ple_ngram=3, ple_heads_per_ngram=4,
                  ple_dim=32, ple_conv_kernel=4)
    # Small prime vocabularies per hash head; the table is padded to a multiple of 512 rows like the converter's.
    primes = [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097]
    offsets = [sum(primes[:i]) for i in range(len(primes))]
    config["ple_rows"] = (offsets[-1] + primes[-1] + 511) // 512 * 512
    arch = "qwen4exp."
    overrides = {"block_count": config["layers"], "embedding_length": config["embedding"],
                 "attention.head_count": config["heads"], "attention.head_count_kv": config["kv_heads"],
                 "attention.key_length": config["head_dim"], "attention.value_length": config["head_dim"],
                 "expert_count": config["experts"], "expert_used_count": config["experts_used"],
                 "expert_feed_forward_length": config["expert_ff"],
                 "expert_shared_feed_forward_length": config["shared_ff"], "context_length": 4096,
                 "ssm.state_size": config["ssm_state"], "ssm.group_count": config["ssm_groups"],
                 "ssm.time_step_rank": config["ssm_dt_rank"], "ssm.inner_size": config["ssm_state"] * config["ssm_dt_rank"],
                 "ssm.conv_kernel": config["conv_kernel"], "full_attention_interval": config["full_attention_interval"],
                 "hyper_connection.count": config["hc"], "hyper_connection.low_rank": config["hc_rank"],
                 "attention.indexer.head_count": config["indexer_heads"],
                 "attention.indexer.key_length": config["indexer_dim"],
                 # Smaller than the test prompts, so the selection is sparse and spans prompt chunks.
                 "attention.indexer.top_k": args.indexer_top_k, "ple.ngram_size": config["ple_ngram"],
                 "ple.heads_per_ngram": config["ple_heads_per_ngram"],
                 "ple.conv_kernel": config["ple_conv_kernel"],
                 "embedding_length_per_layer_input": config["ple_dim"]}
    arrays = {"attention.compress_ratios": (UINT32, [4 if (i + 1) % 4 == 0 else 0 for i in range(config["layers"])]),
              "ple.layers": (UINT32, config["ple_layers"]), "ple.head_offsets": (UINT64, offsets),
              "ple.head_vocab_sizes": (UINT64, primes)}
    metadata, vocab = [], None
    for key, kind, raw in source:
        if key.startswith(DROPPED_PREFIXES):
            continue
        short = key.removeprefix(arch)
        if short in overrides:
            raw = encode_value(kind, overrides.pop(short))
        elif short in arrays:
            stored_element = struct.unpack_from("<I", raw)[0]
            element, values = arrays.pop(short)
            # Keep the file's element width (the head ranges are uint64 in the target).
            raw = encode_array(stored_element if stored_element in (UINT32, UINT64) else element, values)
        elif key in ("general.name", "general.basename"):
            raw = encode_string("eqt-tiny-random-qwen4exp")
        elif key == "general.size_label":
            raw = encode_string("tiny")
        elif key == "tokenizer.ggml.tokens":
            vocab = struct.unpack_from("<Q", raw, 4)[0]
        metadata.append((key, kind, raw))
    if overrides or arrays or not vocab:
        raise ValueError(f"Target header lacks expected keys: {sorted(overrides) + sorted(arrays)}")
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
