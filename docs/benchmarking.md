# Benchmark protocol

## Reproduce

`tools/benchmark.py` runs the same request JSON through the shared native engine, saves `.raw.json`, captures stdout/stderr, and creates a provenance-enriched `.json`. Default output is ignored `results/local/`. Use only versioned public workloads for committed results. ADB serial selection is supported but serial numbers are not written into records. Device snapshots include a whitelist of build/SoC properties, CPU features, GLES identification, memory, free storage, thermalservice and battery data.

```powershell
python tools/benchmark.py --request configs/smoke-request.json --repetitions 3
python tools/benchmark.py --request configs/throughput-request.json --repetitions 3 --duration-seconds 900 --output results/local/soak
python tools/io_benchmark.py --output results/local/io
python tools/verify_runtime.py --target host --output results/local/correctness.json
python tools/verify_runtime.py --target host --model models/Qwen3.5-0.8B-Q8_0.gguf --output results/local/correctness-hybrid.json
```

Natural-EOS tests measure experience. Fixed-output probes set `ignore_eos=true`; tokens after EOS are artificial throughput work and not answer-quality evidence. Never compare their quality to normal chat. The current harness reloads the model each repetition; its sustained workload is repeated requests with load/snapshot overhead, not one uninterrupted decode. Record wall time and inference duty cycle. Do not call first-run mmap data cold: upload, hashing and prior runs warm page cache, and stock Android does not provide a supported cache-drop action here.

For optimization A/B, keep artifact/hash, prompt tokens/template, context, output budget, sampler, backend and thread count identical. Alternate A/B order for at least three pairs, record thermal snapshots, and report median/range plus total latency. Sequence changes can change expert routing. No frequency pinning, root, thermal overrides or forced app termination.

## Schema 1 and units

- `configuration`: applied context/batch/threads, CPU backend, mmap flag, file/tensor bytes, admission budget, feature flags, source revision (with dirty marker), compiler and pinned llama revision. Budget kind is `admission_only`.
- `provenance`: binary/model/workload SHA-256, exact request, source-tree digest, toolchain lock, UTC start time, repetition and before/after device state. Source hash describes recorded files; binary hash identifies the built artifact. Build flags are in CMake/scripts and dependency lock.
- `load_ms`: model plus context/batch construction; excludes app file hashing. `prefill_ms`: native processing of prompt batches. Tokenization/template time is included in request `total_ms` and TTFT, not prefill throughput.
- `ttft_ms`: request start through first sampled non-EOS token, independent of whether it contains visible text. `first_visible_piece_ms`: first non-whitespace raw piece, which may be a reasoning delimiter. `first_answer_content_ms`: null until reasoning parsing exists. These exclude model load and Android rendering delay.
- `decode_ms`: wall time after prefill through the sampling/decode loop, including sampler, callbacks and final EOS computation. `decode_tokens_per_second = generated_tokens / (decode_ms / 1000)`. First token uses prefill logits; the last length-limited output token is not evaluated again. Report token count so short-run inflation is visible.
- `total_ms`: template/tokenize + memory reset + prefill + decode; excludes result serialization/final state clear and load. End-to-end user latency also includes UI dispatch, hashing/load if needed, and rendering.
- `token_latency_ms`: first sample latency from decode start, then successive emission intervals; p50/p95/p99 use nearest rank. First token TTFT is reported separately. Null percentiles mean zero emitted tokens.
- `memory_before/after`: RSS/PSS and process-lifetime high-water RSS in bytes; `/proc/self/io` storage bytes and `getrusage` faults where accessible. Missing Windows/Android counters are null. PSS is a snapshot; peak RSS includes loading and previous operations in that process. OS page cache outside mappings is not fully represented.
- `reasoning_tokens`, `answer_tokens`, cache hit/miss, expert I/O wait and energy remain null. Storage counter deltas are process-accounted physical reads, not requested expert bytes. Warm-cache zero reads must not be interpreted as an ideal cache.
- `text` and `token_ids`: output for fixture equivalence/quality evaluation. `first_logits` is opt-in and large, intended only for correctness tests; enabling it adds overhead and disqualifies that run as a normal performance baseline.

Thermalservice's cached values can differ from current HAL values. Preserve raw distinction. Battery temperature is not SoC temperature. Charging, screen state and user activity are part of conditions; the phone remains a normal personal device. No GPU counter or energy-access assumption follows from having ADB.

## Quality and upcoming coverage

`workloads/quality.jsonl` is a small human-reviewed fixture set, not a public benchmark score. Report instruction following, Italian quality, arithmetic/coding correctness, repetitions and incomplete answers separately. Calibration data must not overlap this set. Reference target quality should use the exact official BF16 revision or a documented high-precision GGUF and the same template/workloads; no target-quality baseline has been run yet.

The 15-minute resident fixture trial and small hybrid host state checks have passed; see progress.md for results and conditions. `eqt-check`, built by the native build script, tests full-prefix replay, repeat requests, seeded sampling, reload and recovery after cancellation. It compares full first-logit vectors and token sequences; it does not test prefix reuse, speculative rollback or a streamed backend. `verify_runtime.py` also checks mmap/allocated loading, prefill chunk sizes and context overflow. Android verification stages both native executables and the selected model, hashes the remote artifacts and restores execute permissions. It replaces the CLI test fixture, without modifying app-private files.

The I/O harness performs three alternating buffered/direct pairs. Each run reads 128 seeded random 2 MiB ranges. It checks matching offsets and edge-byte checksums; that checksum detects some staging errors but is not full-content integrity verification. Full file hashes are checked separately before new runs. Record flag acceptance, requested bytes and `/proc/self/io` deltas; unsupported direct I/O is an error record, never a silent buffered fallback. The older October 5 records predate automatic remote hash verification; fixture identity was checked separately during that session.

Before M2: add measured 512/2K/4K/8K input frontiers using the actual tokenizer, full logits resident/streamed comparisons, cache eviction/short-read/cancellation tests, and app-private versus file-picker storage I/O comparisons. Synthetic I/O and small model tests are not proof of large MoE performance.

## Expert layout planning

Acquire only the pinned 16 MiB header prefix, without downloading the full weights:

```powershell
python tools/acquire_model.py configs/target-model.json --header-only --directory .cache/research
python tools/plan_experts.py .cache/research/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf.header.bin --output results/local/target-expert-plan.json
```

This offline tool accepts only the inspected separate gate/up/down Qwen3.5 MoE layout and K-block quantizations. Its sizes follow the pinned GGML `ggml-common.h` definitions. It validates payloads against tensor spans and derives each expert offset without treating alignment padding as model data. Planning uses 4 KiB rounding; any rounded final read beyond EOF is explicitly reported as `aligned_tail_overrun_bytes` and needs loader tail handling. This is not a filesystem capability probe. Provenance includes the header digest and pinned full-artifact identity, with `full_artifact_hash_verified=false`.

The plan contains no observed routes or hit rate. All-miss byte totals assume default top-eight routing, no coalescing and no cache reuse. A three-bank expert cannot be treated as a single contiguous 2 MiB read; replay must retain its separate offsets before comparing with the initial I/O microbenchmark.
