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

The phone's temperature moves repetitions by up to 20% within six runs (2026-10-07). On the Galaxy, the stock throttling status follows the skin sensor (thresholds 38/40/42/45 °C), so `tools/sweep_target.py --cool-to C` waits before each run until the live HAL skin temperature is at most C, and records `celsius_before`. It only reads `dumpsys thermalservice` and changes no thermal settings. Read the "Current temperatures from HAL" section; the cached list can stay stale for hours.

Results include `decode_process_cpu`: whole-process user/system CPU milliseconds, minor/major faults and voluntary/involuntary context switches over the decode phase only. All threads count, including expert I/O workers. The values are null where `getrusage` is unavailable. On this phone `simpleperf` samples user space only (`cpu-cycles:u`), so system time from these counters is the only view of kernel-side cost (faults, `MADV_DONTNEED`, page zeroing).

`tools/sustained_run.py` measures sustained decoding on battery. It records tokens/s for every minute, and the phone's energy from the stock battery service (`dumpsys battery` charge counter × mean voltage, since the shell user cannot read the `/sys` battery nodes). An idle baseline measured just before the run is subtracted. The result is whole-phone energy per generated token, not engine-only energy. The phone must be unplugged; nothing on the device is changed.

The sustained runner now defaults to 900 seconds of natural-EOS requests on a single loaded model. It finishes the active request at the deadline, cycles the request's `sequence`, and rejects `ignore_eos=true`. Output limits remain safety bounds; the runner preserves context and sampling settings. `configs/flash-next-sustained-request.json` supplies four synthetic workloads (Italian, English, code and reasoning) for evaluation; keep predictor calibration data separate. Supply the phone's existing pack path explicitly through load options. For the baseline, use the same context and pack options in every cache arm and admit a larger cache only after measuring available memory.

```powershell
python tools/sustained_run.py --run-as org.equity.app --model files/models/qwen3.8-flash-next-ud-q2kxl/Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf --load '{"expert_pack":"/data/user/0/org.equity.app/files/models/qwen3.8-flash-next-ud-q2kxl/Qwen3.8-Flash-Next-UD-Q2_K_XL.eqtpack"}' --powered --output results/local/flash-next-natural.json
```

Reports retain every request's native result (including termination, applied configuration, TTFT, counters, memory and token latencies), the exact input sequence, and source/binary hashes. Complete 60-second windows concatenate active decode time, including final EOS computation, and exclude prefill/gaps. `sequence_wall_seconds` and `end_to_end_tokens_per_second` include prefill and request transitions but exclude initial model load; `wall_seconds` covers the runner interval including load and transport. Energy covers the battery snapshot interval including load, and is null when powered or required counters are missing. Missing useful-compute and exposed-wait estimates stay null: summed worker wait is not wall time. Per-bank traffic and drained-I/O attribution still require additional native instrumentation before the full feasibility analysis.

The CLI supports this same protocol directly with a nonempty `sequence` and a positive `sustained_seconds` (at most 86,400). A zero/absent duration retains the existing one-pass sequence behavior. `--target host` in the Python runner permits functional smoke checks without Android; these establish no mobile speed.

Expert I/O attribution follows each job's submission phase through request draining. `decode_io_bytes_by_bank` splits gate/up/down/other ranges by demand/speculative queue at dispatch (promoted jobs count as demand). Its sum equals `decode_io_bytes`. Packed ranges include alignment padding; non-pack ranges report logical bank payloads rather than physical aligned reads. Failed jobs are counted separately in `decode_io_failed_jobs`; ranges on a failed job are not evidence that the full read succeeded. `decode_io_busy_ms` is the union of intervals with at least one decode-submitted read active, including reads completed after generation. `expert_io_drain_ms` reports the final wait outside `decode_ms`, and sequence wall time includes it. Workers publish counters before exposing completion to `end_request`.

`decode_hits`, `decode_prefetch_hits` and `decode_inflight_hits` exclude prompt calls. Prefetch hits are a subset of ready hits; `decode_hits + decode_inflight_hits + decode_misses` equals demanded experts. The ready demand-cache fraction is `(decode_hits - decode_prefetch_hits) / decode_demanded_experts`; do not use request-wide hit counts to derive a decode-only cache floor. These changes affect measurement attribution, not routing or stored weights.

`eqt-kernel-bench` checks that batched MUL_MAT and MUL_MAT_ID results equal one-column products bit for bit (`batch_bitwise_equal`), and measures all batch widths interleaved within each repetition.

## Schema 1 and units

- `configuration`: applied context/batch/threads, CPU backend, mmap flag, file/tensor bytes, admission budget, feature flags, source revision (with dirty marker), compiler and pinned llama revision. Budget kind is `admission_only`.
- `provenance`: binary/model/workload SHA-256, exact request, source-tree digest, toolchain lock, UTC start time, repetition and before/after device state. Source hash describes recorded files; binary hash identifies the built artifact. Build flags are in CMake/scripts and dependency lock.
- `load_ms`: model plus context/batch construction; excludes app file hashing. `prefill_ms`: native processing of prompt batches. Tokenization/template time is included in request `total_ms` and TTFT, not prefill throughput.
- `ttft_ms`: request start through first sampled non-EOS token, independent of whether it contains visible text. `first_visible_piece_ms`: first non-whitespace raw piece, which may be a reasoning delimiter. `first_answer_content_ms`: with thinking, the first non-whitespace piece after `</think>` (null if no answer started); otherwise equal to `first_visible_piece_ms`. These exclude model load and Android rendering delay.
- `decode_ms`: wall time after prefill through the sampling/decode loop, including sampler, callbacks and final EOS computation. `decode_tokens_per_second = generated_tokens / (decode_ms / 1000)`. First token uses prefill logits; the last length-limited output token is not evaluated again. Report token count so short-run inflation is visible.
- `total_ms`: template/tokenize + memory reset + prefill + decode; excludes result serialization/final state clear and load. End-to-end user latency also includes UI dispatch, hashing/load if needed, and rendering.
- `token_latency_ms`: first sample latency from decode start, then successive emission intervals; p50/p95/p99 use nearest rank. First token TTFT is reported separately. Null percentiles mean zero emitted tokens.
- `memory_before/after`: RSS/PSS and process-lifetime high-water RSS in bytes; `/proc/self/io` storage bytes and `getrusage` faults where accessible. Missing Windows/Android counters are null. PSS is a snapshot; peak RSS includes loading and previous operations in that process. OS page cache outside mappings is not fully represented.
- With thinking, `reasoning_tokens` counts generated tokens through `</think>` and `answer_tokens` the rest; `reasoning_ms` is decode time until that tag (the whole decode, with `reasoning_complete=false`, when the limit or a stop came first). They are null without thinking. Cache hit/miss, expert I/O wait and energy remain null. Storage counter deltas are process-accounted physical reads, not requested expert bytes. Warm-cache zero reads must not be interpreted as an ideal cache.
- `termination`: `eos`, `length` (the request's `max_tokens`), `context` (the loaded window filled first) or `cancelled`. A request is refused before evaluation when its prompt leaves fewer than `output_reserve` tokens (default: all of `max_tokens`) of the window; the app sends a reserve of at most 2,048 and enlarges the window when a prompt needs it.
- `op_profile` (request `profile_ops`): decode wall time per token by GGML operation and weight type, measured on compute thread 0 from one node's start to the next, barrier included; the gap across a graph split is counted on the split's last node. Instrumentation, not a baseline run. `decode_compute_lent` (load option `decode_reclaims_compute`): the prompt's compute buffer, the part decoding touched, and what was lent to the expert cache.
- `session` (request `session`, D033): `status` (`in_memory`, `restored`, `none`, `not_requested`, `other_model`, `mismatch`, `unreadable`, `incompatible`, `failed`), `restored_tokens`, `restore_ms`, and after the turn `saved_tokens`, `saved_bytes`, `save_ms`, `save_error` or `save_skipped`. A restored prompt counts its restored tokens in `reused_prompt_tokens`; restore time is inside `total_ms`, save time is not. `configuration.kv_on_storage` (load option `kv_file_dir`, D034): the directory and the bytes reserved there; null when keys and values are in memory.
- `text` and `token_ids`: output for fixture equivalence/quality evaluation. `first_logits` is opt-in and large, intended only for correctness tests; enabling it adds overhead and disqualifies that run as a normal performance baseline.

Thermalservice's cached values can differ from current HAL values. Preserve raw distinction. Battery temperature is not SoC temperature. Charging, screen state and user activity are part of conditions; the phone remains a normal personal device. No GPU counter or energy-access assumption follows from having ADB.

## Quality and upcoming coverage

`workloads/quality-v1.jsonl` and `tools/eval_quality.py` score quality and speed in the same run, so a speedup can be checked for quality loss. The set has 27 greedy, natural-EOS items:

- Italian: sentence count, translation, word limit, grammar, Unicode round-trip;
- instruction following: exact JSON keys, line format, yes/no, uppercase, language switch, word constraint;
- reasoning with exact numeric or single-word answers;
- coding, with Python tests;
- stable knowledge;
- a multi-turn recall;
- long context: a needle at about 4k and 8k tokens, and a three-record aggregation at about 4k, generated from a seed so no dataset is stored.

Checks are deterministic. Coding tests execute model-written code on the host only with `--execute-code`, in a subprocess with a timeout. Each record keeps answers, per-check details, prompt and generated tokens, prefill, TTFT and decode speed. `eval_quality.py compare A.json B.json` lines up two builds item by item. A build may replace the reference only if its scored checks do not regress; review the stored answers too, because string checks cannot judge fluency.

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

## Expert streaming, CPU variants and threadpool — 2026-10-06

Streaming is enabled per load: `expert_streaming`, `expert_cache_mib`, `io_threads`, `direct_io`, `prefetch`, `expert_layer_weights`, `expert_trace` (CLI only). Admission becomes non-routed tensor bytes + expert budget + 512 MiB reserve. `configs/target-stream-request.json` is the target starting point (1024 MiB cache, 4 I/O threads, buffered reads, no prefetch).

`expert_cache` fields (per request, cache contents stay warm across requests): `hits`, `misses`, `prefetch_hits`, `inflight_prefetch_hits` (demand waited on a speculative read), `demand_read_bytes`, `prefetch_read_bytes`, `prefetch_issued`, `prefetch_wasted` (evicted unused), `predicted`/`predicted_correct` (decode lookahead accuracy), `evictions`, `overflow_events` (a batch exceeded a layer quota), `io_wait_ms` (decode thread blocked on demand reads), `predict_ms`, `resident_bytes`, `peak_resident_bytes`. Requested bytes are payload bytes; physical storage reads remain the `/proc/self/io` delta. `expert_cache_hits`, `expert_cache_misses` and `expert_io_wait_ms` are now filled when streaming.

Exactness: `python tools/verify_streaming.py --target host|linux|android [--direct-io]` compares each streamed configuration with a resident run using the same prefill chunk and requires bitwise-equal first-token logits and identical greedy tokens. It runs eviction-every-layer, chunked prefill, fully cached, prefetch and non-uniform quota cases on two random-weight qwen35moe fixtures plus the lifecycle checker. Build the fixtures from the pinned target header with `python tools/make_moe_fixture.py` and `--expert-ff 96 --output models/eqt-tiny-qwen35moe-ff96.gguf` (unaligned Q8_0 down banks). They exercise graph, routing and storage paths; their text and routing locality are meaningless.

POSIX path without a phone: `docker run --rm -v <repo>:/src python:3.11-slim sh /src/scripts/linux_check.sh` builds natively on Linux x86-64 and runs the SIMD test and streaming checks, including real `O_DIRECT` on the container filesystem. NEON kernels: run the static `build/android/bin/eqt-simd-test` under `docker run --platform linux/arm64`; emulated timing is not a measurement.

Routing traces and replay: `tools/collect_traces.py` runs one process per prompt with `expert_trace` and concatenates JSONL traces. Collect calibration traces from `workloads/routing-calibration.jsonl` and evaluation traces from a different workload, then `tools/simulate_cache.py --calibrate A --evaluate B --plan results/2026-10-06/target-expert-plan.json --experts 256 --experts-used 8 --budget-mib ...`. Policies: global LRU, global random, uniform layer LRU, optimized layer LRU and a global farthest-next-use bound, all with the native call semantics (transient overflow, trim after the call). Replay predicts hits and requested bytes only; wall time, reclaim and storage latency require device runs with the exported `native_expert_layer_weights`.

CPU variants: Android CLI runs need the shared libraries next to the tools; `tools/runner.py` stages and hashes them and sets `LD_LIBRARY_PATH`. `--cpu-variant android_armv8.6_1` (benchmark/verify tools) forces one variant through `EQT_CPU_VARIANT`; results record `cpu_variant` and HWCAP `cpu_features`. Compare variants with identical requests, alternating order, and report both decode and prefill; different kernels may reorder FP32 reductions, so cross-variant logits are compared with the existing absolute tolerance, not bitwise.

Threadpool: `threadpool` (default true), `poll` (0–100) and `cpu_mask` (hex CPU-id mask) are load options; `configuration.threadpool` records them. A/B `threadpool=false` reproduces GGML's per-graph disposable pool. Pinning or polling changes energy and phone responsiveness; report thermal state and battery drain alongside speed.

Galaxy sequence: `python tools/galaxy_session.py preflight simd variants threadpool streaming`, then explicitly `acquire-target` (streams 22 GB from the pinned URL to `/data/local/tmp/eqt` without a host copy, hash-verified on the stream and on the device), then `target` and `traces`.

## Qwen3.8-Flash-Next (qwen4exp) — 2026-10-19

Artifact: `configs/flash-next-model.json`. The repository ships three gguf-split shards; the artifact is their merge. `python tools/gguf_merge.py configs/flash-next-model.json --local <merged file> [--verify-data]` fetches only the shard headers (ranged requests), rebuilds the merged header as `gguf_merge` writes it and compares it with the pinned header hash and a local merge; `--verify-data` also rehashes each shard from the local file. The app downloads the shards into the merged file directly (same layout, no second copy) and checks the merged header before writing weights and the whole file at the end.

Copying a local merge into the installed debug app: `python tools/acquire_model.py configs/flash-next-model.json --push-local models/flash-next/Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf --model-id qwen3.8-flash-next-ud-q2kxl`. It appends 256 MiB segments through `/data/local/tmp` as the app user, resumes after the last confirmed segment, needs only the model's size free on the phone, and installs the file only after its on-device SHA-256 matches. CLI tools reach models in the app's storage with `--run-as org.equity.app` and app-relative model paths (`files/models/<id>/<file>`): binaries are copied into `files/bench/` and run there.

Fixture: `python tools/make_qwen4exp_fixture.py` (from `.cache/research/Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf.header.bin`, the first 11,024,320 bytes of the merge). `verify_streaming.py` includes it; `--fixture` and `--skip-mtp` select a subset.

Load options added:

- `prefill_prefetch_min_coverage` (0–1, default 0): read the next layer ahead during a prompt batch only when the batch is expected to use at least that share of its experts, 1 − (1 − k/E)^rows.
- `lookahead_gate` (default true): with hyper-connection mixers, the lookahead applies the mixer's low-rank gate; false averages the normed streams.

Request option: `prompt_cache` (default true) reuses the state the previous request left when the new prompt starts with all of its tokens (D027).

Result fields added: `reused_prompt_tokens`; `expert_cache.borrowed_evictions` (entries other layers lent to a prompt batch), `expert_cache.predict_wait_ms` (decode-thread time spent waiting for the lookahead). `prefill_tokens_per_second` counts evaluated tokens only.

Decode-speed options (D029), all exact:

- `speculative_pool` (default false): lookahead reads go to a pool beside the layer quota instead of evicting cached experts. A pooled expert joins the cache when routed; otherwise it is dropped after its layer has run.
- `lookahead_layers` (1–2, default 1): with 2, the predictor also reads the top `lookahead2_count` (default 5) candidates of layer l+2. It uses the gate-less mixer and goes after the urgent l+1 reads.
- `decode_uses_prompt_allowance` (default true): while decoding, `prefill_prefetch_mib` counts as cache quota.
- `prefetch_count` (default 0, all): reads ahead only the first N predicted ranks per row.
- `bank_spin_us` (default 20000): spin time before a compute thread sleeps on a pending bank.

Read scheduling and predictor options (D030), all exact; routing never depends on them:

- `read_priority` (default false): three read queues served in order, demand → next layer → layer after. Rules:
  - a queued speculative read moves to the demand queue when its layer routes to the expert;
  - a read for the layer after moves up a queue when the next layer's own prediction lists it;
  - a pooled read no worker has started is dropped when its layer has run without it.
- `prefetch_inflight` (0–16, default 0 = no limit): while decoding, at most this many speculative reads in flight, so demanded reads queue behind fewer of them in the device. Prompts are not capped.
- `lookahead_router_q8` (default false): the lookahead scores routers from a Q8_0 copy (1.39 MB instead of 5.24 MB per Flash-Next layer). The copies count toward admission (67 MB for Flash-Next). On the Galaxy it left the predictor's time unchanged (D030), so the catalog does not use it.

Whatever the options, the predictor queues the next layer's reads before it scores the layer after. The hyper-connection gate's sigmoid uses a vectorized exponential on NEON (`eqt-simd-test` checks it against double precision).

New `expert_cache` fields:

- `decode_demand_jobs` and `decode_demand_job_ms`: bank reads demanded while decoding, timed from demand (or promotion) to completion;
- `promoted_reads`, `confirmed_reads` and `cancelled_prefetches`.

New `expert_cache` fields:

- `predicted_by_rank` / `predicted_correct_by_rank`, and the `early_` variants for the l+2 prediction, by rank 0–15;
- `decode_prefetch_issued` and `early_prefetch_issued`;
- `decode_io_bytes`, and `decode_io_busy_ms` (decode wall time with at least one expert read in flight);
- `decode_budget_bytes`.

`eqt-io-bench` takes `threads` (1–32): concurrent readers, each with its own buffer, over the same random offsets; `bytes_per_second` is the aggregate rate. Flash-Next experts are read as 473,600-byte (gate, up) and 921,600-byte (down) slices, at most 117 and 226 pages once rounded to pages, so sweep `read_bytes` 479232 and 925696 with `direct: true` over `threads` 1, 4, 8, 16 and 32 on the model file itself.

Decode-only replay of routing traces: `python tools/replay_decode.py --calibrate <trace> --evaluate <other trace> --entry-bytes 1884160 --expert-bytes 1875200 --layers 48 --budget-mib ...`. Prompt calls update the cache with native semantics; only single-token calls are scored.

Four-row kernels and the verification replay (D037):

- `row_kernels` (load option, list of weight type names, default empty): single-column MUL_MAT and MUL_MAT_ID products of these types use four-row kernels where the CPU backend variant has them (AArch64 dotprod: `iq2_xs`, `iq4_nl`). They are bitwise equal to GGML's one-row kernels. The switch is process-wide; every load sets it. `configuration.row_kernels` lists the types that got one (empty on x86 or the armv8.0 variant, null when the backend has no switch).
- `rollback_slots` (load option, 0–8, default 0): recurrent-state snapshots without an MTP head, as `mtp` sets them, so a context can roll back rejected positions (`configuration.rollback_slots`). Slots bring back the recurrent-state gather that patch 0007 removes for single-sequence decoding.
- `verify_replay` (request, greedy, text only, a load without `mtp`): `{"tokens": [reference token ids], "acceptance": 0.65, "draft_max": 3}`. Every step evaluates an anchor and up to `draft_max` proposals (1–7, default 1). Both `batch` and `rollback_slots` must be at least `draft_max + 1`. Correct proposals follow the reference; the first wrong proposal uses a later reference token, and later inputs still follow the reference, making rejected steps optimistic. Each reached draft depth has its own evenly spread conditional acceptance schedule. The answer is teacher-forced; rejected rows are rolled back. The result reports `draft_max`, `drafted_tokens`, `accepted_drafts`, `acceptance_rate` (accepted/proposed), `verification_steps`, `tokens_per_step`, `step_ms`, and `target_agreement` (the target's greedy token against the emitted reference). The short final block shrinks to the remaining output/reference tokens. No draft head runs: `draft_head_ms` is null and its cost is not in `decode_ms`. Termination `replay_end` means the reference ran out. This is a verifier measurement, not speculative inference or proof of numerical equivalence.

Short verification screening: `python tools/bench_verification.py --target host --model models/Qwen3.5-0.8B-Q8_0.gguf --output results/local/verification`. The default is five runs of 32 tokens: plain, perfect free proposals with 8/4/2 verification rows (including the anchor), then plain again to expose drift. Use `--load '{"expert_streaming":true,...}'` to pass a model's measured streaming configuration. Android needs an existing model path and optionally `--run-as org.equity.app`; it never pushes weights. The tool retains raw results, source/artifact hashes, thermal readings and actual expert read bytes per emitted token. It performs no cooldown or sustained run. A promising short result needs an alternating repeat and numerical comparison before changing inference defaults.

Greedy prompt lookup (D039/D040, CLI default off): request `"prompt_lookup": 3` proposes up to three tokens from the request's prompt and generated history. It selects the most recent occurrence of the longest matching suffix (3–8 tokens) and can extend periodic overlapping matches. Every proposal is verified by the target; the first disagreement ends the accepted prefix and later state rows are rolled back. No match evaluates one token. The load needs `rollback_slots` and `batch` of at least `prompt_lookup + 1`; the supported proposal range is 1–7. Sampling, images, steering, MTP and verification replay cannot be combined with lookup. Since D040 the full prompt and first sample use the normal path, and prompt/session reuse follows its usual eligibility rules. Result `prompt_lookup` reports proposed/accepted tokens, verification steps, tokens per step and lookup time; the initial sample is outside verification steps. Later batch shapes can change floating-point results: no general bitwise-equivalence claim.

In the app, Settings → Advanced → Prompt lookup is enabled by default for the pinned Qwen3.5 9B Q4_K_M. Other model profiles, nonzero creativity, image input, steering and insufficient snapshot memory use normal generation. The four slots need 192 MiB beyond the existing admission requirement; lookup is dropped before reducing context or rejecting a load. Disabling applies to the next answer; enabling needs a reload. `python tools/verify_lookup.py --model models/Qwen3.5-0.8B-Q8_0.gguf` compares first logits, token IDs, turn/session eligibility, the disabled path and a one-token limit with plain decoding. Its report distinguishes actual reuse/restore from a matching fallback.

Real lookup comparison on an existing phone model: `python tools/bench_verification.py --run-as org.equity.app --model files/models/qwen3.5-9b-q4km/Qwen3.5-9B-Q4_K_M.gguf --request configs/lookup-copy-request.json --lookup --rows 4 --output results/local/lookup-copy`. This uses actual greedy generation, without teacher forcing. It retains raw output and summary before refusing a token mismatch. The example is a prompt-copy workload and is not a general chat benchmark.

`eqt-kernel-bench THREADS --experts` times Flash-Next's expert products (gate and up 2560→640 IQ2_XS, down 640→2560 IQ4_NL, 160 distinct experts, one decoded token) with the four-row kernels off and on, interleaved, and checks them bit for bit, also with three tokens and row counts that leave one-row edges. `eqt-kernel-bench THREADS --sustain=SECONDS` repeats the raw DRAM read rate (256 MiB), Q5_K/Q8_0/Q6_K products and the expert products back to back and prints one sample per 0.5 s with the CPU clocks of cpu0, cpu4, cpu7 and cpu9 (null when unreadable).

`tools/sustained_run.py --powered` runs on a charging phone: speed and thermal behavior only; energy fields are null.

Image input with a streamed model (D038):

- `images_release_expert_cache` (load option, default false; needs `mmproj` and `expert_streaming`): before a request with images loads the projector, the expert cache is emptied and its pages returned, so the encoder fits where the cache was. Exact. The result reports `expert_cache_released_bytes` (null when not used), and `configuration.vision.releases_expert_cache` shows the option.
- `eqt-bench` request `"sequence": [request, …]` runs the requests one after another on the same load (the top-level `load` applies) and writes `{"results": [...]}`.
- `vision_flash_attn` (load option, default true): the CPU vision encoder uses flash attention. Without it a 1,024-token image holds about 1 GiB of attention matrices (D038). GPU mode keeps `gpu_vision_flash_attn`.
