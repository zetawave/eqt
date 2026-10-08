# Progress — 2026-10-06


## Resume here (state on 2026-10-08)

**Goal:** the fastest exact engine for Qwen3.5/3.6 MoE models larger than the phone's RAM, on a stock Galaxy S24+ (Exynos 2400). Quality, context length and speed advance together; nothing is traded for tokens/s.

**State:**

- Qwen3.6-35B-A3B decoding:
  - UD-IQ2_M (11.9 GB): 5.5 tokens/s mean in 64-token runs on a cool phone; about 3.5 tokens/s sustained on battery (thermal status 4), about 1.07 J per token.
  - UD-Q4_K_M (22.7 GB): about 2.6 tokens/s median on the quality set.
- A 3,803-token prompt prefills in 164 s. Quality-v1: both builds 26/27.
- Defaults in the engine and app: expert slots, tiled i8mm prefill, coalesced reads, prefill prefetch, prompt batch 512, context bounded by the model.
- Decisions D015–D020 describe today's engine.

**Device layout (changed on 2026-10-08):**

- All models live in the app's private storage: `/data/user/0/org.equity.app/files/models/<id>/`, each with a `.sha256` marker (Qwen3-0.6B, UD-IQ2_M, UD-Q4_K_M MTP). `/data/local/tmp/eqt` no longer holds models.
- The installed app is the debug build. Uninstalling it, or switching to a release signature, deletes the models.
- CLI access works through `adb shell run-as org.equity.app`. It can read `/data/local/tmp` but cannot execute from it, so binaries and libraries must be copied into `files/bench/` and executed there. Executables staged in `/data/local/tmp` need mode 755 so `run-as` can copy them.
- **First task next session:** teach `tools/runner.py` this run-as mode (stage into `files/bench`, models under `files/models`, results read back with `run-as cat`). The sweep, quality and sustained tools depend on it.

**Next performance goals, in order:**

1. **Energy per token,** which sets sustained speed:
   - the spin-wait share (35% of cycles); WFE lost in short runs (D016) but may win once throttled;
   - CPU work per token;
   - why storage wait doubles in long runs (82 against about 31 ms per token).
2. **Prefill:** i8mm microkernel register blocking (the kernels are about 42% of prefill cycles), DeltaNet and attention prefill kernels, 8k–32k context sweeps.
3. **Cache policy:** eviction informed by router scores (the Belady gap is about 15 points), and a speculative pool that never displaces demanded experts.
4. **Exact Q4_K/Q8_0 tiles,** so MTP on the reference build is lossless and faster.
5. **Quality:** a larger evaluation set before UD-IQ2_M can replace the reference.

**Features requested and not started:** multimodal input (Qwen3.6 vision needs the `mmproj` projector and llama.cpp `mtmd`; models should advertise their capabilities in the app), and directional steering as a per-chat option.

## Working implementation

The shared C++20 CPU engine, Windows/Android CLI and Kotlin APK run real inference. The stock Galaxy SM-S926B / s5e9945 / Android 16 has completed chat, Unicode output, descriptor loading, reload, cancellation and recovery tests. No root, unrelated app termination or thermal override was used. The user freed storage to about 30 GiB; this does not increase available RAM.

M0 research and the small-model M1 vertical path are implemented. M2 bounded expert streaming is now **implemented and verified for exactness on host, Linux and under emulation, but has not run on the Galaxy**. The 35B checkpoint has not been acquired and has not run anywhere. The 5 tokens/s beyond-RAM target has not been demonstrated.

## Measured baseline: 2026-10-05

| Experiment | Observed result | Scope |
|---|---|---|
| Three natural-EOS Galaxy smoke runs | 29 tokens each, 21.01–23.32 tokens/s; TTFT 642.9–668.6 ms | Qwen3-0.6B Q8; short resident fixture. Italian response failed the two-sentence instruction. No target-quality claim. |
| Fifteen-minute repeated-request run | 53 requests, 13,568 tokens; weighted decode 19.19 tokens/s; range 17.94–20.29; median TTFT 827.9 ms; peak RSS 928,067,584 bytes (~885 MiB) | CPU NEON, four threads, context 2048, batch 128, 256 forced tokens, greedy, USB charging. All token sequences identical. |
| Buffered random reads | 2.95–3.53 GB/s; process-accounted physical reads zero | Warm page-cache result, not storage bandwidth. |
| Direct random reads | 0.63–0.96 GB/s, median 0.87 GB/s; ~256 MiB physical reads each run | Three buffered/direct pairs, alternating order, 128 × 2 MiB requests per run. O_DIRECT accepted; CLI `/data/local/tmp` only. |

The request window covered 910.97 seconds, with load + native-request duty fraction 85.57%. Each repetition reloads the model and collects device snapshots; this is not uninterrupted decode. Page cache was uncontrolled. Android thermal status was 0 or 1 during the run; battery and SoC temperatures are distinct. Normal phone activity was not controlled.

Records: `results/2026-10-05/soak-summary.json` retains per-request measurements and provenance; first/last full records are `soak-sample-000.json` and `soak-sample-052.json`. The complete local series remains in ignored `results/local/galaxy-soak`. I/O records and summary are in `results/2026-10-05/io` and `io-summary.json`. These exploratory builds preceded the first implementation commit; binary hashes identify the measured artifacts. Do not describe them as a frozen clean-source release baseline.

## Correctness and preparation: 2026-10-06

The host verifier now checks both official Qwen3-0.6B Q8 and pinned Qwen3.5-0.8B Q8. All mapped/allocated and batch 128/16 comparisons have maximum first-logit difference zero and identical 24-token sequences. Context overflow is rejected. Seven additional checks pass on each fixture: repeat request, new chat after conversation, full conversation replay, recovery after decode cancellation, recovery after pre-cancellation, seeded sampling repeat and reload. These compare full first-logit vectors and generated token IDs. The hybrid fixture exercises recurrent state but has no routed experts. Results are in `results/2026-10-06/correctness*-host.json`.

The expanded native checker builds for Android but has not yet run there. Earlier Galaxy dense-fixture numerical and JNI lifecycle checks passed. The verifier now stages its own binaries/model, verifies their remote SHA-256 and restores execute permissions after ADB push. The I/O harness now also stages and hashes its input instead of assuming the last CLI model is the dense fixture.

`tools/plan_experts.py` derives exact routed tensor byte ranges from the pinned 35B GGUF header. Its 120 separate gate/up/down banks contain 19,568,525,312 payload bytes. A complete expert is 1,900,544 bytes in 37 layers and 2,039,808 bytes in three layers. Without cache hits or coalescing, top-eight routing across 40 layers requires **960 independent reads/token**, totaling 611,516,416 payload bytes or 615,448,576 bytes with independently rounded 4 KiB reads. No planned target range crosses EOF. Unknown formats, malformed dimensions and incomplete banks are rejected. Synthetic range/alignment/error tests pass. This is a header-derived plan, not measured expert traffic.

## PC-only optimization pass: 2026-10-06

Implemented (decisions D006–D010):

- **Exact bounded expert streaming.** Metadata-mode load without the 22 GB mmap populate, routed banks in a reserved virtual range, residency enforced at `ffn_moe_topk` before `MUL_MAT_ID`, layer-partitioned LRU quotas, a prioritized I/O worker pool, optional direct I/O and optional router-lookahead prefetch. Per-request cache counters and opt-in routing traces.
- **One recorded llama.cpp patch** (absent optional tensors in metadata mode), applied and verified by bootstrap.
- **Android CPU ISA variants** (armv8.0 → 9.2) built as GGML dynamic backends, selected from HWCAP at runtime, forced with `EQT_CPU_VARIANT` for A/B. The APK/CLI now use `c++_shared`.
- **Persistent threadpool** with `poll` and `cpu_mask` options, replacing GGML's per-graph thread creation.
- **NEON intrinsics** for Equity's only hot loop (router lookahead); GGML kernels unchanged.
- Tools: random-weight qwen35moe fixtures from the pinned target header, streaming verifier, trace collector, cache replay with optimized quotas, direct-to-device target acquisition with resume, Linux container check and a staged Galaxy runner.

Verified locally (records in `results/2026-10-06/`):

| Check | Result | Scope |
|---|---|---|
| Streamed versus resident, 2 fixtures × 5 configurations | Bitwise-equal first-token logits and identical tokens; lifecycle checks pass with streaming + prefetch | Windows host and native Linux x86-64 container; Linux adds real `O_DIRECT`. Random weights: exactness only. |
| Native cache versus replay | Identical hits/misses (1003/1083 over 512 calls) | Validates that replay predicts the native policy's hits. |
| Replay on fixture traces | Global LRU 0% at every budget; layer quotas 24–51% | Mechanics only; no target locality. |
| NEON kernels | 0 failures versus double reference under QEMU arm64 | Correctness only; emulated timing meaningless. |
| Persistent threadpool | Host decode medians +2% resident, +11% streamed; identical tokens | x86 host indication only. |
| Existing resident checks | Qwen3-0.6B mapped/allocated, chunked prefill, context overflow and 7 lifecycle checks pass | Host, after all changes. |

Not verified: anything on the Galaxy after these changes (variant loading in the APK namespace, shared-library packaging, `MAP_NORESERVE` behaviour under Android memory pressure, direct I/O in app-private or picker storage), target routing locality, throughput, quality and energy.

## Galaxy session: 2026-10-06 evening

Device: SM-S926B, s5e9945, Android 16, kernel 6.1, `/data` on f2fs. CPU domains from sysfs: cpu0–3 1.96 GHz, cpu4–6 2.59 GHz, cpu7–8 2.90 GHz, cpu9 3.21 GHz. HWCAP: dotprod, fp16, SVE, SVE2, i8mm, bf16; no SME. After the user's restart: MemAvailable 5,847 MiB, 31.4 GiB free. The target was streamed to `/data/local/tmp/eqt` during these checks, so no timing below is a clean performance result.

| Check | Result | Scope |
|---|---|---|
| Streamed versus resident on device, 2 fixtures × 6 configurations incl. `O_DIRECT` | All bitwise-equal, identical tokens; lifecycle checks pass | CLI, `/data/local/tmp`, auto-selected `android_armv9.0_1`. |
| CPU variant loading | CLI and APK namespace both load `android_armv9.0_1` by soname; instrumentation (inference, reload, cancellation, recovery) passes | First run of the dynamic backends on Android. |
| NEON kernels on Exynos | 0 failures; router GEMV 256×2048 47.6 µs | Warm-cache microbenchmark during the download. |
| Variant numerics, Qwen3-0.6B Q8_0, 55-token prompt | Every ARM variant is RMS 0.103–0.114 (max 0.49–0.62) from the x86 build; v8.0 and v8.2 are chunk-invariant. i8mm variants differ with 16-token chunks: RMS 0.145 (v8.6) and 0.182 (v9.0), max chunk delta 0.82/1.02; v9.0 changes one greedy token versus x86 at batch 1/128 | The x86 build is another quantized implementation, not a high-precision reference. i8mm batched kernels depend on chunk shape: a numerical deviation needing quality evaluation, not a proven defect. |

### Qwen3.6-35B-A3B streamed on the Galaxy (first beyond-RAM runs)

The pinned UD-Q4_K_M artifact (22,134,528,992 bytes) was streamed to `/data/local/tmp/eqt` with `acquire_model.py --adb-stream`; SHA-256 `ac0e2c11…` matched on the received stream and on the device. **The 22 GB model ran on the 11 GB phone with ~3.7–3.9 GiB RSS** (2.45 GB shared weights read once at load, 4.3 s).

First natural-EOS run (1024 MiB expert cache, 4 I/O threads, buffered): 52-token Italian prompt, 51-token answer, correct and exactly two sentences; prefill 10.3 s (5.0 tokens/s), decode **2.78 tokens/s** (p50 339 ms), expert hit rate 47.9%. Page cache was warm from acquisition (20.2 GB requested, 14.8 GB physically read): not a cold result.

Sweeps use 64 forced greedy tokens; every arm produced identical token sequences, so all arms demanded identical experts. Thermal status rose from 0 to 2–3 during the session; arms alternate order, but results remain thermally confounded. Records: `results/2026-10-06/galaxy/target-*.json`.

| Experiment | Observation (decode tokens/s, per repetition) | Interpretation |
|---|---|---|
| I/O threads 4/8/16, buffered vs `O_DIRECT` | 2.18–2.81; 1.0–1.45 GB/s effective while waiting; 16 threads best for direct I/O | Decode I/O wait ~12–15 s of 23–29 s; the remaining ~180 ms/token is not I/O. |
| simpleperf (`cpu-clock:u`, 16 direct I/O threads) | 44% of user CPU samples in GGML threadpool polling, 39% in Q8_0/Q4_K/Q5_K/Q6_K dot products, 7.9% `memmove` (direct-I/O bounce) | Polling burns CPU while waiting for I/O; bounce copies cost CPU and memory bandwidth. |
| `poll` 50 vs 0; CPU masks | poll50 3.02/2.65; poll0 2.38/2.27; 6 big cores 2.34/2.35; cpu6–9 **0.28/0.37** | Wake-ups per graph split cost more than polling; manual pinning is harmful here (negative result). |
| Router lookahead prefetch (no extra graph split) | 3.07/3.05 vs 3.05/2.56; prediction accuracy **81.8%**; decode I/O wait −40%; 75% of prefetches used, 23% wasted; +10 GB physical reads | I/O overlap works, but extra I/O threads and bounce copies slow compute by a similar amount. |
| Compute threads 4 vs 6; cache 1024 vs 2048 MiB (prefetch on) | t4: 3.30/2.81; t6: 2.57/2.44; **2048 MiB: 3.38/3.58** | 4 threads stay best; doubling the cache cuts physical reads from 34.8 to 26.5 GB. |

Page-congruent zero-copy direct reads followed: banks sit at addresses congruent to their file offsets, so whole pages are read in place and only ≤2 partial pages per slice are copied. They are exact on the fixtures, on host and device. Target, same binary, 2048 MiB, prefetch, 16 I/O threads: **direct zero-copy 4.19 tokens/s (p50 232 ms, thermal 1→2) and 3.54 (thermal 2); buffered 2.54/2.71**. Direct I/O therefore beats buffered by 30–65% in the same session. The gain over the earlier bounce-buffer direct path (3.38/3.58 at thermal 2) is plausible but not established, because thermal state dominates the variation.

Best measured decode: **4.19 tokens/s** (64 forced tokens, thermal status 1→2). Sustained performance is lower: repeated runs reach thermal status 2–3 and 2.5–3.6 tokens/s. The 5 tokens/s target is not reached. During heavy runs the phone discharged on PC USB (33% → 30%, up to −1.3 A), so long runs need a stronger charger.


## Current constraints and next experiment

Arithmetic, not prediction: at 5 tokens/s, an all-miss decode needs ~3.06 GB/s of expert payload. Paging the Experts measured 38–48% LRU hit rates on iPhone at 576 MiB–1 GiB for this model; even 50% reuse leaves ~1.5 GB/s, above the 0.63–0.96 GB/s single-queue direct reads measured earlier. Queue depth (I/O threads), hit rate (quotas, budget) and compute (ISA variant) must all be measured on the phone before choosing the next optimization. Available RAM was 3–4 GiB; non-routed target weights are ~2.39 GiB, so the cache budget is the main free variable.

Next, ordered by measured bottleneck:

1. **Hit rate.** Collect calibration/evaluation traces on the target (`galaxy_session.py traces`) and A/B optimized against uniform quotas; also try 2560–3072 MiB caches while MemAvailable allows.
2. **Thermal-controlled sustained runs.** Use a wall charger, cool to status ≤1 between arms and run a 15-minute sustained decode with the best configuration; report median and decay.
3. **Compute.** About 150–200 ms/token is non-I/O. Profile again with zero-copy reads: Q8_0 dot products for the shared attention weights dominate. Then A/B CPU variants on the target (v8.6 i8mm versus v9.0 SVE2) and evaluate the i8mm chunking deviation for quality.
4. **Polling energy.** Threadpool polling consumed 44% of user CPU samples while waiting for I/O. Measure an adaptive pause during long I/O waits for energy, without losing the latency benefit of polling.
5. **Prefill** (10 s for 52 tokens) is dominated by reading most experts of every layer once; larger prefill chunks and prefetch for prefill are candidates.

System-picker selection remains pending. The phone was reconnected on the evening of 2026-10-06 for the session above. One known fixture is in `Download/eqt-smoke-Qwen3-0.6B-Q8_0.gguf`; the private staged fixture is separate. No personal files were removed. Build artifacts and all weights stay outside Git.

## Galaxy session: 2026-10-07 — MTP, repack, wireless ADB

The MTP-enabled UD-Q4_K_M (`configs/target-mtp-model.json`, 22,663,387,424 bytes, SHA-256 `0b21525e…`) was transferred over wireless ADB and verified on the device. Wireless `adb exec-in` silently lost stream tails, and orphaned remote writers kept appending after truncation. Acquisition now pushes 64 MiB segments with `adb push`, confirms each by device size, and still requires the final on-device SHA-256 (D013).

All runs: 1 GiB expert cache, zero-copy direct I/O, 16 I/O threads, 64 forced greedy tokens, MemAvailable ~4.5 GiB, thermal status 0–2. Records are in `results/2026-10-07/galaxy/`.

| Arm | Decode tokens/s (2 reps) | Acceptance | Tokens/step | Physical reads |
|---|---|---|---|---|
| plain | 2.79 / 2.81 | – | 1 | 24.2 GB |
| plain + prefetch | 2.92 / 2.83 (later 3.05 / 2.85) | – | 1 | 34.2 GB |
| **MTP k=1** + prefetch | **3.02 / 2.92** (later 2.94 / 2.90) | 0.73 | 1.73 | 36.4 GB |
| MTP k=2 | 2.56 / 2.57 | 0.61 | 2.21 | 43.0 GB |
| MTP k=3 | 2.44 / 2.41 | 0.51 | 2.46 | 45.3 GB |
| repack (plain / MTP k=1) | 2.93 / 2.79; 2.96 / 2.91 | – / 0.73 | – | same |

A k=1 step costs 572 ms: 32 ms drafting, 19 ms head catch-up, 220 ms blocked on I/O and about 300 ms compute for two verified rows. Per accepted token, I/O wait falls from about 180 ms to 127 ms, but compute does not fall: batching two tokens does not amortize decode compute on four Exynos cores. Larger drafts read the union of more experts faster than acceptance grows. Repacking resident weights for i8mm GEMV, which was +30% on the x86 host, gives no measurable gain on the phone.

The zero-copy profile shows `memmove` at 1.7% (it was 7.9% with bounce copies) and Q8_0 dot products as the largest compute item (21.7%). Decode compute is bound by the bytes of the Q8_0 shared weights, not by instructions.

Token sequences: with the auto-selected i8mm variant (v9.0), MTP k=1 and k=3 diverge from plain greedy at token 6, while k=2 matches. With the v8.2 dotprod variant, plain and MTP k=1 are identical over 32 tokens. The divergence comes from i8mm batched kernels (the chunk-shape dependence measured on 2026-10-06), not from the speculative logic. MTP is exact up to batch-shape floating-point effects.

Next lever, from these measurements: fewer bytes per token. The asymmetric UD-IQ2_M MTP artifact (11.9 GB) has experts at IQ2_XXS/IQ3_XXS (9.4 GiB versus 18.2 GiB) and attention/shared weights at Q5_K/Q6_K instead of Q8_0. It needs a separate quality evaluation against the UD-Q4_K_M answers on the same workloads.

### Asymmetric quantization: UD-IQ2_M + MTP — 2026-10-07

Artifact `configs/target-mtp-iq2m-model.json` (11,882,969,376 bytes, SHA-256 `b989da14…`, verified on device). Routed experts are mostly IQ2_XXS/IQ3_XXS (9.4 GiB); attention and the shared expert are Q5_K/Q6_K; the MTP block is included. Same protocol as the Q4 sweeps; all arms produced identical tokens; thermal status 1–2.

| Arm | Decode tokens/s | Decode storage wait |
|---|---|---|
| plain + prefetch, 1 GiB cache | 4.56 / 3.92 | ~4.2 s of 14–16 s |
| MTP k=1, 1 GiB | 3.71 / 3.77 | ~4.3 s |
| **plain + prefetch, 2 GiB** | **4.32 / 4.66** | ~2.9 s of 14 s |
| MTP k=1, 2 GiB | 4.26 / 4.09 | ~3.0 s |

Natural-EOS quality smoke set (six prompts, greedy, against the UD-Q4_K_M answers in `results/2026-10-07/galaxy/quality-*.json`): five matched the rubric. The two-sentence Italian instruction got one long sentence (Q4: two sentences); the JSON answer used the requested lowercase keys (Q4 did not). Speed on these prompts: 3.2–4.4 tokens/s versus 2.3–3.0 for Q4. Six prompts cannot establish quality parity; a larger evaluation set is required before recommending this build.

Storage wait is now about 20–30% of decode. During decode the four compute threads run on cpu6–9, but the X4/A720 cores run at about 2.0 GHz (maximum 3.2/2.9 GHz) under stock thermal and power management. The remaining compute cost is about 170 ms/token. The shell process uses the root cpuset (CPUs 0–9), so CPU placement is not the limiter.

Next:

1. A larger quality evaluation of UD-IQ2_M versus UD-Q4_K_M (Italian, instruction following, coding, reasoning), with calibration data kept separate from test data.
2. Sustained 15-minute runs.
3. Prediction-informed eviction (the Belady gap is about 15 points).
4. Reducing compute per token (graph splits, threadpool wake-ups, output-head cost).

## Galaxy session: 2026-10-07 evening — batched kernels, profile, negative results

Principle restated by the project owner: **quality and speed advance together**. No optimization may trade answer quality, context length or other capabilities for tokens/s. Numerically exact work (kernels, I/O, scheduling, an Equity-owned engine) comes first. Lossy candidates such as UD-IQ2_M stay candidates until a quality evaluation clears them against UD-Q4_K_M.

### Done and kept

- **Exact 2×2 dotprod tiles** for Q5_K, IQ2_XXS and IQ3_XXS, plus column pairing for odd batches in MUL_MAT and same-expert token pairing in MUL_MAT_ID (llama.cpp patch 0002, D015). Batched outputs are bitwise equal to one-column outputs on the Galaxy (v9.0 and v8.2 variants) and the host.
  - The cost of two columns fell from 1.87–1.99× to 1.15–1.39× (`results/2026-10-07/galaxy/kernel-bench-tiles.json`).
  - MTP k=1 on UD-IQ2_M moved from a loss to equal or up to +9%, with identical tokens.
- `eqt-kernel-bench` checks batched-versus-single bitwise equality and interleaves batch widths against thermal drift.
- `bootstrap.ps1` compares recorded patches per file, so several patches can coexist. `.gitattributes` keeps `*.patch` LF, so their pinned hashes hold on Windows checkouts.
- The engine resolves the GGML threadpool through the CPU backend registry (no `dlopen` of a variant soname).
- New counters: `decode_process_cpu` in results (user/system ms, minor/major faults, context switches over decode only; null on Windows).
- `sweep_target.py --cool-to C` waits until the live HAL skin temperature drops to the given value. Skin drives the stock throttling status here (thresholds 38/40/42/45 °C); the cached `dumpsys` list can be stale.
- `prefetch_extra` load option, default 0 (D017).

### Measured on the Galaxy

- `simpleperf` (user space only; the kernel denies kernel samples) on plain UD-IQ2_M decoding:
  - 35% of cycles are GGML threads spinning: 22% polling between graphs, including during expert I/O, and 13% in barriers;
  - dot kernels: Q5_K 17%, IQ2_XXS 14%, IQ3_XXS 9%, Q6_K 6%, Q4_K 5%;
  - `memmove` 1.8%; gated DeltaNet 1.4%.
- Expert I/O: 91% hits with a 2 GiB cache. Each miss blocks decode for about 1.46 ms, against a QD1 O_DIRECT floor of 0.64 ms per 320 KiB and 1.09 ms per 1 MiB. Storage wait is about 42–45 ms per token, about 23% of decode.
- Plain UD-IQ2_M decode ranged from 4.25 to 5.24 tokens/s across this session's sweeps. The phone's thermal state alone moved repetitions by up to 20%, which is why sweeps now cool down.

### Tried and reverted (negative results)

- **WFE waits instead of spinning:** −13%, and −4% in barriers only (D016).
- **Fused MTP draft/catch-up protocol:** no gain (D015).
- **Wider lookahead prefetch:** pollutes the cache (D017).
- **Layer-skip self-speculation:** not implemented. With measured costs it would give about 0.8×, versus about 1.05× for the shipped MTP head, because skipped-layer drafts still cost about half a token and read their experts.

### Next, in order

1. **Quality baseline before any further lossy choice.**
   - A real evaluation set: Italian, instruction following, coding, reasoning, long context.
   - Compare UD-Q4_K_M (reference) with UD-IQ2_M.
   - Track speed and quality in the same table.
2. **Context length as a first-class metric.** Benchmarks and the app default to 2048 tokens. Qwen3.6 attends fully in only 10 of 40 layers, so KV memory per token is small. Measure decode speed and memory at 8k–32k contexts, and stop accepting short-context-only results.
3. **Controlled re-measurement** with `--cool-to` and `decode_process_cpu`:
   - kernel time spent on expert eviction and refill (`MADV_DONTNEED`, refault, zeroing);
   - page recycling between evicted and loaded experts, if that time is significant.
4. **Fewer spin cycles that still keep clocks high.** Fewer graph splits per token, and a scheduler that overlaps attention and expert work instead of idling workers during I/O. This may justify an Equity-owned decode graph for the Qwen3.5/3.6 MoE family.
5. **Exact Q4_K/Q8_0 tiles**, so that UD-Q4_K_M (the quality reference) also gets exact, faster batched verification.
6. **Cache policy from router scores** (the Belady gap is about 15 points), with a speculative pool that never displaces demanded experts.

## Galaxy session: 2026-10-08 — long context, prompt processing, quality yardstick

Goal: make long context usable and measurable without touching answer quality. Every change is exact (bitwise-equal outputs) or differs only in float rounding, with tokens compared on the phone.

### Done and kept

- **Context bounded by the model, not by the engine** (D019). The old 8,192-token cap is gone; admission counts the KV cache estimated from GGUF metadata (22 KiB per token for the 35B).
- **Prompt processing 3.2× faster on a 3,803-token prompt: 521.6 s → 164.4 s** (D018):
  - coalesced expert reads;
  - whole-layer prefetch for prompt batches;
  - upstream's tiled GEMM enabled on AArch64 with a new i8mm (USMMLA) microkernel;
  - prompt batches up to 2,048;
  - a NEON `simd_gemm` that ends the scalar attention fallback in SVE builds (bitwise-identical logits).
- **52-token chat prompt:** 5.78 s → 3.95 s. Decode speed unchanged.
- **Quality yardstick:** `tools/eval_quality.py` with `workloads/quality-v1.jsonl` (27 auto-checked items, including generated 4k/8k long-context items). A host smoke run on Qwen3.5-0.8B scored 12/26; every failure was a genuine model error.
- **Correctness** (streaming, MTP, coalesced reads, whole-layer prefetch, direct I/O, tiled path): 28/28 on the Galaxy, 28/28 in the Linux container with real O_DIRECT, 22/22 on the host. `scripts/linux_check.sh` now copies the MTP fixture.
- **Tools:**
  - `sweep_target.py --quality-item` (long prompts from the quality set) and `--cool-to` (skin temperature);
  - `eqt-kernel-bench` takes batch widths and reports the relative error next to the bitwise flag;
  - `eqt-io-bench` has a `fresh_pages` mode;
  - the store reports `evict_ms`.

### First quality table: UD-IQ2_M versus UD-Q4_K_M

`quality-v1` was run on the Galaxy with the same D018 load options for both builds: greedy, natural EOS, thinking off as in the app. Records: `results/2026-10-08/galaxy/quality-v1-ud-*.json`.

| | UD-IQ2_M (11.9 GB) | UD-Q4_K_M (22.7 GB) |
|---|---|---|
| Checks passed | 26/27 | 26/27 |
| Italian / instructions / coding / knowledge / conversation / long context | 5/5, 6/6, 3/3, 3/3, 1/1, 3/3 | 5/5, 6/6, 3/3, 3/3, 1/1, 3/3 |
| Reasoning | 5/6 | 5/6 |
| Median decode over the 27 answers | 3.48 tokens/s | 2.61 tokens/s |
| Total prefill over the 27 prompts | 813 s | 1,256 s |
| Needle at 7,857 tokens (prefill) | correct (365 s) | correct (550 s) |

Both builds miss the same item: a 15% discount followed by a 10% markup on 80 (74.8). UD-IQ2_M answered 68, UD-Q4_K_M 78.2.

Two checks were too strict and rejected valid answers from both builds:

- "archivio" as a translation of "storage";
- "dei gelati buonissimi" as a grammar fix.

They were widened and both reports re-scored from the stored answers (`eval_quality.py rescore`); the six coding answers were read before their tests were executed.

Scope: on this 27-item set the 2.7-bit build shows no measurable quality loss and is about 33% faster. Twenty-seven items are a first signal, not proof of parity; harder reasoning and longer generations need a larger set before UD-IQ2_M can replace the reference.

### Slot-addressed expert cache (D020, now the default)

Decode faulted about 21,000 pages and spent about 125 ms of system CPU per token, because every expert reload landed on freshly zeroed pages. Fresh pages made 320 KiB direct reads 55–67% slower.

With persistent slots resolved through a new GGML hook, faults fall to 19 per token and system CPU to 36–40 ms. UD-IQ2_M decode goes from a mean of 5.05 to 5.54 tokens/s (p50 196 → 174 ms per token), with identical tokens. Best runs reach 5.81–5.84 tokens/s with a cool phone.

### First sustained run on battery

UD-IQ2_M, slots on, D018 load options, 2,400 greedy tokens of a chat answer continued past EOS. Run with `tools/sustained_run.py`, unplugged; record `results/2026-10-08/galaxy/sustained-ud-iq2m-slots-battery.json`.

- Tokens/s by minute: 4.55, 3.96, 3.98, 3.67, 3.57, 3.53, 3.51, 3.49, 3.43, 3.48. The mean over 649 s of decoding was 3.70.
- The stock thermal status rose from 0 to 4 (severe): skin 36.7 → 47.3 °C, application processor 60 °C. No thermal setting is touched.
- Whole-phone draw was about 7.4 W, of which 3.55 W is the idle baseline measured just before the run. Net energy was about 1.07 J per token. Battery went from 86% to 79%.
- Storage wait was 197 s of 649 s (82 ms per token), against 30–35 ms per token in 64-token runs. Unexplained so far; candidates are routing over long, post-EOS text and storage throttling as the phone heats.

So the 5 tokens/s target is met in short runs on a cool phone (5.5–5.8) but not sustained: over minutes, power and heat set the speed. Energy per token is now a first-class metric next to tokens/s and quality.

### Measured, not yet acted on

- After D018, prefill is dominated by the matrix microkernels (about 42% of cycles). Interleaving operands once per panel instead of once per microtile (kernel "v3") brought no measurable gain (171 s against 164 s) and was not kept; register blocking is the next candidate.

### Next

1. Quality: UD-IQ2_M and UD-Q4_K_M on `quality-v1` with the D018 load options (running), then compare item by item.
2. Energy per token: the spin-waiting share (35% of cycles; spinning beat WFE in short runs, D016, but heat now caps sustained speed), CPU work per token, and why storage wait doubles in long runs.
3. Microkernel register blocking; DeltaNet and attention prefill kernels.
4. Long-context sweeps at 8k–32k tokens: prefill time, decode speed, memory.
