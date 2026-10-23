# Flash-Next: an implementation route toward 5 sustained tokens/s

Date: 2026-10-23. Status: engineering proposal, not an achieved speed.

Device follow-up: [the natural-answer session](../results/2026-10-23/galaxy/flash-next-natural-feasibility.json) measured 2.36 tokens/s over 3,140 tokens and about 23 minutes, with 628 MB/token and thermal status 4. Its final ten full decode windows averaged 2.04 tokens/s. Two answers ended at EOS; the code answer hit the 1,536-token cap. Short cache screening reduced traffic at 2,304/2,816 MiB, but 2,816 MiB produced 2.57 million major faults over 32 tokens and only 0.69 tokens/s. A subsequent 1,792 MiB run recovered to 3.08 tokens/s at a cooler thermal state. This is evidence against adopting a larger cache from a free-memory estimate alone; it does not isolate the exact kernel reclaim policy. Cgroup limits were not readable. No larger-cache default or 3 GiB test was adopted.

The 3 GiB scenario below remains conditional arithmetic, not an established implementation route. Develop and measure scheduling/bank-prefetch candidates at 1,792 MiB first; increased capacity requires a separate fault/residency gate. Useful-compute and exposed-wait wall times still need attribution and remain null rather than being inferred from summed worker waits.

A first bounded scheduling experiment moved only the shared-expert computation between native routing and routed products. Six host comparisons retained bitwise logits/tokens; a short Galaxy ABBA gave plain 2.91/2.79 versus prototype 2.76/2.62 tokens/s, identical tokens. No speed benefit was demonstrated, so the source changes were restored. [The rejected candidate diff](../results/2026-10-23/galaxy/shared-expert-first-rejected.patch) and the measurement record preserve the result. This does not evaluate complete per-expert FFN fusion, which remains a separate implementation task.

## Decision

Pursue an exact CPU decode path that schedules complete expert computations as their banks become ready, combined with selective bank prefetching and a larger expert cache when actual memory admission permits it. Improve dense kernels only against the remaining measured compute deficit. Keep the current model, native top-10 routing, sampling, context, and reference decoder.

This is a concrete implementation candidate with a physically plausible budget, **not evidence that 5 sustained tokens/s is attainable on this phone**. Existing measurements do not establish that result. The candidate needs all three of these conditions together:

1. About 400 MB/token or less of actual expert reads, with adequate sustained storage throughput.
2. At most 170 ms/token of useful CPU work and synchronization in the integrated decoder.
3. At most 25 ms/token of additional, exposed I/O wait.

These are engineering acceptance budgets, not predicted gains. A 195 ms token would give 5.13 tokens/s. Meeting either the storage budget or the compute budget alone is insufficient. No multiplication of unrelated microbenchmark speedups is used here.

Start with the measurement and feasibility ticket below. Do not commission an MTP integration, GPU rewrite, quantization change, or a complete new engine first.

## Evidence and limits

Repository inspected: `a8979b083a558d37f28732e6fdb39d022c1ac43a`; pinned llama.cpp: `6c59c40076c00eab49754dc955d7652d93f9e125`, with the repository patches. Model identity is in [the manifest](../configs/flash-next-model.json).

| Evidence | Result | Interpretation |
| --- | --- | --- |
| D032, Galaxy, four short runs | 2.89-3.06 tokens/s | Real inference, approximately 330 ms/token; not sustained. |
| D035/D037, Galaxy profiles | About 100 ms exposed expert wait, 145 ms dense products, 55 ms expert compute, plus other work | Removing all exposed wait still leaves more than 200 ms in the current implementation. |
| D037, pack I/O microbench | About 2.4 GB/s for whole experts at queue depth 8-16 | An isolated storage rate, not guaranteed with simultaneous inference or smaller reads. |
| D037, 1,792 MiB cache | About 545 MB/token read | At 2.4 GB/s, storage alone takes about 227 ms/token: below 4.5 tokens/s even with perfect overlap. |
| D039, 1,024 MiB cache, perfect free proposals | 3.74 tokens/s with four verification rows | Teacher-forced screening without drafting; already below the requested speed. Two/four-row verification also changed a greedy prediction. |
| D037, 1,800-token run | 2.19 tokens/s, thermal status 4 | Most output was forced beyond EOS; not a valid natural-answer sustained baseline. |
| D035, held-out routing replay | LRU 48.4%, best tested online policy 50.7%, Belady 68.0% | Belady is an offline bound. A policy switch does not deliver its gain. |

Relevant records: [D032](../results/2026-10-20/galaxy/flash-next-d032-pack.json), [diagnostics](../results/2026-10-21/galaxy/flash-next-diagnostics.json), [D037](../results/2026-10-22/galaxy/flash-next-d037-session.json), [sustained run](../results/2026-10-22/galaxy/flash-next-sustained-usb.json), [verification](../results/2026-10-23/galaxy/verification-lookup.json), [cache replay](../results/2026-10-19/host/flash-next-replay-decode.json), and [decisions](decisions.md).

### New read-only checks in this investigation

The pinned header was verified against its SHA-256. Excluding routed tensors and the two mapped embedding tables, stored tensor spans total **3,536,007,680 bytes = 3,372.20 MiB = 3.293 GiB**. This is a weight-storage sum, not complete runtime RSS. The routed payload totals 46,084,915,200 bytes; selecting ten experts from every layer without reuse requires 900,096,000 payload bytes/token, before pack alignment.

Sampled one expert from every layer, all three banks: 144 samples, 90,009,600 bytes. zlib level 1 retained 99.219% of the bytes, a saving of only 0.781%; every decompression was byte-identical. IQ2_XS was essentially incompressible with this codec; IQ4_NL saved 1.59%. Its sampled marginal nibble entropy was 3.911 bits out of four. This is not a bound on every possible codec, but provides no support for making lossless compression the main speed strategy.

Provenance, exact sample offsets and hashes: [host probe](../results/2026-10-23/host/flash-next-feasibility-probe.json). The full 79 GB hash was not recomputed. No weights were changed. ADB was unavailable during this investigation, so there are no new phone timings or live memory measurements.

## Why this specific change

There are two different forms of wasted opportunity in the current source.

**Whole-expert speculative reads.** `ExpertStore::queue_reads()` splits demanded reads, but speculative pack reads fetch every bank. A prediction that turns out to be wrong therefore reads gate, up, and down. D035 attributes about 135-140 MB/token to unused speculation.

**Projection-wide scheduling.** `build_moe_ffn()` constructs the up product for all selected experts, then gate, activation, and down. The CPU backend orders ready experts before pending experts within one `MUL_MAT_ID`, but constructs that order once. It does not finish a ready expert's entire FFN while another expert's first projection waits. The shared expert is also constructed after the routed branch even though it depends on the same FFN input and can be computed independently before the final addition.

Consequently, simply lowering `prefetch_count` creates demand stalls; D036 already measured that failure. The proposed change supplies useful work during those stalls before making prefetch more selective.

The bounded implementation is one optional single-token MoE execution path plus changes to the existing expert store. Keep the attention, PLE, hyper-connection, KV, tokenizer, session, JNI, and sampler implementations. Do not replace the complete engine.

## Feasibility arithmetic

Use decimal MB/GB for traffic and binary MiB/GiB for memory. Let `D` be actual physical expert bytes read per emitted token, `C` the all-miss expert payload, `h` the demand-cache hit fraction excluding speculative hits, and `W` wasted speculative bytes. Approximately:

```text
D = C * (1 - h) + W + alignment/duplicate-read overhead
T_decode >= max(T_useful_compute_and_sync, D / sustained_storage_rate)
T_decode = T_useful_compute_and_sync + exposed_wait
```

The maximum is a lower bound, not a latency prediction: per-layer dependencies, queue latency and DRAM contention can make the real token much slower. `compute_wait_thread_ms` is summed across threads; do not add it directly to wall time or subtract it from a profile without establishing the conversion.

| Conditional scenario | Approximate expert bytes/token | Storage floor at 2.4 GB/s |
| --- | ---: | ---: |
| Current traffic | 545 MB | 227 ms |
| Held-out 1,792 MiB LRU hit rate 48.4%, only 35 MB waste | 499 MB | 208 ms |
| 3 GiB replay hit rate 59.5%, 35 MB waste | 400 MB | 167 ms |

The third row becomes about 187 ms at 2.14 GB/s, leaving little margin. About 2.14 GB/s is the busy-time rate derived from the old sustained run, not a forecast for a new read-size mix. At 2.0 GB/s, 400 MB/token consumes the entire 200 ms budget before dependency gaps. Thus the plan requires measuring storage under the proposed schedule and thermal state.

This also explains why an unchanged 1.75 GiB cache plus a few small kernels is not a credible 5-token/s plan on the held-out traces. Approximately 3 GiB of cache is a feasibility candidate, not RAM the phone is assumed to have.

## Implementation tickets, in order

### Ticket 0: establish a valid workload and two independent ceilings

Files: `tools/sustained_run.py`, `tools/sweep_target.py`, `tools/benchmark.py`, `native/engine.cpp` diagnostics, existing result formats.

1. Respect natural EOS and repeat independently authored long-answer requests to maintain load for at least 15 minutes. Account for prefill and pauses separately; also report end-to-end emitted-token throughput. Never concatenate post-EOS garbage to reach a duration. The first implementation increment corrected the harness's unconditional `ignore_eos=True` override and added repeated requests on one load; see progress.md for validation and remaining instrumentation.
2. Use public/synthetic prompts across Italian exposition, English exposition, code, and reasoning. Keep calibration and held-out evaluation disjoint. Record template, rendered prompt hash, thinking/sampling settings, context, actual output length and termination. A finite token cap is a safety/context bound, not permission to continue past EOS.
3. Reproduce the D032 configuration, then test 1,792 / 2,304 / 3,072 MiB caches only when admitted. Keep context and all other options equal. Use short alternating A/B/A runs before long runs. Do not clear user data, kill apps, or use privileged memory/cache/thermal controls.
4. Count actual decode I/O by bank and by demand/speculation; distinguish queued-but-cancelled reads from bytes already transferred. Collect wall-time compute, exposed wait, late-bank events, completed-but-unused bytes, process faults/RSS/PSS, committed slots, and thermal status. A decode result must drain/account outstanding work so that late completions do not disappear from the byte count.
5. Add an explicitly diagnostic known-route replay only if needed to measure the best possible overlap. Record a reference run's routes, preload only its next one/two layers within the same byte budget, and compare the target's own routes/logits during replay. This uses future information and must be labeled an oracle, never inference. It can reject an impossible schedule but cannot validate a predictor or establish a production speed.

The first deliverable is a table of safe cache capacity, required bytes/token, sustained read rate, and useful compute time. If even an optimistic storage floor exceeds 200 ms at the largest safe cache, the exact single-token candidate has failed that workload's storage gate. If useful compute remains above 200 ms, eliminating I/O stalls alone cannot succeed.

### Ticket 1: complete ready experts before blocking

Files: `third_party/llama.cpp/src/models/qwen4exp.cpp::build_layer_ffn`, `third_party/llama.cpp/src/llama-graph.cpp::build_moe_ffn`, CPU dispatch in `ggml/src/ggml-cpu/`, `native/expert_store.{h,cpp}`; record modifications through the existing patch/bootstrap mechanism.

Implement an opt-in, single-sequence, single-token path:

1. Run the native router, selection, weights and normalization unchanged; enqueue all missing demanded banks immediately.
2. Keep a small ready-task set for selected experts: up, gate, activation, down. Recheck readiness between tasks instead of waiting on the first pending expert in a fixed list. Compute the resident shared-expert branch while demanded reads run.
3. Reuse the existing numerical kernels first. Preserve activation quantization granularity, floating-point accumulation order, scale handling and FMA behavior. There is no need to invent faster arithmetic in this ticket.
4. Write each expert's output to its original selected-expert slot. Preserve the reference weighting and final reduction order, regardless of which expert finishes first. Ten 2,560-wide F32 outputs need only 102,400 bytes, plus bounded intermediates; account for every buffer.
5. Use the existing worker pool cooperatively, rather than creating one OS thread per expert. Never hold the store/queue mutex while executing arithmetic or waiting for I/O. Slot pins must last until every reader and writer finishes.

The first version can select a ready complete expert and process its whole FFN. Add finer per-bank scheduling only if profiling shows that coarse readiness still blocks useful work. Keep the ordinary path for prefill, verification batches, unsupported layouts, adapters and other cases until separately covered.

Correctness gate: compare intermediate outputs, logits and generated tokens to the reference on qwen4exp fixtures and the real pinned artifact. Timing-only scheduling should be bitwise equal. Exercise cancellation during a read, a late completion, a failed/short read, unload/reload and subsequent generation. Never publish an optimization merely because top-1 happened to match once.

Performance gate: lower exposed wait at the same bytes/token. This change cannot remove the storage-volume ceiling by itself. Measure its interaction with later partial prefetching rather than demanding a standalone 5-token/s result.

### Ticket 2: prefetch only the banks that earn their cost

Files: `native/expert_store.{h,cpp}` (`Entry`, `begin_load`, `queue_reads`, `read_slots`, `ensure`, `reap`, `cancel_unread`, `worker`), `native/expert_pack.{h,cpp}` only if a format change proves necessary.

Keep the existing pack and original tensor bytes. Its bank offsets permit an initial implementation without rewriting 46 GB of experts.

- High-confidence predictions may request complete experts. For less certain candidates, request only the first-needed projection, or gate+up; defer down until the exact router confirms the expert. Select the first-needed bank from the actual execution schedule, not from a guessed tensor order.
- A regular expert's down bank is about half its payload. Fetching only one of gate/up for a wrong prediction spends about a quarter of a full expert. Holding prediction decisions constant, 135 MB of current waste would become roughly 34 MB with one-bank speculation. This is arithmetic, not a measured speedup or a guarantee of unchanged I/O bandwidth.
- Request every missing bank when the native router confirms demand. Prediction confidence must never suppress required execution or substitute another expert.
- Use explicit valid, queued and in-flight bank masks. `pending == 0` alone no longer means the expert is complete. A partially loaded slot must not become an ordinary complete cache hit or be handed to an unready consumer.
- Preserve the separate speculative pool. Charge its full committed slot capacity even when only one bank has arrived; fewer bytes read do not automatically mean less allocated memory.
- Prevent duplicate reads when a speculative bank becomes demanded. Preserve publication ordering, wake cancellation/failure waiters, and prevent slot reuse until outstanding writes finish. Cancelling a queued read does not cancel an already submitted OS read.
- Make admission sensitive to measured completion deadlines and byte budget, with demand ahead of speculation. Calibrate confidence on held-out data and distinguish accuracy among all predictions from accuracy among actual cache misses. A fixed cutoff on rank already failed in D036.

Benchmark real 0.47/0.92/1.88 MB reads under decode, not only whole-expert microbenchmarks. Smaller transfers can lose enough bandwidth to erase all byte savings. Gate on total token time, exposed wait, energy and actual bytes together.

### Ticket 3: close the compute and power deficit

Files: `native/kernel_bench.cpp`, `native/simd.cpp`, CPU quant kernels and patch 0002, `native/engine.cpp` profiling. Keep the same weights and one-column reference arithmetic.

After the new schedule, profile again. D037's expert row kernels saved only about 3-4 ms/token; their isolated 11-15% gain cannot be applied to the whole decoder. Dense products and synchronization must supply the remaining reduction.

Prioritize exact row blocking for the actually dominant Q5_K, Q6_K and Q8_0 shapes, reuse of activation quantization where inputs truly match, and avoiding redundant dispatch/barriers. Quant format names alone do not identify the expensive shapes. Keep temporary expansion/repacking within the memory budget: saving CPU time while displacing expert cache can lose more time to storage.

Measure sustained kernels and integrated decoding at the same thermal state. Test wait strategies only after ready work is exhausted; blanket sleeping/WFE previously lost performance. An advisory stock-Android performance API would be a separate measured experiment, not a way to override thermal limits.

Gate: useful compute plus synchronization <=170 ms/token in the integrated sustained workload, exposed wait <=25 ms/token, and no quality/lifetime regression. If the best exact kernels still exceed 200 ms, stop calling this single-token schedule a route to five; it needs a materially different compute design.

### Conditional ticket: a learned predictor, only if missed deadlines remain decisive

A small external predictor may rank transfers while the original router alone determines execution. This preserves model semantics even when predictions fail. It is different from changing the model's router and different from drafting answer tokens.

First collect current-layer four-stream features and next-layer ground-truth selections on synthetic/public calibration traces. Compare a small low-rank predictor against the existing mixer/router lookahead on held-out Italian, code, reasoning, long-context and image-derived text states. Account for predictor weights, feature buffers, time, and cache displaced. Freeze and hash any predictor artifact.

The pass criterion is improved ready-bank coverage at the same physical-read budget and lower end-to-end latency, including recovery. An increase in top-k recall alone is insufficient. No predictor may update target weights, alter routing, or omit an expert. Do not fund training until the byte and compute gates show it could change the outcome.

Primary research: [SpecPrefetch v1](https://arxiv.org/html/2607.24787v1) separates transfer prediction from execution routing and schedules by transfer windows. Its reported mobile gains concern a different model and Snapdragon system, with short runs; they establish no Exynos rate. Use the design principle as a candidate, not its performance numbers as our forecast. A port must independently inspect and pin its code/license before reuse.

## Memory admission is a gate, not an optimization claim

The 1,792 MiB `speedCacheMiB` in `Models.kt` is not a fixed maximum. `EngineController.streamPlan()` already allocates available memory up to 3,072 MiB in Auto; `speedCacheMiB` also influences automatic context planning. Do not file a fix claiming that replacing 1792 with 3072 automatically creates more cache.

At 3 GiB cache, the weight sum alone is about 6.293 GiB. Add recurrent state (about 113 MiB), prompt/decode workspace, the speculative pool, committed-but-idle slots, KV/indexer state, mapped embedding and KV pages, app/UI memory, libraries and the existing reserve. At a 512-token prompt batch, workspace was measured around 518 MiB on the host. A lower batch or releasing pages can affect prefill cost; neither is an assumed free gain. D032's `decode_reclaims_compute` already failed to show a speed benefit.

Respect the requested context, vision support and the engine's phase-specific admission checks. Do not obtain a speed figure by shrinking context, clearing other apps, counting mmap as free, or ignoring page cache/zRAM. Measure available memory and reclaim behavior on the phone before committing to a 3 GiB cache. If it does not fit, use the measured smaller-cache floor to decide whether this branch remains viable.

## Branches not selected for the first implementation

| Candidate | Reason |
| --- | --- |
| MTP / more verification rows | Current zero-draft-cost bound is below five; drafts, rollback and displaced cache add cost. Revisit only after an improved verifier demonstrates sufficient margin with realistic acceptance. |
| Prompt lookup | Helps copy/repetition tasks; the ordinary-question 9B check proposed zero tokens. No Flash-Next free-form result. |
| Default Vulkan decode or NPU | D025 measured an unfavorable GPU decode path on the 35B and the engine refuses qwen4exp GPU mode. This does not prove every new GPU kernel impossible, but supplies no supported shortcut. No usable Exynos NPU path is demonstrated. |
| Cache-policy switch | Tested online policies gain only about two points; the Belady result requires future knowledge. |
| More I/O threads or larger reads everywhere | Demand latency, bandwidth, useful bytes and dependencies matter together; prior sweeps already rejected simple versions. |
| Lossless compressed expert pack | The new sampled zlib probe saves under 1%; a specialized codec would first need a convincing measured compression/decompression result. |
| Quantizing the 27 GiB n-gram table further | Only a few rows are read per token; reducing total file size is not equivalent to reducing decode traffic. |

[DwarfStar's inspected Flash-Next documentation](https://raw.githubusercontent.com/antirez/ds4/main/docs/QWEN38_FLASH_NEXT.md) describes a dedicated Metal/CUDA path with different quantization and hardware, and states that SSD expert streaming for this model is not implemented there. It is not an Android drop-in. This URL was inspected on 2026-10-23; it is not a dependency pin.

[EntroLLM](https://arnabsanyal.github.io/icassp2026/entrollm.html) changes the quantization scheme to make weights compressible before lossless coding. That does not establish lossless compression of our existing IQ bytes. [The ANS compression paper](https://arxiv.org/html/2606.15789v1) is another research lead, not a demonstrated phone implementation for these formats.

### Quantization is a separate contingency, not part of the exact promise

Most shared bytes are already Q5_K/Q6_K, not all Q8_0. The four groups of Q8_0 hyper-connection matrices total about 637.5 MiB; halving them cannot free multiple GiB. Several input dimensions are 320 or 640, incompatible with a naive 256-element K-block conversion. Padding a 640-column projection to 768 costs 20% more elements before bit-width savings.

Changing experts, shared projections, routing or output vocabulary requires a separately pinned artifact and quality evaluation. Prefer original higher-precision sources to repeated requantization. Keep sensitive recurrent/mixer/router paths protected until their errors are measured. The current 27-item quality smoke set cannot prove parity for a new Flash-Next quantization.

If exact scheduling fails its feasibility gates, report that fact. Do not silently replace this task with fewer experts, a smaller model, reduced context or a degraded quantization and call it the same result.

## Final acceptance and handoff

Qualify the result on the stock Galaxy S24+ Exynos 2400 using the installed-app execution conditions as well as CLI:

- Same pinned target artifact and pack, native experts and sampling. Test short, 8k and 32k context frontiers where memory admits the original configuration; report each separately. Preserve longer-context functionality through the reference fallback.
- At least three 15-minute natural-EOS sessions with alternating baseline/candidate order, recorded initial conditions and no cooldown within a sustained session. Include the owner's usual charging condition; report battery-powered and charging results separately. Energy is null while charging or when counters are missing.
- Require >=5 emitted target tokens/s over measured decode and in every complete 60-second decode window of the final ten minutes of each qualifying run. Also report per-workload results, p50/p95 latency, TTFT, wall-time throughput, faults, memory, bytes/token, thermal state and energy. Do not hide long prefill gaps behind a decode-only number.
- Demonstrate numerical equivalence appropriate to every change, error/cancellation recovery, no memory overflow and no lost capabilities. Do not average a warm copy benchmark with slow ordinary answers to claim success.

Instructions for the implementing model:

> Read AGENTS.md, docs/progress.md and this plan. Implement Ticket 0 only, preserving all existing work. Do not run forced post-EOS sustained generation. Produce the memory/I/O/compute feasibility table and source/artifact provenance. Then proceed one ticket at a time only while its measured gates justify the next change. Keep all runtime experiments opt-in and preserve the reference decoder. Do not report 5 tokens/s until the final on-device acceptance criteria pass.

No inference code, model weights, app defaults or installed binaries were changed during this planning task. The new outputs are this plan and a small read-only host analysis record.
