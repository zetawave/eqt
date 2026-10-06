# Progress — 2026-10-06

## Working implementation

The shared C++20 CPU engine, Windows/Android CLI and Kotlin APK run real inference. The stock Galaxy SM-S926B / s5e9945 / Android 16 has completed chat, Unicode output, descriptor loading, reload, cancellation and recovery tests. No root, unrelated app termination or thermal override was used. The user freed storage to about 30 GiB; this does not increase available RAM.

M0 research and the small-model M1 vertical path are implemented. Full M1 still requires the principal Qwen3.6-35B-A3B checkpoint. M2 bounded expert streaming is not implemented. The 5 tokens/s beyond-RAM target has not been demonstrated.

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

## Current constraints and next experiment

The 2 MiB random-read proxy does not represent 960 scattered bank reads. The conditional arithmetic at 5 tokens/s requires about 3.06 GB/s before alignment overhead, shared weights and compute. Using the observed direct-read range would require roughly 69–80% byte reuse even before those costs; this is a feasibility constraint, not a throughput prediction. Initial available RAM was only about 3–4 GiB, while non-routed target weight spans alone are about 2.38 GiB. A full target copy also exceeds available host disk space. Only its 16 MiB header prefix has been acquired, and its full advertised hash is not locally verified.

Next: collect exact routing on a manageable MoE reference, replay the target-sized bank reads under an explicit cache budget, then integrate a bounded storage backend and compare resident/streamed logits. Keep default routing unchanged. Measure the actual app-private and picker storage paths; the shell-path direct-I/O result is insufficient. Context frontiers (512/2K/4K/8K input), target quality, GPU/NPU, prefetch, MTP and steering remain unverified.

The phone was disconnected on 2026-10-06 and the user requested PC-only work. System-picker selection remains pending: the private-file descriptor path passed, but selecting a Downloads file was not completed. One known fixture was moved from the unused ADB-created external app directory to `Download/eqt-smoke-Qwen3-0.6B-Q8_0.gguf` for that test; the private staged fixture is separate. No personal files were removed. Build artifacts and all weights stay outside Git.
