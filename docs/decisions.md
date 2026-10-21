# Decisions — 2026-10-05

## D001: Qwen3.6-35B-A3B remains the principal candidate

The official [configuration](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json) confirms the Qwen3.5 MoE architecture: 40 layers, 256 routed experts, eight selected per layer, one shared expert, 2048 hidden width, 512 expert width and a 3:1 linear/full attention pattern. The [model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/995ad96eacd98c81ed38be0c5b274b04031597b0) supplies weights under Apache 2.0. Its published evaluations are not evidence for Equity's quantization, short context, Italian quality or phone speed.

Choose this family for capacity, fine-grained MoE and a usable existing CPU graph. This is provisional: actual quality per response time and storage traffic must decide continuation. Qwen3-0.6B Q8 is only a build/lifecycle fixture; it does not exercise hybrid state or MoE routing. A separately pinned Qwen3.5-0.8B Q8 hybrid fixture in `configs/hybrid-model.json` now passes mapped/allocated and prefill-chunk comparisons on the host. It still has no routed experts.

The candidate quant is [Unsloth UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/a483e9e6cbd595906af30beda3187c2663a1118c), fixed in `configs/target-model.json`. It is mixed precision, not uniform four-bit. A range download of its first 16 MiB and `tools/inspect_gguf.py` found 733 tensors: 19,568,525,312 bytes of routed-expert storage spans and 2,555,013,632 other weight bytes, including embeddings/head/shared blocks. Total file: 22,134,528,992 bytes (20.61 GiB). No MTP tensors were observed in this artifact; the official source configuration has one MTP layer. Vision needs a separate projector and is out of the initial text path. The full file's advertised SHA-256 has not been locally verified because only the header was downloaded.

**First target hypothesis:** measured expert locality must reduce traffic enough to make storage latency tolerable. The all-miss estimate from this artifact is roughly `19,568,525,312 × 8 / 256 = 611,516,416 bytes/token`, excluding shared weights and overhead. At 5 token/s that alone requires ~3.06 GB/s. This is arithmetic, not measured storage bandwidth or a prediction. An observed 3–4 GiB available RAM leaves little expert-cache capacity once ~2.38 GiB of other weights, recurrent/KV state, buffers and UI are included.

## D002: Reuse an immutable upstream runtime before a narrow extension

Pin llama.cpp `6c59c40076c00eab49754dc955d7652d93f9e125`, MIT. Reuse GGUF, tokenizer, Jinja template interpreter, sampling and CPU kernels. Equity owns resource admission, lifecycle, metrics, benchmarks and the future expert-storage path. Do not fork or rewrite numerical kernels until a profile and equivalence test justify it.

Source inspection of `src/models/qwen35moe.cpp` confirms gated DeltaNet/full attention, routed/shared expert graph and an MTP graph. This is source-level support, not successful execution of the main checkpoint. MTP is disabled. `LLAMA_LAZY_MODE_*` only handles architecture-marked tensors and is not a bounded expert cache; the Qwen3.5 expert tensors inspected are not marked lazy. mmap baseline is explicitly not M2 streaming.

Keep the dependency's C++17 requirements local; Equity compiles as C++20. Compile baseline ARM64 NEON without global SVE/i8mm flags. The phone advertises additional features, but instruction-specific dispatch and benefits are separate experiments. Disable weight repacking initially to avoid unmeasured copies. CPU only; no NPU access or Vulkan speed claim.

## D003: Minimal app and honest memory admission

Use Kotlin platform views for the first vertical slice to keep dependencies/build memory small; Compose remains a later UI option. Native work is serialized on a worker. Cancellation alone crosses threads. JNI passes UTF-8 bytes, avoiding modified-UTF-8 corruption. The app holds a seekable file descriptor for the loaded model and does not duplicate weights. No background service: leaving the foreground cancels generation.

Default: 2048 context, 128-token chunks, four threads, F16 context defaults, 2048 MiB admission budget with 512 MiB reserve. The app additionally leaves 512 MiB beyond its requested budget in reported available memory. This is an admission heuristic; it does not constrain page cache/allocators and is not sufficient for loading the 35B. Refuse oversized models until explicit streaming exists. 4K/8K require measured workspace and quality checks; short contexts must not inherit long-context benchmark scores.

## D004: Capacity constraints

Initial device storage: ~17.0 GiB free. After the user's cleanup: ~30.1 GiB free, enough for one target artifact with some remaining space. Initial host `X:` free: ~9.4 GiB, insufficient for a full target copy. Keep only small fixture weights locally. Do not remove personal files or alter stock Android settings. A later explicit streamed acquisition directly to the phone can avoid requiring two 22 GB copies, after the runtime can actually use the target.

## D005: Preserve bank layout in the next storage experiment — 2026-10-06

The initial direct-read proxy measured 0.63–0.96 GB/s for random 2 MiB requests in the shell's storage path. Buffered reads were page-cache hits. Neither is an inference forecast. The verified target header places gate, up and down in separate banks: `tools/plan_experts.py` derives 960 independent ranges per all-miss token, not 320 contiguous complete-expert reads. Exact payload is 611,516,416 bytes/token; independent 4 KiB rounding raises this to 615,448,576 bytes/token.

Use these actual tensor offsets/strides when preparing a replay. Start with exact routing and a bounded byte cache; include alignment amplification, eviction, short reads, shared-weight pressure and total time. Record any coalescing separately. The previous ~2 MiB request size is close to a complete expert's payload but does not capture its three-bank access pattern. Synthetic or small-MoE routes cannot establish target locality; target traces and resident/streamed numerical checks remain exit criteria.

The header prefix now has its own pinned SHA-256 and an explicit range-acquisition option. Servers that ignore the range are rejected before reading their body. This makes the metadata experiment reproducible without acquiring the full 22 GB checkpoint or presenting its advertised full hash as locally verified.

## D006: Exact bounded expert streaming inside the unmodified GGML graph — 2026-10-06

Load streamed models through `llama_model_init_from_user` with the file's own GGUF metadata. This avoids llama.cpp's mmap path, which on Linux/Android maps the whole file with `MAP_POPULATE` and `WILLNEED` (a 22 GB read at load). `TensorLoader` reads every non-routed tensor once into ordinary CPU buffers. Routed banks (`blk.N.ffn_{gate,up,down}_exps.weight`) are placed by `tensor_buft_overrides` in `EQT_Experts`, a host buffer type backed by a reserved address range (`MAP_NORESERVE` anonymous memory; `VirtualAlloc` reserve/commit on Windows hosts). Banks keep the exact GGUF layout, so GGML kernels are unchanged.

The scheduler evaluation callback observes `ffn_moe_topk-N` after routing and before `MUL_MAT_ID`; the CPU `MUL_MAT_ID` reads only experts with assigned rows. Missing (layer, expert) slices are read into place by an I/O worker pool and the callback blocks until they are resident. Routing is never altered. Eviction releases only pages fully inside a slice (`MADV_DONTNEED`), so non-page-multiple strides stay correct.

Cache policy: **layer-partitioned LRU with per-layer quotas**, not a global LRU. The decode access pattern cycles over all layers each token; a global LRU smaller than one token's working set evicts each entry just before reuse. Paging the Experts (arXiv 2609.29032) reports a 0% LRU hit rate at 512 MiB for this model family on iPhone, and the replay test reproduces it. Quotas are uniform by default or derived from calibration traces (`expert_layer_weights`). A prefill batch whose distinct experts exceed a quota overflows transiently and is trimmed before the next layer; peak resident bytes are reported. Direct I/O is optional; unaligned GGUF offsets are widened into a per-worker aligned bounce buffer, and an unsupported `O_DIRECT` is an error, never a silent fallback.

Router lookahead prefetch is optional and off by default: layer `l+1`'s F32 router applied to layer `l`'s post-attention residual (RMS-normalized with layer `l+1`'s weight) predicts speculative reads. Prefetch never refreshes recency, enters with the oldest stamp and yields to demand reads. Published results range from high accuracy (Fate) to no benefit (Paging the Experts, 61% accuracy); keep it as a measured A/B only.

Verified so far: bitwise-identical first-token logits and token IDs versus resident on two random-weight qwen35moe fixtures (target tokenizer/template/quant types), across forced eviction, chunked prefill, prefetch, non-uniform quotas and direct I/O; lifecycle checks pass with streaming. Not yet verified: any Galaxy run, target-model routing locality, throughput.

## D007: One recorded llama.cpp patch — 2026-10-06

In metadata mode llama.cpp created every optional tensor absent from the GGUF (biases, FP8/NVFP4 scales). Filling them with identities would be exact but would add hundreds of graph nodes per token. `patches/llama.cpp/0001-*.patch` returns `nullptr` for an absent optional tensor when the metadata lists tensors, matching file-mode semantics. Bootstrap applies it and verifies that the tree equals revision + recorded patches (hash-pinned in `dependencies.json`). It is an upstreaming candidate; numerical kernels remain unmodified.

## D008: Runtime-selected CPU ISA variants on Android — 2026-10-06

The previous build used baseline Armv8.0 for all GGML kernels, leaving dotprod/i8mm unused on Exynos. Android now builds GGML's upstream variant set (`GGML_BACKEND_DL`, `GGML_CPU_ALL_VARIANTS`: armv8.0, 8.2+dotprod, 8.2+fp16, 8.6+i8mm, 9.0+SVE2, 9.2+SME). Equity reads `AT_HWCAP/AT_HWCAP2`, chooses the highest-scoring variant whose features are all present (GGML's own order) and loads it by soname, which works for the CLI (`LD_LIBRARY_PATH`) and inside the APK namespace. GGML additionally rejects a variant whose score is zero. `EQT_CPU_VARIANT` forces one for A/B; the applied variant and HWCAP features are recorded. The shared libraries require `c++_shared`. GGML's automatic choice on Exynos would be the SVE2 variant; whether 128-bit SVE beats the NEON i8mm variant is an open measurement, not an assumption.

## D009: Persistent GGML threadpool — 2026-10-06

Without OpenMP and without an attached pool, GGML creates and joins worker threads for every graph compute. Expert streaming splits each token graph at every routed layer. Equity attaches one pool per context (`threadpool`, `poll`, `cpu_mask` options). Host x86 indication: +2% resident and +11% streamed decode medians, identical tokens. The Android benefit and the big-core mask/poll trade-off (energy versus wake latency) are pending device A/B.

## D010: Intrinsics where Equity owns a hot loop; no hand-written assembly yet — 2026-10-06

Equity's own numerical code is only the router-lookahead predictor (RMS scaling and a 256×2048 F32 GEMV per layer). It uses NEON intrinsics with eight independent FMA accumulators and a scalar fallback; it is checked against a double-precision reference natively and under QEMU arm64. Quantized GEMV/GEMM stay in GGML, whose kernels already use NEON/dotprod/i8mm/SVE intrinsics once the right ISA variant is selected (D008). Hand-written assembly is deferred until a device profile (simpleperf) shows a specific kernel hotspot that intrinsics cannot schedule well; the beyond-RAM bottleneck is expected to be storage, which must be measured first.

## D011: Galaxy-informed defaults — 2026-10-06 evening

Measured on the target (records in `results/2026-10-06/galaxy/`):

- **Direct I/O with page-congruent placement.** Banks are placed at addresses congruent to their GGUF offsets modulo 4 KiB (buffer `init_tensor`, one extra page per bank), so `O_DIRECT` reads whole pages in place without realigning the file. In the same session it beat buffered reads by 30–65%; the earlier bounce copy had cost 7.9% of user CPU. Recommended streamed configuration: `direct_io=true`, `io_threads=16`.
- **Keep `poll=50`, 4 compute threads, no CPU mask.** Sleeping workers (`poll=0`) were ~20% slower, 6 threads were slower, and a 4-core mask collapsed throughput (0.3 tokens/s), likely by colliding with Android cpusets. Pinning stays available only as an experiment.
- **Router lookahead.** Its prediction is now taken in the top-k callback (the residual is still allocated), adding no graph split. It was 81.8% accurate on the target and cut decode I/O wait by 40%. With bounce-buffer reads, compute slowed by about as much; the zero-copy A/B with and without prefetch is pending. It stays opt-in.
- **Budget.** 2048 MiB beat 1024 MiB at ~3.9 GiB RSS with ~5.8–7 GiB available after a restart. Budgets must still be chosen from measured available RAM, never assumed.

## D012: Native MTP self-speculation, not a separate draft model — 2026-10-06

Speculative decoding fits an I/O-bound MoE engine. One verification batch reads the resident shared weights once and only the union of the experts its tokens select. The draft must share the target's tokenizer; Qwen3-0.6B (151,936 tokens) cannot draft for Qwen3.6 (248,320 tokens). Qwen3.6 ships its own MTP block. The pinned `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` UD-Q4_K_M (`configs/target-mtp-model.json`) adds `blk.40`: 20 tensors, 504 MiB, of which 486 MiB are 256 routed experts. It reuses the target embedding and head. Its trunk tensors match the non-MTP artifact in names, types, shapes and spans; the imatrix chunk count differs, so trunk values may differ.

Equity drives llama.cpp's MTP context (`LLAMA_CONTEXT_TYPE_MTP`) through the public batch API and the `llama-ext.h` hidden-state staging calls, without linking llama.cpp's `common` library (HTTP client, chat parsers). The protocol follows `common/speculative.cpp` draft-mtp for a single head:

1. Prefill all prompt tokens except the last; the head catches up on each chunk with target hidden states shifted by one position.
2. Draft up to `draft_max` tokens greedily on the head.
3. Verify `[last, drafts…]` in one target batch.
4. Accept the matching prefix plus the target's next token.
5. Roll both contexts back past the accepted position. Hybrid DeltaNet state rolls back through `n_rs_seq = draft_max + 1` recurrent snapshots, with no re-evaluation.

The MTP experts stream through the same cache as layer 40. Speculation is greedy-only for now; sampling requests fall back to plain decoding, until exact speculative sampling is implemented. On random-weight fixtures the speculative output equals plain greedy output on host, Linux and the Galaxy. Those fixtures reject nearly every draft, so rollback runs on every step.

The recorded llama.cpp patch also skips `TENSOR_SKIP` tensors in metadata mode, so an MTP artifact can stream with `mtp=false` as a baseline.

## D013: Segment-confirmed device acquisition — 2026-10-06

During the first MTP transfer over wireless ADB, the remote writer stopped after 236 MB while the local `adb exec-in` client kept accepting the rest of the stream. The stream hash matched; the device-side hash check rejected the file. Device acquisition now writes 512 MiB HTTP range segments, one `exec-in` each. It confirms the device file size after every segment, truncates and retries a short segment up to five times (restoring the incremental SHA-256 state), resumes after re-hashing the device prefix, and still requires the final on-device SHA-256.

## D014: Compose app with explicit in-app model downloads — 2026-10-07

The app moves to Jetpack Compose and Material 3. It uses a dark palette and the bundled Inter font (OFL; `licenses/Inter-OFL.txt`). The new Equity mark is an equals sign on a teal-to-violet disc, used for the adaptive icon and the in-app logo. Three tabs: Chat, Models, Settings.

The app now declares INTERNET, but only to download the two pinned catalog artifacts: Qwen3-0.6B Q8_0 and Qwen3.6-35B-A3B UD-Q4_K_M. Downloads start only from an explicit tap. Inference, prompts and conversations stay offline. Each download is a WorkManager data-sync foreground job:

- resumable HTTP range transfer, with redirects followed manually so the range header survives;
- partial file re-hashed on resume, full SHA-256 checked against the pin;
- storage checked before writing;
- the 22 GB model waits for an unmetered network.

Downloads can be paused, resumed and discarded, and continue in the background with a notification.

Loading picks per-model options:

- the small model stays resident;
- the large model streams with an automatic expert cache sized from free RAM, direct I/O (visible fallback to buffered reads) and lookahead prefetch;
- admission includes per-model context cost, after an emulator run showed that ignoring the KV cache invites the low-memory killer.

A large model is released when the app goes to the background. Leaving the app stops generation.

Verified on an x86_64 emulator with ARM translation (UI only, no performance meaning): the Models screen, download with progress, pause and resume, then completion with a matching SHA-256. Not yet verified: loading and chat inside the new app on the Galaxy, and the release (R8) build at runtime. `scripts/android-app.ps1 -Variant Debug|Release [-Install]` builds both; release signing uses an untracked local keystore.

## D015: Exact batched kernels for speculative verification — 2026-10-07

Verifying `k` drafted tokens is one batch of `k+1` columns. GGML's AArch64 dot kernels for Q5_K, IQ2_XXS and IQ3_XXS took one column per call, so every extra column decoded each weight block again. In the DRAM regime (16 distinct 2048×4096 matrices, 4 threads), a 2-column product cost 1.87× (Q5_K), 1.99× (IQ2_XXS) and 1.95× (IQ3_XXS) of one column. These three formats carry UD-IQ2_M's routed experts, attention and shared expert.

The second llama.cpp patch adds 2×2 dotprod tiles (two weight rows × two activation columns) for these types. Each block is decoded once per row and reused for both columns. Every output keeps the single-column accumulation order: integer group sums are exact, and the float expressions are the same, so tiled results are bitwise identical to one-column calls. `eqt-kernel-bench` checks this for MUL_MAT, MUL_MAT_ID with one shared input and MUL_MAT_ID with per-slot inputs, at 1–4 columns. It also measures all batch widths interleaved, because the phone's clock and thermal drift otherwise bias the later widths.

The same patch changes two dispatch paths:

- MUL_MAT used 2×2 calls only when the column count and both chunk ranges were even, so a 3-token batch fell back to one column. It now pairs adjacent columns of the same matrix and handles odd edges with single calls.
- MUL_MAT_ID always used one column. It now pairs consecutive tokens routed to the same expert.

On the Galaxy (v9.0 variant, interleaved medians), the cost of two columns relative to one fell:

| Format | Before | After |
|---|---|---|
| IQ2_XXS | 1.99× | 1.26–1.29× |
| IQ3_XXS | 1.95× | 1.15–1.19× |
| Q5_K | 1.87× | 1.34–1.39× |

Q6_K (SVE i8mm, 1.25×) was already column-exact. Q4_K and Q8_0 still use upstream i8mm 2×2 kernels on v8.6 and later. Those kernels accumulate with separate multiply and add (`vmlaq_f32`), while the one-column path is FMA-contracted, so they remain the only batch-shape-dependent formats. In UD-IQ2_M that is only the Q4_K output head; in UD-Q4_K_M it is most of the experts. On the v8.2 dotprod variant, every format is bitwise equal.

On UD-IQ2_M with a 2 GiB cache, MTP k=1 was slower than plain decoding before the tiles (4.26/4.09 versus 4.32/4.66 tokens/s). With the tiles it ranges from equal to +9%; every arm still produced the plain greedy tokens.

| Sweep | MTP k=1 (tokens/s) | Plain (tokens/s) | Thermal status |
|---|---|---|---|
| `tiles-mtp-sweep.json` | 5.30 / 4.99 | 5.24 / 4.59 | 0, then 1 |
| `fused-mtp-sweep.json` (warmer phone) | 4.48 / 4.51 / 4.53 | 4.74 / 4.31 / 4.44 | 1–2 |

Both records are in `results/2026-10-07/galaxy/`. k=2 stays slower (3.91/4.26 tokens/s): its acceptance is 0.54, and the union of experts grows faster than the accepted tokens.

A k=1 step spends about 27 ms in the MTP head: about 21 ms drafting, mostly the 273 MiB Q4_K output head, and about 5 ms catching the head up on the verified rows. A fused protocol caught up only on accepted drafts inside the draft batch, removing one recomputed head row per step. It measured the same (28 ms per step; the second sweep above) and was reverted to llama.cpp's draft-mtp order.

The limit is the verification step, about 1.49× a single-token step. Two tokens share only about 3.3 of 8 experts per layer, so most expert compute and I/O still doubles.

## D016: Spin-waiting stays; WFE waits were slower — 2026-10-07

A user-space `cpu-cycles:u` profile of plain UD-IQ2_M decoding (`simpleperf`; the kernel denies kernel samples) put 35% of cycles in GGML's thread loops:

- 22% in workers polling for the next graph, between graph computes and during expert I/O waits;
- 13% in barrier spins.

The kernels took 17% (Q5_K), 14% (IQ2_XXS), 9% (IQ3_XXS), 6% (Q6_K), 5% (Q4_K) and 2% (F32).

The spin replacement was an AArch64 `LDXR` + `WFE` wait: the core sleeps until another core writes the watched line. The kernel advertises `evtstrm`, so a missed event costs at most about 100 µs. It was A/B tested on the same binaries, alternating ABBAAB, with identical tokens:

| Variant | Spin, tokens/s | WFE, tokens/s | Change |
|---|---|---|---|
| WFE in barriers and time-bounded polling | 5.04 / 4.77 / 4.42 | 4.23 / 4.04 / 4.12 | −13% |
| WFE in barriers only | 4.82 / 4.35 / 4.33 | 4.44 / 4.37 / 4.18 | −4% |

Spinning keeps the cores busy and their clocks high under stock DVFS; sleeping workers pay wake-up and frequency ramp-up costs. The change was reverted. Power was not measured: the battery reports no discharge while charging. A sustained run on battery could still favour lower-power waits.

## D017: Lookahead prefetch stays at the routed top-k — 2026-10-07

In plain UD-IQ2_M decoding with a 2 GiB cache, 91% of demanded experts hit the cache. The 9% that miss each cost about 1.46 ms of blocked decode. That is 1.4–2.3× the QD1 floor measured with `eqt-io-bench` on the same file (O_DIRECT, p50 0.64 ms per 320 KiB read, 1.09 ms per 1 MiB). Prefetches rarely arrive late (228 in-flight hits). Most misses are experts the router lookahead did not predict: it was right for 80.4% of them.

The `prefetch_extra` option adds the next-ranked candidates of the lookahead score to the prefetch, issued after the top-k. Raising it improved recall but evicted useful entries, so demand misses rose:

| `prefetch_extra` | Correct predictions | Prefetches issued | Wasted | Decode misses | Tokens/s |
|---|---|---|---|---|---|
| 0 | 15,801 | 4,115 | 580 | 1,839 | 5.14 / 4.25 |
| 4 | 17,898 | 8,386 | 3,103 | 1,935 | 4.26 / 4.24 |
| 8 | 18,625 | 13,705 | 7,610 | 2,484 | 3.68 / 3.95 |

Source: `results/2026-10-07/galaxy/prefetch-extra-sweep.json`. Thermal drift was uncontrolled, and the two baseline runs differ by 17%. The option stays, defaulting to 0, with a correctness case in `tools/verify_streaming.py`.

A wider prefetch would need a separate speculative pool that never displaces demanded entries, or eviction informed by router scores (the Belady gap is still about 15 points).

## D018: Prompt processing (prefill) for streamed MoE — 2026-10-08

On the Galaxy with UD-IQ2_M and a 2 GiB cache, a 3,803-token prompt took 521.6 s to prefill (`needle-4k` from `workloads/quality-v1.jsonl`). Only 71 s of that was storage wait, so for long prompts compute, not I/O, was the limit. Four changes, all measured on the phone:

1. **Coalesced expert reads** (`coalesce_kib`, `coalesce_gap`). Adjacent missing experts of one layer load with one read per bank. Runs may bridge up to `coalesce_gap` resident experts, whose bytes are re-read unchanged. Decode misses are rarely adjacent, so decode is unaffected.
2. **Whole-layer prefetch for prompt batches** (`prefill_prefetch_mib`). For batches of more than 8 rows, the next layer's absent experts are read while the current layer computes, beyond the cache quota within a transient allowance. Admission counts that allowance. The reads never bridge: the next layer's entries can be evicted before such a read completes.
3. **Tiled GEMM on AArch64.** Upstream GGML's tiled K-quant/IQ matrix product decodes each weight block once for up to 256 columns, but was enabled only for x86. The second patch enables it on AArch64 with i8mm, using a new USMMLA microkernel: column pairs are interleaved once per microtile, and sums stay exact in integers. Fully decoding a block once and multiplying it against many columns needs at least 8 rows (per expert for MUL_MAT_ID). The engine's prompt batch limit therefore rises from 512 to 2048, so that experts receive enough rows. Decode and speculative verification (at most 9 rows) keep the bitwise-exact 2×2 tiles (D015). `tiled_mm` toggles the path per process, for A/B runs.
4. **NEON GEMM for attention in SVE builds.** The CPU flash-attention tiles call `simd_gemm`, which GGML disables when `__ARM_FEATURE_SVE` is defined: sizeless SVE vectors cannot form its accumulator arrays. The armv9.0 (SVE) variant that the phone selects therefore ran attention on a scalar loop; it was 53% of prefill cycles. The patch adds a fixed 128-bit NEON kernel for all AArch64 builds. Each output element accumulates in the same order, and all 248,320 first-token logits of a 588-token prompt are bitwise identical before and after.

| 3,803-token prompt, UD-IQ2_M | Prefill |
|---|---|
| baseline (batch 128) | 521.6 s |
| + coalesced reads | 540.6 s |
| batch 512, coalesced reads + whole-layer prefetch, vec_dot kernels | 453.7 s |
| + tiled i8mm GEMM | 335.6 s |
| batch 2048, tiled i8mm GEMM | 309.8 s |
| + NEON attention GEMM (default armv9.0 variant) | **164.4 s** |

The 52-token chat prompt prefills in 3.95 s instead of 5.78 s (mean of two cooled runs each): coalesced reads 4.96 s, plus whole-layer prefetch. Records: `results/2026-10-08/galaxy/prefill-*.json` and `neon-gemm-check.json`. Every run produced the same tokens; the needle answer was correct in all of them. The tiled path differs from the one-column kernels only in float rounding: relative error at most 3.4·10⁻⁷ in `eqt-kernel-bench`. Streamed and resident results stay bitwise equal, because both use the same kernels.

At 128 columns, after warming, tiled i8mm against vec_dot gives IQ3_XXS ×3.8, IQ2_XXS ×3.0, Q6_K ×2.0, Q5_K ×1.7 and Q4_K ×1.6. The microkernel runs at about a quarter of the estimated i8mm peak. The remaining prefill profile is:

- matrix microkernels about 42%;
- flash attention 9%;
- the gated DeltaNet recurrence 6%.

Costs:

- **Reads on short prompts.** Whole-layer prefetch reads every absent expert, so a short prompt reads more bytes: 12.3 GB versus 6.4 GB over a 52-token run. Batch 2048 reads each layer once per 2,048 tokens: 19.7 GB versus 65.4 GB at batch 512 for the long prompt.
- **Memory.** Peak RSS was 4.5–4.8 GiB, including the 512 MiB allowance.

## D019: Context bounded by the model, KV cache in admission — 2026-10-08

The engine capped `context` at 8,192 tokens, which blocked long-context work and contradicts the goal of not trading context for speed. The limit is now the model's trained context (`<arch>.context_length`: 262,144 for Qwen3.6-35B-A3B, 40,960 for Qwen3-0.6B); larger requests are rejected with that number.

Admission now counts the KV cache. The estimate comes from GGUF metadata: F16 keys and values of full-attention layers. For hybrid models, those are every `full_attention_interval`-th layer; MTP blocks count conservatively. For Qwen3.6-35B-A3B this gives 22 KiB per token (about 5.5 GiB at 262k); for Qwen3-0.6B, 112 KiB per token, matching its 28 full-attention layers. Results record `trained_context` and `kv_bytes_per_token_estimate` (null when the metadata is missing). `memory_budget_mib` may now reach 16,384 MiB. The 2,048-token cap on `max_tokens` is also gone: a request is bounded only by prompt plus generation fitting the context.

## D020: Slot-addressed expert cache by default — 2026-10-08

Measured cost of the GGUF-shaped expert banks during decoding (UD-IQ2_M, 2 GiB cache):

- about 21,000 minor page faults and 125 ms of system CPU per token;
- about 1.1–1.3 s of `MADV_DONTNEED` on the decode thread per 64-token request.

Every reloaded expert landed on pages its previous eviction had released, so the kernel faulted, zeroed and mapped them inside the read. `eqt-io-bench` with `fresh_pages` measured the penalty on the phone with O_DIRECT, warm device: 320 KiB reads took 0.49 ms into reused pages and 0.75–0.82 ms into fresh ones; 1 MiB reads took 1.18 ms against 1.62 ms.

The slot cache (`expert_slots`, now the default) keeps every resident expert in a persistent, page-aligned slot of an Equity-owned arena. MUL_MAT_ID resolves expert addresses through a new GGML hook (`ggml_cpu_set_expert_address`, also used by the tiled path).

- **Reads.** A slot holds the page-rounded file range of each bank, and the GGUF expert strides are multiples of 4 KiB, so direct reads land whole in the slot without a bounce buffer. Coalesced reads become one `preadv`, plus a 4 KiB copy where neighbours share a page.
- **Eviction** returns the slot to a free list; nothing is unmapped.
- **Memory.** Free slots left by a prompt are released once when decoding starts, so steady-state memory is the cache plus one page per bank per expert.
- **Lifetime.** The hook is installed after `finalize()` and removed before the store is freed.
- **Safety.** An expert that is not resident aborts instead of reading zeros.

A/B on the Galaxy: same binary, alternating, cooled to 37 °C skin before each run, identical tokens in all runs.

| | GGUF-layout banks | Slots |
|---|---|---|
| Decode, tokens/s | 4.91 / 5.28 / 4.96 (mean 5.05) | 5.81 / 5.84 / 4.97 (mean 5.54) |
| p50 per token | 200 / 182 / 196 ms | 170 / 171 / 180 ms |
| Storage wait per 64 tokens | 2.5–2.6 s | 1.9–2.2 s |
| Minor faults per token | 21,013 | 19 |
| System CPU per token | 123–126 ms | 36–40 ms |
| Eviction time per request | 1,059–1,283 ms | about 2 ms |
| Peak RSS | 4.22 GiB | 4.66 GiB |

The higher peak RSS comes from warm slots kept during the prompt; they are released when decoding starts. Correctness: streamed outputs are bitwise equal to resident ones with slots on the host, in Linux with real O_DIRECT, and on the Galaxy, including coalesced `preadv`, shared boundary pages, whole-layer prefetch and MTP. The GGUF-layout path stays available (`expert_slots: false`) and remains in the verification suite. Record: `results/2026-10-08/galaxy/slots-decode.json`.

Tried and not kept: a "v3" i8mm microkernel that interleaved src0/src1 row pairs once per panel and band, through GGML's repack hooks, instead of once per microtile. The long prompt prefilled in 171.2 s against 164.4 s for the v2 kernel, inside run-to-run noise, so the simpler v2 stays.

## D021: Image input through mtmd, projector resident only while encoding — 2026-10-16

Qwen3.6 reads images with its vision projector (`mmproj-F16.gguf`, 899 MB, Unsloth's F16 conversion from the same pinned revision as the language weights). The engine links llama.cpp's `mtmd` library alone (`LLAMA_BUILD_MTMD`, video off), with no change to the recorded patches.

- **Request format.** A user message's `content` may be an array of `{"type":"text"}` and `{"type":"image","path"}` parts, at most 8 images per request. Parts are flattened into the template's string content with mtmd's media marker, so text-only requests render, tokenize and evaluate exactly as before: first-token logits and 32 generated tokens are bitwise equal to `main` on Qwen3-0.6B and Qwen3.5-0.8B, single- and multi-turn (host).
- **Memory.** The projector is loaded per request, every image is encoded first, the embeddings are copied (tokens × `n_embd_inp` floats) and the projector is freed before the prompt is evaluated. Its weights and encoder buffers therefore never coexist with the prompt-time expert prefetch or with decoding. The app reserves `max(prefill allowance, projector + 512 MiB)` instead of their sum and loads text-only, with a visible note, when even that does not fit.
- **Positions.** Image embeddings are decoded by the engine with mtmd's M-RoPE (t, y, x, z) grid positions, the same layout as `mtmd-helper`'s `decode_embd_batch`; the result is token-identical to `mtmd_helper_eval_chunk_single` on Qwen3.5-0.8B (host). The KV cache holds `count` cells while positions advance by each image's `n_pos`.
- **Limits.** `image_max_tokens` defaults to 1024 per image (about one megapixel; the app also stores attachments at most 1536 px on the longer side). Older images in a chat beyond the 4 most recent are described in text instead of re-encoded. Prompts with images decode without MTP drafting, because the head has no token to pair with an embedding row.
- **Cancellation.** Encoding one image does not poll cancellation; the request stops after it.

Galaxy, UD-IQ2_M, app in the foreground, 508 MiB expert cache: a 300-token newspaper photo encoded in about 30 s; a follow-up question including re-encoding answered with the first token after 26 s and decoded at 3.6 tokens/s. The answer read the date and headline correctly. No quality evaluation of image understanding exists yet.

## D022: Persistent chats and answers that finish in the background — 2026-10-16

- **Chats** are stored in app-private storage, `files/chats/<id>/chat.json` plus a `media` folder for attachments, with an `index.json` for the drawer; writes are atomic and ordered on one IO thread, and backups stay excluded. A side drawer lists, opens, creates and deletes chats. A turn is sent back to the model as context only when its answer completed; a half-written answer found on disk after a crash is shown as interrupted.
- **Attachments:** photos (system photo picker), camera captures (FileProvider in the cache), and files. Images are re-encoded as JPEG (EXIF-oriented, longer side at most 1536 px); text files up to 4 MB become text; PDFs give their text on Android 15+ or, when they have none, their first three pages as images.
- **Background.** Loading and answering run under a foreground service of type `specialUse` with an ongoing notification (progress, Stop) and a partial wake lock released when the engine is idle. An answer finished while Equity is not visible posts a notification that opens its chat. Leaving the app no longer stops generation; a large model is still released on the system's background trim once nothing is running.
- **Allowed CPUs.** In the background, even with a foreground service, One UI moves the process to the `moderate` cpuset (cpu0–3, the little cores) and the `low` blkio group. Six spinning GGML workers on four cores stalled each other: 0.2 tokens/s. The engine now checks the allowed CPUs (`sched_getaffinity`) before each graph and attaches a pool of `min(configured threads, allowed CPUs)`; pools are kept per size until unload, because GGML's CPU backend pauses the previously attached pool. The image encoder takes the same limit. Measured in the background with the screen on: 1.8 tokens/s for an image answer (first token 72 s). Equity does not and cannot raise these system limits; foreground speed is unchanged. Changing the thread count mid-answer may change float rounding in reductions; tokens across a change have not been compared.

## D023: Markdown and LaTeX answers rendered natively — 2026-10-16

Answers are CommonMark with GFM tables and strikethrough, parsed by commonmark-java 0.27.0 (BSD-2) and rendered by Equity's Compose code (`ui/MarkdownText.kt`): headings, emphasis, inline code, nested ordered and bullet lists, block quotes, rules, links, horizontally scrolling code blocks with a copy button, and tables whose columns size to their content up to a cap and scroll sideways. Text stays selectable.

Formulas (`$…$`, `$$…$$`, `\(…\)`, `\[…\]`) are taken out before Markdown parsing, which would otherwise consume LaTeX backslashes, underscores and asterisks, and replaced by private-use placeholders. A single `$` opens a formula only when followed by a non-space and closed on the same line by a `$` after a non-space and not before a digit, so "$5 and $10" stays text; code spans and fenced blocks are left alone; an unclosed formula in a streaming answer stays text until it closes. Formulas are drawn by AndroidMath (MIT, iosMath port, FreeType under the FTL) into cached bitmaps: inline ones as text placeholders, display ones as centred blocks; a formula it cannot parse is shown as its source.

JLaTeXMath (used by Markwon's LaTeX plugin) was not chosen because it is GPL v2 with a linking exception; KaTeX would need a WebView per message. AndroidMath is fetched from JitPack, restricted to its group; its POM's runtime AppCompat, test and JitPack placeholder dependencies are excluded, and its `libc++_shared.so` is byte-identical to the NDK 28 one the engine ships. R8 keeps its JNI-resolved classes; the minified release build compiles.

Checked on the Galaxy with an injected answer covering every element above. User messages stay plain text.

## D024: Decode time accounted, six threads, overlapped expert reads — 2026-10-16

A per-op wall-time profile (temporary GGML diagnostic, thread 0, single-token graphs) split one UD-IQ2_M token (2 GiB expert cache, four threads, cool phone, 128 greedy tokens) into: 38.2 ms host time waiting for missed experts, 4.7 ms lookahead prediction and ~6 ms other host work between the 41 graph segments; 41.7 ms expert GEMVs (IQ2_XXS 23.7, IQ3_XXS 16.2); 55.1 ms dense GEMVs (Q5_K 29.1, Q4_K output head 12.4, Q6_K 10.5, F32 routers 3.1); 19.6 ms barrier waits over 2,207 nodes; ~18 ms small ops. Total 183.5 ms (5.45 tokens/s). Per token the model reads ~1.33 GB of dense weights and ~0.29 GB of experts.

`eqt-kernel-bench` on the Galaxy (weights beyond the caches): K-quants reach 33 GB/s with four threads and 37–38 GB/s with six, close to the CPU's DRAM ceiling (Q8_0 39.5 GB/s); IQ2_XXS/IQ3_XXS are compute-bound at 10–11 GB/s (four) and 14–15 GB/s (six); eight threads including little cores add nothing.

Kept, each with identical token sequences on the Galaxy and bitwise-equal host streaming checks (29/29):

- **Six compute threads** (all big cores; app default 6): 5.45 → 6.85 tokens/s, barrier waits 20 → 4 ms. The earlier "four is best" (D011) predates slots and no longer holds.
- **Overlapped expert reads.** In decode, `ensure()` queues missing experts without waiting. A GGML hook (patch 0002) makes MUL_MAT_ID compute ready experts first; the address hook waits only for the bank it reads (per-bank ready bits). Waits spin: sleeping idled the big cores and the governor's lower clocks slowed every later op about twofold (1.72 tokens/s). Host time between segments 47 → 10 ms; 6.85 → 7.63 tokens/s (cool start).
- **Lookahead on a predictor thread**: the residual is copied (8 KiB) and the next layer's router is scored off the decode thread; cache state keeps one writer because every decode-thread entry point waits for the predictor. Segment gaps 10 → 4–6 ms; 7.64 → 8.01 tokens/s in the cleanest pair.
- **Token embedding mapped from the file** (streaming loads): one 1.4 KB row is read per token, so the table no longer occupies anonymous RAM. RSS −332 MiB on UD-IQ2_M (table 333 MiB) and −515 MiB by size on UD-Q4_K_M; the app's admission gives that memory to the expert cache.
- **Four I/O threads reserved for demanded reads**: less waiting in MUL_MAT_ID (146–155 against 172–179 thread-ms per token); 8.22 against 7.82 tokens/s in the cleanest pair.

Best cool-start result: 8.22 tokens/s (121.7 ms per token) against 5.45 at the start of the session. Thermal state and background activity (Play Store updates, system_server) moved single runs by up to 15%; the second pair of each A/B was disturbed and is reported, not dropped. No sustained or in-app measurement of the combined build exists yet.

Rejected: speculative reads in the Linux idle I/O class (6.58 against 6.94 at matched temperature) and a cap of four speculative reads in flight (5.67): both delayed prefetches more than they helped demand reads.

Measured but not yet acted on: demanded reads still wait ~0.45 ms before starting (thread wake-up) and take 1.3–1.8 ms. Awake demand threads (WFE on the queue counter, pinned to little cores) are implemented, default off (`awake_demand_io_threads`), unmeasured.

In the app the same build first decoded at 4.24 tokens/s. Two app-side causes, both fixed:

- **Cache too small.** Admission kept 1,024 MiB of margin on top of a sum that already matched the measured peak RSS, and reserved 512 MiB of encoder work for images (measured: 35 MiB), so the expert cache fell to its 508 MiB minimum and images were switched off. CLI at 512/1,024/1,536 MiB: 6.37/6.81/7.57 tokens/s (74/81/85% hits). The margin is now 384 MiB and image work 128 MiB.
- **UI competing for the big cores.** During generation the UI thread used 47% and the RenderThread 20% of a core at higher priority than the six compute threads, whose barriers then waited for the preempted one. The status dot animated permanently, a typing animation ran during the prompt, and the answer was re-laid out every 60 ms. Now: no animation while answering, updates every 200 ms; the UI thread uses about 6%.

Same prompt in the app afterwards: 6.87 tokens/s, 88% hits, 2,038 MiB cache with images on, 4,684 MiB peak, thermal status 1 at the start. The cache size follows the free memory at load.

## D025: GPU mode for prompts and images; decoding stays on the CPU — 2026-10-17

Question: does the Xclipse 940 GPU (Vulkan, GGML's backend) make Qwen3.6-35B-A3B faster on the Galaxy? Record: `results/2026-10-17/galaxy/gpu-mode.json`.

Decoding: no. The GPU reads DRAM at the CPU's ceiling (row sum 39.3 GB/s; CPU Q8_0 GEMV 39.4) over the same memory bus: CPU and GPU together reach 39.2 GB/s on Q5_K, no more than the CPU alone. GGML's matrix-vector shaders read K-quants at 19–25 GB/s (best of nine tunings) against the CPU's 31–39. Only the compute-bound IQ expert kernels add up (IQ2_XXS 24.8 GB/s together against 11.1), but they are about a quarter of a token, and a streamed decode needs the routed experts on the CPU every layer: 40 CPU–GPU round trips per token at 0.2–0.4 ms each (measured) cost more than the ~12 ms they could save. The driver reports no cooperative matrix and no accelerated int8 dot products, so GGML's integer shaders do not apply (a current glslc was built and checked).

Prompts and images: yes. Prompt-sized GEMMs run at 0.74–1.37 TFLOPS on the GPU against 0.34–0.46 on six cores. The GPU mode, selected in Settings ("CPU + GPU", next model load), keeps the CPU graph and computes chosen nodes on the GPU through a hook in GGML's CPU backend (patch 0002): every compute thread calls it, thread 0 copies the node's sources into staging host memory imported into Vulkan (CPU-cached, so results read back at memory speed), the others wait in WFE, and the result lands in the CPU tensor. Weights stay in their CPU buffers and the expert slots; a routed product stages only the experts its tokens use. No model copy is made for the GPU; staging is freed after each prompt.

Measured choices (929-token prompt unless noted):

- Waiting threads: sleeping 23.1 s (the governor lowered CPU clocks for the following small operations), spinning 21.6 s (it took power from the GPU: GPU time 10.1 → 12.8 s), WFE 19.5 s. Kept WFE.
- Batch 2048 in GPU mode: 16.5 s (fewer passes over the streamed experts, fuller GPU tiles). The CPU path gains nothing from it (24.0 against 24.5 s) and keeps 512.
- DeltaNet recurrences and conv-state concatenations on the GPU: 2.1 → 0.5 s and 1.3 → 0.3 s; total 14.5 s. Flash attention on the GPU was rejected (0.7 → 7.0 s without matrix cores). Sizing MUL_MAT_ID tiles by rows per expert was rejected (no gain).
- Thresholds: dense products and nodes from 128 tokens, routed products from 640. Smaller prompts lost time to staging and synchronization (35 tokens: 4.2 → 5.4 s; routed products at 425 tokens: 4.9 s either way).
- Images: the vision encoder runs on the GPU without flash attention (33.5 → 8.8 s), and the whole prompt (text tokens and image rows) is read in shared mixed batches, one pass over the experts instead of one per text chunk.

Result, CPU path against GPU mode: 35 tokens 4.20 → 4.12 s (nothing offloaded); 237 tokens 6.87 → 6.38 s; 929 tokens 24.5 → 14.6 s (1.68x); one 1024x768 image, first token 69.2 → 23.9 s (2.9x). Decode speed is unchanged (CPU). Peak RSS +0.4–0.45 GiB while a long prompt is read. First-token logits against the CPU path at the same batch size: same top-1 and top-10, KL 1.8e-3 (text) and 2.7e-2 (image), first 8 tokens equal; a full quality evaluation of the GPU mode is still to do. The CPU path is unchanged: the hook is not installed, and the model never uses the Vulkan device.

## D026: Qwen3.8-Flash-Next (qwen4exp) on the streaming engine — 2026-10-19

Model, from the pinned GGUF header (`configs/flash-next-model.json`, Unsloth UD-Q2_K_XL, 78.9 GB): `qwen4exp`, 48 layers, 512 routed experts with 10 per token plus a shared expert, Gated DeltaNet in 36 layers and sparse attention (QSA: a block indexer keeps the top 2,048 cells plus the tail) in 12, a four-stream hyper-connection residual with low-rank mixers, and a 51B-parameter n-gram table (PLE) read at layer 1. Byte budget per token:

| Part | Size | Per token |
|---|---|---|
| Routed experts (IQ2_XS gate/up, IQ4_NL down) | 43.9 GiB, 1.875 MB per expert | 10 × 48 experts, 897 MB if all miss |
| n-gram table (IQ4_NL, 320M rows × 160) | 27.5 GiB | 16 rows of 90 bytes |
| Shared weights (Q5_K/Q6_K/Q8_0, F32 routers) | 3.37 GiB | all of it |
| KV and indexer cache | 30 KiB per context token | |
| Recurrent and PLE convolution state | 113 MiB | |

Engine changes:

- **Architecture accepted.** The pinned llama.cpp already implements `qwen4exp`. Patch 0004 only names the post-attention residual streams `attn_residual`, as qwen35 does.
- **The n-gram table is always mapped from the file**, like the token embedding, never copied or counted as resident. Without the file's tensor list (metadata mode) the model sizes it by the rows its hash heads can address, a prefix of the converter's padded table; the loader accepts exactly that prefix. Windows hosts now map tensors too (placeholder reservations at 64 KiB granularity, `MapViewOfFile3`), so the 79 GB model runs on a 16 GB PC.
- **Lookahead through the mixer.** The router reads the FFN mixer's output, not a normed residual. The predictor applies the next layer's mixer to the current post-attention streams: per-stream RMS norm, the low-rank gate as two Q8_0 products on int8-quantized activations (NEON, SDOT where HWCAP has it), the mean of the gated streams, then the F32 router. Host, 64 tokens: 77.6% of predicted experts are routed, against 71.3% when the streams are averaged without the gate (13.5% more decode misses). The gate stays; `lookahead_gate: false` keeps the cheaper variant, and `predict_wait_ms` reports any time the decode thread waits for the predictor.
- **Prompt batches stay inside the cache budget.** A 512-token batch uses almost all 512 experts of a layer (0.9 GiB). The store used to hold that overflow on top of the budget; now other layers lend entries, in the order a layer sweep reuses them last (D020's slots make this an index change). Peak memory during prompts is the budget plus `prefill_prefetch_mib`, unless one layer alone needs more than the budget.
- **Next-layer reads for prompts only when they pay.** Reading the whole next layer during a prompt wastes reads when the batch uses few of its experts: 52 tokens use about 64% of 512, and the host read 54 GB instead of 31. `prefill_prefetch_min_coverage` reads ahead only when 1 − (1 − k/E)^rows reaches the threshold. It defaults to 0 (unchanged for the 35B, where 52 tokens already use 81%); the catalog sets 0.9 for Flash-Next.
- **GPU mode refused for qwen4exp.** Its node selection was measured on qwen35moe graphs (8,192-channel DeltaNet concatenations). The engine refuses before loading, and the app falls back to the CPU path as for a phone without Vulkan.

Correctness, all with tokens and first-token logits compared:

- **Fixture.** `tools/make_qwen4exp_fixture.py` builds a random-weight qwen4exp fixture from the pinned header: same tokenizer, template and stored types, eight layers, sparse attention below the prompt length, one PLE layer. Its 12 streamed configurations are bitwise equal to the resident run, and the lifecycle checks pass. The qwen35moe and MTP fixtures still pass all 29 cases.
- **Real model, host.** A 4 GiB cache with lookahead and prompt prefetch, and a 1.5 GiB cache without either, gave identical tokens and bitwise-equal logits over 248,320 entries.

Routing locality, from exact traces of the real model on the host: 8 calibration prompts and 6 held-out ones, decode steps only, uniform layer quotas (`tools/replay_decode.py`, `results/2026-10-19/host/flash-next-replay-decode.json`):

| Cache | LRU (native) | Decayed LFU | Belady bound | Expert bytes read per token, LRU |
|---|---|---|---|---|
| 512 MiB | 17.9% | 23.0% | 36.7% | 739 MB |
| 1 GiB | 37.0% | 38.2% | 56.1% | 567 MB |
| 1.5 GiB | 45.2% | 46.3% | 65.0% | 493 MB |
| 2 GiB | 51.3% | 52.3% | 70.5% | 439 MB |
| 3 GiB | 59.5% | 60.4% | 78.0% | 365 MB |

Calibration-optimized quotas add about one point. Pinning the most frequent experts loses points. Neither is worth a policy change now. Consecutive decode tokens share few experts: two tokens use 16.4 distinct experts per layer, three 21.9, four 26.8. MTP drafting accepts 64–66% per draft according to Unsloth's measurements. A k=1 verification would then read about 16.4 experts per layer for 1.65 tokens, only 3% less per token, and k=2 would read about the same. While storage bounds decoding, MTP is not pursued; its separate GGUF also needs an unmerged llama.cpp branch.

Arithmetic, not a measurement: with the 1–1.5 GiB cache the Galaxy can hold, decoding reads about 0.5–0.57 GB of experts per token. Compute is estimated at 160–170 ms per token (3.4 GiB of shared weights at the K-quant GEMV rates of D025, plus IQ2_XS/IQ4_NL experts). The storage rate at queue depth decides the speed: about 2 tokens/s at 1.2 GB/s, about 3.5–4 tokens/s at 2–2.5 GB/s with full overlap. `eqt-io-bench` now takes `threads` to measure exactly that.

Download: the repository ships the build as three gguf-split shards (the first holds only metadata). Their merge is rebuilt from the shard headers alone: the first shard's metadata with `split.count` moved last and set to 0, all tensor infos with offsets recomputed at 32 bytes, then the shards' data sections in order (`tools/gguf_merge.py`, the reference for the app). Built from the official shard headers, the merged header is byte-identical to the local `llama-gguf-split --merge` output (11,024,320 bytes, SHA-256 `2106fcb2…`), and the planned size matches the file. The app's download writes that header, after checking its pinned hash, and appends the shards' data sections as they stream, so the phone needs the file's size, not twice it. The finished file must match the merge's SHA-256 (`f26a5e67…`). Records: `results/2026-10-19/host/`.

Memory fit, measured on the host: a prompt batch's compute buffers are linear in its tokens (518 MiB at 512, 259 at 256, 130 at 128, 65 at 64, at 4,096 context; 534 at 512 and 16,384). The experts one layer holds at once fall with the batch: 1 − (1 − 10/512)^b of 916 MiB, so 843 MiB at 128 tokens and 657 at 64. When the phone's free memory is short, the app first drops the next-layer prompt allowance, then reads prompts in batches of 256, 128, 64 or 32 tokens before refusing the load. For Flash-Next the minimum falls from about 5,440 MiB at 512 tokens to 4,980 at 128. Answers are the same up to float rounding. A prompt longer than the batch takes more passes over the experts: a short first message costs the same at 128 tokens, a 1,000-token prompt about four times the reads.

Not verified: anything on the Galaxy (load, memory, speed, the dotprod path, prompt-cache timing), quality of this quantization, and the in-app download.

## D027: A chat turn reuses the previous turn's state — 2026-10-19

Every request used to clear the context and evaluate the whole conversation again. With Qwen3.8-Flash-Next a prompt batch reads almost every expert of every layer (about 43 GB at 512 tokens), so each turn of a long chat re-read most of the model, once per 512 tokens of history.

The engine now keeps the context's keys, values and recurrent state after a text request that ends at EOS or at the output limit. A cancelled request, or one with model-output transformations, images or MTP drafting, does not keep them. It also keeps the exact tokens they hold: the prompt and the evaluated answer. An answer cut by the limit has its last token sampled but not evaluated, and EOS is never part of it. A new prompt that starts with all of those tokens evaluates only the rest.

Recurrent state cannot be rewound, so any other prompt clears the context and evaluates everything, as before: a new chat, an edited or regenerated message, or a template that renders earlier turns differently. `prompt_cache: false` in a request forces that rebuild. `reused_prompt_tokens` reports the reuse, and `prefill_tokens_per_second` now counts evaluated tokens.

When the reuse applies: turn n+1's rendered prompt must start with turn n's prompt and generated tokens.

- **Qwen3.8 (Flash-Next):** its template keeps an empty `<think></think>` block in earlier assistant turns by default (`preserve_thinking`), exactly what the non-thinking generation prompt produced. Non-thinking chats therefore reuse their state whenever the answer re-tokenizes to the generated tokens.
- **Qwen3.6 (35B):** its template drops that block from earlier turns, so its chats keep rebuilding, unchanged.
- **Thinking mode:** chats rebuild, because Equity does not send the reasoning back.

Numerics. A reused state evaluates the same tokens in other batch shapes: answer tokens one by one (autoregressive DeltaNet) instead of in a prompt batch (chunked DeltaNet). It differs by float rounding, which can flip near-ties among 512 experts. On the real model, host, the lifecycle check's second turn reused 75 prompt tokens. Against a full rebuild of the same conversation: identical 24 answer tokens, the same top-1, 9 of the top-10 shared, KL 2.5e-3, largest logit difference 1.55.

That is the same class of difference that batch shape alone produces without any cache. One 122-token prompt evaluated in one batch and token by token gave KL 4.3e-4, largest difference 0.93, 8 of the top-10 shared, and identical tokens. All other lifecycle checks on the real model (repeat, new chat, conversation replay, cancellation recovery, seeded sampling, reload) remain bitwise equal.

On the random-weight fixtures the generated text does not re-tokenize to the generated tokens, so they exercise the rebuild path only; all lifecycle checks pass. A multi-turn quality evaluation with reuse is still to do. Records: `results/2026-10-19/host/flash-next-lifecycle.json`, `flash-next-batch-shape-sensitivity.json`.

## D028: Hyper-connection model compatibility (qwen4exp) — 2026-10-19

Qwen3.8-Flash-Next keeps four residual streams per token. Model-specific configuration handles every stream, and offline package validation accounts for this layout.

Checks on the qwen4exp fixture cover resident and streamed inference, exact recovery when configuration is disabled, cancellation, reload and invalid requests. Nine offline-tool tests pass. A real-model host smoke check produced a finite, dimensionally compatible 47-block package (blocks 1..47, width 2560).

Several sequences in one batch can change sparse-attention block selection when prompts exceed the model's attention budget. Offline collection therefore evaluates such prompts separately. Equity's inference runs one sequence.

These are technical compatibility checks. Effects on behavior and answer quality have not been evaluated for this model.

## D029: Flash-Next decoding on the Galaxy: speculative pool, two-layer lookahead, decode-phase cache — 2026-10-19

First controlled measurements of Qwen3.8-Flash-Next on the Galaxy S24+ (Exynos 2400). The CLI runs as the app user on the model in app storage (`tools/runner.py --run-as`), with the same 52-token prompt, 64 greedy tokens, 6 threads, 16 I/O threads, direct I/O and a 1 GiB expert cache unless stated. Each arm ran once. Two runs of the same baseline gave 1.58 and 1.47 tokens/s, so differences under about 8% are noise. Records: `results/2026-10-19/galaxy/flash-next-decode-ab.json` and `flash-next-storage-sweep.json`.

**Storage.** `eqt-io-bench`, direct random reads over the model file. Values are GB/s, with p50 latency in parentheses:

| Read size | 1 thread | 4 | 8 | 16 |
|---|---|---|---|---|
| 479 KB (one IQ2_XS gate or up slice) | 0.60 | 1.38 | 1.45 | 1.78 (3.9 ms) |
| 926 KB (one IQ4_NL down slice) | 1.05 | 1.90 | 2.01 | 2.00 (6.2 ms) |
| 1.88 MB | 1.23 | 2.17 | 2.19 | 2.10 |
| 3.76 MB | 1.50 | 2.14 | 2.31 | 2.20 |

Storage saturates at about 2–2.3 GB/s from queue depth 8. That is the upper end of D026's arithmetic, not a 1.2 GB/s phone.

**Baseline: storage is busy, but on the wrong reads.** At 1 GiB the cache served only 26% of demanded experts and 35% missed. Lookahead reads evicted cached experts to make room for predictions, and 30% of those predictions were never used. A simpleperf profile of the decode threads put about 41% of samples in `wait_bank`, 18% in GGML barriers and 30% in kernels.

**Speculative pool** (`speculative_pool`). Lookahead reads no longer evict. They go to a pool beside the layer's quota, at most one prediction (top-k plus `prefetch_extra`) per layer, about 60 MiB in total. A pooled expert becomes an ordinary cache entry when the router selects it, and unused ones are dropped once their layer has run. Quota trimming and excess counts exclude pooled entries. Same series:

- 1.47 → 1.61 tokens/s;
- cache hits 26% → 44%;
- misses 34% → 17%;
- expert reads 52 → 41 GB.

**Two-layer lookahead** (`lookahead_layers: 2`, `lookahead2_count`). After the urgent prediction for layer l+1, the predictor applies layer l+2's gate-less mixer and router to the same streams and reads that layer's top `lookahead2_count` candidates into its pool. Reads that were still in flight when demanded fell from 23% to 16% of demand at 1 GiB, for 1.68 tokens/s; that step alone is within noise. Precision of the l+2 prediction by rank: 0.91, 0.88, 0.80, 0.75, 0.69, 0.61, 0.55, 0.49, 0.42, 0.35. Five candidates keep the precise ranks.

**Cache size.** With pool and two-layer lookahead, a 1,792 MiB cache measured:

| Metric | Value |
|---|---|
| Speed | 1.92 tokens/s |
| p50 token latency | 488 ms |
| Cache hits | 54% |
| Prefetch hits | 20% |
| In flight when demanded | 12% |
| Misses | 14% |
| Peak RSS | 6.03 GB |

**Rejected:**

| Variant | Speed | Why |
|---|---|---|
| `prefetch_count` 6 or 8 | 1.48 / 1.44 tokens/s | Fewer wasted reads, more misses. |
| 8 demand I/O threads | 1.53 tokens/s | No gain. |
| `bank_spin_us` 1000 | 1.11 tokens/s | Sleeping waits lose clocks, as in D024. |
| Two-layer lookahead with 8 candidates plus 2 extra | 1.918 against 1.915 tokens/s | Same speed, with 21% more reads and 73% more unused predictions. |

**Decode-phase cache** (`decode_uses_prompt_allowance`, default on). The next-layer prompt allowance (`prefill_prefetch_mib`) is cache quota while decoding. A prompt takes it back through D026's borrowing, layer by layer. In these CLI runs the allowance was 0, so this change is unmeasured on the phone; in the app it adds the allowance to the decode cache.

**Also in this change:**

- **n-gram row prefetch.** Patch 0004 now prefetches a token's 16 n-gram rows also when Equity maps the table itself. It is a madvise hint, with no numerical effect, and was not measured separately.
- **Ready-first order.** MUL_MAT_ID already computes ready experts first (patch 0002, D024); nothing changed there.

Where the time goes at 1,792 MiB:

- 522 ms per token;
- storage is busy 67% of decode wall time, at about 1.65 GB/s while busy, reading 0.58 GB per token including unused predictions;
- compute threads wait about 170 ms per token.

That leaves about 350 ms of compute and synchronization per token, about twice D026's 160–170 ms estimate. Reads alone would allow about 3.5 tokens/s at 2 GB/s with full overlap. That is arithmetic, not a measurement. The next levers are therefore CPU time per token (barriers, IQ2_XS/IQ4_NL expert kernels) as much as fewer reads.

Correctness: on the qwen4exp and qwen35moe fixtures, all 46 streamed-versus-resident and lifecycle cases are bitwise equal, including the new `speculative_pool_prefetch`, `speculative_pool_chunk16_ranked` and `lookahead_two_layers_pool` (`results/2026-10-19/host/streaming-correctness-pool-lookahead2.json`). On the phone every arm produced the same 64 tokens.

Catalog: Flash-Next sets `speculative_pool`, `lookahead_layers: 2` and `lookahead2_count: 5`, with `prefill_prefetch_min_coverage: 0.9`. The app's automatic cache, plus the decode-phase allowance, decides the cache size. With 6.5 GB available, about 1.9 GiB is expected while decoding; this is an estimate.

Not verified:

- speed in the app with this build;
- sustained and thermal behavior beyond 64 tokens (thermal status rose from 0–1 to 1–2 within each run);
- energy per token;
- quality of the quantization.

## D030: Flash-Next read scheduling: demand first, few speculative reads in flight — 2026-10-20

**Where decoding waited** (D029, 1,792 MiB):

- Compute threads waited about 5.4 ms on every layer with a missing expert. A demanded read queued in the storage device behind up to 12 speculative reads.
- GGML barrier time was mostly the other threads waiting for that one.
- The predictor took about 6 ms per layer of the ~11 available.

**Changes.** All are exact: routing never depends on them.

- **Read priority** (`read_priority`). Reads are served from three queues: demand, then the next layer's predictions, then the two-layer lookahead's.
  - A queued speculative read moves to the demand queue when its layer routes to the expert (promoted).
  - A lookahead read that the next layer's own prediction also lists moves up a queue (confirmed).
  - A pooled read no worker has started is dropped when its layer has run without it (cancelled). Only reads with every bank still queued are dropped, so no compute thread can wait on a missing bank.
- **Speculative reads in flight** (`prefetch_inflight`). While decoding, at most N speculative reads are in the device at once. Prompt reads are not capped.
- **Predictor.** The next layer's reads are queued before the two-layer score is computed, and the mixer gate's sigmoid uses a NEON polynomial exponential. `eqt-simd-test` on the Galaxy passes against double precision.
- **Q8_0 router copy** (`lookahead_router_q8`, kept off). It lowers predictor bytes by 44%, but the predictor's time per layer on the Galaxy did not change (5.94 against 5.97 ms). The predictor is not limited by router bytes, so the 67 MB copy is not used.
- **Cache policy, offline.** On the real decode traces at 1,792 MiB, segmented LRU adds at most 1.3 hit points and a cache shared across layers loses 3, so neither is pursued.

**Galaxy, same session.** One repetition per arm, 64 tokens; all nine arms generated the same tokens. Records: `results/2026-10-20/galaxy/flash-next-d030-ab.json`.

| Arm (1,792 MiB, pool, two-layer lookahead) | Tokens/s | p50 ms | Demand bank read, ms | Compute wait per token, ms | Decode reads |
|---|---|---|---|---|---|
| reference (D029) | 2.12 | 448 | 4.13 | 158 | 37.1 GB |
| + Q8_0 router | 2.08 | 474 | 4.06 | 155 | 36.9 GB |
| + read priority | 2.21 | 429 | 4.12 | 155 | 36.1 GB |
| + at most 6 in flight | 2.31 | 394 | 3.44 | 132 | 34.4 GB |
| + at most 3 in flight | 2.41 | 386 | 3.13 | 127 | 32.3 GB |
| read priority, 3 in flight, F32 router | 2.39 | 399 | | | |
| same, 2 in flight | 2.38 | 400 | | | |
| same, 3 in flight + 2 extra candidates | 2.40 | 399 | | | |

**Why it helps.** With 3 reads in flight, demanded reads finish a quarter sooner. Decode reads fall by 4.8 GB, mostly because 2,760 unused predictions are cancelled before any read, and the storage stays busy 78% of decode time. Cache hits do not change (54%). From 2 to 6 in flight, and with extra candidates, the gain levels off at 2.38–2.41 tokens/s.

**Correctness:**

- all 75 fixture cases are bitwise equal, including the capped and uncapped read-priority cases with promotion, confirmation and cancellation;
- on the real model on the host, the reference, Q8_0-router and all-options arms give identical tokens and bitwise-equal first-token logits (`results/2026-10-19/host/flash-next-d030-host-check.json`);
- with the Q8_0 router, 77.14% of predicted experts are routed, against 77.22% with F32.

**Catalog.** Flash-Next adds `read_priority` and `prefetch_inflight: 3`.

**Next lever.** The predictor still takes about 6 ms of a ~8 ms layer on its small core, so the next layer's reads start late. Faster prediction, or splitting it across cores, would leave more time for those reads.

**Not verified:** speed in the app with this build, sustained and thermal behavior beyond 64 tokens, energy per token.

## D031: Model-specific configuration import — 2026-10-20

- Each model's card in Models offers Import, Replace and Remove for its configuration package. Association uses the exact model-file fingerprint; incompatible packages are rejected.
- A local GGUF opened from the phone's files gets its own card while loaded. Verifying its fingerprint enables import.
- Chats show the configuration button only while the loaded model has a verified package. Incompatible settings are disabled when another model is loaded, and answer details record the effective configuration.
- Configuration values, maximum answer length and creativity can be set with sliders or typed fields.


## D032: Flash-Next experts in an expert pack, moved in place; a faster lookahead — 2026-10-20

**Where decoding waited** (D030, 2.4 tokens/s): storage was busy 80% of decode time at only about 1.6 GB/s. Each expert arrived as three small reads (479, 479 and 926 KB, one per GGUF tensor) at low queue depth. `eqt-io-bench` on the same file gave 2.17 GB/s for one 1.88 MB read at queue depth 4, against about 1.6 GB/s for that mix.

**Expert pack** (`native/expert_pack.h`, tool `eqt-pack`):
- A file beside the GGUF holds the routed experts expert-major. Within a layer, an expert's banks follow each other (gate, up, down), each padded to 4 KiB, so one page-aligned read fetches a whole expert.
- The pack records the GGUF offsets, types and shapes it was built from. The store refuses a pack built from any other file.
- With `expert_pack`, a slot mirrors the packed expert byte for byte, so direct reads land in place with no partial-page copies.
  - Speculative reads take whole experts.
  - `expert_pack_split` reads a demanded expert as 2 or 3 concurrent parts (whole banks, split by size).
- The GGUF stays the source of every other tensor.

**Moving the experts in place** (`eqt-pack move`). The phone had 28 GB free, and a copy needs 46 GB. For each layer, `move`:
1. writes the layer into the pack;
2. re-reads it from storage (direct I/O) and compares both the packed bytes and the source bytes against what it read from the GGUF;
3. records the layer atomically in `PACK.move`;
4. only then deallocates the whole blocks of its ranges in the GGUF (hole punch). The partial blocks at a range's ends keep their bytes.

Peak extra space is about one layer. An interrupted move resumes.
- `eqt-pack unpack` writes the experts back, last layer first, truncating the pack behind it. It is also resumable, and leaves the GGUF byte-identical to the file before the move.
- `eqt-pack sha256` hashes the logical model (the GGUF with its experts read from the pack), for comparison with the pinned SHA-256.
- The engine refuses a GGUF with holes in its routed ranges unless a pack is given (SEEK_HOLE), so a moved model never computes with zeroed experts.
- The app passes `<model>.eqtpack` automatically when it exists beside the model, and deleting the model deletes it too.

On the Galaxy the move took 199 s for 46.08 GB of experts (48 layers). Free space went from 28.3 to 27.7 GB. The GGUF now occupies 32 GB, the pack 46.2 GB.

**Lookahead kernel.** The Q8_0 GEMV of the predictor processes four rows per pass, with the FP16 scales converted as one vector; it used to work row by row with a scalar conversion per block. Predictor time per layer on the Galaxy:
- 5.8–5.9 ms → 4.1 ms with the F32 router;
- 3.05 ms with `lookahead_router_q8`, which D030 rejected because the old kernel gained nothing from it. The decode thread's wait for the predictor fell from 11–14 to 2–3 ms per token.

**Galaxy, 64 greedy tokens, one repetition per arm, the same 64 tokens in every arm** (1,792 MiB cache, pool, two-layer lookahead, read priority). Records: `results/2026-10-20/galaxy/flash-next-d032-pack.json`.

| Arm | Tokens/s | p50 ms | Compute wait ms/token | Decode reads |
|---|---|---|---|---|
| D030 options, before the move | 2.45 | 385 | 117 | 32.8 GB |
| pack, 3 in flight | 2.59 | 369 | 104 | 37.0 GB |
| pack, 2 in flight, split 2 | 2.65 | 360 | 99 | 35.3 GB |
| pack, 1 in flight, split 2 | 2.70 | 357 | 108 | 31.6 GB |
| **pack, 2 in flight, split 2, Q8_0 router** | **2.89 / 3.06 / 2.94 / 3.05** | 323–334 | 87–92 | 35.2 GB |

The best configuration measured 2.99 tokens/s on average over four runs, against 2.45 before (+22%). The 52-token prompt took 10.4–11.7 s, against 13.9 s.

**Why the pack gave less than estimated.** Storage still delivered about 1.6 GB/s while busy: the queue stays shallow, at 1–3 speculative reads plus the demanded ones. Higher depth raises throughput but delays demanded reads in the device. Misses stayed at 13.7% of demanded experts in every arm.

**Rejected** (each within noise or worse):
- awake demand threads (2.81: they compete with the predictor on the little cores);
- 8 demand threads (2.97);
- split 3 (3.00);
- 1 in flight (2.99);
- the predictor split over 3 threads on the little cores (`lookahead_threads`: 3.01 against 3.05);
- the same with polling (`lookahead_spin`: 2.27, CPU contention).

Both options are implemented, default off, and produce identical predictions for any thread count.

**Late lookahead, measured and not pursued** (host, real model, `results/2026-10-20/host/flash-next-late-lookahead.json`). Scoring layer l+1 from layer l's output streams, just before l+1's attention, raises mean precision only from 77% to 81% (rank 10: 0.43 → 0.45). Layer l+1's attention moves routing more than layer l's MoE does. `lookahead_late: 1` records the precision only.

**A GET_ROWS patch withdrawn.** Splitting a single-row F32 `GET_ROWS` (the 3 MB DeltaNet state gather) across threads was bitwise equal on the host. On the Galaxy, with identical tokens, it measured:
- with one predictor thread, 2.25 tokens/s with the patch against 3.05 without;
- with three, 2.96 against 3.01.

The interaction is not explained, so the patch is withdrawn.

**Also in this change, not yet measured on the phone:**
- `profile_ops`: decode wall time by operation through the CPU backend's node hook.
- `decode_reclaims_compute`. After a prompt, the compute buffer's pages are returned (`MADV_DONTNEED`, the buffer holds no state between evaluations). After the first decode step, the pages decoding touched are counted (`mincore`). The rest, minus a 32 MiB margin, becomes expert cache until decoding ends; then the cache trims to the prompt quota and releases the freed slots before the next prompt. Patch 0005 adds the scheduler's buffer accessors.

**Correctness:**
- Host: the 17 earlier streaming cases plus the pack and parallel-lookahead cases are bitwise equal to the resident path on the three fixtures.
- Galaxy (f2fs, direct I/O), on fixture copies (`tools/verify_pack_move.py`):
  - move, interrupted move, resume, unpack and interrupted unpack;
  - SHA-256 through the pack equal to the original;
  - refusal without the pack;
  - 18 bitwise streaming cases.
- Linux container (real O_DIRECT, `madvise`, `mincore`): all 29 cases on each of the three fixtures are bitwise equal, including the pack and `decode_reclaims_compute` cases, and so are the 5 MTP cases; move and unpack pass as on the Galaxy. Decoding touched about 1 MB of a fixture's 128 MB compute buffer and lent 94 MB to the cache.
- Records: `results/2026-10-20/galaxy/pack-move-galaxy.json`, `streaming-correctness-pack-galaxy.json`, `results/2026-10-20/host/streaming-correctness-pack-parallel.json`, `results/2026-10-20/linux/`.

**Not verified:**
- the in-app load with the pack;
- `eqt-pack sha256` on the phone's 79 GB model;
- `decode_reclaims_compute` on the phone;
- sustained and thermal behavior;
- energy.

The phone's AP reached 58 °C and the battery 47 °C during this session on USB power. The slow runs with the withdrawn patch (2.25–2.42 tokens/s, one of them without cooling) are excluded from the table.

**Attribution, 2026-10-21** (`results/2026-10-21/galaxy/flash-next-tonight.json`). Same harness; the battery was at 28% and 42 °C on USB, and every arm produced the same 64 tokens. The options are pool, two-layer lookahead, 1,792 MiB cache, read priority, 2 in flight and the Q8_0 router.

| Arm | Tokens/s | Prefill, 52 tokens |
|---|---|---|
| without the pack (`eqt-pack unpack`, 144 s) | 2.71 / 2.79 | 12.1 / 11.7 s |
| with the pack, split 2 (`move` again, 190 s) | 2.86 / 2.74 / 2.61 / 2.69 | 11.0–11.5 s |
| with the pack and `decode_reclaims_compute` | 2.80 / 2.33 | 11.7 / 13.1 s |

- The pack's decode gain is not measurable this session: 2.72 against 2.75 mean. D032's +22% came mostly from the Q8_0 router and the in-flight limit. Prefill is about 5% faster with the pack.
- `decode_reclaims_compute` shows no gain and stays off.
- Every arm ran slower than on 2026-10-20 (2.99 for the best one), with the phone warmer and charging.

A cool `profile_ops` run (2.76 tokens/s, 362 ms per token profiled) puts the time in:
- the IQ2_XS expert products (132 ms, including their wait for reads);
- Q5_K (58 ms), Q8_0 (31 ms), IQ4_NL experts (23 ms), Q6_K (22 ms);
- the Q4_K output head, a single node (17 ms);
- ARGSORT (12 ms over 47 nodes);
- F32 products (12 ms), GET_ROWS (10 ms) and Gated DeltaNet (9 ms).

## D033: Saved chat sessions, opt-in per chat — 2026-10-21

**Why.** D027 keeps a chat's state in memory between turns, but only while the model stays loaded and no other chat answers. Coming back to a chat later meant evaluating the whole conversation again. On Flash-Next that costs about 11 s per 52 prompt tokens on the Galaxy (D032).

**Request.** `session: {path, restore = true, save = true}`. The engine keeps two files:
- `path`: llama.cpp's sequence-state file. It holds the tokens, the KV cache cells and the recurrent state.
- `path.json`: a record with `format` (`equity-session`), `version`, `engine`, `model_sha256`, `architecture`, `tokens` and `bytes`. `engine` is the llama.cpp revision, a 12-hex hash of each applied patch (from `dependencies.json`, at configure time) and the state-file version.

**Restore.** It runs before the reuse decision, only when the context does not already hold a usable prefix. The result reports one `session.status`:
- `in_memory`: the context holds a usable prefix, so the file is not read;
- `restored`: the file was read (it also reports `restored_tokens`, `bytes` and `restore_ms`);
- `none`: no file;
- `not_requested`: reuse does not apply (images, steering, MTP, `prompt_cache: false`), or the request said not to restore.

A file is read only when:
- its record matches this engine and this model's SHA-256;
- its tokens are a strict prefix of the new prompt;
- the tokens loaded back equal those in the header.

Otherwise:
- a session of another model is kept for that model (`other_model`);
- an unreadable, incompatible or non-extending session (an edited or regenerated turn) is deleted (`unreadable`, `incompatible`, `mismatch`), and so is one that fails to load (`failed`). The prompt is then rebuilt as without a session.

**Save.** After a turn whose state D027 keeps, and only for a model with a known SHA-256. The state is written to `path.partial` and the record to `path.json.partial`, then both are renamed. A crash therefore leaves the previous session (an earlier prefix of the same chat) or none. The result reports `saved_tokens`, `saved_bytes` and `save_ms`, or `save_error`.

**Saved only when the next turn will use it.** Before writing, the engine renders the conversation with this answer and a following user message, and checks that the template reproduces the context's tokens. If it does not, the result says `save_skipped: not_extendable`. That happens when the template renders earlier answers differently (the 35B and the small Qwen3 models, D027) or when the answer re-tokenizes differently. Without this check, those chats would write a file every turn only to delete it on the next. `force: true` writes anyway; the checks use it on the random-weight fixtures.

**App.**
- The Chat settings dialog (the tune button, beside directional steering) has a "Remember this chat" switch per chat, off by default. It shows the saved size and token count and has a "Delete saved state" button.
- The file is `chats/<id>/session.state`. It is deleted:
  - with the chat;
  - when the switch is turned off;
  - from the dialog;
  - when the model is deleted (`deleteSessionsOf` matches the record's SHA-256).
- Thinking turns send no session: their history drops the reasoning, so the next prompt never extends the state.
- A resumed answer's stats say "resumed N saved tokens".

**Size.** On Flash-Next a session holds the 36 DeltaNet layers' recurrent state, which has a fixed size, plus the attention and indexer cells (30 KiB per token, an estimate from the cache shapes). The host check's 75-token session measured 120.3 MB. A fixture's is 0.27 MB.

**Correctness** (`eqt-check`, case `session_save_restore`):
1. The first turn is saved with model hash `a`.
2. With hash `b`, the session is kept and not read.
3. After a reload with `a`, the restored second turn must equal the in-memory reuse bitwise, with the same reused count. If the template allows no reuse, the session must be deleted and the rebuild must equal the plain rebuild.
4. The check also requires the save decision to match the in-memory reuse.

Results:
- Real Flash-Next on the host: the state was judged extendable and saved, then `restored` (75 tokens), bitwise equal.
- Host fixtures, with the attention keys and values in memory and on storage: the not-extendable, forced-save, other-model and mismatch paths pass.
- The Linux container passes the same lifecycle checks with real `mmap`, `fallocate` and hole punching.
- Galaxy (f2fs, fixtures): lifecycle 9/9 on each fixture, with the keys and values in memory and on storage. The first run there found a bug: a session path without a directory (`result.json.session`) wrote nothing, because libc++ fails `create_directories("")`. The app always passes absolute paths.

**Galaxy, Flash-Next with the pack** (`results/2026-10-21/galaxy/flash-next-tonight.json`):
- Turn 1 (52 prompt tokens, 64 generated) saved 115 tokens, 121.6 MB, in 139 ms.
- A new process restored them in 72 ms. Turn 2 then evaluated 25 of its 140 prompt tokens: time to first token 8.7 s.
- The same turn rebuilt from scratch took 27.8 s, with the same tokens.

Long sessions are not measured.

## D034: Attention keys and values on storage for sparse attention — 2026-10-21

**Why.** On Flash-Next the context costs 30 KiB per token:
- the 12 QSA layers' keys and values, 24 KiB (2 KV heads × 256 × 2 × F16 × 12);
- the block indexer's pooled keys, 6 KiB.

At 128K tokens that is 3.75 GiB, on a phone where the expert cache and 3.4 GiB of shared weights already compete for memory. QSA reads, per layer and decoded token, only the 2,048 cells its indexer selects (512 blocks of 4) plus the tail. The CPU flash-attention kernel skips masked cells without touching their keys or values (`ggml_compute_forward_flash_attn_ext_f16_one_chunk`, the decode path).

**Design** (patch 0006, `native/kv_file.cpp`, load option `kv_file_dir`):
- Patch 0006 adds `llama_set_kv_buft_override`, with which the host chooses the buffer type of each KV cache layer by cache tag. Equity hooks only the attention cache (tag `""`). The indexer (`idx_`) is scored over every cell for every token, so it stays in memory.
- The buffer type maps a file shared, from `kv_file_dir`:
  - The file is created with `mkostemp` and unlinked immediately (Windows: `FILE_FLAG_DELETE_ON_CLOSE`). Its storage is released when the context is freed or the process ends, also after a crash. No file outlives the model load.
  - The whole size is reserved at load (`posix_fallocate`), so a full device fails the load, not a write during generation.
  - Clearing the cache releases and re-reserves the file's blocks (punch hole), so the zeros cost no writes.
  - `MADV_RANDOM`: reads fetch the selected pages, not their neighbours.
- The result:
  1. the indexer and the recurrent state stay in RAM;
  2. the remaining keys and values live in flash;
  3. the indexer chooses the blocks;
  4. only the missing blocks are read.

  The block cache is the kernel's page cache. A 4-cell block of one layer's keys is 4 KiB, one page, and so is its block of values. Blocks reused across tokens stay cached. Under memory pressure the kernel evicts them, instead of the app being killed. The kernel writes new tokens' cells back to storage (24 KiB per token).
- The data are the bytes llama.cpp would hold in RAM, read by the same kernels, so outputs are bitwise equal by construction. That is tested, not assumed.

**Admission and sizing.**
- The engine counts the attention keys and values as at most 128 MiB of memory (the page allowance). It reports `kv_on_storage: {directory, reserved_bytes}` in the configuration.
- The app enables it for catalog models with `kvStorageMiBPer1k` (Flash-Next: 24 MiB on storage and 6 MiB in memory per 1K tokens). Auto and Max then size the window from the 6 MiB of memory per 1K tokens. The window is also bounded by free storage minus 2 GiB. A 128K window, for example, needs 768 MiB of indexer memory plus the allowance, and reserves 3 GiB of storage while loaded.

**Correctness:**
- Host, 24 greedy tokens with logits: bitwise equal with the cache in memory and on storage, on 4 models (tiny qwen4exp with QSA, tiny qwen35moe, Qwen3.5-0.8B, Qwen3-0.6B). No file was left after exit.
- Real Flash-Next on the host, 4,096-token window, 16 greedy tokens: logits and tokens bitwise equal. The file reserved 96 MiB, exactly 4,096 × 24 KiB, so only the attention keys and values went to storage. Decode measured 1.75 and 1.68 tokens/s, one host run each: not a phone measurement.
- `verify_streaming` adds `kv_on_storage_prefetch` and a second lifecycle check (cancellation, reload, sessions) with `kv_file_dir`. Every case passes on the three fixtures, on the host, in the Linux container and on the Galaxy (f2fs, `/data/local/tmp`).

**Galaxy, short context** (4,096-token window, a 96 MiB file in the app's `files/kv`, arms alternated): 2.87 and 2.83 tokens/s on storage, against 2.61 and 2.69 in memory, with the same tokens. At this length the file costs nothing measurable. The first attempt failed because `files/kv` did not exist yet (the app creates it), and the error blamed storage or memory. The engine now creates `kv_file_dir` and reports the allocation's own error.

**Not yet measured, and how it could cost speed:**
- Long-context decode. Pages that expert reads evict must be read back. The worst case is 2,048 cells × 2 KiB × 12 layers = 48 MiB per token, against about 900 MB of experts if every one misses.
- Prefill. The tiled prompt kernel reads every tile that holds a selected cell for any query of the tile.

These measurements belong to a later session. A long prompt is expensive to measure on the phone: Flash-Next reads about 46 GB of experts per 512-token batch.

## D035: Where Flash-Next decoding goes on the Galaxy; recurrent state read in place — 2026-10-21

Records: `results/2026-10-21/galaxy/flash-next-diagnostics.json`, `results/2026-10-21/host/flash-next-cache-policies.json`.

**Per token, from the 64-token profile and the store's counters** (about 345 ms):
- about 100 ms waiting for expert reads;
- about 145 ms in the shared dense weights;
- about 55 ms computing experts;
- the rest is mostly per-layer graph splits (12 ms, counted on the router's ARGSORT) and the recurrent-state copies.

Storage is busy about 80% of decoding. It reads about 559 MB per token: 124 MB demanded and 435 MB speculative, of which about 140 MB is never used.

**Patch 0007, kept.** For one token of one sequence, llama.cpp gathered each DeltaNet layer's 3 MiB state out of the cache on a single thread (`GET_ROWS`), then wrote the new state back. The state is now read in place from the cache row it is written back to. A small custom op first copies the source row there when it is another row: a new sequence's zeroed row.
- The pattern applies to every DeltaNet model (qwen35, qwen35moe, qwen3next, qwen4exp). MTP rollback slots and multi-token batches keep the gather.
- Each state row has one reader, which runs before the write-back; graph reuse already keys on the head cell.
- Builds with and without the patch are bitwise equal: first logits and 48 greedy tokens, on the three fixtures and Qwen3.5-0.8B, for a long and a one-token prompt. The full host suite (69 cases, lifecycle with sessions, MTP) passes.
- Galaxy profile: `GET_ROWS` fell from 10.45 to 0.92 ms per token, and the custom op takes 0.16 ms. `CPY` fell from 5.4 to 4.6 ms. The profiled total fell from 362 to 347 ms per token.
- Decode speed is within run-to-run noise: 2.60 and 2.90 tokens/s with the patch, against 2.61–2.86 earlier the same evening.

**Cache policy from the router's ranking, rejected** (host, exact traces of the real model, `ranking` = the router's top 64 per token, recorded by the store's trace). At the phone's 1,792 MiB (20 experts per layer), on held-out prompts:

| Policy | Decode hit rate |
|---|---|
| LRU (native) | 48.4% |
| decayed LFU | 49.4% |
| evict the worst rank at the latest token | 49.8% |
| a calibrated next-token probability from the latest rank | 48.4% |
| that probability plus a per-token decayed frequency (best of 4 settings) | 50.7% |
| Belady bound | 68.0% |

The rank predicts the next token's reuse only weakly: 61% for rank 0, 23% for rank 9, 8% for rank 20. Belady's advantage comes from reuse further ahead, which the past does not reveal. Two points are not worth a policy change.

**Dense weights run below their isolated rate.** In decoding the Q5_K, Q8_0 and Q6_K products read about 24–27 GB/s. `eqt-kernel-bench` reads the same formats at 35–39 GB/s with six threads on the same phone the same evening. Two explanations were ruled out:
- **The predictor's DRAM traffic.** A diagnostic run without the lookahead lowered the dense products by only 3–5% (Q5_K 56.5 → 55.0 ms), while decoding slowed to 2.31 tokens/s.
- **Thread placement.** Compute threads on cpus 4–9 (X4 and A720), with or without every I/O thread on the A520 cores, measured 2.75/2.78 and 2.64/2.89 tokens/s against 2.60/2.88 for the defaults. Prefill was slower: 14–16 s against 12. The experimental option was removed.

The cause of the gap is still open. Identical configurations differ by up to 10% between runs, so effects of a few percent need repeated arms.
