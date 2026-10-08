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
