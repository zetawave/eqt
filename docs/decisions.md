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
