# Progress — 2026-10-06

## Flash-Next known-route and memory feasibility screening — 2026-10-23

- Added opt-in diagnostic known-route prefetch for one/two future routed layers, using the existing cache, speculative pool, read priority and in-flight limit. The native router still selects every expert; prompt/decode routes must match the recording in order. Every pre-sampling logit vector can be SHA-256 checked. This is an oracle with free decode prediction, not a production optimization or a bound on every possible scheduler. No app default, weight or dependency patch changed.
- A corrupted-route check exposed that a scheduler callback can stop without a nonzero process return code. Store failures now set the abort flag, and the engine checks it before accepting an evaluation. Truncated/mismatched routes, invalid future layers, repeated oracle requests and oracle options without streaming are rejected.
- Galaxy, USB-powered, D032 options, 1,792 MiB cache, 8k context, batch 128, six compute threads, one synthetic English request capped at 32 tokens with natural EOS enabled: plain **1.96 / 2.16 tokens/s**, one-layer oracle **2.56**, two-layer oracle **2.49 / 2.24**. All 32 logit-vector hashes and token IDs matched; every oracle matched all 1,536 routed calls. Plain reads were **578.1 / 578.5 MB per emitted token**; oracle reads **462.6–464.4 MB**. Two-layer pooled rate rose from 2.052 to 2.361 tokens/s (+15.0%), with 19.94% fewer bytes. These are descriptive short-run results with drift, not a sustained or production speedup. The 37 C cooldown target was not reached within 60 s; measured skin temperatures span 37.2–41.7 C, thermal status 0–1. All requests reached the length cap; none continued past EOS.
- The 32 outputs contain **31 decode forward evaluations**. Oracle traffic is approximately **477.5–479.4 MB per evaluated token**, so even the isolated 2.4 GB/s arithmetic leaves almost no margin below 200 ms. Actual oracle pack busy-time rates were 1.44–1.65 GB/s in this session. These rates describe this schedule, not a universal storage ceiling. Useful-compute and exposed-wait wall time remain null; summed worker waits are not subtracted from wall time.
- Added `eqt-memory-bench` and `tools/bench_memory.py`: persistent workers, verified CPU affinity, checked sums or explicit AArch64 loads, 256/512 MiB buffers, static and dynamic work distribution. Checks pass, but achieved reads are roughly **16–28 GB/s** across these arms while some synthetic dense products remain faster (early Q8_0 median 34.5 GB/s). The probe is **inconclusive as a DRAM roofline**; it cannot establish a 40 GB/s ceiling or support the proposed 55–60 GB/s margin. Memory-controller counters and USB energy remain null.
- Validation: host/Android builds passed; full per-position logit and route comparisons passed on qwen4exp, qwen35moe and the in-graph top-k path, including Android pack/direct I/O and four invalid-route cases. The host regression passed 21 streaming cases across three fixtures plus lifecycle/session checks. Android pack regression and lifecycle checks are recorded in the result file. These fixture timings are not device model performance claims.
- Decision: an I/O opportunity exists (about 20% fewer reads with future knowledge), but these tests do not justify commissioning complete-expert scheduling plus bank masks as a demonstrated route to five. A bounded opt-in I/O prototype may target incremental gains; sustained CPU/energy and workload-specific bytes remain separate gates. The second external analysis is directionally useful, but its 300–320 ms sustained compute, 5–10% scheduler gain, universal 3.5-token/s oracle cutoff and 40-versus-60 GB/s binary verdict are hypotheses. The historical 35% spin profile belongs to the 35B, not a Flash-Next measurement.
- Reproduction: [benchmark instructions](benchmarking.md#known-route-and-cpu-bandwidth-feasibility-probes), [request](../configs/flash-next-oracle-request.json). Measurements/provenance: [feasibility gates](../results/2026-10-23/galaxy/flash-next-feasibility-gates.json). No APK was installed.

## Shared-expert-first screening: rejected increment — 2026-10-23

- Prototyped a single-token graph ordering that expands the resident shared-expert branch after native routing and before routed expert products, to overlap its work with pending expert reads. The original kernels, weighting and reduction order were retained. This was not complete per-expert FFN fusion.
- Six host fixture comparisons (batches 1/16, resident and streamed) retained bitwise first logits and token IDs. A short Galaxy ABBA screening on the real model, 1,792 MiB cache: plain 2.91/2.79 tokens/s; prototype 2.76/2.62. All 32 token IDs agreed. Thermal drift and one repetition per arm prevent attributing a precise slowdown, but there is no demonstrated speed benefit.
- Restored exactly the three source files backed up before the experiment; no new dependency patch or default was retained. The candidate diff is preserved as `results/2026-10-23/galaxy/shared-expert-first-rejected.patch`; measurements and artifact hashes are appended to `flash-next-natural-feasibility.json`. The full expert-FFN scheduler and partial-bank prefetching remain unimplemented.

## Flash-Next device feasibility and I/O counters — 2026-10-23

- Galaxy CLI as the app user, USB-powered, 8k context, batch 128, six compute threads and D032 pack tuning. Natural-EOS sequence: 3,140 tokens over 22.97 minutes of sequence wall time; 2.36 tokens/s decode, 2.28 end-to-end, final ten full decode windows 2.04 tokens/s. Two answers ended at EOS; the code answer reached its 1,536-token cap. The reasoning request was not reached. Expert traffic averaged 628 MB/token; thermal status reached 4. No post-EOS generation. Energy is null while charging.
- Short profiled A/B/A screening: 1,792 MiB cache 2.61 / 2.75 tokens/s and 547 / 555 MB/token; 2,304 MiB 2.32 tokens/s and 517 MB/token. At 2,816 MiB, reads fell to 476 MB/token but speed collapsed to 0.69, with 2,574,547 decode major faults over 32 tokens. The free-memory estimate admitted this arm but did not protect residency. A subsequent 1,792 MiB check recovered to 3.08 tokens/s at a cooler thermal state. These single short runs do not isolate a repeatable speedup or the precise reclaim mechanism. Cgroup memory limits were unreadable; no root or thermal controls were used. No 3 GiB arm was attempted after the reclaim failure.
- Corrected I/O instrumentation: jobs retain their submission phase through draining; decode bank ranges are split by gate/up/down/other and demand/speculative dispatch class. Decode-only ready/prefetch/in-flight hit counters exclude prompts. Workers now publish counters under the completion mutex before exposing zero outstanding jobs; final drain time is reported separately. Non-pack byte accounting explicitly remains logical payload, not physical aligned traffic.
- Host/Android builds passed. Host: 21 streaming cases across three fixtures were bitwise equal, including bank-counter sums and decode-outcome accounting; lifecycle checks and five MTP comparisons also passed. Android: four qwen4exp pack/direct-I/O cases and lifecycle checks passed. Four Python reporting checks passed. Real Flash-Next short cache arms and recovery emitted identical token IDs.
- Record: `results/2026-10-23/galaxy/flash-next-natural-feasibility.json`. The long run used the previous binary while instrumentation was developed; its recorded artifact hashes identify that build, and its late I/O attribution retains the old caveat. Short cache arms used the corrected binary. No APK, model, chat, app default or routing change. Five sustained tokens/s remains unachieved; the plan now rejects free-memory-only cache growth and prioritizes work within 1,792 MiB.

## Flash-Next implementation: natural-EOS measurement increment — 2026-10-23

- `eqt-bench` can repeat a nonempty sequence for `sustained_seconds` on one loaded model, finishing the active request at the deadline. Timed runs reject forced post-EOS generation and require positive output safety limits. Ordinary single requests and one-pass sequences retain their behavior.
- `tools/sustained_run.py` now preserves context/sampling, uses natural EOS, and reports per-request results, termination counts, active-decode minute windows (including final EOS work), prefill, sequence wall time, end-to-end throughput and provenance. Energy is null on the host, while powered, or with missing battery counters. Added four synthetic long-answer evaluation prompts. Explicit pack load options are still required for a moved model.
- Host and Android CLI builds passed. Three focused Python tests passed; five invalid native timed configurations were rejected before model load. Real Qwen3-0.6B host inference produced two natural-EOS 29-token answers on one load, identical to ordinary single-request decoding. This is a functional check, not a Flash-Next/mobile speed result. Local report: `results/local/sustained-natural-host.json`.
- Ticket 0 remains in progress: safe cache capacity and natural-workload speed need the phone. Per-bank/drained-I/O instrumentation and useful-compute/exposed-wait attribution are not implemented; unavailable estimates remain null. Expert scheduling, weight formats and app defaults are unchanged.

## Flash-Next performance handoff: 2026-10-23

- Planning only: [the implementation route](flash-next-5tps-plan.md) specifies feasibility gates and incremental tickets for complete-expert scheduling, selective bank prefetching, safe cache capacity and exact dense-kernel work. Five sustained tokens/s remains unproven; current measurements do not justify a guaranteed result.
- Read-only inspection found two relevant source constraints: speculative pack reads fetch whole experts, while ready-first execution operates within each projection rather than through a complete expert FFN. Auto cache already uses available memory up to 3 GiB; the catalog's 1,792 MiB speed target is not a fixed cap.
- The sustained harness currently forces `ignore_eos=True`. The handoff requires natural-EOS workloads before further sustained-speed claims; the harness itself was not changed.
- New host evidence: verified pinned header; 3,372.20 MiB of non-routed, non-mapped tensor spans. zlib level 1 saved only 0.781% across one sampled expert per layer (144 banks, 90 MB), with byte-identical decompression. Record: `results/2026-10-23/host/flash-next-feasibility-probe.json`. This is a compression-size probe, not inference or Android performance.
- ADB was unavailable. No phone benchmark, runtime implementation, model transformation, app default change or installation was performed.

## Stopped-answer chat history: 2026-10-23

- Explicitly stopped answers now keep their user question and generated text in subsequent requests. Previously, the request builder silently omitted both, so repeated stops made a conversation appear to lose its context. The change also applies to existing saved chats without rewriting them.
- Failed and still-streaming turns remain excluded; reasoning stays outside the model's conversation history. Native decoding, cancellation and cache/session lifetime rules are unchanged.
- Debug Vulkan APK and instrumentation APK built successfully. The focused request-builder check passed on the Galaxy with synthetic messages: completed turns, stopped partial answers, stops during reasoning, repeated stops and mixed failed/completed turns. No model was loaded or generation run. Installed the debug update; saved chat content hashes are unchanged.

## Chat header cleanup: 2026-10-23

- Removed the last-model "Load model" shortcut from the chat header. Models are loaded from the Models screen. No temperature indicator or logo animation was added.
- Debug Vulkan APK built successfully and installed as an update on the connected Galaxy. This UI-only removal does not change inference.

## Ordinary-question lookup check: 2026-10-23

- The owner observes about 4 tokens/s asking questions on the 9B Q4. A short Galaxy CLI check on an ordinary sky-color question gives plain → lookup → plain: 4.61 → 4.33 → 3.70 tokens/s, identical 32 token IDs. Lookup proposes **zero** tokens, so it accelerates none of this answer. The plain drift brackets the lookup rate; it does not establish a precise overhead or slowdown.
- Record: `results/2026-10-23/galaxy/lookup-question.json`. D039/D040's copy gains do not apply to general question answering. No inference code, app default or user preference was changed during this check. Reserving snapshots and scanning without useful matches remains a potential cost; default enablement was premature as a general performance measure.

## App prompt lookup enabled by default: 2026-10-23

- Settings now has Advanced → Prompt lookup, persisted and enabled by default. The app applies it to the pinned Qwen3.5 9B Q4_K_M (catalog or hash-verified local file), with creativity 0, text input and no steering. Models without a measured profile keep their existing decoder.
- Four recurrent rollback slots reserve an additional 192 MiB when they fit beside the existing context and memory margin. Otherwise the model loads normally without lookup; no context is reduced for the optimization. Allocation failure also retries without snapshots. Disabling applies to the next answer; enabling needs a model reload.
- Lookup now uses the normal full-prompt prefill and first sample. First logits match plain decoding on the host fixtures and Qwen3.5-0.8B. Prefix/session eligibility follows the normal path instead of being disabled by lookup. Later batched calculations still carry the numerical caveat of D039.
- Debug APK with Vulkan built; Android lint passed (0 errors, 32 warnings). Focused lookup checks and the small-model lifecycle comparison passed. The tested tiny fixture and 0.8B templates decline prefix/session reuse in both paths; these checks establish matching fallback, not a successful session restore on those models.
- Final Galaxy CLI copy check with the integrated prefill: 6.77 tokens/s with lookup against 4.24/3.58 plain, the same 32 token IDs. The phone entered thermal status 1 at the end; these absolute rates are not comparable to the cooler D039 session. Record: `results/2026-10-23/galaxy/lookup-app.json`. Installed the debug APK as an update and launched it successfully; no weights or chats were removed.

## Short verification screening: 2026-10-23

- Extended `verify_replay` from one proposal to 1–7 proposals (2–8 verification rows including the anchor). Each reached depth has its own deterministic conditional acceptance schedule; rejected suffixes roll back through recurrent snapshots. The final block shrinks to the remaining reference/output tokens. This remains teacher-forced measurement without a drafter, not speculative inference.
- `tools/bench_verification.py` performs five short runs by default: 32-token plain decode, perfect free proposals with 8/4/2 rows, and plain decode again to expose drift. It records actual expert I/O bytes per emitted token, numerical agreement, thermal readings and source/artifact hashes. No cooldown or sustained test. The English workload is `configs/verification-request.json`.
- Host and Android CLI builds passed. Focused host checks passed on the qwen35moe and qwen4exp fixtures: perfect proposals at 2/4/8 rows, 8-row verification at conditional acceptance 0.65 and 0, and rejection of invalid widths, insufficient rollback slots and invalid token IDs (16 checks). Every emitted reference position matched the target's greedy token. The five-arm tool smoke check also passed. These are correctness checks, not mobile performance results.
- Galaxy screening is complete (D039, `results/2026-10-23/galaxy/verification-lookup.json`). Perfect free proposals do not establish a useful MTP gain on the 35B; Flash-Next has modest headroom and some batch-dependent greedy differences. On the resident 9B, four verification rows are promising and eight lose.
- Implemented opt-in greedy `prompt_lookup`: longest matching suffix (3–8 tokens), most recent occurrence, overlapping periodic continuations, full target verification and recurrent rollback. It defaults off and does not use a second model. Prefix/session reuse is disabled while it runs.
- Real 9B Q4_K_M copy decoding on the Galaxy: 9.27 tokens/s against 6.18/5.44 plain at 32 tokens. One 64-token repeat in reversed order gave 9.71/9.36 against 5.37 plain (+74–81%), identical token IDs, thermal status 0. This is a short prompt-copy result, not a free-form or sustained-speed claim. No APK was installed and no inference default changed.
- Focused lookup checks cover accepted proposals, no-match fallback, invalid inputs and a rejected proposal at EOS. Host fixture first logits are bitwise equal; Qwen3.5-0.8B retains sampled token IDs but its first logits differ by up to 0.347. The lookup path is experimental, not claimed bitwise equivalent to plain decoding.


## App UI refinements: 2026-10-22

- Help dialogs now use opaque surfaces; the image viewer has an opaque black background. Chat configuration explicitly uses the opaque dialog surface.
- Settings and chat/model options show short, localized descriptions without opening help. Smaller, muted question marks retain the detailed explanations; simple headings and slider hints no longer add redundant help buttons.
- Model cards restore inline expandable details. Load, download, resume and chat actions sit on the right, with compact gradient/outlined buttons and minimum touch-target spacing; grouped actions wrap on narrow screens.
- Removed the Privacy footer below the composer and the empty-chat motto; the Equity icon and wordmark remain.
- Debug APK built successfully with Vulkan support; Kotlin compilation and Android lint passed (0 errors, 30 warnings). No device installation or visual verification has been performed during this change.

## App contextual help: 2026-10-22

- Explanations in Settings, model details, image support, vector import and chat configuration now open from contextual question-mark buttons. Privacy and context usage use the same custom Compose help dialog.
- The shared help dialog follows the existing dark theme, with gradient accents, a bounded scroll area and a dedicated Close action. Help buttons have localized accessibility labels; existing English and Italian explanations are preserved.
- Operational status, memory shortfalls and invalid numeric input remain visible beside their controls. Inference and settings behavior are unchanged.
- Debug APK built successfully. Kotlin compilation and Android lint passed; lint reports 0 errors and 29 warnings. Visual verification is left to the owner; the APK has not been installed during this change.

## App localization: 2026-10-22

- English is the default Android resource language; Italian follows the device language automatically. Both are declared for Android's per-app language settings.
- Screen copy, accessibility labels, starter prompts, catalog descriptions, notifications and app-owned error/status messages use localized resources. Model prompts, generated answers, saved conversations and native diagnostic output keep their original content.
- Built the debug APK and installed it as an update on the connected phone. No tests or lint were run after the owner requested skipping verification.

## Resume here (state on 2026-10-08; Flash-Next on 2026-10-20)

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
- `tools/runner.py` has this run-as mode since 2026-10-19: `--run-as org.equity.app` on the sweep, quality, sustained and trace tools, with app-relative model paths. It stages into `files/bench`, reads models under `files/models` and reads results back with `run-as cat`. The Flash-Next measurements of D029 ran through it.

**Qwen3.8-Flash-Next (2026-10-19/20):** supported end to end and verified on the host. Controlled CLI decoding on the Galaxy, with identical tokens throughout:

- 1.47–1.58 tokens/s at the start;
- 1.92 with a speculative pool, a two-layer lookahead and a 1,792 MiB cache (D029);
- 2.39–2.41 with demand-first read scheduling, against 2.12 for D029's configuration in the same session (D030);
- 2.89–3.06 (2.99 mean over four runs) with the experts in an expert pack, 2 whole-expert reads in flight, demanded experts in two parts and the Q8_0 lookahead router with a four-row kernel, against 2.45 for D030's options in the same session (D032).

The owner's app observations are 1.8 tokens/s (252 tokens) and, after D029, 2.1 tokens/s (224 tokens). Chat turns reuse the previous turn's state where the template allows it (D027). A chat can keep that state on storage between sessions (D033, opt-in). Flash-Next keeps its attention keys and values in a file on storage, so long windows need only the indexer in memory (D034). Neither has been measured on the phone yet.

**Flash-Next on the phone is moved (2026-10-20, D032).** `eqt-pack move` moved its experts into `files/models/qwen3.8-flash-next-ud-q2kxl/Qwen3.8-Flash-Next-UD-Q2_K_XL.eqtpack`; the GGUF beside it has holes in its routed ranges and loads only with the pack. The installed debug app (2026-10-21, 17:54) passes the pack automatically, with D032's tuning, saved sessions and keys and values on storage. `eqt-pack unpack MODEL PACK` restores the original GGUF byte for byte. `eqt-pack sha256` against the pinned hash has not been run on the phone yet.

**Flash-Next and 5 tokens/s (D037, measured on the Galaxy 2026-10-22):**
- **Storage:** with today's reads (545 MB per token in 64-token runs) and the measured 2.4 GB/s for whole experts, plain decoding cannot pass about 4.4 tokens/s however fast compute gets. 5 tokens/s needs more cache hits or almost no wasted speculation.
- **MTP k=1 is not pursued:** a two-row verification step costs 2.1–2.6 plain tokens even with every draft accepted, and reads rise to 587 MB per token at 0.65.
- **Four-row IQ2_XS/IQ4_NL kernels** (`row_kernels`): bitwise equal on the phone and 1.11–1.15× in isolation, but only about 3–4 ms per token in decoding; off by default.
- **Dense products** fall with the big-core clock under the thermal cap (1,824 MHz within 25 s of full load).
- **First sustained run** (1,800 tokens): 2.19 tokens/s mean, 2.0 after six minutes, thermal status 4. Cache hits fell from 54% to 39% and reads rose to 861 MB per token, but most of that text was degenerate (past EOS), so it needs a natural long-answer prompt.

**Image input for Flash-Next (D038, 2026-10-22):**
- The pinned projector (904 MB, `configs/flash-next-mmproj.json`) is in the catalog as "Add image support". Loads attach it on their own, and "Reload with images" serves a model already loaded.
- A streamed model empties its expert cache before encoding images, so the encoder does not cost text speed.
- On the Galaxy (CLI as the app user) Flash-Next read a synthetic image correctly: first token after 85 s, 4.63 GiB RSS while encoding.
- The debug APK with it is installed, and the projector is already in the app's storage. The in-app flow still needs a look on the phone.

**Catalog, 2026-10-22:** Qwen3.5 9B joins the app in two resident builds, both with the same optional image support:
- Q4_K_M, 5.7 GB;
- UD-IQ3_XXS, 4.0 GB, for less free memory.

`configs/qwen35-9b-model.json` pins them. Model descriptions no longer quote speeds or times (they change with each measurement) and are localized. The 9B has not run on the Galaxy yet.

**Next performance goals, in order:**

1. **Energy per token,** which sets sustained speed:
   - the spin-wait share (35% of cycles); WFE lost in short runs (D016) but may win once throttled;
   - CPU work per token;
   - why storage wait doubles in long runs (82 against about 31 ms per token).
2. **Prefill:** i8mm microkernel register blocking (the kernels are about 42% of prefill cycles), DeltaNet and attention prefill kernels, 8k–32k context sweeps.
3. **Cache policy:** studied on Flash-Next's real traces with the router's ranking (D035). The best online policy gains 2 points over LRU against Belady's 20; not pursued. The speculative pool that never displaces cached experts is in place (D029).
4. **Flash-Next, from D032's measurements** (about 330 ms per token: ~90 ms waiting for expert reads, ~240 ms compute and synchronization):
   - D035: the recurrent-state gather is gone (patch 0007, −10 ms per token in the profile). The 12 ms of ARGSORT are the per-layer graph splits;
   - the dense products read 24–27 GB/s while decoding against 33–38 in short bench runs: mostly the thermal clock cap (big cores 2.3–2.5 GHz within seconds at thermal status 1, D036). Energy per token is the lever, item 1;
   - a quarter of the storage traffic is unused speculation, but filtering it (D036) cut reads 4% without speed;
   - `topk_in_graph` removes the per-layer graph split (exact; −2 ms per token on the Galaxy), default off;
   - `decode_reclaims_compute` was measured and gained nothing;
   - the graph split per layer (about 0.28 ms each);
   - the IQ2_XS/IQ4_NL expert kernels: four-row kernels measured, bitwise equal, about 3–4 ms per token (D037);
   - a sustained run on a prompt that asks for a long answer: does the cache hit rate fall from 54% to 39% in natural long text too (D037)?
   - the BF16 indexer GEMV (scalar, about 7 ms per token; vectorizing changes summation order, so it needs a numerical comparison);
   - misses stay at 13.7% of demanded experts: the late lookahead adds only 4 points of precision.
5. **Exact Q4_K/Q8_0 tiles,** so MTP on the reference build (the 35B) is lossless and faster. Not for Flash-Next, where two-row steps lose (D037).
6. **Quality:** a larger evaluation set before UD-IQ2_M can replace the reference.

## Flash-Next: 2026-10-21 night — the 5 tokens/s analysis, four-row kernels, verification replay (D037)

Checked against the records of D029–D036:
- **Holds:** the per-token breakdown (≈100 ms waiting for reads, ≈145 ms dense weights, ≈55 ms expert compute, the rest overhead), and that storage bytes per token become the binding limit.
- **Does not hold:** that MTP k=1 lowers storage reads per token. Lookahead and demand reads work per row, so a two-row step reads the union of both tokens' experts.
- **Does not hold:** that the dense weights cost "once" for two rows. Exact tiles cost 1.25–1.39× per extra column (D015), and Q8_0/Q4_K have no exact tile.
- **Does not hold:** that IQ2_XS was benched at 12.2 GB/s. That figure is the IQ2_XXS/IQ3_XXS bench.

Implemented (not yet on the phone):
- **`row_kernels`** (patch 0002): four-row IQ2_XS and IQ4_NL GEMV kernels, bitwise equal to the one-row kernels. A static aarch64 check under QEMU passes 8,000 of 8,000 random cases for v8.2 and v9.0; a non-fused variant fails 3,064, so the check is sensitive. Off by default.
- **`verify_replay` with `rollback_slots`:** the cost of k=1 verification steps without a draft head. On the fixtures, target agreement is 1.000 at acceptance 1, 0.65 and 0, resident and streamed. On the real model on the host it measured 1.25 tokens/s against 1.39 plain (record in `results/2026-10-21/host/`).
- **`eqt-kernel-bench --experts` / `--sustain=S`**, and `sustained_run.py --powered`.
- Host: 72/72 streaming cases and 5/5 MTP cases pass; the Android and host builds are clean.

**Phone session** (run on 2026-10-22, results in D037): `python .cache/phone_kernels.py` (USB, phone unlocked, Equity app closed; about 75 minutes, of which about 14 are the sustained run). Steps, in order:
1. `state`, `stage`;
2. `exact`: fixtures, four-row kernels against the resident path;
3. `kernels`: bitwise and GB/s;
4. `sustain_kernels`: the thermal cap, memory against clocks;
5. `io`: storage ceiling for 1.9–15 MB reads;
6. `rows`: decode A/B;
7. `ops`;
8. `replay`: plain, plain with slots, replays at 0.65 and 1.0, alternated twice;
9. `sustained`: 1,800 tokens.

Raw results are in `results/local/galaxy-flash-next/kernels/`; records are `results/2026-10-22/galaxy/flash-next-d037-session.json` and `flash-next-sustained-usb.json`.

## App: 2026-10-21 — answer length, context window, compact Models, context gauge, saved sessions, KV on storage

Built and linted. Not yet seen on the phone.

- **Answer length.** It has no limit by default: an answer runs until the model ends it or the loaded window is full (the engine's new `context` termination; stats say "context full"). Turning the limit on gives a slider and typed field up to 32,768 tokens. Requests leave at most 2,048 tokens of room for the answer (`output_reserve`); the window still grows when a prompt needs more.
- **Context window.** The choices are Auto · 2K · 4K · 8K · Max, Auto by default.
  - Auto gives the KV cache half of the free memory left once a streamed model has its full-speed expert cache (`speedCacheMiB`: Flash-Next 1,792, 35B 2,048), or once a resident model fits.
  - Max takes all of it, with the cache at its floor.
  - Both stop at the trained context (catalog `trainedContext`; for picked files the engine caps the window, `context_up_to_trained`) and never start below 4K.
- **Models screen.** Compact cards show a badge, the name, size and kind, a status, and the primary action. Description, tagline, license, memory details and the Wi-Fi note sit in a collapsible row. Image support, steering and deletion moved to an options sheet behind the ⋮ button. A memory shortfall still shows on the card.
- **Chat.** A small circle beside Send shows the share of the loaded window the chat uses: the last answer's prompt plus answer tokens (saved per message as `context_tokens`), plus tokens arriving. A tap shows the numbers.
- **Saved sessions (D033).** "Remember this chat" in the Chat settings dialog (tune button, beside directional steering), off by default. After each answer the next turn can continue from, the state is written to `chats/<id>/session.state`. Coming back to the chat restores it and evaluates only the new text. The dialog shows the size and deletes the state. Deleting the chat, turning the switch off or deleting the model also deletes it. Thinking turns are not saved.
- **Keys and values on storage (D034).** Flash-Next maps its attention keys and values from an unlinked file in `files/kv`, which disappears with the load. Auto and Max size its window from the indexer's 6 MiB per 1K tokens and from free storage.

Engine: `output_reserve`, the `context` termination and `context_up_to_trained`, checked on the host with Qwen3-0.6B.
- A 256-token window filled exactly (22 prompt + 234 answer tokens).
- A reserve that does not fit is refused with the usual overflow error.
- 100,000 is capped to the trained 40,960.

Engine: sessions and keys and values on storage are bitwise equal to the in-memory paths on the host fixtures, on Qwen3.5-0.8B and Qwen3-0.6B (keys and values) and on the real Flash-Next (a 75-token session restored), and in the Linux container (30/30 cases per fixture, lifecycle 9/9 with both).

**Galaxy session, 2026-10-21 evening** (`.cache/phone_tonight.py`; `results/2026-10-21/galaxy/flash-next-tonight.json`; the latest APK is installed):
- **Fixtures on f2fs:** all cases pass, and the lifecycle is 9/9 with keys and values in memory and on storage. This run found the relative session path bug (D033).
- **Pack attribution** (D032): the pack gives no measurable decode gain this session (2.72 against 2.75 tokens/s mean) and prefill is about 5% faster. `decode_reclaims_compute` gains nothing.
- **Keys and values on storage at 4K:** 2.87 and 2.83 against 2.61 and 2.69 tokens/s, alternated, same tokens (D034).
- **Saved session:** saved in 139 ms (122 MB), restored in 72 ms. Turn 2's first token came in 8.7 s, against 27.8 s rebuilt (D033).
- **Profile:** in D032.

The phone keeps the experts in the pack. Later the same evening (D035): patch 0007 measured on the phone, the cache-policy study, and the dense-weight diagnostics. The installed APK (19:57) includes patch 0007. The UI still needs a manual look.

## Qwen3.8-Flash-Next: 2026-10-19 (branch `experimental/005-Flash-Next-120b`)

**Model.** Qwen3.8-Flash-Next is a `qwen4exp` model: 125B parameters plus a 51B n-gram table, 512 experts with 10 per token, a hyper-connection residual, and Gated DeltaNet with sparse attention. The build is UD-Q2_K_XL, 78.9 GB (`configs/flash-next-model.json`). The local file `models/flash-next/Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf` is the merge of the three official shards; its header, rebuilt from the official shard headers, is byte-identical. What it costs per token is in D026: 3.37 GiB of resident shared weights, 897 MB of experts if every one misses, 16 n-gram rows.

**Engine (D026, D027):**

- streaming support for qwen4exp;
- the n-gram table mapped from the file, also on Windows hosts;
- a lookahead that recomputes the next layer's FFN mixer (77.6% of predicted experts routed on the host);
- prompt batches kept inside the cache budget, since one layer's experts take 0.9 GiB;
- next-layer prompt reads only when the batch uses at least 90% of the experts;
- GPU mode refused for qwen4exp;
- chat turns that reuse the previous turn's state, for every model whose template allows it.

**App:**

- Catalog entry "Qwen3.8 Flash Next · Q2_K_XL" (badge 125B, Qwen Community License 1.0).
- The download fetches the shard headers, checks the merged header's pinned hash, writes the merged file in place while the shards stream, resumes inside the current shard, and verifies the whole-file SHA-256. The phone needs about 80 GB free, not twice that.
- Admission: the cache never starts below one layer (916 MiB), because prompts need that much resident at once. When memory is short, the next-layer prompt allowance is dropped first, then prompts are read in batches of 256, 128, 64 or 32 tokens before the load is refused. Flash-Next's minimum falls from about 5,440 to 4,980 MiB at 128 tokens; the configuration line shows the batch. A picked local file of this exact size gets the catalog model's memory figures.
- `CatalogModel.tuning` passes measured per-model load options.

**Tools:**

- `tools/make_qwen4exp_fixture.py` builds the random-weight fixture;
- `tools/gguf_merge.py` plans and verifies the merge;
- `tools/replay_decode.py` replays decode-only cache behaviour;
- `tools/acquire_model.py --push-local` copies a local model into the installed app: resumable 256 MiB segments, device-side SHA-256, then the app's marker;
- `--run-as org.equity.app` on `sweep_target`, `eval_quality`, `sustained_run` and `collect_traces` runs the CLI as the app user on models in its storage. This was the open task from 2026-10-08; it is written but has not been run on the phone yet.
- `eqt-io-bench` takes `threads`.

**Verified on the host** (16 GB PC; not a Galaxy measurement), records in `results/2026-10-19/host/`:

- all 41 streamed-versus-resident cases are bitwise equal on the four fixtures, including the new qwen4exp one, and the lifecycle checks pass;
- on the real model, two very different cache configurations give bitwise-equal logits and identical tokens;
- the 8 lifecycle checks pass on the real model, bitwise where the evaluation path is the same. There, the second turn reused 75 prompt tokens: identical answer tokens, KL 2.5e-3 against a full rebuild. Batch shape alone gives KL 4.3e-4 on this model (D027);
- the merge plan matches the local file, both in Python and through the app's own Kotlin `GgufMerge`, compiled and run on the JVM against the official shard headers;
- the debug APK (with GPU mode) and the Android CLI tools build without new warnings.

**Routing locality** (real traces, held-out prompts, decode steps):

- LRU serves 37% of expert requests with a 1 GiB cache, 45% with 1.5 GiB and 51% with 2 GiB; that is 567, 493 and 439 MB of expert reads per token;
- decayed LFU and optimized quotas each add about one point; the Belady bound is about 19 points higher;
- two consecutive tokens share only about 4 of 10 experts per layer, so MTP would cut I/O per token by about 3% and is not pursued.

**Model compatibility (D028):** qwen4exp supports model-specific configuration across its hyper-connection streams. Fixture checks and a real-model host smoke check pass. These establish technical compatibility; effects on behavior and answer quality remain unevaluated.

**Owner-reported Galaxy observation (2026-10-19):** the model runs in the app; the supplied statistics screenshot shows 1.8 tokens/s, 252 generated tokens and 6.1 s to first token. Record: `results/2026-10-19/galaxy/flash-next-owner-observation.json`. Prompt length, cache budget, batch, prefix reuse, thinking state, thermal state, peak RSS and energy were not captured and remain null. This is a single app observation, not a controlled speed, sustained or quality result. The website now includes the model alongside both 35B builds, separates host checks from this observation, and retains the 35B scope on its earlier performance figures. Local preview from the repository root: `npm start`, then `http://localhost:8377`.

**Galaxy decode speed (D029, branch `experimental/flash-next-speed-001`):** these are CLI runs as the app user, 64 greedy tokens, one repetition per arm. Records: `results/2026-10-19/galaxy/`.

- **Storage:** direct random reads reach 1.78 GB/s for 479 KB slices and 2.0–2.3 GB/s for 0.9–3.8 MB reads, from queue depth 8.
- **Baseline at 1 GiB:** 1.47–1.58 tokens/s. 26% cache hits, because lookahead reads evicted cached experts.
- **Speculative pool:** lookahead reads sit beside the quota and join the cache only when routed. Result: 1.61 tokens/s, 44% hits, 17% misses.
- **Two-layer lookahead (5 candidates):** 1.68 tokens/s at 1 GiB. At 1,792 MiB it reaches 1.92 tokens/s: p50 488 ms, 54% hits, peak RSS 6.03 GB. Every arm generated the same tokens.
- **Rejected:** `prefetch_count` 6/8, 8 demand I/O threads, a 1 ms spin limit (1.11 tokens/s), and 8+2 two-layer candidates (same speed, 21% more reads).
- **Decode-phase cache:** decoding now uses the prompt-time next-layer allowance as cache. It is unmeasured on the phone because the CLI runs had no allowance.
- **n-gram row prefetch:** applies also to Equity's own mapping. Not measured separately.
- **Catalog:** Flash-Next enables the pool and the two-layer lookahead.
- **Correctness:** 46/46 fixture cases are bitwise equal with the new options.

**Owner app observation after D029:** "Di che colore è il cielo?": 2.1 tokens/s over 224 tokens, first token after 6.4 s. This is a single observation.

**D030 (2026-10-20, Galaxy CLI, 1,792 MiB, 64 tokens):**

- **Read scheduling.** Demand reads come first. Queued predictions are promoted when routed, and unused ones are cancelled before any read. At most 3 speculative reads are in flight. Result: 2.12 → 2.39–2.41 tokens/s in the same session, p50 448 → 386–399 ms.
- **Why.** Demand bank reads take 4.13 → 3.13 ms, compute threads wait 158 → 127 ms per token, and decode reads fall by 4.8 GB. Nine arms gave identical tokens.
- **Plateau.** From 2 to 6 in flight, and with 2 extra candidates, results are within 2.38–2.41.
- **Q8_0 router copy.** It left the predictor's 5.9 ms per layer unchanged on the phone and is not used. The predictor is the next lever.
- **Vectorized gate.** `eqt-simd-test` passes on the Galaxy.
- **Correctness.** 75/75 fixture cases are bitwise equal, and the real model on the host gives identical tokens and bitwise-equal logits. Records: `results/2026-10-20/galaxy/flash-next-d030-ab.json`, `results/2026-10-19/host/flash-next-d030-host-check.json`.
- **Catalog.** Flash-Next adds `read_priority` and `prefetch_inflight: 3`. The installed debug build has it.

**Model configuration in the app (D031):** import is per model, in each model's card in Models. Chats show the configuration button only while the loaded model has a verified package. Configuration values, answer length and creativity each have a slider and a typed field.

Model-specific packages are stored in the app's private storage, using the same layout as an in-app import. The app verifies compatibility natively when the model loads.

Not yet observed in the app: the new configuration screens and their effect on generated answers.

**Not verified:** Flash-Next speed in the app with this build, sustained and thermal behaviour beyond 64 tokens, energy per token, the in-app download, and the quality of this quantization.

**Device session, when ~80 GB are free on the phone** (steps 1–2 and the storage and A/B parts of step 4 are done; see D029):

1. Install the debug build in place, which keeps the installed models: `adb install -r android/app/build/outputs/apk/debug/app-debug.apk`, or `./scripts/android-app.ps1 -Variant Debug -Install`.
2. Copy the model: `python tools/acquire_model.py configs/flash-next-model.json --push-local models/flash-next/Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf --model-id qwen3.8-flash-next-ud-q2kxl`. It resumes if interrupted, and hashing 79 GB on the phone takes a few minutes.
3. Load it in the app (Models). The configuration line shows the expert cache it got; expect about 1–1.5 GiB on a phone with 5.5–6 GiB available.
4. Measure in this order:
   - `eqt-simd-test` (NEON + dotprod Q8_0 path);
   - `eqt-io-bench` over the model file in app storage, `direct: true`, `read_bytes` 479232 and 925696, `threads` 1/4/8/16/32. This number decides the expected speed: about 2 tokens/s at 1.2 GB/s, 3.5–4 at 2–2.5 GB/s (arithmetic, D026);
   - `sweep_target.py --run-as org.equity.app --model files/models/qwen3.8-flash-next-ud-q2kxl/Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf` with `--cool-to`, arms for `io_threads`, `prefetch` on/off and `lookahead_gate`, watching `predict_wait_ms`;
   - a `quality-v1` subset;
   - a short sustained run.

## Thinking mode: 2026-10-19

- The engine reports `thinking_supported` when the model's chat template has `enable_thinking`, and rejects thinking requests otherwise. With thinking on, tokens through `</think>` are reasoning. Results report `reasoning_tokens`, `answer_tokens`, `reasoning_ms` (decode time to the tag), `reasoning_complete` and `first_answer_content_ms`; these were previously null placeholders (`docs/benchmarking.md`). The tag is matched only where the newest piece can complete it. Sampling and prompts without thinking are unchanged.
- The composer now has the text field above a tool row: Attach, Chat configuration, a **Think** chip (saved preference; disabled when the loaded template has no thinking mode) and Send/Stop. Reasoning is stored apart from the answer and never sent back as context. While it streams, a collapsed block shows "Thinking… · N tokens" and only the newest lines, fading upward. Afterwards it collapses to "Thought for X s" and expands on tap. The answer follows below; stats include the reasoning time. The progress notification says "Equity is thinking". A response that ends while reasoning explains that no answer was written.
- `eqt-thinking-check` (Qwen3-0.6B, temperature 0.6, seed 42) passed: 1,124 reasoning and 124 answer tokens, the correct answer after `</think>`, identical seeded repeats, an unfinished 4-token limit and a cancelled run with consistent fields. Qwen3.5-0.8B did not close its reasoning within 1,536 tokens greedy or at 0.6 (repetitive self-checking); this was treated as a small-model limitation and not investigated further. On the Galaxy, the `chat-ui` case covers the streaming preview, completed header, expansion, reasoning-without-answer message, notification and the chip's model-dependent state with synthetic messages. The user observed thinking working on the 35B IQ2_M model in the app; no on-device thinking speed or quality was measured.
- Thinking uses the Settings temperature (default 0, greedy) and output limit (default 512). Qwen recommends temperature 0.6 for thinking, and reasoning consumes output tokens, so long reasoning can end before an answer.

## Chat configuration and blank output: 2026-10-17

- Read-only inspection of a reported blank answer found 2,062 line-feed characters and zero visible characters; its saved stats reported 2,048 generated tokens. The observed cursor-only display was consistent with whitespace-only model output. The original response was preserved and its prompt was not replayed.
- Streaming chat and notifications now distinguish token generation without visible text, expose the live token count, and explain a completed blank response. Whitespace/control/format-only content does not leave a lone cursor. Native generation, model weights and sampling are unchanged.
- The composer has a Chat configuration icon immediately beside Attach. One modal holds the per-chat configuration, with explicit Save/Cancel and read-only controls during active operations. Settings opens the same form. The previous header row and nested Adjust dialog were removed.
- Debug app/instrumentation builds and Android lint passed (0 errors). Galaxy accessibility/UI checks passed waiting state, blank streaming status, live count updates, visible Unicode text updates, blank completion, notification consistency, composer icon placement and the single modal. These use synthetic benign UI messages and no model inference or chat/settings writes. The app was updated in place; the existing blank-response chat was reopened for visual verification. Record: `results/2026-10-17/galaxy/chat-ui-validation.json`.

## Attachment context recovery: 2026-10-17

- Valid document attachments could fail because their tokenized text plus the reserved answer exceeded the loaded context. The engine now reports exact input/output/context counts through a typed JNI exception. The app retries once with a larger context, rounded to 2,048 tokens and bounded by the model and the existing memory admission checks. It preserves the request, sampling options and imported document text.
- Reloading runs on the existing native worker, frees the previous allocations before computing the new expert-cache budget, and retains local-file descriptors until the replacement model is unloaded. Memory budgets now include the larger context for both resident and streamed models. A failed reload leaves an empty engine rather than reporting it ready.
- Generation and attachment-read errors remain visible in the saved answer. Snackbar errors are cleared after display, so clearing state does not cancel the coroutine that displays them. Settings describe the context selector as the starting size.
- Galaxy checks: native inference/reload/typed overflow/cancellation/recovery passed. The attachment integration check imported a synthetic JSON without changing its staged text, expanded 4,096 → 10,240 context for 7,628 input tokens, generated text, rejected a request beyond the model's trained context with a saved explanation, then generated again. Generation was stopped after at least 24 emitted chunks; this is a functional check, not a long-answer quality or performance result. Temporary test chats were removed and settings restored.
- A 21 KB user JSON on UD-IQ2_M with GPU mode grew to 18,432 context and reached generated output, but exceeded the integration test's ten-minute deadline before its 24-chunk stop condition. This is not a passed full-run or a latency claim; long-document prefill remains slow. The private input is not included in the repository.
- Unloading and memory-pressure teardown no longer reset the cancellation flag while generation is running. The unloading phase stays busy until the native worker has released the model. The Galaxy unload-during-prefill regression passed (201 ms in this single functional check).

## GPU session: 2026-10-17 — GPU mode for prompts and images (D025)

- Settings › Performance › Processor: "CPU + GPU" reads prompts, chat history and images on the Xclipse 940 through GGML's Vulkan backend (`libggml-vulkan.so`, built when Visual Studio's C++ environment is present; `scripts/android-app.ps1` sets it up). Answers are still written by the CPU path, unchanged.
- UD-IQ2_M on the Galaxy: 929-token prompt 24.5 → 14.6 s; first token after one image 69.2 → 23.9 s; short prompts unchanged. Record: `results/2026-10-17/galaxy/gpu-mode.json`.
- Decoding on the GPU was measured and rejected: shared memory bus, slower GEMV shaders, per-layer round trips (D025).
- Not verified: the in-app GPU mode beyond loading and a smoke test, UD-Q4_K_M in GPU mode, a quality evaluation of GPU-read prompts, sustained runs.
- Every chat turn still re-reads the whole conversation (no prefix or recurrent-state reuse across turns): the next large prompt-time saving for both modes.

## Decode session: 2026-10-16 — 5.45 → 8.22 tokens/s on UD-IQ2_M (D024)

- Per-op profile of a token, kernel roofline, and four exact changes: six threads, overlapped expert reads (patch 0002 ready-first MUL_MAT_ID hook), lookahead on a predictor thread, file-mapped token embedding (−332/−515 MiB RSS), reserved demand I/O threads. Tokens identical in every Galaxy run; host streaming 29/29 bitwise; text requests bitwise equal to `main`.
- In the app: 4.24 → 6.87 tokens/s after enlarging the auto expert cache (admission margin) and removing UI animation/relayout that preempted compute threads.
- Next, by measured size: demanded-read latency (~0.45 ms wake-up + 1.3–1.8 ms read; awake demand threads implemented, default off, unmeasured); fused per-expert gate/up/down so resident experts overlap a pending read beyond the gate projection; IQ2_XXS/IQ3_XXS NEON kernels (≈30 ms per token, compute-bound at 14 GB/s against 38 for K-quants); the DeltaNet state gather/scatter copies (`GET_ROWS`/`CPY`, ~6 ms; MTP rollback needs care); then sustained and in-app measurements of the combined build.

## App session: 2026-10-16 — images, chats, background answers (branch `feature/multimodal`)

- **Image input** (D021): mtmd linked into the engine; the Qwen3.6 projector is a separate pinned download ("Add image support", 899 MB) shared by both 35B builds and installed on the Galaxy. Text requests are bitwise equal to `main` (host A/B). On the Galaxy the 35B read a newspaper photo correctly; foreground follow-up: first token 26 s, 3.6 tokens/s.
- **Chats** (D022): persistent, side drawer with new/open/delete; photos, camera, text files and PDFs as attachments.
- **Markdown and LaTeX** (D023): answers render headings, lists, tables, code blocks and formulas (commonmark-java + AndroidMath).
- **Background answers** (D022): foreground service + wake lock + completion notification; the thread pool follows the CPUs Android allows (background: 4 little cores, 1.8 tokens/s instead of 0.2).
- Not verified: answers with the screen off after the thread fix, the release (R8) build, the in-app projector download, camera capture, PDF import, and image understanding quality beyond one photo.
- Device: `/data` has under 1 GB free. About 17 GB are old `runner-*` staging folders in `files/bench` and `/data/local/tmp/eqt`; they were left untouched pending the owner's decision.
- The `third_party/llama.cpp` tree had the experimental branch's patches 0003–0005 applied; it was reset to this branch's 0001–0002 (the diff equalled the committed patches exactly; a copy is in `build/backup/`). Switching back to the experimental branch needs its patches re-applied.

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
