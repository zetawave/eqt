# Third-party notices

Equity's original source is MIT licensed under `LICENSE`. Model weights have separate terms and are not included in this repository or APK.

| Component | Revision / version | Terms and retained notices |
|---|---|---|
| llama.cpp, GGML and the selected upstream Jinja/JSON support sources | `6c59c40076c00eab49754dc955d7652d93f9e125` | MIT, `licenses/llama.cpp-MIT.txt`; original source notices remain intact in the checkout. |
| nlohmann JSON, vendored by the pinned llama.cpp checkout | Version in that checkout | MIT, `licenses/nlohmann-json-MIT.txt`. |
| Kotlin standard library/build plugin | 2.1.20 | Apache 2.0, [JetBrains license](https://github.com/JetBrains/kotlin/blob/v2.1.20/license/LICENSE.txt); original dependency JARs retain their notices. |
| Gradle wrapper/tooling | 8.13 | Apache 2.0, [Gradle license](https://github.com/gradle/gradle/blob/v8.13.0/LICENSE); wrapper scripts preserve upstream headers. |
| Official Qwen3-0.6B GGUF fixture | `23749fefcc72300e3a2ad315e1317431b06b590a` | Apache 2.0, `licenses/Qwen3-0.6B-Apache-2.0.txt`. Acquisition manifest retains model source, revision and SHA-256. |
| Qwen3.5-0.8B GGUF hybrid fixture, quantized by Unsloth | `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` | Apache 2.0; [pinned quantizer repository](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/tree/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0), [original model](https://huggingface.co/Qwen/Qwen3.5-0.8B). No weights are redistributed. See `configs/hybrid-model.json`. |
| Qwen3.6 candidate and Unsloth quantization | `configs/target-model.json`, official revision in decisions.md | Publisher metadata says Apache 2.0; no full weights distributed/acquired. Preserve source license and quantizer attribution on future distribution. |

The license texts in `licenses/` are included as APK assets. This does not convert other models' licenses into MIT/Apache: in particular Qwen3.8 has its own named community license and is not part of the implementation.
