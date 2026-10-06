# Equity (eqt)

An offline Android inference research engine targeting Galaxy S24+ / Exynos 2400. The goal is useful inference with a capable model larger than available RAM; **5 sustained tokens/s is an experimental target, not a promise**.

## Current status

- A shared C++20 CPU engine and native benchmark CLI run real Qwen inference on the stock Galaxy.
- A 15-minute Qwen3-0.6B Q8 repeated-request run completed 53 requests / 13,568 tokens: weighted decode **19.19 tokens/s**, range 17.94–20.29, peak RSS about 885 MiB. This is a small resident fixture, not the 35B or beyond-RAM performance.
- The Kotlin app builds and runs real chat on the Galaxy. Native instrumentation passes reload, cancellation and recovery checks.
- Qwen3.6-35B-A3B is the main research candidate. Its pinned mixed Q4 GGUF is 22,134,528,992 bytes. Only its metadata prefix has been acquired; target inference, a bounded expert cache and streaming are not implemented yet.
- Dense and hybrid fixtures pass host numerical/state checks. Target expert byte ranges are now validated from the header; storage tests and next steps are recorded in [progress](docs/progress.md).
- M0 research is documented. The small-model M1 vertical path works; the principal checkpoint and M2–M4 remain pending. See [progress](docs/progress.md), [decisions](docs/decisions.md), [research](docs/research.md) and [benchmark protocol](docs/benchmarking.md).

## Build and run

Windows prerequisites: Git, Python 3.12, Android SDK 36, NDK `28.2.13676358`, SDK CMake `3.22.1`, JDK 17+ (tested with 21), `ANDROID_HOME`, and ADB on PATH. Host build additionally needs Visual Studio 2022 C++ tools. Dependencies are pinned in `dependencies.json`; Gradle wrapper uses 8.13.

```powershell
./scripts/bootstrap.ps1
python tools/acquire_model.py configs/smoke-model.json
./scripts/build.ps1 -Target Android -Jobs 2
python tools/device.py
python tools/benchmark.py --target android --repetitions 3
./android/gradlew.bat -p android assembleDebug assembleDebugAndroidTest
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

Acquire weights explicitly; build and inference do not download models. App runtime has no Internet permission, analytics, account requirement or cloud inference. Model acquisition requires Internet; inference is offline. App backup is disabled.

Use **Open GGUF** to select a local seekable file. For a reproducible debug fixture, run the CLI once as above, then install and run the instrumentation APK. It explicitly copies the verified fixture from `/data/local/tmp/eqt` into app-private storage using test-only shell access:

```powershell
adb install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w org.equity.app.test/org.equity.app.NativeSmokeTest
adb shell am start -W -n org.equity.app/.MainActivity
```

Tap **Staged model**, then **Send**. Production selection needs neither ADB nor test privileges. The picker attempts direct descriptor access without copying weights; providers whose descriptors cannot be reopened are rejected. Actual system-picker selection is still unverified; the private-file descriptor path passed instrumentation. ADB-created external files were not readable by this app on the tested firmware, so pushing into Android/data alone is insufficient. A single GGUF is supported; pipes, split models and multimodal input are unsupported. Hashing reads the entire model and warms filesystem caches. Debug staging makes one explicit additional ~610 MiB fixture copy, not a copy of arbitrary imported models.

CPU, mmap, context, threads and the resident admission budget are reported after load. The budget is a conservative file-size admission check with workspace reserve, **not an enforced RSS limit**. Expert-cache controls are absent until there is a real bounded cache. Stop cancels native CPU execution; each subsequent request rebuilds the prefix. Leaving the activity cancels ongoing work. Conversations live in process memory; export is explicit.

```powershell
./scripts/build.ps1 -Target Host -Jobs 2
python tools/benchmark.py --target host --repetitions 3
python tools/verify_runtime.py --target host
python tools/acquire_model.py configs/hybrid-model.json
python tools/verify_runtime.py --target host --model models/Qwen3.5-0.8B-Q8_0.gguf
python -m unittest discover -s tools -p 'test_*.py'
```

Results default to ignored `results/local/`. Only public workloads and small selected records belong in Git. The app's export contains generated text; review before sharing. Full model weights and build artifacts remain ignored.

Code license and weight conditions are separate. See [third-party notices](THIRD_PARTY_NOTICES.md). No license is inferred from a model's name; use the pinned artifact's license.
