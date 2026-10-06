# Equity engineering rules

Read PROMPT_INIZIALE.md and docs/progress.md before substantial changes. Chat in Italian; write code, comments, documentation and UI text in English.

- Preserve existing work. Implement small, executable increments; never label a scaffold as inference.
- Use C++20/NDK/CMake for the shared engine, Kotlin for Android and Python/PowerShell for offline tools.
- Keep ownership explicit and JNI thin. Run inference/import off the UI thread. Serialize native lifetime operations; cancellation must remain callable concurrently.
- Keep hot paths free of avoidable allocations, copies and locks. Optimize only against a measured bottleneck.
- Use descriptive names, focused functions and standard formatting. Comments explain numerical/layout/lifetime invariants and non-obvious constraints only.
- Pin dependency and model revisions. Preserve license notices. Do not commit weights, personal conversations, large datasets, builds or device identifiers.
- Runtime is offline: no network permission, telemetry, accounts or silent downloads. Acquisition and export are explicit operations.
- Stock Android only. Never root, change firmware, kill unrelated apps, clear user data or override thermal limits.
- Report missing counters as null. Distinguish author results, local measurements, estimates and hypotheses; never extrapolate desktop/Snapdragon results to Exynos.
- Memory mapping is not a bounded expert cache. Account for page cache, shared weights, recurrent/KV state, workspace and temporary copies.
- Preserve routing/sampling semantics in the reference path. Compare numerical outputs before claiming equivalent optimization. Quantization/pruning/steering require separate quality evaluation.
- Validate inputs and failure paths. Run relevant checks after changes; do not add tests that merely duplicate trivial implementation.
- Keep 5 sustained tokens/s for a model larger than available RAM as an experimental target, never a promised result.
