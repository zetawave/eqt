#!/bin/sh
# Native Linux x86-64 build and checks for the POSIX expert-store path (mmap reservation, madvise,
# pread, O_DIRECT). Run inside a Debian container with the repository mounted at /src:
#   docker run --rm -v <repo>:/src python:3.11-slim sh /src/scripts/linux_check.sh
set -eu
apt-get update -qq
apt-get install -y -qq --no-install-recommends g++ cmake ninja-build > /dev/null
cmake -S /src -B /src/build/linux -G Ninja -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build /src/build/linux --target eqt-bench eqt-check eqt-simd-test
/src/build/linux/bin/eqt-simd-test
mkdir -p /tmp/models
cp /src/models/eqt-tiny-qwen35moe.gguf /src/models/eqt-tiny-qwen35moe-ff96.gguf \
    /src/models/eqt-tiny-qwen35moe-mtp.gguf /tmp/models/
cd /src
EQT_FIXTURE_DIR=/tmp/models python3 tools/verify_streaming.py --target linux --direct-io \
    --output results/local/streaming-correctness-linux.json
