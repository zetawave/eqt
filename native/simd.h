#pragma once

#include <cstddef>
#include <cstdint>

namespace eqt::simd {
// Single-precision kernels for Equity-owned hot paths (router lookahead). Weight GEMV/GEMM stays in GGML.
float dot(const float *a, const float *b, size_t n);
float sum_squares(const float *x, size_t n);
// out[i] = x[i] * scale * weight[i]
void scale_mul(const float *x, const float *weight, float scale, float *out, size_t n);
// scores[r] = dot(rows + r * n, x) for row-major rows with stride n.
void gemv_rows(const float *rows, const float *x, float *scores, size_t row_count, size_t n);
const char *implementation();
} // namespace eqt::simd
