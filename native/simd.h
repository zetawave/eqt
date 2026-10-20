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
// Quantizes x to signed 8-bit blocks of 32 with one scale each (amax / 127, round to nearest), as GGML prepares
// the activation of a Q8_0 dot product. n must be a multiple of 32; q holds n values, scales n / 32.
void quantize_blocks(const float *x, size_t n, int8_t *q, float *scales);
// scores[r] = sum over blocks b of d(r, b) * scales[b] * dot(q(r, b), q(b)) for GGML Q8_0 rows (34-byte blocks of
// an FP16 scale and 32 int8 values) of width n, row r starting at rows + r * row_stride.
void gemv_q8_0(const uint8_t *rows, size_t row_stride, const int8_t *q, const float *scales, float *scores,
               size_t row_count, size_t n);
// out[i] = mean over s < streams of normed[s * width + i] * sigmoid(gate[s * width + i]); the NEON path
// evaluates e^x with a polynomial (a few ulp), the scalar path with std::exp.
void gated_mean(const float *normed, const float *gate, float *out, size_t width, size_t streams);
const char *implementation();
} // namespace eqt::simd
