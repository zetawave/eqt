#include "simd.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define EQT_NEON 1
#endif

namespace eqt::simd {
#ifdef EQT_NEON
// Four-wide FMA latency is ~4 cycles with two or more pipes on Cortex-A720/X4, so loops keep at
// least eight independent accumulators live. Summation order differs from the scalar path.
float dot(const float *a, const float *b, size_t n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float sum_squares(const float *x, size_t n) {
    return dot(x, x, n);
}

void scale_mul(const float *x, const float *weight, float scale, float *out, size_t n) {
    const float32x4_t factor = vdupq_n_f32(scale);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), factor), vld1q_f32(weight + i)));
    }
    for (; i < n; ++i) {
        out[i] = x[i] * scale * weight[i];
    }
}

void gemv_rows(const float *rows, const float *x, float *scores, size_t row_count, size_t n) {
    size_t r = 0;
    // Four rows share each activation load; two accumulators per row cover FMA latency.
    for (; r + 4 <= row_count && n % 8 == 0; r += 4) {
        const float *w0 = rows + r * n, *w1 = w0 + n, *w2 = w1 + n, *w3 = w2 + n;
        float32x4_t a0 = vdupq_n_f32(0), b0 = a0, a1 = a0, b1 = a0, a2 = a0, b2 = a0, a3 = a0, b3 = a0;
        for (size_t i = 0; i < n; i += 8) {
            const float32x4_t xl = vld1q_f32(x + i), xh = vld1q_f32(x + i + 4);
            a0 = vfmaq_f32(a0, vld1q_f32(w0 + i), xl);
            b0 = vfmaq_f32(b0, vld1q_f32(w0 + i + 4), xh);
            a1 = vfmaq_f32(a1, vld1q_f32(w1 + i), xl);
            b1 = vfmaq_f32(b1, vld1q_f32(w1 + i + 4), xh);
            a2 = vfmaq_f32(a2, vld1q_f32(w2 + i), xl);
            b2 = vfmaq_f32(b2, vld1q_f32(w2 + i + 4), xh);
            a3 = vfmaq_f32(a3, vld1q_f32(w3 + i), xl);
            b3 = vfmaq_f32(b3, vld1q_f32(w3 + i + 4), xh);
        }
        scores[r] = vaddvq_f32(vaddq_f32(a0, b0));
        scores[r + 1] = vaddvq_f32(vaddq_f32(a1, b1));
        scores[r + 2] = vaddvq_f32(vaddq_f32(a2, b2));
        scores[r + 3] = vaddvq_f32(vaddq_f32(a3, b3));
    }
    for (; r < row_count; ++r) {
        scores[r] = dot(rows + r * n, x, n);
    }
}

const char *implementation() {
    return "neon";
}
#else
float dot(const float *a, const float *b, size_t n) {
    float sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float sum_squares(const float *x, size_t n) {
    return dot(x, x, n);
}

void scale_mul(const float *x, const float *weight, float scale, float *out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[i] * scale * weight[i];
    }
}

void gemv_rows(const float *rows, const float *x, float *scores, size_t row_count, size_t n) {
    for (size_t r = 0; r < row_count; ++r) {
        scores[r] = dot(rows + r * n, x, n);
    }
}

const char *implementation() {
    return "scalar";
}
#endif
} // namespace eqt::simd
