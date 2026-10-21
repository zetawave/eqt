#include "simd.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define EQT_NEON 1
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#endif

namespace eqt::simd {
namespace {
constexpr size_t q8_block = 32;
constexpr size_t q8_block_bytes = 2 + q8_block; // FP16 scale, then the quants (GGML block_q8_0)

// IEEE binary16 to binary32, including subnormals (the FP16 library's bit-exact conversion that GGML uses).
float half_to_float(const uint8_t *bytes) {
    uint16_t h;
    std::memcpy(&h, bytes, sizeof(h));
    const auto as_float = [](uint32_t bits) {
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    };
    const auto as_bits = [](float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        return bits;
    };
    const uint32_t w = static_cast<uint32_t>(h) << 16, sign = w & 0x80000000u, two_w = w + w;
    const float normalized = as_float((two_w >> 4) + (0xE0u << 23)) * 0x1.0p-112f;
    const float denormalized = as_float((two_w >> 17) | (126u << 23)) - 0.5f;
    return as_float(sign | (two_w < (1u << 27) ? as_bits(denormalized) : as_bits(normalized)));
}
} // namespace

void quantize_blocks(const float *x, size_t n, int8_t *q, float *scales) {
    for (size_t b = 0; b < n / q8_block; ++b) {
        const float *block = x + b * q8_block;
        float amax = 0;
        for (size_t j = 0; j < q8_block; ++j) {
            amax = std::max(amax, std::abs(block[j]));
        }
        const float d = amax / 127.0f, inverse = d ? 1.0f / d : 0.0f;
        scales[b] = d;
        for (size_t j = 0; j < q8_block; ++j) {
            q[b * q8_block + j] = static_cast<int8_t>(std::nearbyint(block[j] * inverse));
        }
    }
}
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

namespace {
bool has_dotprod() {
#if defined(__linux__) && defined(HWCAP_ASIMDDP)
    static const bool available = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
    return available;
#else
    return false;
#endif
}

__attribute__((target("arch=armv8.2-a+dotprod"))) inline int32x4_t block_dot(const uint8_t *block, int8x16_t x0,
                                                                              int8x16_t x1) {
    const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), vld1q_s8(reinterpret_cast<const int8_t *>(block + 2)), x0);
    return vdotq_s32(acc, vld1q_s8(reinterpret_cast<const int8_t *>(block + 18)), x1);
}

// SDOT adds four int8 products into each 32-bit lane per instruction (at most 4 * 128 * 128 per step). Four rows
// share each activation load; their block sums and FP16 scales are combined as one vector per block, so the
// in-order little cores the predictor usually runs on issue no scalar conversions or reductions per row.
__attribute__((target("arch=armv8.2-a+dotprod"))) void gemv_q8_0_dotprod(const uint8_t *rows, size_t row_stride,
                                                                          const int8_t *q, const float *scales,
                                                                          float *scores, size_t row_count, size_t n) {
    const size_t blocks = n / q8_block;
    size_t r = 0;
    for (; r + 4 <= row_count; r += 4) {
        const uint8_t *row0 = rows + r * row_stride, *row1 = row0 + row_stride, *row2 = row1 + row_stride,
                      *row3 = row2 + row_stride;
        float32x4_t total = vdupq_n_f32(0);
        for (size_t b = 0; b < blocks; ++b) {
            const size_t at = b * q8_block_bytes;
            const int8x16_t x0 = vld1q_s8(q + b * q8_block), x1 = vld1q_s8(q + b * q8_block + 16);
            // Lane i of the pairwise sums is row i's block total.
            const int32x4_t sums = vpaddq_s32(vpaddq_s32(block_dot(row0 + at, x0, x1), block_dot(row1 + at, x0, x1)),
                                              vpaddq_s32(block_dot(row2 + at, x0, x1), block_dot(row3 + at, x0, x1)));
            uint16_t halves[4];
            std::memcpy(&halves[0], row0 + at, 2);
            std::memcpy(&halves[1], row1 + at, 2);
            std::memcpy(&halves[2], row2 + at, 2);
            std::memcpy(&halves[3], row3 + at, 2);
            const float32x4_t d = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(halves)));
            total = vfmaq_f32(total, vcvtq_f32_s32(sums), vmulq_n_f32(d, scales[b]));
        }
        vst1q_f32(scores + r, total);
    }
    for (; r < row_count; ++r) {
        const uint8_t *row = rows + r * row_stride;
        float sum = 0;
        for (size_t b = 0; b < blocks; ++b) {
            const uint8_t *block = row + b * q8_block_bytes;
            const int8x16_t w0 = vld1q_s8(reinterpret_cast<const int8_t *>(block + 2));
            const int8x16_t w1 = vld1q_s8(reinterpret_cast<const int8_t *>(block + 18));
            int32x4_t acc = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(q + b * q8_block));
            acc = vdotq_s32(acc, w1, vld1q_s8(q + b * q8_block + 16));
            sum += half_to_float(block) * scales[b] * static_cast<float>(vaddvq_s32(acc));
        }
        scores[r] = sum;
    }
}
} // namespace

void gemv_q8_0(const uint8_t *rows, size_t row_stride, const int8_t *q, const float *scales, float *scores,
               size_t row_count, size_t n) {
    if (has_dotprod()) {
        gemv_q8_0_dotprod(rows, row_stride, q, scales, scores, row_count, n);
        return;
    }
    const size_t blocks = n / q8_block;
    for (size_t r = 0; r < row_count; ++r) {
        const uint8_t *row = rows + r * row_stride;
        float sum = 0;
        for (size_t b = 0; b < blocks; ++b) {
            const uint8_t *block = row + b * q8_block_bytes;
            const int8x16_t w0 = vld1q_s8(reinterpret_cast<const int8_t *>(block + 2));
            const int8x16_t w1 = vld1q_s8(reinterpret_cast<const int8_t *>(block + 18));
            const int8x16_t x0 = vld1q_s8(q + b * q8_block), x1 = vld1q_s8(q + b * q8_block + 16);
            // Each int8 product fits int16; pairwise accumulation into int32 cannot overflow.
            int32x4_t acc = vpaddlq_s16(vmull_s8(vget_low_s8(w0), vget_low_s8(x0)));
            acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w0), vget_high_s8(x0)));
            acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w1), vget_low_s8(x1)));
            acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w1), vget_high_s8(x1)));
            sum += half_to_float(block) * scales[b] * static_cast<float>(vaddvq_s32(acc));
        }
        scores[r] = sum;
    }
}

namespace {
// e^x = 2^n e^r with n = round(x / ln 2), r = x - n ln 2 in two parts and Cephes' degree-6 polynomial for e^r
// (a few ulp). x is clamped to [-87, 88], so 2^n stays a normal float built in the exponent field.
float32x4_t exp4(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(88.0f));
    const float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    const float32x4_t e = vaddq_f32(vfmaq_f32(r, p, vmulq_f32(r, r)), vdupq_n_f32(1.0f));
    const int32x4_t power = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(e, vreinterpretq_f32_s32(power));
}
} // namespace

void gated_mean(const float *normed, const float *gate, float *out, size_t width, size_t streams) {
    const float inverse = 1.0f / static_cast<float>(streams);
    const float32x4_t one = vdupq_n_f32(1.0f);
    size_t i = 0;
    for (; i + 4 <= width; i += 4) {
        float32x4_t sum = vdupq_n_f32(0);
        for (size_t s = 0; s < streams; ++s) {
            const float32x4_t g = vld1q_f32(gate + s * width + i);
            const float32x4_t sigmoid = vdivq_f32(one, vaddq_f32(one, exp4(vnegq_f32(g))));
            sum = vfmaq_f32(sum, vld1q_f32(normed + s * width + i), sigmoid);
        }
        vst1q_f32(out + i, vmulq_n_f32(sum, inverse));
    }
    for (; i < width; ++i) {
        float sum = 0;
        for (size_t s = 0; s < streams; ++s) {
            sum += normed[s * width + i] / (1.0f + std::exp(-gate[s * width + i]));
        }
        out[i] = sum * inverse;
    }
}

const char *implementation() {
    return has_dotprod() ? "neon+dotprod" : "neon";
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

void gemv_q8_0(const uint8_t *rows, size_t row_stride, const int8_t *q, const float *scales, float *scores,
               size_t row_count, size_t n) {
    const size_t blocks = n / q8_block;
    for (size_t r = 0; r < row_count; ++r) {
        const uint8_t *row = rows + r * row_stride;
        float sum = 0;
        for (size_t b = 0; b < blocks; ++b) {
            const uint8_t *block = row + b * q8_block_bytes;
            const auto *weights = reinterpret_cast<const int8_t *>(block + 2);
            int32_t acc = 0;
            for (size_t j = 0; j < q8_block; ++j) {
                acc += static_cast<int32_t>(weights[j]) * q[b * q8_block + j];
            }
            sum += half_to_float(block) * scales[b] * static_cast<float>(acc);
        }
        scores[r] = sum;
    }
}

void gated_mean(const float *normed, const float *gate, float *out, size_t width, size_t streams) {
    const float inverse = 1.0f / static_cast<float>(streams);
    for (size_t i = 0; i < width; ++i) {
        float sum = 0;
        for (size_t s = 0; s < streams; ++s) {
            sum += normed[s * width + i] / (1.0f + std::exp(-gate[s * width + i]));
        }
        out[i] = sum * inverse;
    }
}

const char *implementation() {
    return "scalar";
}
#endif
} // namespace eqt::simd
