#include "simd.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
double reference_dot(const float *a, const float *b, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * b[i];
    }
    return sum;
}

// FP32 reordering error grows with the sum of |a_i b_i|; 8 ulp-scale slack per term is generous.
bool close(double expected, double actual, double magnitude, size_t n) {
    return std::abs(expected - actual) <= 1e-6 * magnitude * std::sqrt(static_cast<double>(n)) + 1e-6;
}
} // namespace

int main() {
    std::mt19937 random(1234);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    int failures = 0;
    for (size_t n : {1u, 3u, 4u, 7u, 8u, 15u, 16u, 17u, 31u, 64u, 255u, 2048u, 2051u}) {
        std::vector<float> a(n), b(n), out(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = normal(random);
            b[i] = normal(random);
        }
        double magnitude = 0;
        for (size_t i = 0; i < n; ++i) {
            magnitude += std::abs(static_cast<double>(a[i]) * b[i]);
        }
        if (!close(reference_dot(a.data(), b.data(), n), eqt::simd::dot(a.data(), b.data(), n), magnitude,
                   n)) {
            std::printf("dot mismatch n=%zu\n", n);
            ++failures;
        }
        eqt::simd::scale_mul(a.data(), b.data(), 0.5f, out.data(), n);
        for (size_t i = 0; i < n; ++i) {
            if (out[i] != a[i] * 0.5f * b[i]) {
                std::printf("scale_mul mismatch n=%zu i=%zu\n", n, i);
                ++failures;
                break;
            }
        }
    }
    for (size_t rows : {1u, 4u, 5u, 256u}) {
        for (size_t n : {8u, 12u, 2048u}) {
            std::vector<float> w(rows * n), x(n), scores(rows);
            for (auto &v : w) {
                v = normal(random);
            }
            for (auto &v : x) {
                v = normal(random);
            }
            eqt::simd::gemv_rows(w.data(), x.data(), scores.data(), rows, n);
            for (size_t r = 0; r < rows; ++r) {
                double magnitude = 0;
                for (size_t i = 0; i < n; ++i) {
                    magnitude += std::abs(static_cast<double>(w[r * n + i]) * x[i]);
                }
                if (!close(reference_dot(&w[r * n], x.data(), n), scores[r], magnitude, n)) {
                    std::printf("gemv mismatch rows=%zu n=%zu r=%zu\n", rows, n, r);
                    ++failures;
                    break;
                }
            }
        }
    }
    // Q8_0 rows (FP16 scale + 32 int8) against quantized activations: the result must equal the double-precision
    // sum of the same integer products and scales up to FP32 accumulation error.
    for (size_t rows : {1u, 3u, 320u}) {
        for (size_t n : {32u, 320u, 10240u}) {
            std::vector<float> x(n), scores(rows), x_scales(n / 32);
            std::vector<int8_t> xq(n);
            std::vector<uint8_t> w(rows * (n / 32) * 34);
            for (auto &v : x) {
                v = normal(random);
            }
            std::uniform_int_distribution<int> quant(-127, 127);
            for (size_t block = 0; block < rows * (n / 32); ++block) {
                const uint16_t half = 0x2000 + static_cast<uint16_t>(quant(random) & 0x3ff); // about 0.008..0.016
                std::memcpy(&w[block * 34], &half, 2);
                for (size_t j = 0; j < 32; ++j) {
                    w[block * 34 + 2 + j] = static_cast<uint8_t>(static_cast<int8_t>(quant(random)));
                }
            }
            eqt::simd::quantize_blocks(x.data(), n, xq.data(), x_scales.data());
            for (size_t i = 0; i < n; ++i) {
                if (std::abs(xq[i] * x_scales[i / 32] - x[i]) > x_scales[i / 32] * 0.5f + 1e-6f) {
                    std::printf("quantize mismatch n=%zu i=%zu\n", n, i);
                    ++failures;
                    break;
                }
            }
            eqt::simd::gemv_q8_0(w.data(), (n / 32) * 34, xq.data(), x_scales.data(), scores.data(), rows, n);
            for (size_t r = 0; r < rows; ++r) {
                double expected = 0, magnitude = 0;
                for (size_t b = 0; b < n / 32; ++b) {
                    const uint8_t *block = &w[(r * (n / 32) + b) * 34];
                    uint16_t half;
                    std::memcpy(&half, block, 2);
                    // binary16 normal numbers only in this test: (1 + m/1024) * 2^(e - 15)
                    const double d = std::ldexp(1.0 + (half & 0x3ff) / 1024.0, ((half >> 10) & 0x1f) - 15);
                    long long dot = 0;
                    for (size_t j = 0; j < 32; ++j) {
                        dot += static_cast<long long>(static_cast<int8_t>(block[2 + j])) * xq[b * 32 + j];
                    }
                    expected += d * x_scales[b] * static_cast<double>(dot);
                    magnitude += std::abs(d * x_scales[b] * static_cast<double>(dot));
                }
                if (!close(expected, scores[r], magnitude, n / 32)) {
                    std::printf("gemv_q8_0 mismatch rows=%zu n=%zu r=%zu\n", rows, n, r);
                    ++failures;
                    break;
                }
            }
        }
    }
    // Gated stream mean (hyper-connection mixer lookahead), against double precision, gates over the range the
    // polynomial exponent covers and beyond its clamp.
    for (size_t width : {3u, 4u, 2560u}) {
        const size_t streams = 4;
        std::vector<float> normed(streams * width), gate(streams * width), out(width);
        std::uniform_real_distribution<float> wide(-100.0f, 100.0f);
        for (size_t i = 0; i < normed.size(); ++i) {
            normed[i] = normal(random);
            gate[i] = i % 3 == 0 ? wide(random) : 4.0f * normal(random);
        }
        eqt::simd::gated_mean(normed.data(), gate.data(), out.data(), width, streams);
        for (size_t i = 0; i < width; ++i) {
            double expected = 0, magnitude = 0;
            for (size_t s = 0; s < streams; ++s) {
                const double term = normed[s * width + i] / (1.0 + std::exp(-static_cast<double>(gate[s * width + i])));
                expected += term;
                magnitude += std::abs(term);
            }
            expected /= streams;
            if (std::abs(expected - out[i]) > 2e-6 * magnitude + 1e-7) {
                std::printf("gated_mean mismatch width=%zu i=%zu\n", width, i);
                ++failures;
                break;
            }
        }
    }
    // Router-shaped timing (256 experts x 2048 width); informative only, not a device benchmark.
    std::vector<float> router(256 * 2048), hidden(2048), scores(256);
    for (auto &v : router) {
        v = normal(random);
    }
    for (auto &v : hidden) {
        v = normal(random);
    }
    const auto start = std::chrono::steady_clock::now();
    constexpr int repeats = 200;
    for (int i = 0; i < repeats; ++i) {
        eqt::simd::gemv_rows(router.data(), hidden.data(), scores.data(), 256, 2048);
    }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / repeats;
    std::printf("{\"implementation\":\"%s\",\"failures\":%d,\"router_gemv_us\":%.1f,\"checksum\":%.3f}\n",
                eqt::simd::implementation(), failures, us, scores[0]);
    return failures ? 1 : 0;
}
