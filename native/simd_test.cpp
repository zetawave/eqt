#include "simd.h"

#include <chrono>
#include <cmath>
#include <cstdio>
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
