// Measures how GGML's CPU matrix multiplication scales with the number of activation columns (tokens) for
// the weight formats Equity streams. A cost ratio near N for N columns means weight blocks are decoded again
// for every token, which is what limits batched speculative verification.
#include "engine.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstring>
#include <thread>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

std::vector<float> read(const ggml_tensor *tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

struct BatchCheck {
    bool bitwise = true;          // every batched output equals its one-column product bit for bit
    double max_relative = 0;      // max |batched - single| / max |single| over all outputs
};

// Small batches use exact tiles: every column of a batched MUL_MAT, and every token of a MUL_MAT_ID whose
// experts are shared between tokens, must equal the one-column product bit for bit. Prompt-sized batches may
// take the tiled GEMM path, which is exact in integers but rounds floats differently; the relative error
// measures that.
BatchCheck check_batch(ggml_backend_t backend, ggml_type type, const std::vector<uint8_t> &packed, int64_t width,
                    int64_t rows, int columns, std::mt19937 &random) {
    constexpr int64_t experts = 4, expert_rows = 512, used = 2;
    ggml_init_params params = {ggml_tensor_overhead() * (16 + 8 * columns) + ggml_graph_overhead(), nullptr, true};
    ggml_context *ctx = ggml_init(params);
    ggml_tensor *weight = ggml_new_tensor_2d(ctx, type, width, rows);
    ggml_tensor *bank = ggml_new_tensor_3d(ctx, type, width, expert_rows, experts);
    ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, columns);
    ggml_tensor *slots = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, used, columns); // per-slot inputs, as for down
    ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, columns);
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_tensor *batched = ggml_mul_mat(ctx, weight, x);
    ggml_tensor *batched_ids = ggml_mul_mat_id(ctx, bank, ggml_reshape_3d(ctx, x, width, 1, columns), ids);
    ggml_tensor *batched_slots = ggml_mul_mat_id(ctx, bank, slots, ids);
    for (ggml_tensor *t : {batched, batched_ids, batched_slots}) {
        ggml_build_forward_expand(graph, t);
    }
    std::vector<ggml_tensor *> single, single_ids, single_slots;
    for (int c = 0; c < columns; ++c) {
        ggml_tensor *column = ggml_view_2d(ctx, x, width, 1, x->nb[1], c * x->nb[1]);
        ggml_tensor *id = ggml_view_2d(ctx, ids, used, 1, ids->nb[1], c * ids->nb[1]);
        single.push_back(ggml_mul_mat(ctx, weight, column));
        single_ids.push_back(ggml_mul_mat_id(ctx, bank, ggml_reshape_3d(ctx, column, width, 1, 1), id));
        single_slots.push_back(ggml_mul_mat_id(
            ctx, bank, ggml_view_3d(ctx, slots, width, used, 1, slots->nb[1], slots->nb[2], c * slots->nb[2]), id));
        for (ggml_tensor *t : {single.back(), single_ids.back(), single_slots.back()}) {
            ggml_build_forward_expand(graph, t);
        }
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(weight, packed.data(), 0, ggml_nbytes(weight));
    ggml_backend_tensor_set(bank, packed.data(), 0, ggml_nbytes(bank));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (ggml_tensor *t : {x, slots}) {
        std::vector<float> values(ggml_nelements(t));
        for (auto &v : values) {
            v = normal(random);
        }
        ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
    }
    // Consecutive tokens share expert 1 and alternate the other, so experts see both paired and lone rows.
    std::vector<int32_t> routing;
    for (int c = 0; c < columns; ++c) {
        routing.insert(routing.end(), {c % 2 ? 2 : 0, 1});
    }
    ggml_backend_tensor_set(ids, routing.data(), 0, ggml_nbytes(ids));
    ggml_backend_graph_compute(backend, graph);
    BatchCheck check;
    const std::vector<float> batches[] = {read(batched), read(batched_ids), read(batched_slots)};
    for (int c = 0; c < columns; ++c) {
        const std::vector<float> singles[] = {read(single[c]), read(single_ids[c]), read(single_slots[c])};
        for (int k = 0; k < 3; ++k) {
            const size_t count = singles[k].size();
            check.bitwise &= std::memcmp(batches[k].data() + c * count, singles[k].data(), count * sizeof(float)) == 0;
            float scale = 0, error = 0;
            for (size_t i = 0; i < count; ++i) {
                scale = std::max(scale, std::fabs(singles[k][i]));
                error = std::max(error, std::fabs(batches[k][c * count + i] - singles[k][i]));
            }
            check.max_relative = std::max(check.max_relative, scale > 0 ? double(error) / scale : 0.0);
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return check;
}
} // namespace

int main(int argc, char **argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 4;
    // Optional comma-separated batch widths; prompt-sized widths exercise the tiled GEMM path when enabled.
    std::vector<int> columns = {1, 2, 3, 4};
    if (argc > 2) {
        columns.clear();
        for (const char *p = argv[2]; *p;) {
            columns.push_back(std::atoi(p));
            while (*p && *p != ',') {
                ++p;
            }
            p += *p == ',';
        }
    }
    const int64_t width = 2048, rows = 4096; // a 2048 -> 4096 projection, as in Qwen3.6 attention
    eqt::Engine engine;                      // selects and loads the CPU backend variant
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        std::fprintf(stderr, "no CPU backend\n");
        return 1;
    }
    auto *reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    if (auto set_threads = reinterpret_cast<void (*)(ggml_backend_t, int)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"))) {
        set_threads(backend, threads);
    }
    if (auto features = reinterpret_cast<ggml_backend_get_features_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features"))) {
        for (const ggml_backend_feature *feature = features(reg); feature->name; ++feature) {
            std::fprintf(stderr, "%s=%s ", feature->name, feature->value);
        }
        std::fprintf(stderr, "\n");
    }
    // A persistent pool, as the engine uses; otherwise each compute call spawns its worker threads.
    auto create = reinterpret_cast<decltype(&ggml_threadpool_new)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
    auto attach = reinterpret_cast<void (*)(ggml_backend_t, ggml_threadpool *)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
    if (create && attach) {
        auto pool_params = ggml_threadpool_params_default(threads);
        attach(backend, create(&pool_params));
    }
    // Memory read roofline for the same thread count: sum a 64 MiB buffer.
    {
        std::vector<uint64_t> buffer((64u << 20) / 8, 1);
        std::vector<double> rates;
        for (int r = 0; r < 5; ++r) {
            std::vector<std::thread> workers;
            std::vector<uint64_t> sums(threads);
            const auto start = Clock::now();
            for (int t = 0; t < threads; ++t) {
                workers.emplace_back([&, t] {
                    const size_t n = buffer.size() / threads, begin = t * n;
                    uint64_t a = 0, b = 0, c = 0, d = 0;
                    for (size_t i = begin; i + 4 <= begin + n; i += 4) {
                        a += buffer[i]; b += buffer[i + 1]; c += buffer[i + 2]; d += buffer[i + 3];
                    }
                    sums[t] = a + b + c + d;
                });
            }
            for (auto &w : workers) {
                w.join();
            }
            rates.push_back(buffer.size() * 8 / std::chrono::duration<double>(Clock::now() - start).count() / 1e9);
        }
        std::fprintf(stderr, "read bandwidth %d threads: %.1f GB/s\n", threads, median(rates));
    }
    std::mt19937 random(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> source(width * rows), importance(width, 1.0f);
    for (auto &v : source) {
        v = normal(random) * 0.02f;
    }
    const ggml_type types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                               GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS};
    std::printf("{\"threads\": %d, \"shape\": [%lld, %lld], \"results\": [\n", threads, (long long)width, (long long)rows);
    bool first = true;
    for (const ggml_type type : types) {
        ggml_quantize_init(type);
        std::vector<uint8_t> packed(ggml_row_size(type, width) * rows);
        ggml_quantize_chunk(type, source.data(), packed.data(), 0, rows, width,
                            ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr);
        const int batches = static_cast<int>(columns.size());
        std::vector<BatchCheck> exact(batches);
        for (int b = 0; b < batches; ++b) {
            exact[b] = check_batch(backend, type, packed, width, rows, columns[b], random);
        }
        // Sixteen distinct matrices (~70-140 MiB) keep the working set far beyond the CPU caches, as in real
        // decode where each token reads gigabytes of weights from DRAM. One graph per batch width shares them.
        constexpr int copies = 16;
        ggml_init_params params = {ggml_tensor_overhead() * (copies * (batches + 1) + batches) +
                                       batches * ggml_graph_overhead(), nullptr, true};
        ggml_context *ctx = ggml_init(params);
        std::vector<ggml_tensor *> weights, inputs;
        std::vector<ggml_cgraph *> graphs;
        for (int c = 0; c < copies; ++c) {
            weights.push_back(ggml_new_tensor_2d(ctx, type, width, rows));
        }
        for (const int n : columns) {
            inputs.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, n));
            graphs.push_back(ggml_new_graph(ctx));
            for (auto *w : weights) {
                ggml_build_forward_expand(graphs.back(), ggml_mul_mat(ctx, w, inputs.back()));
            }
        }
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        for (auto *w : weights) {
            ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
        }
        for (auto *x : inputs) {
            std::vector<float> activation(ggml_nelements(x));
            for (auto &v : activation) {
                v = normal(random);
            }
            ggml_backend_tensor_set(x, activation.data(), 0, ggml_nbytes(x));
        }
        // Batch widths alternate within each repetition so frequency and thermal drift affect them equally.
        std::vector<std::vector<double>> times(batches);
        for (int i = 0; i < 25; ++i) {
            for (int b = 0; b < batches; ++b) {
                const auto start = Clock::now();
                ggml_backend_graph_compute(backend, graphs[b]);
                const double us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
                if (i >= 5) {
                    times[b].push_back(us);
                }
            }
        }
        const double base = median(times[0]) / copies;
        for (int b = 0; b < batches; ++b) {
            const double t = median(times[b]) / copies; // per matrix
            std::printf("%s  {\"type\": \"%s\", \"tokens\": %d, \"us\": %.1f, \"ratio_vs_1\": %.2f, \"weight_gb_s\": %.1f, "
                        "\"batch_bitwise_equal\": %s, \"batch_max_relative_error\": %.2e}",
                        first ? "" : ",\n", ggml_type_name(type), columns[b], t, t / base, packed.size() / t / 1e3,
                        exact[b].bitwise ? "true" : "false", exact[b].max_relative);
            first = false;
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    std::printf("\n]}\n");
    ggml_backend_free(backend);
    return 0;
}
