// Measures how GGML's CPU matrix multiplication scales with the number of activation columns (tokens) for
// the weight formats Equity streams. A cost ratio near N for N columns means weight blocks are decoded again
// for every token, which is what limits batched speculative verification.
//
// eqt-kernel-bench THREADS [WIDTHS] [--experts] [--sustain=SECONDS]
//   WIDTHS       comma-separated batch widths for the column-scaling table (default 1,2,3,4)
//   --experts    only the routed-expert products of Qwen3.8-Flash-Next with the four-row kernels off and on
//   --sustain=S  only a sustained loop: raw DRAM read rate, dense and expert products and CPU clocks over S seconds
#include "engine.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstring>
#include <fstream>
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
using RowKernelSetter = uint64_t (*)(uint64_t);

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

std::vector<float> read(const ggml_tensor *tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

std::vector<uint8_t> quantize(ggml_type type, int64_t width, int64_t rows, std::mt19937 &random) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> source(width * rows), importance(width, 1.0f);
    for (auto &v : source) {
        v = normal(random) * 0.02f;
    }
    ggml_quantize_init(type);
    std::vector<uint8_t> packed(ggml_row_size(type, width) * rows);
    ggml_quantize_chunk(type, source.data(), packed.data(), 0, rows, width,
                        ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr);
    return packed;
}

void fill_normal(ggml_tensor *t, std::mt19937 &random) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> values(ggml_nelements(t));
    for (auto &v : values) {
        v = normal(random);
    }
    ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
}

// Sum of a 64 MiB buffer on `threads` threads: the CPU's DRAM read rate at the current clocks.
double read_bandwidth(int threads, const std::vector<uint64_t> &buffer) {
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
    return buffer.size() * 8 / std::chrono::duration<double>(Clock::now() - start).count() / 1e9;
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
    for (ggml_tensor *t : {x, slots}) {
        fill_normal(t, random);
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

void column_scaling(ggml_backend_t backend, const std::vector<int> &columns, std::mt19937 &random) {
    const int64_t width = 2048, rows = 4096; // a 2048 -> 4096 projection, as in Qwen3.6 attention
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const ggml_type types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                               GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS};
    std::printf("\"shape\": [%lld, %lld], \"results\": [\n", (long long)width, (long long)rows);
    bool first = true;
    for (const ggml_type type : types) {
        const std::vector<uint8_t> packed = quantize(type, width, rows, random);
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
            fill_normal(x, random);
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
    std::printf("\n]");
}

// Qwen3.8-Flash-Next's routed experts: gate and up 2560 -> 640 (IQ2_XS), down 640 -> 2560 (IQ4_NL, one input row
// per routed slot). One decoded token runs 10 experts per product; 16 products read 160 distinct experts (76 and
// 147 MB), so the weights come from DRAM as in decoding.
struct ExpertProduct {
    const char *name;
    ggml_type type;
    int64_t width, rows;
    bool per_slot;
};
constexpr ExpertProduct expert_products[] = {{"gate_up", GGML_TYPE_IQ2_XS, 2560, 640, false},
                                             {"down", GGML_TYPE_IQ4_NL, 640, 2560, true}};

struct ExpertGraph {
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph *graph = nullptr;
    std::vector<ggml_tensor *> outputs;
    size_t weight_bytes_read = 0; // per compute
    ~ExpertGraph() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

// `tokens` decoded rows; token t routes to experts (k*used + t*shift + j) % experts in product k.
void build_expert_graph(ExpertGraph &g, ggml_backend_t backend, const ExpertProduct &p, int tokens, int shift,
                        std::mt19937 &random) {
    constexpr int64_t used = 10, products = 16, experts = used * products, distinct = 4;
    g.ctx = ggml_init({ggml_tensor_overhead() * (4 * products + 4) + ggml_graph_overhead(), nullptr, true});
    ggml_tensor *bank = ggml_new_tensor_3d(g.ctx, p.type, p.width, p.rows, experts);
    std::vector<ggml_tensor *> inputs, ids;
    g.graph = ggml_new_graph(g.ctx);
    for (int k = 0; k < products; ++k) {
        inputs.push_back(p.per_slot ? ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, p.width, used, tokens)
                                    : ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, p.width, 1, tokens));
        ids.push_back(ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, used, tokens));
        g.outputs.push_back(ggml_mul_mat_id(g.ctx, bank, inputs.back(), ids.back()));
        ggml_build_forward_expand(g.graph, g.outputs.back());
    }
    g.buffer = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    const size_t expert_bytes = ggml_row_size(p.type, p.width) * p.rows;
    for (int d = 0; d < distinct; ++d) {
        const std::vector<uint8_t> packed = quantize(p.type, p.width, p.rows, random);
        for (int64_t e = d; e < experts; e += distinct) {
            ggml_backend_tensor_set(bank, packed.data(), e * expert_bytes, expert_bytes);
        }
    }
    std::vector<bool> touched(experts);
    for (int k = 0; k < products; ++k) {
        fill_normal(inputs[k], random);
        std::vector<int32_t> routing;
        for (int t = 0; t < tokens; ++t) {
            for (int j = 0; j < used; ++j) {
                const int e = static_cast<int>((k * used + t * shift + j) % experts);
                routing.push_back(e);
                touched[e] = true;
            }
        }
        ggml_backend_tensor_set(ids[k], routing.data(), 0, ggml_nbytes(ids[k]));
    }
    g.weight_bytes_read = std::count(touched.begin(), touched.end(), true) * expert_bytes;
}

std::vector<std::vector<float>> outputs_of(const ExpertGraph &g) {
    std::vector<std::vector<float>> values;
    for (const auto *t : g.outputs) {
        values.push_back(read(t));
    }
    return values;
}

// Four-row kernels against the one-row kernels on the same graphs: bitwise equality (one token; three tokens that
// share experts; row counts that leave one-row edges) and the time of one decoded token's products.
void expert_kernels(ggml_backend_t backend, RowKernelSetter set_rows, std::mt19937 &random) {
    std::printf("\"expert_products\": [\n");
    const auto apply = [set_rows](uint64_t mask) {
        if (set_rows) {
            set_rows(mask);
        }
    };
    bool first = true;
    for (const ExpertProduct &p : expert_products) {
        const uint64_t mask = 1ull << p.type;
        const bool available = set_rows && (set_rows(mask) & mask);
        bool bitwise = true;
        for (const ExpertProduct &edge : {p, ExpertProduct{p.name, p.type, p.width, p.rows + 6, p.per_slot}}) {
            if (!available) {
                break;
            }
            for (const auto [tokens, shift] : {std::pair{1, 0}, std::pair{3, 5}}) {
                ExpertGraph g;
                build_expert_graph(g, backend, edge, tokens, shift, random);
                apply(0);
                ggml_backend_graph_compute(backend, g.graph);
                const auto reference = outputs_of(g);
                apply(mask);
                ggml_backend_graph_compute(backend, g.graph);
                const auto rows4 = outputs_of(g);
                for (size_t k = 0; k < reference.size(); ++k) {
                    bitwise &= std::memcmp(reference[k].data(), rows4[k].data(), reference[k].size() * sizeof(float)) == 0;
                }
            }
        }
        ExpertGraph g;
        build_expert_graph(g, backend, p, 1, 0, random);
        std::vector<double> one_row, four_rows;
        for (int i = 0; i < 30; ++i) {
            for (const bool rows4 : {false, true}) {
                apply(rows4 ? mask : 0);
                const auto start = Clock::now();
                ggml_backend_graph_compute(backend, g.graph);
                const double us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
                if (i >= 5) {
                    (rows4 ? four_rows : one_row).push_back(us);
                }
            }
        }
        apply(0);
        constexpr double products = 16;
        const double one = median(one_row), four = median(four_rows);
        std::printf("%s  {\"product\": \"%s\", \"type\": \"%s\", \"shape\": [%lld, %lld], \"row_kernel_available\": %s, "
                    "\"rows4_bitwise_equal\": %s, \"one_row_us\": %.1f, \"four_rows_us\": %.1f, "
                    "\"one_row_gb_s\": %.1f, \"four_rows_gb_s\": %.1f, \"speedup\": %.3f}",
                    first ? "" : ",\n", p.name, ggml_type_name(p.type), (long long)p.width, (long long)p.rows,
                    available ? "true" : "false", available ? (bitwise ? "true" : "false") : "null", one / products,
                    four / products, g.weight_bytes_read / one / 1e3, g.weight_bytes_read / four / 1e3, one / four);
        first = false;
    }
    std::printf("\n]");
}

// Current clock of one CPU in MHz as JSON, null when the sysfs file cannot be read.
std::string cpu_mhz(int cpu) {
    std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/scaling_cur_freq");
    long khz = 0;
    return file >> khz ? std::to_string(khz / 1000) : "null";
}

// Repeats the same work for `seconds`: if the dense products slow down with the raw read rate, the thermal cap
// limits memory; if they slow down while the raw rate holds, they are limited by the capped CPU clocks.
void sustain(ggml_backend_t backend, int threads, RowKernelSetter set_rows, double seconds, std::mt19937 &random) {
    constexpr int copies = 16;
    const int64_t width = 2048, rows = 4096;
    const ggml_type dense_types[] = {GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K};
    struct Dense {
        ggml_context *ctx;
        ggml_backend_buffer_t buffer;
        ggml_cgraph *graph;
        size_t bytes;
    };
    std::vector<Dense> dense;
    for (const ggml_type type : dense_types) {
        const std::vector<uint8_t> packed = quantize(type, width, rows, random);
        ggml_context *ctx = ggml_init({ggml_tensor_overhead() * (2 * copies + 1) + ggml_graph_overhead(), nullptr, true});
        ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
        ggml_cgraph *graph = ggml_new_graph(ctx);
        std::vector<ggml_tensor *> weights;
        for (int c = 0; c < copies; ++c) {
            weights.push_back(ggml_new_tensor_2d(ctx, type, width, rows));
            ggml_build_forward_expand(graph, ggml_mul_mat(ctx, weights.back(), x));
        }
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        for (auto *w : weights) {
            ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
        }
        fill_normal(x, random);
        dense.push_back({ctx, buffer, graph, packed.size() * copies});
    }
    ExpertGraph experts[2];
    for (int k = 0; k < 2; ++k) {
        build_expert_graph(experts[k], backend, expert_products[k], 1, 0, random);
    }
    // 256 MiB per raw-rate pass, so starting its threads is a small part of the time.
    const std::vector<uint64_t> buffer((256u << 20) / 8, 1);
    std::printf("\"sustain_seconds\": %.0f, \"samples\": [\n", seconds);
    const auto begin = Clock::now();
    const auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - begin).count(); };
    bool first = true;
    // Items 0: raw read, 1-3: dense, 4-7: experts one-row/four-row. Each sample averages a window of about 0.5 s
    // of the work running back to back: bytes and seconds per item.
    constexpr int items = 8;
    while (elapsed() < seconds) {
        const double window_start = elapsed();
        double bytes[items] = {}, time[items] = {};
        const auto timed = [&](int item, size_t amount, auto &&work) {
            const auto start = Clock::now();
            work();
            time[item] += std::chrono::duration<double>(Clock::now() - start).count();
            bytes[item] += static_cast<double>(amount);
        };
        while (elapsed() - window_start < 0.5) {
            timed(0, buffer.size() * 8, [&] { read_bandwidth(threads, buffer); });
            for (int k = 0; k < 3; ++k) {
                timed(1 + k, dense[k].bytes, [&] { ggml_backend_graph_compute(backend, dense[k].graph); });
            }
            for (int k = 0; k < 2; ++k) {
                for (const int rows4 : {0, 1}) {
                    if (set_rows) {
                        set_rows(rows4 ? 1ull << expert_products[k].type : 0);
                    }
                    timed(4 + 2 * k + rows4, experts[k].weight_bytes_read,
                          [&] { ggml_backend_graph_compute(backend, experts[k].graph); });
                }
            }
        }
        double gb_s[items];
        for (int i = 0; i < items; ++i) {
            gb_s[i] = bytes[i] / time[i] / 1e9;
        }
        std::printf("%s  {\"t\": %.2f, \"raw_read_gb_s\": %.1f, \"q5_K\": %.1f, \"q8_0\": %.1f, \"q6_K\": %.1f, "
                    "\"iq2_xs\": %.1f, \"iq2_xs_rows4\": %.1f, \"iq4_nl\": %.1f, \"iq4_nl_rows4\": %.1f, "
                    "\"mhz\": [%s, %s, %s, %s]}",
                    first ? "" : ",\n", window_start, gb_s[0], gb_s[1], gb_s[2], gb_s[3], gb_s[4], gb_s[5], gb_s[6],
                    gb_s[7], cpu_mhz(0).c_str(), cpu_mhz(4).c_str(), cpu_mhz(7).c_str(), cpu_mhz(9).c_str());
        std::fflush(stdout);
        first = false;
    }
    if (set_rows) {
        set_rows(0);
    }
    std::printf("\n]");
    for (auto &d : dense) {
        ggml_backend_buffer_free(d.buffer);
        ggml_free(d.ctx);
    }
}
} // namespace

int main(int argc, char **argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 4;
    std::vector<int> columns = {1, 2, 3, 4};
    bool experts_only = false;
    double sustain_seconds = 0;
    for (int a = 2; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--experts") {
            experts_only = true;
        } else if (arg.rfind("--sustain=", 0) == 0) {
            sustain_seconds = std::atof(arg.c_str() + 10);
        } else {
            // Comma-separated batch widths; prompt-sized widths exercise the tiled GEMM path when enabled.
            columns.clear();
            for (const char *p = arg.c_str(); *p;) {
                columns.push_back(std::atoi(p));
                while (*p && *p != ',') {
                    ++p;
                }
                p += *p == ',';
            }
        }
    }
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
    const auto set_rows = reinterpret_cast<RowKernelSetter>(ggml_backend_reg_get_proc_address(reg, "ggml_cpu_set_row_kernels"));
    std::mt19937 random(7);
    std::printf("{\"threads\": %d, ", threads);
    if (sustain_seconds > 0) {
        sustain(backend, threads, set_rows, sustain_seconds, random);
    } else {
        // Memory read roofline for the same thread count.
        std::vector<double> rates;
        const std::vector<uint64_t> buffer((64u << 20) / 8, 1);
        for (int r = 0; r < 5; ++r) {
            rates.push_back(read_bandwidth(threads, buffer));
        }
        std::printf("\"read_bandwidth_gb_s\": %.1f, ", median(rates));
        if (!experts_only) {
            column_scaling(backend, columns, random);
            std::printf(", ");
        }
        expert_kernels(backend, set_rows, random);
    }
    std::printf("}\n");
    ggml_backend_free(backend);
    return 0;
}
