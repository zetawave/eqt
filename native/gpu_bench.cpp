// Feasibility measurements for a GPU decode path through GGML's Vulkan backend: dispatch latency, matrix-vector
// weight bandwidth per format (device memory and imported host memory), routed-expert products and prompt-sized
// matrix products. Prints one JSON object; compare with eqt-kernel-bench for the CPU.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

long rss_kib() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            return std::atol(line.c_str() + 6);
        }
    }
    return -1;
}

// Median wall time of a synchronized graph compute, in microseconds, after warm-up runs.
double time_graph(ggml_backend_t backend, ggml_cgraph *graph, int runs = 20, int warmup = 5) {
    std::vector<double> times;
    for (int i = 0; i < warmup + runs; ++i) {
        const auto start = Clock::now();
        ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const double us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        if (i >= warmup) {
            times.push_back(us);
        }
    }
    return median(times);
}

std::vector<uint8_t> quantized(ggml_type type, int64_t width, int64_t rows, std::mt19937 &random) {
    std::normal_distribution<float> normal(0.0f, 0.02f);
    std::vector<float> source(width * rows), importance(width, 1.0f);
    for (auto &v : source) {
        v = normal(random);
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

// Allocates the context's tensors either in a device buffer or in page-aligned host memory imported into the
// device (VK_EXT_external_memory_host), which is how CPU-owned expert slots would be shared without copies.
struct Placement {
    ggml_backend_buffer_t buffer = nullptr;
    void *host = nullptr;
    size_t host_bytes = 0;

    bool allocate(ggml_context *ctx, ggml_backend_t backend, bool imported) {
        if (!imported) {
            buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            return buffer != nullptr;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        const size_t align = 1 << 16;
        size_t bytes = 0;
        for (ggml_tensor *t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            bytes += t->view_src ? 0 : GGML_PAD(ggml_nbytes(t), 256);
        }
        host_bytes = GGML_PAD(bytes, align);
#ifdef __linux__
        host = mmap(nullptr, host_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (host == MAP_FAILED) {
            host = nullptr;
            return false;
        }
#else
        return false;
#endif
        buffer = ggml_backend_dev_buffer_from_host_ptr(dev, host, host_bytes, host_bytes);
        if (!buffer) {
            return false;
        }
        auto *base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer));
        size_t offset = 0;
        for (ggml_tensor *t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (!t->view_src) {
                ggml_backend_tensor_alloc(buffer, t, base + offset);
                offset += GGML_PAD(ggml_nbytes(t), 256);
            }
        }
        for (ggml_tensor *t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (t->view_src) {
                ggml_backend_view_init(t);
            }
        }
        return true;
    }
    ~Placement() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
#ifdef __linux__
        if (host) {
            munmap(host, host_bytes);
        }
#endif
    }
};

bool first_row = true;
void row(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void row(const char *fmt, ...) {
    std::printf("%s  ", first_row ? "" : ",\n");
    first_row = false;
    va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
    std::fflush(stdout);
}

// One tiny op versus a chain of tiny ops: the fixed submit/fence cost and the per-dispatch cost.
void dispatch_latency(ggml_backend_t backend) {
    for (const int ops : {1, 16, 128, 512}) {
        ggml_init_params params = {ggml_tensor_overhead() * (ops + 4) + ggml_graph_overhead_custom(1024, false), nullptr, true};
        ggml_context *ctx = ggml_init(params);
        ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
        ggml_tensor *y = x;
        for (int i = 0; i < ops; ++i) {
            y = ggml_scale(ctx, y, 1.0001f);
        }
        ggml_cgraph *graph = ggml_new_graph_custom(ctx, 1024, false);
        ggml_build_forward_expand(graph, y);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<float> host(2048, 1.0f);
        ggml_backend_tensor_set(x, host.data(), 0, ggml_nbytes(x));
        const double graph_us = time_graph(backend, graph, 50, 10);
        // Upload the input and read the output back, as a CPU/GPU split would each layer.
        std::vector<double> round;
        for (int i = 0; i < 60; ++i) {
            const auto start = Clock::now();
            ggml_backend_tensor_set(x, host.data(), 0, ggml_nbytes(x));
            ggml_backend_graph_compute(backend, graph);
            ggml_backend_tensor_get(y, host.data(), 0, ggml_nbytes(y));
            const double us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
            if (i >= 10) {
                round.push_back(us);
            }
        }
        row("{\"test\": \"dispatch\", \"ops\": %d, \"graph_us\": %.1f, \"with_8k_upload_and_readback_us\": %.1f}", ops,
            graph_us, median(round));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
}

// Streaming read roofline: row sums of a 256 MiB F32 matrix read every byte once and write almost nothing.
void read_bandwidth(ggml_backend_t backend) {
    ggml_init_params params = {ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true};
    ggml_context *ctx = ggml_init(params);
    ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 16384);
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_sum_rows(ctx, x));
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_buffer_clear(buffer, 0);
    const double us = time_graph(backend, graph);
    row("{\"test\": \"read_bandwidth\", \"bytes\": %zu, \"us\": %.1f, \"gb_s\": %.1f}", ggml_nbytes(x), us,
        ggml_nbytes(x) / us / 1e3);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

// Sixteen distinct matrices per graph keep the working set beyond the GPU caches, as in decode.
void matvec(ggml_backend_t backend, ggml_type type, int64_t width, int64_t rows, int columns, int copies, bool imported,
            std::mt19937 &random) {
    const std::vector<uint8_t> packed = quantized(type, width, rows, random);
    ggml_init_params params = {ggml_tensor_overhead() * (2 * copies + 2) + ggml_graph_overhead(), nullptr, true};
    ggml_context *ctx = ggml_init(params);
    std::vector<ggml_tensor *> weights;
    for (int c = 0; c < copies; ++c) {
        weights.push_back(ggml_new_tensor_2d(ctx, type, width, rows));
    }
    ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, columns);
    ggml_cgraph *graph = ggml_new_graph(ctx);
    for (auto *w : weights) {
        ggml_build_forward_expand(graph, ggml_mul_mat(ctx, w, x));
    }
    Placement placement;
    if (!placement.allocate(ctx, backend, imported)) {
        row("{\"test\": \"matvec\", \"type\": \"%s\", \"imported\": %s, \"error\": \"allocation failed\"}",
            ggml_type_name(type), imported ? "true" : "false");
        ggml_free(ctx);
        return;
    }
    for (auto *w : weights) {
        ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    }
    fill_normal(x, random);
    const double us = time_graph(backend, graph) / copies;
    const double flops = 2.0 * width * rows * columns;
    row("{\"test\": \"%s\", \"type\": \"%s\", \"shape\": [%lld, %lld], \"columns\": %d, \"imported\": %s, \"us\": %.1f, "
        "\"weight_gb_s\": %.1f, \"gflops\": %.1f}",
        columns == 1 ? "matvec" : "matmul", ggml_type_name(type), (long long)width, (long long)rows, columns,
        imported ? "true" : "false", us, packed.size() / us / 1e3, flops / us / 1e3);
    ggml_free(ctx);
}

// Decode-shaped routed experts: eight of 256 experts per token, gate and up products then down, as in one
// Qwen3.6-35B-A3B MoE layer. Expert choice changes every run so caches do not hold the selected experts.
void routed_experts(ggml_backend_t backend, ggml_type gate_type, ggml_type down_type, int layers, bool imported,
                    std::mt19937 &random) {
    constexpr int64_t hidden = 2048, ff = 512, experts = 256, used = 8;
    const std::vector<uint8_t> gate_packed = quantized(gate_type, hidden, ff, random);
    const std::vector<uint8_t> down_packed = quantized(down_type, ff, hidden, random);
    ggml_init_params params = {ggml_tensor_overhead() * (layers * 20 + 4) + ggml_graph_overhead(), nullptr, true};
    ggml_context *ctx = ggml_init(params);
    ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, 1, 1);
    ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, 1);
    std::vector<ggml_tensor *> banks;
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_tensor *h = x;
    for (int l = 0; l < layers; ++l) {
        ggml_tensor *gate = ggml_new_tensor_3d(ctx, gate_type, hidden, ff, experts);
        ggml_tensor *up = ggml_new_tensor_3d(ctx, gate_type, hidden, ff, experts);
        ggml_tensor *down = ggml_new_tensor_3d(ctx, down_type, ff, hidden, experts);
        banks.insert(banks.end(), {gate, up, down});
        ggml_tensor *g = ggml_mul_mat_id(ctx, gate, h, ids);
        ggml_tensor *u = ggml_mul_mat_id(ctx, up, h, ids);
        ggml_tensor *a = ggml_mul(ctx, ggml_silu(ctx, g), u);
        ggml_tensor *d = ggml_mul_mat_id(ctx, down, a, ids); // [hidden, used, 1]
        h = ggml_reshape_3d(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, d, hidden, used)))),
                            hidden, 1, 1);
    }
    ggml_build_forward_expand(graph, h);
    Placement placement;
    if (!placement.allocate(ctx, backend, imported)) {
        row("{\"test\": \"experts\", \"imported\": %s, \"error\": \"allocation failed\"}", imported ? "true" : "false");
        ggml_free(ctx);
        return;
    }
    for (size_t b = 0; b < banks.size(); ++b) {
        const auto &packed = b % 3 == 2 ? down_packed : gate_packed;
        for (int64_t e = 0; e < experts; ++e) {
            ggml_backend_tensor_set(banks[b], packed.data(), e * packed.size(), packed.size());
        }
    }
    fill_normal(x, random);
    std::vector<double> times;
    std::vector<int32_t> chosen(used);
    for (int i = 0; i < 30; ++i) {
        for (int k = 0; k < used; ++k) {
            chosen[k] = (i * 37 + k * 29) % experts;
        }
        ggml_backend_tensor_set(ids, chosen.data(), 0, ggml_nbytes(ids));
        const auto start = Clock::now();
        ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const double us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        if (i >= 6) {
            times.push_back(us);
        }
    }
    const double us = median(times) / layers;
    const double bytes = used * (2.0 * gate_packed.size() + down_packed.size());
    row("{\"test\": \"experts\", \"gate_up\": \"%s\", \"down\": \"%s\", \"layers\": %d, \"imported\": %s, "
        "\"us_per_layer\": %.1f, \"expert_bytes_per_layer\": %.0f, \"weight_gb_s\": %.1f}",
        ggml_type_name(gate_type), ggml_type_name(down_type), layers, imported ? "true" : "false", us, bytes,
        bytes / us / 1e3);
    ggml_free(ctx);
}

// A graph of sixteen distinct matrix-vector products on one backend, rerun until stopped; reports weight GB/s.
struct MatvecLoop {
    ggml_context *ctx = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    size_t bytes = 0;
    double flops = 0;

    MatvecLoop(ggml_backend_t backend, ggml_type type, std::mt19937 &random, int columns = 1, int copies = 16) {
        constexpr int64_t width = 2048, rows = 4096;
        const std::vector<uint8_t> packed = quantized(type, width, rows, random);
        ggml_init_params params = {ggml_tensor_overhead() * (2 * copies + 2) + ggml_graph_overhead(), nullptr, true};
        ctx = ggml_init(params);
        std::vector<ggml_tensor *> weights;
        for (int c = 0; c < copies; ++c) {
            weights.push_back(ggml_new_tensor_2d(ctx, type, width, rows));
        }
        ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, columns);
        graph = ggml_new_graph(ctx);
        for (auto *w : weights) {
            ggml_build_forward_expand(graph, ggml_mul_mat(ctx, w, x));
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        for (auto *w : weights) {
            ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
        }
        fill_normal(x, random);
        bytes = packed.size() * copies;
        flops = 2.0 * width * rows * columns * copies;
    }
    ~MatvecLoop() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    double run(ggml_backend_t backend, const std::atomic<bool> &stop) {
        ggml_backend_graph_compute(backend, graph); // warm-up
        ggml_backend_synchronize(backend);
        size_t done = 0;
        const auto start = Clock::now();
        while (!stop.load(std::memory_order_relaxed)) {
            ggml_backend_graph_compute(backend, graph);
            ggml_backend_synchronize(backend);
            ++done;
        }
        return done * bytes / std::chrono::duration<double>(Clock::now() - start).count() / 1e9;
    }
};

ggml_backend_t cpu_backend(int threads) {
    for (const char *variant : {"android_armv9.0_1", "android_armv8.6_1", "android_armv8.2_2"}) {
        const std::string library = std::string("libggml-cpu-") + variant + ".so";
        if (ggml_backend_reg_t reg = ggml_backend_load(library.c_str())) {
            ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
            if (auto set_threads = reinterpret_cast<void (*)(ggml_backend_t, int)>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"))) {
                set_threads(backend, threads);
            }
            auto create = reinterpret_cast<decltype(&ggml_threadpool_new)>(
                ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
            auto attach = reinterpret_cast<void (*)(ggml_backend_t, ggml_threadpool *)>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
            if (create && attach) {
                auto pool_params = ggml_threadpool_params_default(threads);
                attach(backend, create(&pool_params));
            }
            return backend;
        }
    }
    return nullptr;
}

// CPU and GPU reading weights at the same time: if the sum exceeds either alone, a split decode can use both.
void concurrent(ggml_backend_t gpu, std::mt19937 &random) {
    ggml_backend_t cpu = cpu_backend(6);
    if (!cpu) {
        row("{\"test\": \"concurrent\", \"error\": \"no CPU backend\"}");
        return;
    }
    for (const ggml_type type : {GGML_TYPE_Q5_K, GGML_TYPE_IQ2_XXS}) {
        MatvecLoop cpu_loop(cpu, type, random), gpu_loop(gpu, type, random);
        auto measure = [&](bool use_cpu, bool use_gpu, double &cpu_rate, double &gpu_rate) {
            std::atomic<bool> stop{false};
            cpu_rate = gpu_rate = 0;
            std::thread gpu_thread;
            if (use_gpu) {
                gpu_thread = std::thread([&] { gpu_rate = gpu_loop.run(gpu, stop); });
            }
            std::thread cpu_thread;
            if (use_cpu) {
                cpu_thread = std::thread([&] { cpu_rate = cpu_loop.run(cpu, stop); });
            }
            std::this_thread::sleep_for(std::chrono::seconds(3));
            stop = true;
            if (gpu_thread.joinable()) {
                gpu_thread.join();
            }
            if (cpu_thread.joinable()) {
                cpu_thread.join();
            }
        };
        double c_alone, g_alone, c_both, g_both, unused;
        measure(true, false, c_alone, unused);
        measure(false, true, unused, g_alone);
        measure(true, true, c_both, g_both);
        row("{\"test\": \"concurrent\", \"type\": \"%s\", \"cpu_alone_gb_s\": %.1f, \"gpu_alone_gb_s\": %.1f, "
            "\"cpu_with_gpu_gb_s\": %.1f, \"gpu_with_cpu_gb_s\": %.1f, \"sum_gb_s\": %.1f}",
            ggml_type_name(type), c_alone, g_alone, c_both, g_both, c_both + g_both);
    }
    ggml_backend_free(cpu);
}
} // namespace

int main(int argc, char **argv) {
    const std::string only = argc > 1 ? argv[1] : "all";
    ggml_backend_reg_t reg = ggml_backend_load("libggml-vulkan.so");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        std::fprintf(stderr, "no Vulkan device\n");
        return 1;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    const long rss_before = rss_kib();
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        std::fprintf(stderr, "cannot initialize %s\n", props.name);
        return 1;
    }
    std::printf("{\"device\": \"%s\", \"description\": \"%s\", \"memory_free\": %zu, \"memory_total\": %zu, "
                "\"rss_kib_before_init\": %ld, \"rss_kib_after_init\": %ld, \"results\": [\n",
                props.name, props.description, props.memory_free, props.memory_total, rss_before, rss_kib());
    std::mt19937 random(7);
    if (only == "all" || only == "dispatch") {
        dispatch_latency(backend);
    }
    if (only == "all" || only == "bandwidth") {
        read_bandwidth(backend);
    }
    const ggml_type types[] = {GGML_TYPE_F16,  GGML_TYPE_Q8_0,    GGML_TYPE_Q4_K,   GGML_TYPE_Q5_K,
                               GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS};
    if (only == "all" || only == "matvec") {
        for (const ggml_type type : types) {
            matvec(backend, type, 2048, 4096, 1, 16, false, random);
        }
        // The output head, the largest single matrix per token.
        matvec(backend, GGML_TYPE_Q4_K, 2048, 248320, 1, 1, false, random);
        matvec(backend, GGML_TYPE_Q6_K, 2048, 248320, 1, 1, false, random);
    }
    if (only == "all" || only == "imported") {
        const long before = rss_kib();
        for (const ggml_type type : {GGML_TYPE_Q5_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS}) {
            matvec(backend, type, 2048, 4096, 1, 16, true, random);
        }
        row("{\"test\": \"imported_rss\", \"rss_kib_delta\": %ld}", rss_kib() - before);
    }
    if (only == "all" || only == "experts") {
        for (const bool imported : {false, true}) {
            routed_experts(backend, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, 4, imported, random);
            routed_experts(backend, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 2, imported, random);
        }
    }
    if (only == "tune") {
        for (const ggml_type type : types) {
            matvec(backend, type, 2048, 4096, 1, 16, false, random);
        }
        routed_experts(backend, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, 4, false, random);
    }
    if (only == "cpu_matmul") {
        // Prompt-sized products on the six big cores, for comparison with the "matmul" rows.
        ggml_backend_t cpu = cpu_backend(6);
        for (const ggml_type type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS}) {
            for (const int columns : {64, 512}) {
                MatvecLoop loop(cpu, type, random, columns, 2);
                std::atomic<bool> stop{false};
                double rate = 0;
                std::thread worker([&] { rate = loop.run(cpu, stop); });
                std::this_thread::sleep_for(std::chrono::seconds(4));
                stop = true;
                worker.join();
                row("{\"test\": \"cpu_matmul\", \"type\": \"%s\", \"columns\": %d, \"gflops\": %.1f}", ggml_type_name(type),
                    columns, rate / loop.bytes * loop.flops);
            }
        }
        ggml_backend_free(cpu);
    }
    if (only == "concurrent") {
        concurrent(backend, random);
    }
    if (only == "all" || only == "matmul") {
        for (const ggml_type type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS}) {
            for (const int columns : {8, 64, 512}) {
                matvec(backend, type, 2048, 4096, columns, 4, false, random);
            }
        }
    }
    std::printf("\n], \"rss_kib_at_end\": %ld}\n", rss_kib());
    ggml_backend_free(backend);
    return 0;
}
