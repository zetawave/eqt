#pragma once

#include "engine.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_device;

namespace eqt {
class ExpertStore;

struct GpuOptions {
    // Smallest token count of a dense product or other node (min_rows) and of a routed product (min_expert_rows)
    // computed on the GPU. A routed product copies every expert its tokens use and runs one small tile per expert,
    // so it pays off only once each expert serves many tokens. Qwen3.6-35B-A3B UD-IQ2_M on the Galaxy (prompt
    // time, CPU vs GPU): dense products 3.9 vs 2.6 s at 425 tokens and slower below ~100 tokens; routed products
    // 4.9 vs 4.9 s at 425 tokens and 9.4 vs 6.2 s at 929 tokens.
    int min_rows = 128;
    int min_expert_rows = 640;
    // Helpers for staging copies.
    int copy_threads = 4;
    // How the node's other compute threads wait for the GPU: sleeping lets the governor lower CPU clocks for the
    // operations that follow, spinning takes power the GPU could use, WFE idles the core without a syscall.
    enum class Wait { sleep, spin, wfe } wait = Wait::wfe;
};

// Computes the CPU graph's prompt-sized matrix products, DeltaNet recurrences and conv-state concatenations on
// the GPU through GGML's Vulkan backend, using the CPU backend's external compute hook; every other
// node, and every decode-sized one, stays on the CPU unchanged. Weights stay where the CPU path keeps them: each offloaded product copies its weights (for
// routed experts, only the experts its tokens use) into one reused staging buffer that the GPU reads in place.
// Staging is host memory imported into Vulkan and CPU-cached, so results read back at memory speed. No resident
// copy of the model is made for the GPU.
class GpuOffload {
  public:
    // Loads GGML's Vulkan backend module and opens its first device; throws when there is none.
    GpuOffload(GpuOptions options, const ExpertStore *store);
    ~GpuOffload();
    GpuOffload(const GpuOffload &) = delete;
    GpuOffload &operator=(const GpuOffload &) = delete;

    // Staging bytes for products up to `largest_weight_bytes` of weights and `batch` tokens of `width` features
    // with `experts_used` routed experts per token, for memory admission.
    static uint64_t staging_estimate(uint64_t largest_weight_bytes, int batch, int64_t width, int experts_used);
    // Installs or removes the CPU backend hook; only while no graph computes.
    void install();
    void uninstall();
    // A failed GPU product leaves its result undefined: the engine's abort callback stops the graph, and later
    // products stay on the CPU.
    bool failed() const { return failed_.load(); }
    std::string error() const;
    // Returns the staging memory; the next GPU operation maps it again. Only while no graph computes.
    void release_staging();
    Json configuration() const;
    Json statistics() const;
    void begin_request();

  private:
    struct Staging {
        uint8_t *host = nullptr;
        size_t bytes = 0;
        ggml_backend_buffer *buffer = nullptr;
    };
    struct Copy {
        void *dst;
        const void *src;
        size_t bytes;
    };
    struct Support {
        bool dense = false;
        bool routed = false;
    };
    struct Stats {
        uint64_t dense_ops = 0, routed_ops = 0, node_ops = 0, rows = 0, experts_staged = 0;
        uint64_t weight_bytes = 0, input_bytes = 0, output_bytes = 0;
        double stage_ms = 0, gpu_ms = 0, readback_ms = 0, node_gpu_ms = 0;
        uint64_t peak_staging_bytes = 0;
    };

    static bool hook(ggml_tensor *dst, int ith, int nth, void *self);
    bool eligible(const ggml_tensor *dst) const;
    bool node_eligible(const ggml_tensor *dst) const;
    bool run(ggml_tensor *dst);
    bool run_node(ggml_tensor *dst);
    bool reserve(Staging &staging, size_t bytes);
    void release(Staging &staging);
    void probe_support();
    void copy_all();
    void copy_tasks();
    void helper();

    GpuOptions options_;
    const ExpertStore *store_;
    ggml_backend_device *device_ = nullptr;
    ggml_backend *backend_ = nullptr;
    std::string device_name_;
    size_t alignment_ = 256;
    double init_ms_ = 0;
    std::vector<Support> support_;
    Staging weights_, io_;
    std::vector<uint8_t> graph_memory_;
    std::vector<uint8_t> used_experts_;
    std::vector<const ggml_tensor *> roots_;
    static constexpr size_t max_node_bytes = size_t{768} << 20;
    bool installed_ = false;
    // Set by thread 0 while it computes a node, so the node's other threads sleep instead of spinning.
    std::atomic<const ggml_tensor *> busy_{nullptr};
    std::atomic<bool> failed_{false};
    mutable std::mutex error_mutex_;
    std::string error_;
    Stats stats_;

    // Staging copy pool: tasks_ is written only while no helper works on it (remaining_ == 0, working_ == 0).
    std::vector<Copy> copies_, tasks_;
    std::vector<std::thread> helpers_;
    std::mutex pool_mutex_;
    std::condition_variable wake_, done_;
    uint64_t generation_ = 0;
    int working_ = 0;
    bool stop_ = false;
    std::atomic<size_t> next_{0};
    std::atomic<size_t> remaining_{0};
};
} // namespace eqt
