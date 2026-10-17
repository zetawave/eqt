#include "gpu_offload.h"

#include "expert_store.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <sys/mman.h>
#endif

namespace eqt {
namespace {
using Clock = std::chrono::steady_clock;

double since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

constexpr size_t page = 4096;
constexpr size_t staging_granule = size_t{2} << 20;
constexpr size_t copy_chunk = size_t{1} << 20;
constexpr int graph_tensors = 32;

size_t round_up(size_t value, size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

uint8_t *map_pages(size_t bytes) {
#ifdef _WIN32
    return static_cast<uint8_t *>(_aligned_malloc(bytes, page));
#else
    void *memory = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return memory == MAP_FAILED ? nullptr : static_cast<uint8_t *>(memory);
#endif
}

void unmap_pages(uint8_t *memory, size_t bytes) {
#ifdef _WIN32
    (void)bytes;
    _aligned_free(memory);
#else
    munmap(memory, bytes);
#endif
}

using ExternalCompute = void (*)(bool (*)(ggml_tensor *, int, int, void *), void *);
ExternalCompute external_compute_setter() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    return reg ? reinterpret_cast<ExternalCompute>(ggml_backend_reg_get_proc_address(reg, "ggml_cpu_set_external_compute"))
               : nullptr;
}

// Tokens a prompt-sized node processes, or 0 for nodes that stay on the CPU. The conv-state concatenation of a
// DeltaNet layer joins 8192 channels per token. Attention stays on the CPU: without matrix cores the Vulkan
// flash-attention shader took ten times as long for 929 prompt tokens on the Xclipse 940 (measured).
int64_t node_tokens(const ggml_tensor *node) {
    switch (node->op) {
    case GGML_OP_GATED_DELTA_NET:
        return node->src[0]->ne[2];
    case GGML_OP_CONCAT:
        return ggml_nelements(node) / 8192;
    default:
        return 0;
    }
}

const ggml_tensor *root_of(const ggml_tensor *tensor) {
    return tensor->view_src ? tensor->view_src : tensor;
}

// Waits until `word` may have changed from `old`: on AArch64 the core idles in WFE until another core writes the
// word's cache line (or the kernel's event stream fires), staying runnable so the governor keeps its clock.
void wait_change(const std::atomic<const ggml_tensor *> &word, const ggml_tensor *old) {
#if defined(__aarch64__)
    const ggml_tensor *value;
    __asm__ __volatile__("ldaxr %0, [%1]" : "=&r"(value) : "r"(&word) : "memory");
    if (value == old) {
        __asm__ __volatile__("wfe" ::: "memory");
    }
#else
    (void)word;
    (void)old;
    std::this_thread::yield();
#endif
}

void cpu_relax() {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#else
    std::this_thread::yield();
#endif
}
} // namespace

GpuOffload::GpuOffload(GpuOptions options, const ExpertStore *store) : options_(options), store_(store) {
    const auto start = Clock::now();
    if (!external_compute_setter()) {
        throw std::runtime_error("The CPU backend has no external matrix product hook");
    }
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
    if (!reg) {
        reg = ggml_backend_load("libggml-vulkan.so");
    }
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        throw std::runtime_error("No Vulkan GPU is available");
    }
    device_ = ggml_backend_reg_dev_get(reg, 0);
    device_name_ = ggml_backend_dev_description(device_);
    backend_ = ggml_backend_dev_init(device_, nullptr);
    if (!backend_) {
        throw std::runtime_error("Cannot open the GPU " + device_name_);
    }
    alignment_ = std::max<size_t>(ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(device_)), 64);
    graph_memory_.resize(ggml_tensor_overhead() * graph_tensors + ggml_graph_overhead_custom(graph_tensors, false));
    probe_support();
    // Staging must be importable: the GPU reads CPU-written weights in place and results come back cached.
    if (!reserve(io_, staging_granule)) {
        const std::string name = device_name_;
        ggml_backend_free(backend_);
        throw std::runtime_error("The GPU " + name + " cannot use host memory buffers");
    }
    release(io_);
    for (int i = 0; i < options_.copy_threads; ++i) {
        helpers_.emplace_back([this] { helper(); });
    }
    init_ms_ = since(start);
}

GpuOffload::~GpuOffload() {
    uninstall();
    {
        std::lock_guard lock(pool_mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (auto &thread : helpers_) {
        thread.join();
    }
    release(weights_);
    release(io_);
    ggml_backend_free(backend_);
}

uint64_t GpuOffload::staging_estimate(uint64_t largest_weight_bytes, int batch, int64_t width, int experts_used) {
    // Routed down products dominate the activations: inputs and outputs of `experts_used` rows per token.
    const uint64_t activations = static_cast<uint64_t>(batch) * std::max(experts_used, 1) * width * sizeof(float) * 2;
    return round_up(largest_weight_bytes, staging_granule) + round_up(activations, staging_granule) + staging_granule;
}

void GpuOffload::install() {
    external_compute_setter()(&GpuOffload::hook, this);
    installed_ = true;
}

void GpuOffload::uninstall() {
    if (installed_) {
        external_compute_setter()(nullptr, nullptr);
        installed_ = false;
    }
}

std::string GpuOffload::error() const {
    std::lock_guard lock(error_mutex_);
    return error_;
}

// The Vulkan backend's support depends on the weight type only for these products; one representative shape per
// type is checked once, so compute threads read a fixed table.
void GpuOffload::probe_support() {
    support_.assign(GGML_TYPE_COUNT, {});
    std::vector<uint8_t> memory(ggml_tensor_overhead() * 16 + ggml_graph_overhead());
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const auto type = static_cast<ggml_type>(t);
        const auto *traits = ggml_get_type_traits(type);
        if (!traits || traits->blck_size == 0 || traits->type_size == 0 || 4096 % traits->blck_size != 0) {
            continue;
        }
        ggml_init_params params = {memory.size(), memory.data(), true};
        ggml_context *ctx = ggml_init(params);
        ggml_tensor *a = ggml_new_tensor_2d(ctx, type, 4096, 64);
        ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 64);
        ggml_tensor *bank = ggml_new_tensor_3d(ctx, type, 4096, 64, 8);
        ggml_tensor *rows = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4096, 1, 64);
        ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 64);
        support_[t].dense = ggml_backend_dev_supports_op(device_, ggml_mul_mat(ctx, a, b));
        support_[t].routed = ggml_backend_dev_supports_op(device_, ggml_mul_mat_id(ctx, bank, rows, ids));
        ggml_free(ctx);
    }
}

bool GpuOffload::reserve(Staging &staging, size_t bytes) {
    if (staging.buffer && staging.bytes >= bytes) {
        return true;
    }
    release(staging);
    const size_t size = round_up(std::max(bytes, page), staging_granule);
    staging.host = map_pages(size);
    if (!staging.host) {
        return false;
    }
    staging.bytes = size;
    staging.buffer = ggml_backend_dev_buffer_from_host_ptr(device_, staging.host, size, size);
    if (!staging.buffer) {
        release(staging);
        return false;
    }
    return true;
}

void GpuOffload::release(Staging &staging) {
    if (staging.buffer) {
        ggml_backend_buffer_free(staging.buffer);
    }
    if (staging.host) {
        unmap_pages(staging.host, staging.bytes);
    }
    staging = {};
}

bool GpuOffload::eligible(const ggml_tensor *dst) const {
    if (dst->op != GGML_OP_MUL_MAT && dst->op != GGML_OP_MUL_MAT_ID) {
        return node_eligible(dst);
    }
    const ggml_tensor *weights = dst->src[0];
    const ggml_tensor *input = dst->src[1];
    if (failed_.load(std::memory_order_relaxed) || input->type != GGML_TYPE_F32 || input->nb[0] != sizeof(float) ||
        !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst) || weights->ne[3] != 1) {
        return false;
    }
    const Support &support = support_[weights->type];
    if (dst->op == GGML_OP_MUL_MAT) {
        return support.dense && weights->ne[2] == 1 && input->ne[1] * input->ne[2] * input->ne[3] >= options_.min_rows;
    }
    const ggml_tensor *ids = dst->src[2];
    return support.routed && ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) && input->ne[3] == 1 &&
           ids->ne[1] >= options_.min_expert_rows;
}

// Other prompt-sized nodes run on the GPU when their sources can be rebuilt there: each source's root tensor is
// copied whole and views keep their strides and offsets, so the node reads exactly what the CPU kernel would.
bool GpuOffload::node_eligible(const ggml_tensor *dst) const {
    if (failed_.load(std::memory_order_relaxed) || node_tokens(dst) < options_.min_rows || dst->view_src ||
        !ggml_is_contiguous(dst)) {
        return false;
    }
    size_t bytes = ggml_nbytes(dst);
    for (int i = 0; i < GGML_MAX_SRC && dst->src[i]; ++i) {
        const ggml_tensor *root = root_of(dst->src[i]);
        if (!ggml_is_contiguous(root) || !root->data) {
            return false;
        }
        bytes += ggml_nbytes(root);
    }
    return bytes <= max_node_bytes && ggml_backend_dev_supports_op(device_, dst);
}

bool GpuOffload::hook(ggml_tensor *dst, int ith, int, void *opaque) {
    auto &self = *static_cast<GpuOffload *>(opaque);
    if (!self.eligible(dst)) {
        return false;
    }
    if (ith == 0) {
        self.busy_.store(dst, std::memory_order_release);
        if (!self.run(dst)) {
            self.failed_.store(true);
        }
        self.busy_.store(nullptr, std::memory_order_release);
        self.busy_.notify_all();
        return true;
    }
    // Thread 0 marks the node within microseconds of the previous barrier; a thread arriving after it finished
    // stops looking after a short spin. Correctness never depends on this wait: the node's barrier orders dst.
    const ggml_tensor *seen = self.busy_.load(std::memory_order_acquire);
    for (int spin = 0; seen != dst && spin < 4096; ++spin) {
        cpu_relax();
        seen = self.busy_.load(std::memory_order_acquire);
    }
    while (seen == dst) {
        switch (self.options_.wait) {
        case GpuOptions::Wait::sleep:
            self.busy_.wait(seen, std::memory_order_acquire);
            break;
        case GpuOptions::Wait::spin:
            cpu_relax();
            break;
        case GpuOptions::Wait::wfe:
            wait_change(self.busy_, seen);
            break;
        }
        seen = self.busy_.load(std::memory_order_acquire);
    }
    return true;
}

bool GpuOffload::run_node(ggml_tensor *dst) {
    const auto start = Clock::now();
    roots_.clear();
    for (int i = 0; i < GGML_MAX_SRC && dst->src[i]; ++i) {
        const ggml_tensor *root = root_of(dst->src[i]);
        if (std::find(roots_.begin(), roots_.end(), root) == roots_.end()) {
            roots_.push_back(root);
        }
    }
    size_t offsets[GGML_MAX_SRC];
    size_t used = 0;
    for (size_t r = 0; r < roots_.size(); ++r) {
        offsets[r] = used;
        used = round_up(used + ggml_nbytes(roots_[r]), alignment_);
    }
    const size_t output_offset = used;
    const size_t output_bytes = ggml_nbytes(dst);
    if (!reserve(io_, output_offset + output_bytes)) {
        std::lock_guard lock(error_mutex_);
        error_ = "Cannot allocate GPU staging memory";
        return false;
    }
    ggml_init_params params = {graph_memory_.size(), graph_memory_.data(), true};
    ggml_context *ctx = ggml_init(params);
    auto *io_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(io_.buffer));
    ggml_tensor *gpu_roots[GGML_MAX_SRC];
    copies_.clear();
    for (size_t r = 0; r < roots_.size(); ++r) {
        gpu_roots[r] = ggml_new_tensor(ctx, roots_[r]->type, GGML_MAX_DIMS, roots_[r]->ne);
        ggml_backend_tensor_alloc(io_.buffer, gpu_roots[r], io_base + offsets[r]);
        copies_.push_back({io_.host + offsets[r], roots_[r]->data, ggml_nbytes(roots_[r])});
        stats_.input_bytes += ggml_nbytes(roots_[r]);
    }
    // The node keeps the CPU node's operation and parameters; only its sources move.
    ggml_tensor *node = ggml_new_tensor(ctx, dst->type, GGML_MAX_DIMS, dst->ne);
    node->op = dst->op;
    std::memcpy(node->op_params, dst->op_params, sizeof(dst->op_params));
    for (int i = 0; i < GGML_MAX_SRC && dst->src[i]; ++i) {
        const ggml_tensor *src = dst->src[i];
        const size_t r = std::find(roots_.begin(), roots_.end(), root_of(src)) - roots_.begin();
        if (!src->view_src) {
            node->src[i] = gpu_roots[r];
            continue;
        }
        ggml_tensor *view = ggml_view_4d(ctx, gpu_roots[r], src->ne[0], src->ne[1], src->ne[2], src->ne[3], src->nb[1],
                                         src->nb[2], src->nb[3], src->view_offs);
        view->nb[0] = src->nb[0];
        ggml_backend_view_init(view);
        node->src[i] = view;
    }
    ggml_backend_tensor_alloc(io_.buffer, node, io_base + output_offset);
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, graph_tensors, false);
    ggml_build_forward_expand(graph, node);
    copy_all();
    ++stats_.node_ops;
    const auto compute_start = Clock::now();
    stats_.stage_ms += std::chrono::duration<double, std::milli>(compute_start - start).count();
    const ggml_status status = ggml_backend_graph_compute(backend_, graph);
    ggml_free(ctx);
    const auto readback_start = Clock::now();
    const double gpu_ms = std::chrono::duration<double, std::milli>(readback_start - compute_start).count();
    stats_.gpu_ms += gpu_ms;
    stats_.node_gpu_ms += gpu_ms;
    if (status != GGML_STATUS_SUCCESS) {
        std::lock_guard lock(error_mutex_);
        error_ = std::string("GPU ") + ggml_op_name(dst->op) + " failed (status " + std::to_string(static_cast<int>(status)) + ")";
        return false;
    }
    copies_.clear();
    copies_.push_back({dst->data, io_.host + output_offset, output_bytes});
    copy_all();
    stats_.output_bytes += output_bytes;
    stats_.readback_ms += since(readback_start);
    return true;
}

bool GpuOffload::run(ggml_tensor *dst) {
    if (dst->op != GGML_OP_MUL_MAT && dst->op != GGML_OP_MUL_MAT_ID) {
        return run_node(dst);
    }
    const auto start = Clock::now();
    const ggml_tensor *weights = dst->src[0];
    const ggml_tensor *input = dst->src[1];
    const ggml_tensor *ids = dst->src[2];
    const bool routed = dst->op == GGML_OP_MUL_MAT_ID;
    const size_t weight_bytes = ggml_nbytes(weights);
    const size_t row_bytes = static_cast<size_t>(input->ne[0]) * sizeof(float);
    const int64_t input_rows = input->ne[1] * input->ne[2] * input->ne[3];
    const size_t input_bytes = row_bytes * input_rows;
    const size_t ids_bytes = routed ? static_cast<size_t>(ids->ne[0] * ids->ne[1]) * sizeof(int32_t) : 0;
    const size_t output_bytes = ggml_nbytes(dst);
    const size_t ids_offset = round_up(input_bytes, alignment_);
    const size_t output_offset = round_up(ids_offset + ids_bytes, alignment_);
    if (!reserve(weights_, weight_bytes) || !reserve(io_, output_offset + output_bytes)) {
        std::lock_guard lock(error_mutex_);
        error_ = "Cannot allocate GPU staging memory";
        return false;
    }

    ggml_init_params params = {graph_memory_.size(), graph_memory_.data(), true};
    ggml_context *ctx = ggml_init(params);
    ggml_tensor *a = ggml_new_tensor(ctx, weights->type, GGML_MAX_DIMS, weights->ne);
    ggml_tensor *b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, input->ne);
    ggml_tensor *selection = routed ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1]) : nullptr;
    ggml_tensor *c = routed ? ggml_mul_mat_id(ctx, a, b, selection) : ggml_mul_mat(ctx, a, b);
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, graph_tensors, false);
    ggml_build_forward_expand(graph, c);
    if (!ggml_are_same_shape(c, dst)) {
        ggml_free(ctx);
        std::lock_guard lock(error_mutex_);
        error_ = "GPU product shape differs from the CPU graph";
        return false;
    }
    auto *weights_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(weights_.buffer));
    auto *io_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(io_.buffer));
    ggml_backend_tensor_alloc(weights_.buffer, a, weights_base);
    ggml_backend_tensor_alloc(io_.buffer, b, io_base);
    if (selection) {
        ggml_backend_tensor_alloc(io_.buffer, selection, io_base + ids_offset);
    }
    ggml_backend_tensor_alloc(io_.buffer, c, io_base + output_offset);

    copies_.clear();
    if (routed) {
        // Only the experts these tokens use are staged, at their bank positions; the others are never read.
        const int64_t experts = weights->ne[2];
        used_experts_.assign(static_cast<size_t>(experts), 0);
        auto *selected = reinterpret_cast<int32_t *>(io_.host + ids_offset);
        for (int64_t row = 0; row < ids->ne[1]; ++row) {
            const auto *source = reinterpret_cast<const int32_t *>(static_cast<const char *>(ids->data) + row * ids->nb[1]);
            for (int64_t k = 0; k < ids->ne[0]; ++k) {
                const int32_t expert = source[k];
                if (expert < 0 || expert >= experts) {
                    ggml_free(ctx);
                    std::lock_guard lock(error_mutex_);
                    error_ = "Router selected an out-of-range expert";
                    return false;
                }
                selected[row * ids->ne[0] + k] = expert;
                used_experts_[expert] = 1;
            }
        }
        const size_t expert_bytes = weights->nb[2];
        for (int64_t e = 0; e < experts; ++e) {
            if (!used_experts_[e]) {
                continue;
            }
            const void *source = store_ ? store_->expert_data(weights, e) : nullptr;
            if (!source) {
                source = static_cast<const char *>(weights->data) + e * expert_bytes;
            }
            copies_.push_back({weights_.host + e * expert_bytes, source, expert_bytes});
            ++stats_.experts_staged;
            stats_.weight_bytes += expert_bytes;
        }
        ++stats_.routed_ops;
    } else {
        copies_.push_back({weights_.host, weights->data, weight_bytes});
        stats_.weight_bytes += weight_bytes;
        ++stats_.dense_ops;
    }
    if (ggml_is_contiguous(input)) {
        copies_.push_back({io_.host, input->data, input_bytes});
    } else {
        for (int64_t i3 = 0; i3 < input->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < input->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < input->ne[1]; ++i1) {
                    const int64_t row = (i3 * input->ne[2] + i2) * input->ne[1] + i1;
                    copies_.push_back({io_.host + row * row_bytes,
                                       static_cast<const char *>(input->data) + i1 * input->nb[1] + i2 * input->nb[2] +
                                           i3 * input->nb[3],
                                       row_bytes});
                }
            }
        }
    }
    copy_all();
    stats_.input_bytes += input_bytes;
    stats_.rows += static_cast<uint64_t>(routed ? ids->ne[1] : input_rows);
    const auto compute_start = Clock::now();
    stats_.stage_ms += std::chrono::duration<double, std::milli>(compute_start - start).count();

    const ggml_status status = ggml_backend_graph_compute(backend_, graph);
    ggml_free(ctx);
    const auto readback_start = Clock::now();
    stats_.gpu_ms += std::chrono::duration<double, std::milli>(readback_start - compute_start).count();
    if (status != GGML_STATUS_SUCCESS) {
        std::lock_guard lock(error_mutex_);
        error_ = "GPU computation failed (status " + std::to_string(static_cast<int>(status)) + ")";
        return false;
    }
    copies_.clear();
    copies_.push_back({dst->data, io_.host + output_offset, output_bytes});
    copy_all();
    stats_.output_bytes += output_bytes;
    stats_.readback_ms += since(readback_start);
    return true;
}

void GpuOffload::copy_all() {
    tasks_.clear();
    for (const Copy &copy : copies_) {
        for (size_t at = 0; at < copy.bytes; at += copy_chunk) {
            tasks_.push_back({static_cast<char *>(copy.dst) + at, static_cast<const char *>(copy.src) + at,
                              std::min(copy_chunk, copy.bytes - at)});
        }
    }
    if (tasks_.empty()) {
        return;
    }
    {
        std::lock_guard lock(pool_mutex_);
        next_.store(0);
        remaining_.store(tasks_.size());
        ++generation_;
    }
    if (tasks_.size() > 1) {
        wake_.notify_all();
    }
    copy_tasks();
    std::unique_lock lock(pool_mutex_);
    done_.wait(lock, [&] { return remaining_.load() == 0 && working_ == 0; });
}

void GpuOffload::copy_tasks() {
    for (size_t i; (i = next_.fetch_add(1)) < tasks_.size();) {
        std::memcpy(tasks_[i].dst, tasks_[i].src, tasks_[i].bytes);
        remaining_.fetch_sub(1);
    }
}

void GpuOffload::helper() {
    uint64_t seen = 0;
    std::unique_lock lock(pool_mutex_);
    for (;;) {
        wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) {
            return;
        }
        seen = generation_;
        // A helper that wakes after its round ended must not read tasks_ while the next round fills it.
        if (remaining_.load() == 0) {
            continue;
        }
        ++working_;
        lock.unlock();
        copy_tasks();
        lock.lock();
        --working_;
        done_.notify_all();
    }
}

Json GpuOffload::configuration() const {
    Json dense = Json::array(), routed = Json::array();
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (support_[t].dense) {
            dense.push_back(ggml_type_name(static_cast<ggml_type>(t)));
        }
        if (support_[t].routed) {
            routed.push_back(ggml_type_name(static_cast<ggml_type>(t)));
        }
    }
    return {{"backend", "vulkan"},
            {"device", device_name_},
            {"init_ms", init_ms_},
            {"min_rows", options_.min_rows},
            {"min_expert_rows", options_.min_expert_rows},
            {"copy_threads", options_.copy_threads},
            {"wait", options_.wait == GpuOptions::Wait::sleep ? "sleep" : options_.wait == GpuOptions::Wait::spin ? "spin" : "wfe"},
            {"dense_types", dense},
            {"routed_types", routed},
            {"staging", "imported host memory"}};
}

Json GpuOffload::statistics() const {
    return {{"dense_ops", stats_.dense_ops},
            {"routed_ops", stats_.routed_ops},
            {"node_ops", stats_.node_ops},
            {"node_gpu_ms", stats_.node_gpu_ms},
            {"rows", stats_.rows},
            {"experts_staged", stats_.experts_staged},
            {"weight_bytes", stats_.weight_bytes},
            {"input_bytes", stats_.input_bytes},
            {"output_bytes", stats_.output_bytes},
            {"stage_ms", stats_.stage_ms},
            {"gpu_ms", stats_.gpu_ms},
            {"readback_ms", stats_.readback_ms},
            {"peak_staging_bytes", std::max<uint64_t>(stats_.peak_staging_bytes, weights_.bytes + io_.bytes)},
            {"failed", failed_.load()},
            {"error", failed_.load() ? Json(error()) : Json(nullptr)}};
}

void GpuOffload::release_staging() {
    stats_.peak_staging_bytes = std::max<uint64_t>(stats_.peak_staging_bytes, weights_.bytes + io_.bytes);
    release(weights_);
    release(io_);
}

void GpuOffload::begin_request() {
    stats_ = {};
}
} // namespace eqt
