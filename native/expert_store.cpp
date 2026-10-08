#include "expert_store.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "simd.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace eqt {
namespace {
constexpr size_t page = 4096;
// Lookahead covers decode steps and speculative verification batches, not prefill chunks.
constexpr int64_t max_lookahead_rows = 8;
using Clock = std::chrono::steady_clock;

size_t round_down(size_t value, size_t unit) {
    return value / unit * unit;
}
size_t round_up(size_t value, size_t unit) {
    return (value + unit - 1) / unit * unit;
}
double since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

#ifdef _WIN32
uint8_t *reserve(size_t size) {
    return static_cast<uint8_t *>(VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS));
}
void release(uint8_t *base, size_t) {
    VirtualFree(base, 0, MEM_RELEASE);
}
// Committing an already committed page keeps its contents, so shared boundary pages stay valid.
bool commit(uint8_t *data, size_t size) {
    const auto begin = round_down(reinterpret_cast<uintptr_t>(data), page);
    const auto end = round_up(reinterpret_cast<uintptr_t>(data) + size, page);
    return VirtualAlloc(reinterpret_cast<void *>(begin), end - begin, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}
void discard_pages(uint8_t *begin, size_t size) {
    VirtualFree(begin, size, MEM_DECOMMIT);
}
#else
// MAP_NORESERVE: the routed-bank range is address space only; residency is bounded by the cache.
uint8_t *reserve(size_t size) {
    void *base =
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return base == MAP_FAILED ? nullptr : static_cast<uint8_t *>(base);
}
void release(uint8_t *base, size_t size) {
    munmap(base, size);
}
bool commit(uint8_t *, size_t) {
    return true;
}
// Anonymous private pages are freed immediately and read back as zero.
void discard_pages(uint8_t *begin, size_t size) {
    madvise(begin, size, MADV_DONTNEED);
}
#endif

// Only pages fully inside one slice are released; a boundary page shared with a neighbouring expert
// stays resident, so eviction never clears bytes another resident expert still needs.
void discard(uint8_t *data, size_t size) {
    const auto address = reinterpret_cast<uintptr_t>(data);
    const auto begin = round_up(address, page), end = round_down(address + size, page);
    if (begin < end) {
        discard_pages(reinterpret_cast<uint8_t *>(begin), end - begin);
    }
}

struct Reservation {
    uint8_t *base = nullptr;
    size_t size = 0;
};

[[noreturn]] void unsupported(const char *operation) {
    GGML_ABORT("%s is not supported: expert bank contents are managed by eqt::ExpertStore", operation);
}

const ggml_backend_buffer_i buffer_interface = {
    /* .free_buffer   = */
    [](ggml_backend_buffer_t buffer) {
        auto *reservation = static_cast<Reservation *>(buffer->context);
        if (reservation->base) {
            release(reservation->base, reservation->size);
        }
        delete reservation;
    },
    /* .get_base      = */
    [](ggml_backend_buffer_t buffer) -> void * { return static_cast<Reservation *>(buffer->context)->base; },
    /* .init_tensor   = */
    [](ggml_backend_buffer_t buffer, ggml_tensor *tensor) {
        // Place each bank at an address congruent to its file offset modulo the page size, so direct
        // reads land in place for every whole page of a slice (get_alloc_size reserves one extra page).
        const auto *store = static_cast<const ExpertStore *>(buffer->buft->context);
        tensor->data = static_cast<uint8_t *>(tensor->data) + store->source_offset(ggml_get_name(tensor)) % page;
        return GGML_STATUS_SUCCESS;
    },
    /* .memset_tensor = */
    [](ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t) { unsupported("memset_tensor"); },
    /* .set_tensor    = */
    [](ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) { unsupported("set_tensor"); },
    /* .get_tensor    = */
    [](ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t) { unsupported("get_tensor"); },
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ [](ggml_backend_buffer_t, uint8_t) { unsupported("clear"); },
    /* .reset         = */ nullptr,
};

const ggml_backend_buffer_type_i buffer_type_interface = {
    /* .get_name         = */ [](ggml_backend_buffer_type_t) { return "EQT_Experts"; },
    /* .alloc_buffer     = */
    [](ggml_backend_buffer_type_t type, size_t size) -> ggml_backend_buffer_t {
        auto *reservation = new Reservation;
        if (size) {
            reservation->size = round_up(size, 64 * 1024);
            reservation->base = reserve(reservation->size);
            if (!reservation->base) {
                delete reservation;
                return nullptr;
            }
        }
        return ggml_backend_buffer_init(type, buffer_interface, reservation, size);
    },
    /* .alloc_buffer_n   = */ nullptr,
    // Page alignment keeps every page-multiple expert slice on its own pages.
    /* .get_alignment    = */ [](ggml_backend_buffer_type_t) -> size_t { return page; },
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */
    [](ggml_backend_buffer_type_t, const ggml_tensor *tensor) { return ggml_nbytes(tensor) + page; },
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ [](ggml_backend_buffer_type_t) { return true; },
};

enum class Node { other, topk, residual };
Node classify(const char *name, int &layer) {
    const char *suffix = nullptr;
    Node kind = Node::other;
    if (std::strncmp(name, "ffn_moe_topk-", 13) == 0) {
        kind = Node::topk, suffix = name + 13;
    } else if (std::strncmp(name, "attn_residual-", 14) == 0) {
        kind = Node::residual, suffix = name + 14;
    } else {
        return Node::other;
    }
    char *end = nullptr;
    const long value = std::strtol(suffix, &end, 10);
    if (end == suffix || *end != '\0' || value < 0 || value > 4096) {
        return Node::other;
    }
    layer = static_cast<int>(value);
    return kind;
}

bool parse_tensor_name(const char *name, const char *suffix, int &layer) {
    if (std::strncmp(name, "blk.", 4) != 0) {
        return false;
    }
    char *end = nullptr;
    const long value = std::strtol(name + 4, &end, 10);
    if (end == name + 4 || value < 0 || value > 4096 || std::strcmp(end, suffix) != 0) {
        return false;
    }
    layer = static_cast<int>(value);
    return true;
}
} // namespace

class SourceFile {
  public:
    SourceFile(const std::string &path, bool direct) : direct_(direct) {
#ifdef _WIN32
        if (direct) {
            throw std::runtime_error("direct_io is not implemented on Windows hosts");
        }
        const int length = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring wide(length > 0 ? length : 1, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), length);
        handle_ = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Cannot open expert source");
        }
        LARGE_INTEGER size{};
        GetFileSizeEx(handle_, &size);
        size_ = static_cast<uint64_t>(size.QuadPart);
#else
        fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
        if (fd_ < 0) {
            throw std::runtime_error(std::string("Cannot open expert source") +
                                     (direct ? " with O_DIRECT" : "") +
                                     " (no silent fallback): " + std::strerror(errno));
        }
        struct stat info{};
        if (fstat(fd_, &info) != 0 || !S_ISREG(info.st_mode)) {
            close(fd_);
            throw std::runtime_error("Expert source must be a regular file");
        }
        size_ = static_cast<uint64_t>(info.st_size);
        if (!direct) {
            posix_fadvise(fd_, 0, 0, POSIX_FADV_RANDOM);
        }
#endif
    }
    ~SourceFile() {
#ifdef _WIN32
        CloseHandle(handle_);
#else
        close(fd_);
#endif
    }
    uint64_t size() const {
        return size_;
    }
    bool direct() const {
        return direct_;
    }
    // Reads exactly [offset, offset + size). Direct I/O widens unaligned requests to 4 KiB through the
    // caller's aligned bounce buffer; aligned requests land in place.
    void read(uint8_t *destination, size_t size, uint64_t offset, std::vector<uint8_t> &bounce) const {
        if (offset + size > size_) {
            throw std::runtime_error("Expert slice exceeds source file");
        }
        if (!direct_) {
            read_exact(destination, size, offset, offset + size);
            return;
        }
        if (bounce.size() < 2 * page) {
            bounce.assign(2 * page, 0);
        }
        auto *aligned = reinterpret_cast<uint8_t *>(round_up(reinterpret_cast<uintptr_t>(bounce.data()), page));
        const uint64_t end = offset + size;
        if ((reinterpret_cast<uintptr_t>(destination) - offset) % page != 0) {
            throw std::runtime_error("Expert slice is not page-congruent with its file offset");
        }
        // Whole pages are read in place; at most one partial page at each end goes through the bounce page.
        const uint64_t middle_begin = round_up(offset, page), middle_end = round_down(end, page);
        if (middle_begin >= middle_end) {
            for (uint64_t at = round_down(offset, page); at < end; at += page) {
                read_partial(destination, offset, end, at, aligned);
            }
            return;
        }
        if (offset != middle_begin) {
            read_partial(destination, offset, end, round_down(offset, page), aligned);
        }
        read_exact(destination + (middle_begin - offset), middle_end - middle_begin, middle_begin, middle_end);
        if (end != middle_end) {
            read_partial(destination, offset, end, middle_end, aligned);
        }
    }

    struct Span {
        uint8_t *data;
        size_t size;
    };
    // Reads the contiguous file range [offset, offset + total span size) into consecutive spans with one
    // vectored read where available. Direct I/O spans must be page-aligned and page-sized; the range may end
    // past EOF only beyond required_end.
    void read_vector(const Span *spans, int count, uint64_t offset, uint64_t required_end) const {
#ifdef _WIN32
        for (int i = 0; i < count; ++i) {
            read_exact(spans[i].data, spans[i].size, offset, std::min<uint64_t>(required_end, offset + spans[i].size));
            offset += spans[i].size;
        }
#else
        iovec iov[64];
        if (count > 64) {
            throw std::runtime_error("Too many spans in one expert read");
        }
        for (int i = 0; i < count; ++i) {
            iov[i] = {spans[i].data, spans[i].size};
        }
        int first = 0;
        while (first < count) {
            const ssize_t got = preadv(fd_, iov + first, count - first, static_cast<off_t>(offset));
            if (got < 0 && errno == EINTR) {
                continue;
            }
            if (got < 0) {
                throw std::runtime_error(std::string("Expert read failed: ") + std::strerror(errno));
            }
            if (got == 0) {
                if (offset >= required_end) {
                    return;
                }
                throw std::runtime_error("Unexpected EOF while reading expert");
            }
            offset += static_cast<uint64_t>(got);
            size_t left = static_cast<size_t>(got);
            while (first < count && left >= iov[first].iov_len) {
                left -= iov[first].iov_len;
                ++first;
            }
            if (first < count && left) {
                if (direct_ && left % page) {
                    throw std::runtime_error("Unaligned short direct read");
                }
                iov[first].iov_base = static_cast<uint8_t *>(iov[first].iov_base) + left;
                iov[first].iov_len -= left;
            }
        }
#endif
    }

  private:
    // Reads the page at `at` into the bounce page and copies its overlap with [offset, end).
    void read_partial(uint8_t *destination, uint64_t offset, uint64_t end, uint64_t at, uint8_t *aligned) const {
        read_exact(aligned, page, at, std::min<uint64_t>(at + page, end));
        const uint64_t from = std::max(at, offset), to = std::min(at + page, end);
        std::memcpy(destination + (from - offset), aligned + (from - at), to - from);
    }
    // `required_end` permits the final aligned block to stop at EOF.
    void read_exact(uint8_t *destination, size_t size, uint64_t offset, uint64_t required_end) const {
        size_t done = 0;
        while (done < size) {
#ifdef _WIN32
            OVERLAPPED position{};
            const uint64_t at = offset + done;
            position.Offset = static_cast<DWORD>(at);
            position.OffsetHigh = static_cast<DWORD>(at >> 32);
            DWORD got = 0;
            const DWORD request = static_cast<DWORD>(std::min<size_t>(size - done, 1u << 30));
            if (!ReadFile(handle_, destination + done, request, &got, &position) &&
                GetLastError() != ERROR_HANDLE_EOF) {
                throw std::runtime_error("Expert read failed");
            }
#else
            const ssize_t got =
                pread(fd_, destination + done, size - done, static_cast<off_t>(offset + done));
            if (got < 0 && errno == EINTR) {
                continue;
            }
            if (got < 0) {
                throw std::runtime_error(std::string("Expert read failed: ") + std::strerror(errno));
            }
#endif
            if (got == 0) {
                if (offset + done >= required_end) {
                    return;
                }
                throw std::runtime_error("Unexpected EOF while reading expert");
            }
            done += static_cast<size_t>(got);
            if (direct_ && done < size && done % page) {
                throw std::runtime_error("Unaligned short direct read");
            }
        }
    }

#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
    uint64_t size_ = 0;
    bool direct_ = false;
};

ExpertStore::ExpertStore(const std::string &path, ExpertOptions options, const std::atomic_bool &cancelled)
    : options_(std::move(options)), cancelled_(cancelled),
      file_(std::make_unique<SourceFile>(path, options_.direct_io)) {
    if (options_.io_threads < 1 || options_.io_threads > 16) {
        throw std::invalid_argument("io_threads must be in 1..16");
    }
    auto *cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        throw std::runtime_error("No CPU backend registered");
    }
    buffer_type_.reset(new ggml_backend_buffer_type{buffer_type_interface, cpu, this});
    if (!options_.trace_path.empty()) {
        trace_.open(options_.trace_path, std::ios::binary | std::ios::trunc);
        if (!trace_) {
            throw std::runtime_error("Cannot open expert trace output");
        }
    }
}

ExpertStore::~ExpertStore() {
    if (set_expert_address_) {
        set_expert_address_(nullptr, nullptr);
    }
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        demand_queue_.clear();
        prefetch_queue_.clear();
    }
    queue_ready_.notify_all();
    for (auto &worker : workers_) {
        worker.join();
    }
    if (arena_) {
        release(arena_, arena_size_);
    }
}

ggml_backend_buffer_type *ExpertStore::buffer_type() {
    return buffer_type_.get();
}

bool ExpertStore::owns(const ggml_tensor *tensor) const {
    return tensor->buffer && ggml_backend_buffer_get_type(tensor->buffer) == buffer_type_.get();
}

ExpertStore::Layer *ExpertStore::layer_at(int index) const {
    return index >= 0 && index < static_cast<int>(layers_.size()) ? layers_[index].get() : nullptr;
}

void ExpertStore::set_source_offsets(std::unordered_map<std::string, uint64_t> offsets) {
    source_offsets_ = std::move(offsets);
}

uint64_t ExpertStore::source_offset(const char *name) const {
    const auto it = source_offsets_.find(name);
    return it == source_offsets_.end() ? 0 : it->second;
}

void ExpertStore::add_bank(ggml_tensor *tensor, uint64_t file_offset) {
    int index = -1;
    const char *name = ggml_get_name(tensor);
    if (std::strncmp(name, "blk.", 4) == 0) {
        index = std::atoi(name + 4);
    }
    if (index < 0 || index > 4096 || tensor->ne[2] < 1 || tensor->ne[3] != 1 ||
        tensor->nb[2] * tensor->ne[2] != ggml_nbytes(tensor)) {
        throw std::runtime_error(std::string("Unsupported routed bank layout: ") + name);
    }
    if (file_offset + ggml_nbytes(tensor) > file_->size()) {
        throw std::runtime_error(std::string("Routed bank exceeds file: ") + name);
    }
    if (static_cast<int>(layers_.size()) <= index) {
        layers_.resize(index + 1);
    }
    auto &layer = layers_[index];
    if (!layer) {
        layer = std::make_unique<Layer>();
    }
    if (layer->banks.empty()) {
        layer->experts = static_cast<int>(tensor->ne[2]);
    }
    if (layer->experts != tensor->ne[2]) {
        throw std::runtime_error(std::string("Inconsistent expert count in ") + name);
    }
    const size_t stride = tensor->nb[2];
    // A slot bank holds the page-rounded file range of one expert, whose start may sit up to a page before it.
    const size_t span = options_.slots ? round_up(stride, page) + page : round_up(stride, page);
    layer->banks.push_back({static_cast<uint8_t *>(tensor->data), stride, file_offset, tensor, layer->slot_bytes});
    layer->slot_bytes += span;
    layer->entry_bytes += span;
    routed_bytes_ += ggml_nbytes(tensor);
    max_stride_ = std::max(max_stride_, stride);
}

void ExpertStore::add_router_tensor(ggml_tensor *tensor) {
    int index = -1;
    const char *name = ggml_get_name(tensor);
    const bool router = parse_tensor_name(name, ".ffn_gate_inp.weight", index);
    if (!router && !parse_tensor_name(name, ".post_attention_norm.weight", index)) {
        return;
    }
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
        !ggml_backend_buffer_is_host(tensor->buffer)) {
        return;
    }
    if (static_cast<int>(layers_.size()) <= index) {
        layers_.resize(index + 1);
    }
    if (!layers_[index]) {
        layers_[index] = std::make_unique<Layer>();
    }
    auto &layer = *layers_[index];
    if (router) {
        layer.router = static_cast<const float *>(tensor->data);
        layer.width = static_cast<int>(tensor->ne[0]);
        layer.router_rows = static_cast<int>(tensor->ne[1]);
    } else {
        layer.norm = static_cast<const float *>(tensor->data);
    }
}

void ExpertStore::finalize(int experts_used, float norm_epsilon) {
    experts_used_ = experts_used;
    norm_epsilon_ = norm_epsilon;
    if (!options_.layer_weights.empty() && options_.layer_weights.size() != layers_.size()) {
        throw std::invalid_argument("expert_layer_weights must contain one value per model layer");
    }
    double weight_total = 0;
    int streamed = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto *layer = layers_[i].get();
        if (!layer || layer->banks.empty()) {
            continue;
        }
        ++streamed;
        const double weight = options_.layer_weights.empty() ? 1.0 : options_.layer_weights[i];
        if (!std::isfinite(weight) || weight < 0) {
            throw std::invalid_argument("expert_layer_weights must be finite and non-negative");
        }
        weight_total += weight;
    }
    if (!streamed) {
        throw std::runtime_error("Model has no routed expert banks to stream");
    }
    if (weight_total <= 0) {
        throw std::invalid_argument("expert_layer_weights must not all be zero");
    }
    int max_experts = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto *layer = layers_[i].get();
        if (!layer || layer->banks.empty()) {
            continue;
        }
        const double weight = options_.layer_weights.empty() ? 1.0 : options_.layer_weights[i];
        const double share = static_cast<double>(options_.budget_bytes) * weight / weight_total;
        layer->quota = std::min(layer->experts, static_cast<int>(share / layer->entry_bytes));
        layer->entries = std::make_unique<Entry[]>(layer->experts);
        layer->seen.assign(layer->experts, 0);
        layer->prediction.reserve(static_cast<size_t>(experts_used + options_.prefetch_extra) * max_lookahead_rows);
        effective_budget_ += static_cast<uint64_t>(layer->quota) * layer->entry_bytes;
        max_experts = std::max(max_experts, layer->experts);
        if (layer->router && (layer->router_rows != layer->experts || !layer->norm)) {
            layer->router = nullptr; // Lookahead needs a matching F32 router and RMS weight.
        }
        predictable_layers_ += layer->router != nullptr;
    }
    if (options_.prefetch && predictable_layers_ == 0) {
        throw std::invalid_argument("prefetch requires F32 ffn_gate_inp and post_attention_norm tensors");
    }
    residuals_.assign(layers_.size(), nullptr);
    demand_.reserve(max_experts);
    missing_.reserve(max_experts);
    whole_layer_.reserve(max_experts);
    order_.resize(max_experts);
    scores_.resize(max_experts);
    for (auto &layer : layers_) {
        if (layer && layer->router) {
            scratch_.resize(std::max<size_t>(scratch_.size(), layer->width));
        }
    }
    if (options_.slots) {
        setup_slots();
    }
    for (int i = 0; i < options_.io_threads; ++i) {
        workers_.emplace_back([this] { worker(); });
    }
}

void ExpertStore::setup_slots() {
    auto *reg = ggml_backend_reg_by_name("CPU");
    set_expert_address_ = reg ? reinterpret_cast<decltype(set_expert_address_)>(
                                    ggml_backend_reg_get_proc_address(reg, "ggml_cpu_set_expert_address"))
                              : nullptr;
    if (!set_expert_address_) {
        throw std::runtime_error("CPU backend has no expert address hook; expert_slots is unavailable");
    }
    // Address space for every expert of every layer (never all populated); classes share slots of one size.
    std::vector<size_t> class_capacity;
    for (auto &layer : layers_) {
        if (!layer || layer->banks.empty()) {
            continue;
        }
        arena_size_ += static_cast<size_t>(layer->experts) * layer->slot_bytes;
        auto it = std::find_if(slot_classes_.begin(), slot_classes_.end(),
                               [&](const SlotClass &c) { return c.bytes == layer->slot_bytes; });
        if (it == slot_classes_.end()) {
            slot_classes_.push_back({layer->slot_bytes, {}, {}});
            class_capacity.push_back(0);
            it = slot_classes_.end() - 1;
        }
        layer->slot_class = static_cast<int>(it - slot_classes_.begin());
        class_capacity[layer->slot_class] += layer->experts;
        for (size_t b = 0; b < layer->banks.size(); ++b) {
            bank_index_.push_back({layer->banks[b].tensor, layer.get(), static_cast<int>(b)});
        }
    }
    for (size_t c = 0; c < slot_classes_.size(); ++c) {
        slot_classes_[c].warm.reserve(class_capacity[c]);
        slot_classes_[c].cold.reserve(class_capacity[c]);
    }
    std::sort(bank_index_.begin(), bank_index_.end(),
              [](const BankRef &a, const BankRef &b) { return std::less<const ggml_tensor *>()(a.tensor, b.tensor); });
    arena_ = reserve(arena_size_);
    if (!arena_) {
        throw std::runtime_error("Cannot reserve the expert slot arena");
    }
    set_expert_address_(&ExpertStore::expert_address, this);
}

const void *ExpertStore::expert_address(const ggml_tensor *weights, int64_t expert, void *opaque) {
    const auto &store = *static_cast<const ExpertStore *>(opaque);
    const ggml_tensor *base = weights->view_src ? weights->view_src : weights;
    const auto it = std::lower_bound(store.bank_index_.begin(), store.bank_index_.end(), base,
                                     [](const BankRef &ref, const ggml_tensor *t) {
                                         return std::less<const ggml_tensor *>()(ref.tensor, t);
                                     });
    if (it == store.bank_index_.end() || it->tensor != base) {
        return nullptr; // not a streamed bank
    }
    const Bank &bank = it->layer->banks[it->bank];
    // A view of a bank (none in the target GGUFs) addresses view_offs bytes into the bank's expert layout.
    const size_t offset = weights->view_src ? weights->view_offs : 0;
    const int64_t e = expert + static_cast<int64_t>(offset / bank.stride);
    if (e < 0 || e >= it->layer->experts || !it->layer->entries[e].slot) {
        GGML_ABORT("MUL_MAT_ID read a streamed expert that is not resident");
    }
    return store.data_of(*it->layer, it->bank, static_cast<int>(e)) + offset % bank.stride;
}

uint8_t *ExpertStore::data_of(const Layer &layer, int bank_index, int expert) const {
    const Bank &bank = layer.banks[bank_index];
    if (!options_.slots) {
        return bank.data + static_cast<size_t>(expert) * bank.stride;
    }
    // The slot holds the page-rounded file range, so the expert starts at its file offset's page remainder.
    const uint64_t offset = bank.file_offset + static_cast<uint64_t>(expert) * bank.stride;
    return layer.entries[expert].slot + bank.slot_offset + offset % page;
}

bool ExpertStore::assign_slot(Layer &layer, int expert) {
    auto &slots = slot_classes_[layer.slot_class];
    uint8_t *slot = nullptr;
    if (!slots.warm.empty()) {
        slot = slots.warm.back();
        slots.warm.pop_back();
    } else {
        if (!slots.cold.empty()) {
            slot = slots.cold.back();
            slots.cold.pop_back();
            ++stats_.slot_cold_reuses;
        } else if (arena_used_ + slots.bytes <= arena_size_) {
            slot = arena_ + arena_used_;
            arena_used_ += slots.bytes;
        } else {
            return false;
        }
        if (!commit(slot, slots.bytes)) {
            slots.cold.push_back(slot);
            return false;
        }
    }
    layer.entries[expert].slot = slot;
    return true;
}

void ExpertStore::release_slot(Layer &layer, int expert) {
    auto &entry = layer.entries[expert];
    if (entry.slot) {
        slot_classes_[layer.slot_class].warm.push_back(entry.slot);
        entry.slot = nullptr;
    }
}

// A prompt can leave many warm slots of several sizes behind; decoding recycles its own evictions, so the free
// pages are returned once when decoding starts and reused slots fault in again only if needed.
void ExpertStore::set_decode_phase(bool decode) {
    if (decode && !decode_phase_ && options_.slots) {
        for (auto &slots : slot_classes_) {
            for (uint8_t *slot : slots.warm) {
                discard_pages(slot, slots.bytes);
                slots.cold.push_back(slot);
            }
            slots.warm.clear();
        }
    }
    decode_phase_ = decode;
}

// One job reads a contiguous file range of one bank into its experts' slots. Direct reads take each expert's
// page-rounded range; neighbours whose ranges share a page read it once, and the second copies it afterwards.
void ExpertStore::read_slots(const Job &job) const {
    const Layer &layer = *job.layer;
    const Bank &bank = layer.banks[job.bank];
    SourceFile::Span spans[64];
    const uint64_t first_offset = bank.file_offset + static_cast<uint64_t>(job.first) * bank.stride;
    const uint64_t end = first_offset + static_cast<uint64_t>(job.count) * bank.stride;
    if (!file_->direct()) {
        for (int i = 0; i < job.count; ++i) {
            spans[i] = {data_of(layer, job.bank, job.first + i), bank.stride};
        }
        file_->read_vector(spans, job.count, first_offset, end);
        return;
    }
    bool shares_previous[64] = {};
    uint64_t previous_end = 0;
    for (int i = 0; i < job.count; ++i) {
        const uint64_t offset = first_offset + static_cast<uint64_t>(i) * bank.stride;
        const uint64_t begin = round_down(offset, page), finish = round_up(offset + bank.stride, page);
        uint8_t *base = layer.entries[job.first + i].slot + bank.slot_offset;
        shares_previous[i] = i > 0 && begin < previous_end;
        spans[i] = shares_previous[i] ? SourceFile::Span{base + page, static_cast<size_t>(finish - begin - page)}
                                      : SourceFile::Span{base, static_cast<size_t>(finish - begin)};
        previous_end = finish;
    }
    file_->read_vector(spans, job.count, round_down(first_offset, page), end);
    for (int i = 1; i < job.count; ++i) {
        if (shares_previous[i]) {
            const uint64_t previous = first_offset + static_cast<uint64_t>(i - 1) * bank.stride;
            const size_t previous_size = round_up(previous + bank.stride, page) - round_down(previous, page);
            std::memcpy(layer.entries[job.first + i].slot + bank.slot_offset,
                        layer.entries[job.first + i - 1].slot + bank.slot_offset + previous_size - page, page);
        }
    }
}

void ExpertStore::begin_request() {
    end_request();
    stats_ = {};
    stats_.peak_resident_bytes = resident_bytes_;
    ++requests_;
    if (trace_) {
        trace_ << "{\"request\":" << requests_ << "}\n";
    }
}

void ExpertStore::end_request() {
    {
        std::unique_lock lock(mutex_);
        job_done_.wait(lock, [this] { return inflight_jobs_.load() == 0; });
    }
    for (auto &layer : layers_) {
        if (!layer || layer->banks.empty()) {
            continue;
        }
        for (int e = 0; e < layer->experts; ++e) {
            reap(*layer, e);
        }
        layer->prediction_valid = false;
    }
    if (auto *layer = layer_at(overflow_layer_)) {
        trim(*layer);
    }
    overflow_layer_ = -1;
    prefetch_layer_ = -1;
    if (trace_) {
        trace_.flush();
    }
}

bool ExpertStore::observe(ggml_tensor *tensor, bool ask, void *opaque) {
    auto &store = *static_cast<ExpertStore *>(opaque);
    int layer = -1;
    const Node kind = classify(tensor->name, layer);
    if (ask) {
        if (kind == Node::residual) {
            // No graph split here: the residual stays allocated until the post-MoE add, so the lookahead
            // reads it from the top-k callback of the same layer.
            auto *next = store.layer_at(layer + 1);
            const bool usable = store.options_.prefetch && tensor->ne[1] <= max_lookahead_rows && next && next->router &&
                                next->norm && !next->banks.empty();
            if (layer < static_cast<int>(store.residuals_.size())) {
                store.residuals_[layer] = usable ? tensor : nullptr;
            }
            return false;
        }
        return kind == Node::topk && store.layer_at(layer) && !store.layer_at(layer)->banks.empty();
    }
    if (store.cancelled_.load()) {
        return false;
    }
    if (kind != Node::topk) {
        return true;
    }
    if (layer < static_cast<int>(store.residuals_.size()) && store.residuals_[layer]) {
        store.predict(layer, store.residuals_[layer]);
        store.residuals_[layer] = nullptr;
    }
    return store.ensure(layer, tensor);
}

bool ExpertStore::ensure(int layer_index, const ggml_tensor *ids) {
    auto *layer_pointer = layer_at(layer_index);
    if (!layer_pointer || layer_pointer->banks.empty()) {
        return true;
    }
    auto &layer = *layer_pointer;
    if (overflow_layer_ >= 0 && overflow_layer_ != layer_index) {
        trim(*layers_[overflow_layer_]);
        overflow_layer_ = -1;
    }
    if (ids->type != GGML_TYPE_I32) {
        fail("Expert selection tensor is not I32");
        return false;
    }
    ++stats_.ensure_calls;
    ++epoch_;
    demand_.clear();
    const int64_t selected = ids->ne[0], rows = ids->ne[1];
    if (trace_) {
        trace_ << "{\"layer\":" << layer_index << ",\"tokens\":" << rows << ",\"experts\":[";
    }
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t slot = 0; slot < selected; ++slot) {
            const int32_t expert = *reinterpret_cast<const int32_t *>(static_cast<const char *>(ids->data) +
                                                                      row * ids->nb[1] + slot * ids->nb[0]);
            if (expert < 0 || expert >= layer.experts) {
                fail("Router selected an out-of-range expert");
                return false;
            }
            if (trace_) {
                trace_ << (row || slot ? "," : "") << expert;
            }
            if (layer.seen[expert] != epoch_) {
                layer.seen[expert] = epoch_;
                demand_.push_back(expert);
            }
        }
    }
    if (trace_) {
        trace_ << "]}\n";
    }
    if (layer.prediction_valid) {
        stats_.predicted += layer.prediction.size();
        for (const int expert : layer.prediction) {
            stats_.predicted_correct += layer.seen[expert] == epoch_;
        }
        layer.prediction_valid = false;
    }
    ++clock_;
    stats_.demanded += demand_.size();
    missing_.clear();
    for (const int expert : demand_) {
        auto &entry = layer.entries[expert];
        reap(layer, expert);
        entry.stamp = clock_;
        if (entry.state == State::resident) {
            ++stats_.hits;
            stats_.prefetch_hits += entry.prefetched;
        } else if (entry.state == State::loading) {
            ++stats_.inflight_hits;
        } else {
            ++stats_.misses;
            missing_.push_back(expert);
        }
        entry.prefetched = false;
    }
    int excess = layer.resident + static_cast<int>(missing_.size()) - layer.quota;
    while (excess > 0) {
        const int victim = least_recent(layer, epoch_);
        if (victim < 0) {
            break;
        }
        evict(layer, victim);
        --excess;
    }
    if (excess > 0) {
        // The batch needs more distinct experts than the layer quota; they stay until this layer is done.
        ++stats_.overflow_events;
        overflow_layer_ = layer_index;
    }
    const bool decode = decode_phase_;
    if (decode) {
        stats_.decode_demanded += demand_.size();
        stats_.decode_misses += missing_.size();
        stats_.decode_read_bytes += missing_.size() * layer.entry_bytes;
    }
    load_runs(layer, missing_, true, options_.coalesce_gap);
    if (rows > max_lookahead_rows && options_.prefill_prefetch_mib > 0) {
        if (auto *next = layer_at(layer_index + 1); next && !next->banks.empty()) {
            prefetch_whole_layer(*next);
        }
    }
    if (prefetch_layer_ >= 0) {
        if (auto *next = layer_at(prefetch_layer_)) {
            issue_prefetch(*next);
        }
        prefetch_layer_ = -1;
    }
    const auto wait_start = Clock::now();
    for (const int expert : demand_) {
        if (!wait(layer, expert)) {
            stats_.io_wait_ms += since(wait_start);
            return false;
        }
    }
    const double waited = since(wait_start);
    stats_.io_wait_ms += waited;
    stats_.decode_io_wait_ms += decode ? waited : 0;
    stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, resident_bytes_);
    return true;
}

void ExpertStore::predict(int layer_index, const ggml_tensor *hidden) {
    auto *next = layer_at(layer_index + 1);
    if (!next || !next->router || !next->norm || hidden->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(hidden) || hidden->ne[0] != next->width) {
        return;
    }
    const auto start = Clock::now();
    const size_t width = static_cast<size_t>(next->width);
    const int count = std::min(experts_used_ + options_.prefetch_extra, next->experts);
    next->prediction.clear();
    // Approximates layer l+1's router input with the post-attention residual of layer l, normalized by
    // layer l+1's RMS weight; a small batch (speculative verification) prefetches the union of its rows.
    // It only drives speculative reads; routing still uses the exact top-k. Candidates are issued in score
    // order, so extra candidates beyond top-k are read last.
    for (int64_t row = 0; row < hidden->ne[1]; ++row) {
        const auto *x = reinterpret_cast<const float *>(static_cast<const char *>(hidden->data) + row * hidden->nb[1]);
        const float scale = 1.0f / std::sqrt(simd::sum_squares(x, width) / width + norm_epsilon_);
        simd::scale_mul(x, next->norm, scale, scratch_.data(), width);
        simd::gemv_rows(next->router, scratch_.data(), scores_.data(), next->experts, width);
        for (int e = 0; e < next->experts; ++e) {
            order_[e] = e;
        }
        std::partial_sort(
            order_.begin(), order_.begin() + count, order_.begin() + next->experts,
            [this](int a, int b) { return scores_[a] > scores_[b] || (scores_[a] == scores_[b] && a < b); });
        for (int i = 0; i < count; ++i) {
            if (std::find(next->prediction.begin(), next->prediction.end(), order_[i]) == next->prediction.end()) {
                next->prediction.push_back(order_[i]);
            }
        }
    }
    next->prediction_valid = true;
    prefetch_layer_ = layer_index + 1;
    stats_.predict_ms += since(start);
}

void ExpertStore::issue_prefetch(Layer &layer) {
    for (const int expert : layer.prediction) {
        reap(layer, expert);
        auto &entry = layer.entries[expert];
        if (entry.state != State::absent) {
            continue; // Prediction never refreshes recency of resident entries.
        }
        if (layer.resident >= layer.quota) {
            const int victim = least_recent(layer, 0);
            if (victim < 0 || layer.quota == 0) {
                return;
            }
            evict(layer, victim);
        }
        start_load(layer, expert, false);
        entry.prefetched = true;
        entry.stamp = 0; // Unused speculative entries are the first eviction candidates.
        ++stats_.prefetch_issued;
    }
}

void ExpertStore::start_load(Layer &layer, int expert, bool demand) {
    if (begin_load(layer, expert, demand)) {
        queue_reads(layer, expert, 1, 1, demand);
    }
}

// Marks an expert loading with one pending read per bank and commits its pages. On a commit failure the
// entry stays loading with no pending reads and is evicted as failed when reaped.
bool ExpertStore::begin_load(Layer &layer, int expert, bool demand) {
    auto &entry = layer.entries[expert];
    entry.state = State::loading;
    entry.failed.store(false);
    ++layer.resident;
    resident_bytes_ += layer.entry_bytes;
    if (options_.slots) {
        if (!assign_slot(layer, expert)) {
            entry.failed.store(true);
            entry.pending.store(0);
            fail("Cannot assign an expert slot");
            return false;
        }
    } else {
        for (const auto &bank : layer.banks) {
            if (!commit(bank.data + expert * bank.stride, bank.stride)) {
                entry.failed.store(true);
                entry.pending.store(0);
                fail("Cannot commit expert cache pages");
                return false;
            }
        }
    }
    entry.pending.store(static_cast<int>(layer.banks.size()));
    for (const auto &bank : layer.banks) {
        (demand ? stats_.demand_read_bytes : stats_.prefetch_read_bytes) += bank.stride;
    }
    return true;
}

void ExpertStore::queue_reads(Layer &layer, int first, int count, uint64_t members, bool demand) {
    const int banks = static_cast<int>(layer.banks.size());
    stats_.read_jobs += banks;
    stats_.coalesced_reads += count > 1 ? banks : 0;
    inflight_jobs_.fetch_add(banks);
    {
        std::lock_guard lock(mutex_);
        for (int bank = 0; bank < banks; ++bank) {
            (demand ? demand_queue_ : prefetch_queue_).push_back({&layer, bank, first, count, members});
        }
    }
    queue_ready_.notify_all();
}

// Loads absent experts in ascending order. With coalescing, a run of adjacent experts becomes one read per
// bank into the bank's file-congruent range, so the in-place direct-I/O path covers every interior page.
// Demand runs may bridge up to max_gap resident experts, re-reading their bytes unchanged: no compute reads
// expert pages while ensure() runs, and the layer's runs complete before it returns, so no bridged expert
// can be evicted under an in-flight read. Prefetch runs (another layer, completing later) never bridge.
void ExpertStore::load_runs(Layer &layer, std::vector<int> &experts, bool demand, int max_gap) {
    std::sort(experts.begin(), experts.end());
    size_t widest = 0;
    for (const auto &bank : layer.banks) {
        widest = std::max(widest, bank.stride);
    }
    const size_t cap = static_cast<size_t>(options_.coalesce_kib) << 10;
    const auto admit = [&](int expert) {
        if (!begin_load(layer, expert, demand)) {
            return false;
        }
        if (!demand) {
            auto &entry = layer.entries[expert];
            entry.prefetched = true;
            entry.stamp = 0; // unused speculative entries are the first eviction candidates
            ++stats_.prefetch_issued;
        }
        return true;
    };
    size_t i = 0;
    while (i < experts.size()) {
        const int first = experts[i++];
        if (!admit(first)) {
            continue;
        }
        int count = 1;
        uint64_t members = 1;
        while (cap > 0 && i < experts.size()) {
            const int next = experts[i], gap = next - (first + count), span = next - first + 1;
            if (gap > max_gap || span > 64 || static_cast<size_t>(span) * widest > cap) {
                break;
            }
            bool bridgeable = true;
            for (int e = first + count; e < next && bridgeable; ++e) {
                reap(layer, e);
                bridgeable = layer.entries[e].state == State::resident;
            }
            if (!bridgeable) {
                break;
            }
            if (!admit(next)) {
                ++i; // failed entries are reaped and evicted; they never join a read
                break;
            }
            members |= uint64_t{1} << (next - first);
            stats_.bridged_experts += gap;
            count = span;
            ++i;
        }
        queue_reads(layer, first, count, members, demand);
    }
}

// A prompt batch touches most experts of every layer, so the next layer is read while this one computes,
// beyond its quota within the transient allowance. Its own ensure() evicts the unused (stamp 0) entries
// first, and trim() restores the quota once the layer after it starts.
void ExpertStore::prefetch_whole_layer(Layer &layer) {
    const uint64_t ceiling = effective_budget_ + (static_cast<uint64_t>(options_.prefill_prefetch_mib) << 20);
    uint64_t planned = resident_bytes_;
    whole_layer_.clear();
    for (int e = 0; e < layer.experts && planned + layer.entry_bytes <= ceiling; ++e) {
        reap(layer, e);
        if (layer.entries[e].state == State::absent) {
            whole_layer_.push_back(e);
            planned += layer.entry_bytes;
        }
    }
    load_runs(layer, whole_layer_, false, 0);
}

bool ExpertStore::reap(Layer &layer, int expert) {
    auto &entry = layer.entries[expert];
    if (entry.state != State::loading || entry.pending.load() != 0) {
        return entry.state != State::loading;
    }
    if (entry.failed.load()) {
        evict(layer, expert);
        return true;
    }
    entry.state = State::resident;
    return true;
}

void ExpertStore::evict(Layer &layer, int expert) {
    auto &entry = layer.entries[expert];
    const auto start = Clock::now();
    if (options_.slots) {
        release_slot(layer, expert);
    } else {
        for (const auto &bank : layer.banks) {
            discard(bank.data + expert * bank.stride, bank.stride);
        }
    }
    stats_.evict_ms += since(start);
    stats_.prefetch_wasted += entry.prefetched;
    stats_.evictions += !entry.failed.load();
    entry.state = State::absent;
    entry.prefetched = false;
    --layer.resident;
    resident_bytes_ -= layer.entry_bytes;
}

int ExpertStore::least_recent(const Layer &layer, uint32_t epoch) const {
    int victim = -1;
    for (int e = 0; e < layer.experts; ++e) {
        const auto &entry = layer.entries[e];
        if (entry.state == State::resident && (epoch == 0 || layer.seen[e] != epoch) &&
            (victim < 0 || entry.stamp < layer.entries[victim].stamp)) {
            victim = e;
        }
    }
    return victim;
}

void ExpertStore::trim(Layer &layer) {
    for (int e = 0; e < layer.experts; ++e) {
        reap(layer, e);
    }
    while (layer.resident > layer.quota) {
        const int victim = least_recent(layer, 0);
        if (victim < 0) {
            return;
        }
        evict(layer, victim);
    }
}

bool ExpertStore::wait(Layer &layer, int expert) {
    auto &entry = layer.entries[expert];
    if (entry.state == State::loading) {
        std::unique_lock lock(mutex_);
        while (entry.pending.load() != 0) {
            if (cancelled_.load()) {
                return false;
            }
            job_done_.wait_for(lock, std::chrono::milliseconds(10));
        }
    }
    reap(layer, expert);
    if (entry.state != State::resident) {
        if (error().empty()) {
            fail("Expert load failed");
        }
        return false;
    }
    return true;
}

void ExpertStore::worker() {
    std::vector<uint8_t> bounce;
    if (file_->direct()) {
        bounce.assign(2 * page, 0);
    }
    for (;;) {
        Job job{};
        {
            std::unique_lock lock(mutex_);
            queue_ready_.wait(
                lock, [this] { return stopping_ || !demand_queue_.empty() || !prefetch_queue_.empty(); });
            if (stopping_) {
                return;
            }
            auto &queue = demand_queue_.empty() ? prefetch_queue_ : demand_queue_;
            job = queue.front();
            queue.pop_front();
        }
        const auto &bank = job.layer->banks[job.bank];
        bool failed = false;
        try {
            if (options_.slots) {
                read_slots(job);
            } else {
                file_->read(bank.data + job.first * bank.stride, bank.stride * job.count,
                            bank.file_offset + job.first * bank.stride, bounce);
            }
        } catch (const std::exception &error) {
            failed = true;
            fail(error.what());
        }
        for (int i = 0; i < job.count; ++i) {
            if (job.members >> i & 1) {
                auto &entry = job.layer->entries[job.first + i];
                if (failed) {
                    entry.failed.store(true);
                }
                entry.pending.fetch_sub(1);
            }
        }
        inflight_jobs_.fetch_sub(1);
        {
            std::lock_guard lock(mutex_);
        }
        job_done_.notify_all();
    }
}

void ExpertStore::fail(const std::string &message) {
    std::lock_guard lock(mutex_);
    if (error_.empty()) {
        error_ = message;
    }
}

std::string ExpertStore::error() const {
    std::lock_guard lock(mutex_);
    return error_;
}

Json ExpertStore::configuration() const {
    Json quotas = Json::array();
    uint64_t min_entry = UINT64_MAX, max_entry = 0;
    for (const auto &layer : layers_) {
        if (layer && !layer->banks.empty()) {
            quotas.push_back(layer->quota);
            min_entry = std::min(min_entry, layer->entry_bytes);
            max_entry = std::max(max_entry, layer->entry_bytes);
        }
    }
    return {
        {"policy", "layer_partitioned_lru"},
        {"routing", "exact_top_k"},
        {"requested_budget_bytes", options_.budget_bytes},
        {"effective_budget_bytes", effective_budget_},
        {"budget_kind", "routed_expert_pages; prefill batches may transiently exceed one layer quota"},
        {"layer_quotas", quotas},
        {"layer_weights", options_.layer_weights.empty() ? Json("uniform") : Json(options_.layer_weights)},
        {"routed_bytes", routed_bytes_},
        {"entry_bytes_min", min_entry},
        {"entry_bytes_max", max_entry},
        {"experts_used", experts_used_},
        {"io_threads", options_.io_threads},
        {"direct_io", options_.direct_io},
        {"prefetch", options_.prefetch ? "router_lookahead_1" : "none"},
        {"prefetch_extra", options_.prefetch_extra},
        {"coalesce_kib", options_.coalesce_kib},
        {"coalesce_gap", options_.coalesce_gap},
        {"prefill_prefetch_mib", options_.prefill_prefetch_mib},
        {"slots", options_.slots},
        {"prefetch_layers", predictable_layers_},
        {"simd", simd::implementation()},
        {"trace", !options_.trace_path.empty()}};
}

Json ExpertStore::statistics() const {
    return {{"ensure_calls", stats_.ensure_calls},
            {"demanded_experts", stats_.demanded},
            {"hits", stats_.hits},
            {"prefetch_hits", stats_.prefetch_hits},
            {"inflight_prefetch_hits", stats_.inflight_hits},
            {"misses", stats_.misses},
            {"demand_read_bytes", stats_.demand_read_bytes},
            {"prefetch_read_bytes", stats_.prefetch_read_bytes},
            {"prefetch_issued", stats_.prefetch_issued},
            {"prefetch_wasted", stats_.prefetch_wasted},
            {"predicted", stats_.predicted},
            {"predicted_correct", stats_.predicted_correct},
            {"evictions", stats_.evictions},
            {"overflow_events", stats_.overflow_events},
            {"io_wait_ms", stats_.io_wait_ms},
            {"decode_demanded_experts", stats_.decode_demanded},
            {"decode_misses", stats_.decode_misses},
            {"decode_read_bytes", stats_.decode_read_bytes},
            {"decode_io_wait_ms", stats_.decode_io_wait_ms},
            {"predict_ms", stats_.predict_ms},
            {"evict_ms", stats_.evict_ms},
            {"slot_cold_reuses", stats_.slot_cold_reuses},
            {"slot_arena_committed_bytes", arena_used_},
            {"read_jobs", stats_.read_jobs},
            {"coalesced_reads", stats_.coalesced_reads},
            {"bridged_experts", stats_.bridged_experts},
            {"resident_bytes", resident_bytes_},
            {"peak_resident_bytes", stats_.peak_resident_bytes}};
}
} // namespace eqt
