#include "expert_store.h"

#include "expert_pack.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

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
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sched.h>
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

// File views replace parts of a placeholder reservation in place (Windows 10 1803+). Both the view address and
// its file offset must be multiples of the 64 KiB allocation granularity.
#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif
constexpr size_t map_granularity = 64 * 1024;
struct PlaceholderApi {
    PVOID(WINAPI *alloc)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void *, ULONG) = nullptr;
    PVOID(WINAPI *map)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, void *, ULONG) = nullptr;
};
const PlaceholderApi &placeholder_api() {
    static const PlaceholderApi api = [] {
        PlaceholderApi result;
        if (HMODULE module = GetModuleHandleW(L"kernelbase.dll")) {
            result.alloc = reinterpret_cast<decltype(result.alloc)>(GetProcAddress(module, "VirtualAlloc2"));
            result.map = reinterpret_cast<decltype(result.map)>(GetProcAddress(module, "MapViewOfFile3"));
        }
        return result;
    }();
    return api;
}
uint8_t *reserve_mapped(size_t size) {
    const auto &api = placeholder_api();
    return api.alloc ? static_cast<uint8_t *>(api.alloc(nullptr, nullptr, size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                                        PAGE_NOACCESS, nullptr, 0))
                     : nullptr;
}
// After map_tensor the reservation is a sequence of views and placeholders; each is released separately.
void release_mapped(uint8_t *base, size_t size) {
    for (uint8_t *at = base; at < base + size;) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(at, &info, sizeof(info))) {
            return;
        }
        if (info.Type == MEM_MAPPED) {
            UnmapViewOfFile(info.AllocationBase);
        } else if (info.State == MEM_RESERVE) {
            VirtualFree(info.AllocationBase, 0, MEM_RELEASE);
        }
        at = static_cast<uint8_t *>(info.BaseAddress) + info.RegionSize;
    }
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
constexpr size_t map_granularity = page;
uint8_t *reserve_mapped(size_t size) {
    return reserve(size);
}
void release_mapped(uint8_t *base, size_t size) {
    munmap(base, size);
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

// File-backed tensors: the reservation is replaced in place by read-only mappings of each tensor's file range,
// rounded to the mapping granularity, so a tensor's address is congruent to its file offset modulo that unit.
const ggml_backend_buffer_i mapped_buffer_interface = {
    /* .free_buffer   = */
    [](ggml_backend_buffer_t buffer) {
        auto *reservation = static_cast<Reservation *>(buffer->context);
        if (reservation->base) {
            release_mapped(reservation->base, reservation->size);
        }
        delete reservation;
    },
    /* .get_base      = */
    [](ggml_backend_buffer_t buffer) -> void * { return static_cast<Reservation *>(buffer->context)->base; },
    /* .init_tensor   = */
    [](ggml_backend_buffer_t buffer, ggml_tensor *tensor) {
        const auto *store = static_cast<const ExpertStore *>(buffer->buft->context);
        tensor->data =
            static_cast<uint8_t *>(tensor->data) + store->source_offset(ggml_get_name(tensor)) % map_granularity;
        return store->map_tensor(tensor) ? GGML_STATUS_SUCCESS : GGML_STATUS_ALLOC_FAILED;
    },
    /* .memset_tensor = */
    [](ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t) { unsupported("memset_tensor"); },
    /* .set_tensor    = */
    [](ggml_backend_buffer_t, ggml_tensor *tensor, const void *data, size_t offset, size_t size) {
        std::memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
    },
    /* .get_tensor    = */
    [](ggml_backend_buffer_t, const ggml_tensor *tensor, void *data, size_t offset, size_t size) {
        std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
    },
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ [](ggml_backend_buffer_t, uint8_t) {},
    /* .reset         = */ nullptr,
};

const ggml_backend_buffer_type_i mapped_buffer_type_interface = {
    /* .get_name         = */ [](ggml_backend_buffer_type_t) { return "EQT_Mapped"; },
    /* .alloc_buffer     = */
    [](ggml_backend_buffer_type_t type, size_t size) -> ggml_backend_buffer_t {
        auto *reservation = new Reservation;
        if (size) {
            reservation->size = round_up(size, 64 * 1024);
            reservation->base = reserve_mapped(reservation->size);
            if (!reservation->base) {
                delete reservation;
                return nullptr;
            }
        }
        return ggml_backend_buffer_init(type, mapped_buffer_interface, reservation, size);
    },
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ [](ggml_backend_buffer_type_t) -> size_t { return map_granularity; },
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */
    [](ggml_backend_buffer_type_t, const ggml_tensor *tensor) {
        return round_up(ggml_nbytes(tensor), map_granularity) + map_granularity;
    },
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ [](ggml_backend_buffer_type_t) { return true; },
};

enum class Node { other, topk, residual, layer_out };
Node classify(const char *name, int &layer) {
    const char *suffix = nullptr;
    Node kind = Node::other;
    if (std::strncmp(name, "ffn_moe_topk-", 13) == 0) {
        kind = Node::topk, suffix = name + 13;
    } else if (std::strncmp(name, "attn_residual-", 14) == 0) {
        kind = Node::residual, suffix = name + 14;
    } else if (std::strncmp(name, "l_last-", 7) == 0) {
        kind = Node::layer_out, suffix = name + 7;
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

namespace {
// CPUs of the lowest capacity class on a heterogeneous SoC (the little cores); empty when unknown or uniform.
std::vector<int> lowest_capacity_cpus() {
    std::vector<std::pair<int, int>> capacities;
#ifdef __linux__
    for (int cpu = 0; cpu < 64; ++cpu) {
        const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpu_capacity";
        std::FILE *file = std::fopen(path.c_str(), "r");
        if (!file) {
            continue;
        }
        int capacity = 0;
        if (std::fscanf(file, "%d", &capacity) == 1) {
            capacities.emplace_back(cpu, capacity);
        }
        std::fclose(file);
    }
#endif
    std::vector<int> result;
    if (capacities.empty()) {
        return result;
    }
    int lowest = capacities.front().second, highest = lowest;
    for (const auto &[cpu, capacity] : capacities) {
        lowest = std::min(lowest, capacity);
        highest = std::max(highest, capacity);
    }
    for (const auto &[cpu, capacity] : capacities) {
        if (capacity == lowest && lowest < highest) {
            result.push_back(cpu);
        }
    }
    return result;
}

void pin_to(const std::vector<int> &cpus) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int cpu : cpus) {
        CPU_SET(cpu, &set);
    }
    sched_setaffinity(0, sizeof(set), &set); // a failure keeps the thread unpinned
#else
    (void)cpus;
#endif
}

// Returns the word once it differs from `old`, or after a short wait. On AArch64 the core waits in WFE until
// another core writes the word's cache line (or the kernel's periodic event stream fires), without a syscall.
uint32_t wait_word_change(const std::atomic<uint32_t> &word, uint32_t old) {
#if defined(__aarch64__)
    uint32_t value;
    __asm__ __volatile__("ldaxr %w0, [%1]" : "=&r"(value) : "r"(&word) : "memory");
    if (value == old) {
        __asm__ __volatile__("wfe" ::: "memory");
    }
#else
    for (int i = 0; i < 64 && word.load(std::memory_order_acquire) == old; ++i) {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
    }
#endif
    return word.load(std::memory_order_acquire);
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
      file_(std::make_unique<SourceFile>(options_.pack_path.empty() ? path : options_.pack_path, options_.direct_io)),
      packed_(!options_.pack_path.empty()) {
    if (options_.io_threads < 1 || options_.io_threads > 16) {
        throw std::invalid_argument("io_threads must be in 1..16");
    }
    if (packed_ && !options_.slots) {
        throw std::invalid_argument("expert_pack requires expert_slots");
    }
    if (options_.pack_split < 1 || options_.pack_split > 4) {
        throw std::invalid_argument("expert_pack_split must be in 1..4");
    }
    model_size_ = packed_ ? SourceFile(path, false).size() : file_->size();
    auto *cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        throw std::runtime_error("No CPU backend registered");
    }
    buffer_type_.reset(new ggml_backend_buffer_type{buffer_type_interface, cpu, this});
    mapped_type_.reset(new ggml_backend_buffer_type{mapped_buffer_type_interface, cpu, this});
    path_ = path;
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
    if (set_expert_ready_) {
        set_expert_ready_(nullptr, nullptr);
    }
    if (predictor_.joinable()) {
        {
            std::lock_guard lock(predict_mutex_);
            predict_stopping_ = true;
        }
        predict_ready_.notify_all();
        predictor_.join();
    }
    helpers_stop_.store(true);
    helper_epoch_.fetch_add(1);
    helper_epoch_.notify_all();
    for (auto &helper : helpers_) {
        helper.join();
    }
    awake_stopping_.store(true);
    demand_signal_.fetch_add(1);
    demand_signal_.notify_all();
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        for (auto &queue : queues_) {
            queue.clear();
        }
    }
    queue_ready_.notify_all();
    for (auto &worker : workers_) {
        worker.join();
    }
    if (arena_) {
        release(arena_, arena_size_);
    }
#ifdef _WIN32
    if (map_handle_) {
        CloseHandle(map_handle_);
    }
#else
    if (map_fd_ >= 0) {
        close(map_fd_);
    }
#endif
}

ggml_backend_buffer_type *ExpertStore::buffer_type() {
    return buffer_type_.get();
}

bool ExpertStore::owns(const ggml_tensor *tensor) const {
    return tensor->buffer && ggml_backend_buffer_get_type(tensor->buffer) == buffer_type_.get();
}

ggml_backend_buffer_type *ExpertStore::mapped_buffer_type() {
    return mapped_type_.get();
}

bool ExpertStore::maps(const ggml_tensor *tensor) const {
    return tensor->buffer && ggml_backend_buffer_get_type(tensor->buffer) == mapped_type_.get();
}

bool ExpertStore::map_tensor(ggml_tensor *tensor) const {
    const uint64_t offset = source_offset(ggml_get_name(tensor));
    auto *begin = reinterpret_cast<uint8_t *>(round_down(reinterpret_cast<uintptr_t>(tensor->data), map_granularity));
    const size_t size = round_up(offset % map_granularity + ggml_nbytes(tensor), map_granularity);
#ifdef _WIN32
    const auto &api = placeholder_api();
    if (!api.map || round_down(offset, map_granularity) + size > model_size_) {
        return false; // a view cannot extend past the end of the file
    }
    auto &mapping = const_cast<void *&>(map_handle_);
    if (!mapping) {
        const int length = MultiByteToWideChar(CP_UTF8, 0, path_.c_str(), -1, nullptr, 0);
        std::wstring wide(length > 0 ? length : 1, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path_.c_str(), -1, wide.data(), length);
        HANDLE file = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                  FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            return false;
        }
        mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        CloseHandle(file); // the mapping object keeps the file open
        if (!mapping) {
            return false;
        }
    }
    // Split the placeholder so this tensor's range is one placeholder of its own, then replace it with the view.
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(begin, &info, sizeof(info)) || info.State != MEM_RESERVE) {
        return false;
    }
    if ((info.BaseAddress != begin || info.RegionSize != size) &&
        !VirtualFree(begin, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
        return false;
    }
    return api.map(mapping, GetCurrentProcess(), begin, round_down(offset, map_granularity), size,
                   MEM_REPLACE_PLACEHOLDER, PAGE_READONLY, nullptr, 0) == begin;
#else
    auto &fd = const_cast<int &>(map_fd_);
    if (fd < 0) {
        fd = open(path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            return false;
        }
    }
    // The offset's page remainder equals the tensor's, so the mapping starts on the tensor's first page.
    void *mapped = mmap(begin, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, static_cast<off_t>(round_down(offset, page)));
    if (mapped == MAP_FAILED) {
        return false;
    }
    madvise(mapped, size, MADV_RANDOM);
    return true;
#endif
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
    if (file_offset + ggml_nbytes(tensor) > model_size_) {
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
    if (layer->banks.size() >= 8) {
        throw std::runtime_error(std::string("Too many routed banks in one layer: ") + name);
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
    enum class Kind { router, norm, mixer_norm, mixer_down, mixer_up };
    static constexpr std::pair<const char *, Kind> suffixes[] = {
        {".ffn_gate_inp.weight", Kind::router}, {".post_attention_norm.weight", Kind::norm},
        {".hc_ffn_norm.weight", Kind::mixer_norm}, {".hc_ffn_down.weight", Kind::mixer_down},
        {".hc_ffn_up.weight", Kind::mixer_up}};
    int index = -1;
    const char *name = ggml_get_name(tensor);
    const auto *match = std::find_if(std::begin(suffixes), std::end(suffixes),
                                     [&](const auto &entry) { return parse_tensor_name(name, entry.first, index); });
    if (match == std::end(suffixes)) {
        return;
    }
    const Kind kind = match->second;
    // The mixer projections stay in their stored type; everything else the lookahead reads must be F32.
    const bool projection = kind == Kind::mixer_down || kind == Kind::mixer_up;
    const bool readable = tensor->type == GGML_TYPE_F32 ||
                          (projection && tensor->type == GGML_TYPE_Q8_0 && tensor->ne[0] % 32 == 0);
    if (!readable || !ggml_is_contiguous(tensor) || !ggml_backend_buffer_is_host(tensor->buffer)) {
        return;
    }
    if (static_cast<int>(layers_.size()) <= index) {
        layers_.resize(index + 1);
    }
    if (!layers_[index]) {
        layers_[index] = std::make_unique<Layer>();
    }
    auto &layer = *layers_[index];
    switch (kind) {
    case Kind::router:
        layer.router = static_cast<const float *>(tensor->data);
        layer.width = static_cast<int>(tensor->ne[0]);
        layer.router_rows = static_cast<int>(tensor->ne[1]);
        break;
    case Kind::norm: layer.norm = static_cast<const float *>(tensor->data); break;
    case Kind::mixer_norm:
        // Loaded as [width, streams]: one RMS weight vector per stream.
        layer.mixer_norm = static_cast<const float *>(tensor->data);
        layer.streams = static_cast<int>(tensor->ne[1]);
        break;
    case Kind::mixer_down: layer.mixer_down = tensor; break;
    case Kind::mixer_up: layer.mixer_up = tensor; break;
    }
}

size_t ExpertStore::lookahead_width(const Layer &layer) {
    return static_cast<size_t>(layer.width) * (layer.streams > 0 ? layer.streams : 1);
}

void ExpertStore::attach_pack() {
    PackHeader expected;
    expected.source_size = model_size_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        const Layer *layer = layers_[i].get();
        if (!layer || layer->banks.empty()) {
            continue;
        }
        PackLayer planned;
        planned.layer = static_cast<int>(i);
        planned.experts = layer->experts;
        for (const auto &bank : layer->banks) {
            PackBank entry;
            entry.tensor = ggml_get_name(bank.tensor);
            entry.type = static_cast<int>(bank.tensor->type);
            for (int d = 0; d < 3; ++d) {
                entry.ne[d] = bank.tensor->ne[d];
            }
            entry.source_offset = bank.file_offset;
            entry.stride = bank.stride;
            planned.banks.push_back(std::move(entry));
        }
        expected.layers.push_back(std::move(planned));
    }
    plan_pack(expected);
    const PackHeader pack = read_pack_header(options_.pack_path);
    check_pack_matches(pack, expected);
    if (file_->size() != pack.file_bytes) {
        throw std::runtime_error("Expert pack size differs from its header");
    }
    // Banks take the pack's order, so a job's run of banks is one contiguous range of the packed expert.
    for (const auto &entry : pack.layers) {
        Layer &layer = *layers_[entry.layer];
        std::vector<Bank> ordered;
        for (const auto &packed : entry.banks) {
            const auto it = std::find_if(layer.banks.begin(), layer.banks.end(),
                                         [&](const Bank &bank) { return packed.tensor == ggml_get_name(bank.tensor); });
            ordered.push_back(*it);
            ordered.back().slot_offset = packed.offset;
        }
        layer.banks = std::move(ordered);
        layer.slot_bytes = entry.expert_bytes;
        layer.entry_bytes = entry.expert_bytes;
        layer.pack_base = entry.data_offset;
    }
}

void ExpertStore::finalize(int experts_used, float norm_epsilon) {
    experts_used_ = experts_used;
    norm_epsilon_ = norm_epsilon;
    if (packed_) {
        attach_pack();
    }
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
    const uint64_t decode_extra = options_.decode_uses_prompt_allowance
                                      ? static_cast<uint64_t>(options_.prefill_prefetch_mib) << 20 : 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto *layer = layers_[i].get();
        if (!layer || layer->banks.empty()) {
            continue;
        }
        const double weight = options_.layer_weights.empty() ? 1.0 : options_.layer_weights[i];
        layer->share = weight / weight_total;
        const double share = static_cast<double>(options_.budget_bytes) * layer->share;
        const double decode_share = static_cast<double>(options_.budget_bytes + decode_extra) * layer->share;
        layer->prompt_quota = std::min(layer->experts, static_cast<int>(share / layer->entry_bytes));
        layer->decode_quota = std::min(layer->experts, static_cast<int>(decode_share / layer->entry_bytes));
        layer->quota = layer->prompt_quota;
        layer->entries = std::make_unique<Entry[]>(layer->experts);
        layer->seen.assign(layer->experts, 0);
        layer->prediction.reserve(static_cast<size_t>(experts_used + options_.prefetch_extra) * max_lookahead_rows);
        layer->prediction_rank.reserve(layer->prediction.capacity());
        layer->early_prediction.reserve(layer->prediction.capacity());
        layer->early_rank.reserve(layer->prediction.capacity());
        layer->late_prediction.reserve(layer->prediction.capacity());
        layer->late_rank.reserve(layer->prediction.capacity());
        effective_budget_ += static_cast<uint64_t>(layer->quota) * layer->entry_bytes;
        decode_budget_ += static_cast<uint64_t>(layer->decode_quota) * layer->entry_bytes;
        max_experts = std::max(max_experts, layer->experts);
        // A mixer must map `streams` streams of the router's width through one rank and back.
        const auto *down = layer->mixer_down, *up = layer->mixer_up;
        const int64_t total = static_cast<int64_t>(layer->width) * layer->streams;
        const bool mixer = layer->mixer_norm && down && up && layer->streams > 1 && down->ne[0] == total &&
                           up->ne[1] == total && up->ne[0] == down->ne[1];
        if (!mixer) {
            layer->streams = 0;
        }
        if (layer->router && (layer->router_rows != layer->experts || !(layer->norm || mixer))) {
            layer->router = nullptr; // Lookahead needs a matching F32 router and an RMS weight or a mixer.
        }
        predictable_layers_ += layer->router != nullptr;
        if (layer->router && mixer) {
            mix_normed_.resize(std::max<size_t>(mix_normed_.size(), total));
            mix_gate_.resize(std::max<size_t>(mix_gate_.size(), total));
            mix_low_.resize(std::max<size_t>(mix_low_.size(), down->ne[1]));
            const size_t widest = static_cast<size_t>(std::max<int64_t>(total, down->ne[1]));
            quant_.resize(std::max(quant_.size(), widest));
            quant_scales_.resize(std::max(quant_scales_.size(), widest / 32));
        }
        if (layer->router && options_.lookahead_router_q8 && layer->width % 32 == 0) {
            const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, layer->width);
            layer->router_q8.resize(row_bytes * static_cast<size_t>(layer->experts));
            ggml_quantize_chunk(GGML_TYPE_Q8_0, layer->router, layer->router_q8.data(), 0, layer->experts, layer->width,
                                nullptr);
            quant_.resize(std::max<size_t>(quant_.size(), layer->width));
            quant_scales_.resize(std::max<size_t>(quant_scales_.size(), layer->width / 32));
        }
    }
    if (options_.prefetch && predictable_layers_ == 0) {
        throw std::invalid_argument("prefetch requires an F32 ffn_gate_inp with post_attention_norm or an FFN mixer");
    }
    prompt_budget_ = effective_budget_;
    residuals_.assign(layers_.size(), nullptr);
    graph_topk_.assign(layers_.size(), nullptr);
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
    const int reserved = std::min(options_.demand_threads, options_.io_threads - 1);
    little_cpus_ = lowest_capacity_cpus();
    const int awake = little_cpus_.size() >= 2 ? std::min(options_.awake_demand_threads, reserved) : 0;
    for (int i = 0; i < options_.io_threads; ++i) {
        workers_.emplace_back([this, demand_only = i < reserved, awake = i < awake] { worker(demand_only, awake); });
    }
    if (options_.prefetch && predictable_layers_ > 0) {
        predictor_ = std::thread([this] { predictor(); });
        // Helpers run on the little cores, leaving one of them to the I/O threads; elsewhere they are unpinned.
        const int helpers = little_cpus_.empty() ? options_.lookahead_threads - 1
                                                 : std::min<int>(options_.lookahead_threads - 1, static_cast<int>(little_cpus_.size()) - 1);
        for (int i = 0; i < helpers; ++i) {
            helpers_.emplace_back([this] { helper(); });
        }
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
    set_expert_ready_ = reinterpret_cast<decltype(set_expert_ready_)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_cpu_set_expert_ready"));
    if (set_expert_ready_) {
        set_expert_ready_(&ExpertStore::expert_ready, this);
        overlap_reads_ = options_.overlap_reads;
    }
}

const ExpertStore::BankRef *ExpertStore::bank_of(const ggml_tensor *weights) const {
    const ggml_tensor *base = weights->view_src ? weights->view_src : weights;
    const auto it = std::lower_bound(bank_index_.begin(), bank_index_.end(), base,
                                     [](const BankRef &ref, const ggml_tensor *t) {
                                         return std::less<const ggml_tensor *>()(ref.tensor, t);
                                     });
    return it == bank_index_.end() || it->tensor != base ? nullptr : &*it;
}

bool ExpertStore::expert_ready(const ggml_tensor *weights, int64_t expert, void *opaque) {
    const auto &store = *static_cast<const ExpertStore *>(opaque);
    const BankRef *ref = store.bank_of(weights);
    if (!ref || weights->view_src || expert < 0 || expert >= ref->layer->experts) {
        return true;
    }
    return store.layer_ready(*ref, expert);
}

const void *ExpertStore::expert_address(const ggml_tensor *weights, int64_t expert, void *opaque) {
    const auto &store = *static_cast<const ExpertStore *>(opaque);
    const BankRef *it = store.bank_of(weights);
    if (!it) {
        return nullptr; // not a streamed bank
    }
    const Bank &bank = it->layer->banks[it->bank];
    // A view of a bank (none in the target GGUFs) addresses view_offs bytes into the bank's expert layout.
    const size_t offset = weights->view_src ? weights->view_offs : 0;
    const int64_t e = expert + static_cast<int64_t>(offset / bank.stride);
    if (e < 0 || e >= it->layer->experts || !it->layer->entries[e].slot) {
        GGML_ABORT("MUL_MAT_ID read a streamed expert that is not resident");
    }
    store.wait_bank(*it, static_cast<int>(e));
    return store.data_of(*it->layer, it->bank, static_cast<int>(e)) + offset % bank.stride;
}

bool ExpertStore::layer_ready(const BankRef &ref, int64_t expert) const {
    const auto &entry = ref.layer->entries[expert];
    return entry.slot && (entry.ready.load(std::memory_order_acquire) >> ref.bank & 1);
}

// Called by compute threads inside MUL_MAT_ID. A bank still being read blocks only the threads that reach it.
// They spin like GGML's own barriers: sleeping idles the big cores, and the governor's lower clocks then slowed
// every following operation of the token about twofold (measured on the Galaxy). A failed read sets the bit too,
// and the graph is aborted.
void ExpertStore::wait_bank(const BankRef &ref, int expert) const {
    auto &entry = ref.layer->entries[expert];
    const uint32_t bit = uint32_t{1} << ref.bank;
    uint32_t ready = entry.ready.load(std::memory_order_acquire);
    if (!(ready & bit)) {
        const auto start = Clock::now();
        for (uint32_t spins = 0; !((ready = entry.ready.load(std::memory_order_acquire)) & bit); ++spins) {
            // A read that takes this long is stalled; stop burning the core while it finishes.
            if ((spins & 1023) == 0 && Clock::now() - start > std::chrono::microseconds(options_.bank_spin_us)) {
                entry.ready.wait(ready, std::memory_order_acquire);
                continue;
            }
#if defined(__aarch64__)
            __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
        compute_wait_ns_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    }
    if (entry.failed.load()) {
        aborted_.store(true);
    }
}

uint8_t *ExpertStore::data_of(const Layer &layer, int bank_index, int expert) const {
    const Bank &bank = layer.banks[bank_index];
    if (!options_.slots) {
        return bank.data + static_cast<size_t>(expert) * bank.stride;
    }
    if (packed_) {
        return layer.entries[expert].slot + bank.slot_offset; // packed banks start on a page
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
    wait_predictor();
    if (decode != decode_phase_) {
        // Decoding uses the prompt allowance as cache; a prompt's borrowing gives it back layer by layer.
        for (auto &layer : layers_) {
            if (layer && !layer->banks.empty()) {
                layer->quota = decode ? layer->decode_quota : layer->prompt_quota;
            }
        }
        effective_budget_ = decode ? decode_budget_ : prompt_budget_;
    }
    if (!decode && decode_phase_ && decode_extra_ > 0) {
        // The memory lent while decoding goes back now, before the next prompt's compute buffer needs it.
        for (auto &layer : layers_) {
            if (layer && !layer->banks.empty()) {
                trim(*layer);
            }
        }
        decode_extra_ = 0;
        plan_decode_quotas();
        if (options_.slots) {
            for (auto &slots : slot_classes_) {
                for (uint8_t *slot : slots.warm) {
                    discard_pages(slot, slots.bytes);
                    slots.cold.push_back(slot);
                }
                slots.warm.clear();
            }
        }
    }
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
    awake_decode_.store(decode);
    if (!decode) {
        demand_signal_.fetch_add(1);
        demand_signal_.notify_all();
    }
}

void ExpertStore::plan_decode_quotas() {
    const uint64_t extra = (options_.decode_uses_prompt_allowance ? static_cast<uint64_t>(options_.prefill_prefetch_mib) << 20 : 0) +
                           decode_extra_;
    decode_budget_ = 0;
    for (auto &layer : layers_) {
        if (layer && !layer->banks.empty()) {
            const double share = static_cast<double>(options_.budget_bytes + extra) * layer->share;
            layer->decode_quota = std::min(layer->experts, static_cast<int>(share / layer->entry_bytes));
            decode_budget_ += static_cast<uint64_t>(layer->decode_quota) * layer->entry_bytes;
        }
    }
}

void ExpertStore::set_decode_extra(uint64_t bytes) {
    wait_predictor();
    decode_extra_ = bytes;
    plan_decode_quotas();
    if (decode_phase_) {
        for (auto &layer : layers_) {
            if (layer && !layer->banks.empty()) {
                layer->quota = layer->decode_quota;
            }
        }
        effective_budget_ = decode_budget_;
    }
}

void ExpertStore::set_awake_allowed(bool allowed) {
    awake_allowed_.store(allowed);
}

// One job reads a contiguous file range of one bank into its experts' slots. Direct reads take each expert's
// page-rounded range; neighbours whose ranges share a page read it once, and the second copies it afterwards.
void ExpertStore::read_slots(const Job &job) const {
    const Layer &layer = *job.layer;
    const Bank &bank = layer.banks[job.bank];
    SourceFile::Span spans[64];
    if (packed_) {
        // A slot mirrors the packed expert byte for byte, so every read is page-aligned and lands in place.
        const uint64_t base = layer.pack_base + static_cast<uint64_t>(job.first) * layer.slot_bytes;
        if (job.count == 1) {
            const Bank &last = layer.banks[job.bank + job.banks - 1];
            const size_t begin = bank.slot_offset, end = last.slot_offset + round_up(last.stride, page);
            const SourceFile::Span span{layer.entries[job.first].slot + begin, end - begin};
            file_->read_vector(&span, 1, base + begin, base + end);
            return;
        }
        for (int i = 0; i < job.count; ++i) {
            spans[i] = {layer.entries[job.first + i].slot, layer.slot_bytes};
        }
        file_->read_vector(spans, job.count, base, base + static_cast<uint64_t>(job.count) * layer.slot_bytes);
        return;
    }
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

uint64_t ExpertStore::job_bytes(const Job &job) const {
    const Layer &layer = *job.layer;
    if (!packed_) {
        return layer.banks[job.bank].stride * static_cast<uint64_t>(job.count);
    }
    if (job.count > 1) {
        return layer.slot_bytes * static_cast<uint64_t>(job.count);
    }
    const Bank &last = layer.banks[job.bank + job.banks - 1];
    return last.slot_offset + round_up(last.stride, page) - layer.banks[job.bank].slot_offset;
}

void ExpertStore::begin_request() {
    end_request();
    stats_ = {};
    compute_wait_ns_.store(0);
    stats_.peak_resident_bytes = resident_bytes_;
    ++requests_;
    if (trace_) {
        trace_ << "{\"request\":" << requests_ << "}\n";
    }
}

void ExpertStore::end_request() {
    wait_predictor();
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

uint64_t ExpertStore::release_cache() {
    end_request(); // no read in flight and every finished one reaped
    const uint64_t before = resident_bytes_;
    for (auto &layer : layers_) {
        if (!layer || layer->banks.empty()) {
            continue;
        }
        for (int e = 0; e < layer->experts; ++e) {
            if (layer->entries[e].state == State::resident) {
                evict(*layer, e);
            }
        }
    }
    if (options_.slots) {
        for (auto &slots : slot_classes_) {
            for (uint8_t *slot : slots.warm) {
                discard_pages(slot, slots.bytes);
                slots.cold.push_back(slot);
            }
            slots.warm.clear();
        }
    }
    return before - resident_bytes_;
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
            const bool usable = store.options_.prefetch && next && next->router && !next->banks.empty() &&
                                ggml_nelements(tensor) % lookahead_width(*next) == 0 &&
                                ggml_nelements(tensor) / static_cast<int64_t>(lookahead_width(*next)) <= max_lookahead_rows;
            if (layer < static_cast<int>(store.residuals_.size())) {
                store.residuals_[layer] = usable ? tensor : nullptr;
            }
            return false;
        }
        if (kind == Node::layer_out) {
            // A graph split: the output streams are read before the next layer's attention runs.
            const Layer *next = store.layer_at(layer + 1);
            return store.options_.lookahead_late > 0 && next && next->router && next->streams > 0 &&
                   !next->banks.empty() && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                   ggml_nelements(tensor) % lookahead_width(*next) == 0 &&
                   ggml_nelements(tensor) / static_cast<int64_t>(lookahead_width(*next)) <= max_lookahead_rows;
        }
        if (kind != Node::topk || !store.layer_at(layer) || store.layer_at(layer)->banks.empty()) {
            return false;
        }
        if (store.topk_in_graph_ && tensor->ne[1] == 1 && tensor->view_src && tensor->view_src->op == GGML_OP_ARGSORT &&
            ggml_nrows(tensor->view_src) == 1 && layer < static_cast<int>(store.graph_topk_.size())) {
            store.graph_topk_[layer] = tensor; // handled by compute_hook, without a split
            return false;
        }
        return true;
    }
    if (store.cancelled_.load()) {
        return false;
    }
    if (kind == Node::layer_out) {
        if (auto *next = store.layer_at(layer + 1); next && next->router && next->streams > 0) {
            store.wait_predictor(); // the scratch buffers are the predictor's
            const auto start = Clock::now();
            const size_t width = lookahead_width(*next);
            store.score(*next, static_cast<const float *>(tensor->data), ggml_nelements(tensor) / static_cast<int64_t>(width),
                        width * sizeof(float), store.options_.lookahead_late_gate, next->late_prediction, next->late_rank);
            next->late_valid = true;
            store.stats_.late_ms += since(start);
        }
        return true;
    }
    if (kind != Node::topk) {
        return true;
    }
    return store.on_topk(layer, tensor);
}

bool ExpertStore::compute_hook(ggml_tensor *node, int ith, void *opaque) {
    // Only thread 0 reads graph_topk_ (written by observe on the same thread); the other threads compute the
    // argsort's rows from their index, which for one row is none.
    if (ith != 0 || node->op != GGML_OP_ARGSORT) {
        return false;
    }
    auto &store = *static_cast<ExpertStore *>(opaque);
    if (!store.topk_in_graph_) {
        return false;
    }
    for (size_t layer = 0; layer < store.graph_topk_.size(); ++layer) {
        const ggml_tensor *topk = store.graph_topk_[layer];
        if (!topk || topk->view_src != node) {
            continue;
        }
        store.graph_topk_[layer] = nullptr;
        // GGML's CPU argsort of one F32 row: indices, then std::sort with the same comparison.
        const ggml_tensor *scores = node->src[0];
        GGML_ASSERT(scores->type == GGML_TYPE_F32 && node->type == GGML_TYPE_I32 && ggml_nrows(scores) == 1);
        const auto order = static_cast<ggml_sort_order>(node->op_params[0]);
        const auto *data = static_cast<const float *>(scores->data);
        auto *indices = static_cast<int32_t *>(node->data);
        for (int64_t j = 0; j < scores->ne[0]; ++j) {
            indices[j] = static_cast<int32_t>(j);
        }
        if (order == GGML_SORT_ORDER_ASC) {
            std::sort(indices, indices + scores->ne[0], [data](int32_t a, int32_t b) { return data[a] < data[b]; });
        } else {
            std::sort(indices, indices + scores->ne[0], [data](int32_t a, int32_t b) { return data[a] > data[b]; });
        }
        // A cancelled request stops after this node (the engine's abort callback); a failure stops it too and
        // leaves the store failed, as a failed read does.
        if (!store.cancelled_.load() && !store.on_topk(static_cast<int>(layer), topk) && !store.cancelled_.load()) {
            store.aborted_.store(true);
        }
        return true;
    }
    return false;
}

bool ExpertStore::on_topk(int layer, const ggml_tensor *tensor) {
    // A lookahead still running when its layer is reached holds up the decode thread.
    const auto predictor_wait = Clock::now();
    wait_predictor();
    stats_.predict_wait_ms += since(predictor_wait);
    const ggml_tensor *residual = layer < static_cast<int>(residuals_.size()) ? residuals_[layer] : nullptr;
    if (layer < static_cast<int>(residuals_.size())) {
        residuals_[layer] = nullptr;
    }
    // Rows of the next layer's lookahead width: one residual vector per token, or all its streams with a mixer.
    const Layer *next = layer_at(layer + 1);
    const bool usable = residual && next && residual->type == GGML_TYPE_F32 && ggml_is_contiguous(residual);
    const size_t width = usable ? lookahead_width(*next) : 0;
    const int64_t rows = usable ? ggml_nelements(residual) / static_cast<int64_t>(width) : 0;
    if (!usable || !predictor_.joinable()) {
        if (usable) {
            predict(layer, static_cast<const float *>(residual->data), rows, width * sizeof(float));
            predict_after(layer, static_cast<const float *>(residual->data), rows, width * sizeof(float));
        }
        return ensure(layer, tensor);
    }
    // The residual stays allocated only until the post-MoE add, which the graph may reach before the
    // predictor runs, so it is copied (8-40 KiB per row) first.
    predict_input_.resize(width * rows);
    std::memcpy(predict_input_.data(), residual->data, width * rows * sizeof(float));
    const bool ok = ensure(layer, tensor);
    {
        std::lock_guard lock(predict_mutex_);
        predict_layer_ = layer;
        predict_rows_ = rows;
        predict_pending_ = true;
        predict_flag_.store(true, std::memory_order_release);
    }
    predict_ready_.notify_one();
    return ok;
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
        // The selection is the head of the router's full argsort (its view source): the next ranks are recorded
        // for offline cache-policy studies.
        const ggml_tensor *order = ids->view_src;
        if (order && order->type == GGML_TYPE_I32 && order->ne[0] > selected && order->nb[1] == ids->nb[1]) {
            const int64_t ranks = std::min<int64_t>(order->ne[0], 64);
            trace_ << "],\"ranking\":[";
            for (int64_t row = 0; row < rows; ++row) {
                for (int64_t rank = 0; rank < ranks; ++rank) {
                    trace_ << (row || rank ? "," : "")
                           << *reinterpret_cast<const int32_t *>(static_cast<const char *>(ids->data) + row * ids->nb[1] +
                                                                 rank * order->nb[0]);
                }
            }
        }
        trace_ << "]}\n";
    }
    if (layer.prediction_valid) {
        stats_.predicted += layer.prediction.size();
        for (size_t i = 0; i < layer.prediction.size(); ++i) {
            const bool correct = layer.seen[layer.prediction[i]] == epoch_;
            const int rank = std::min<int>(layer.prediction_rank[i], 15);
            stats_.predicted_correct += correct;
            ++stats_.rank_predicted[rank];
            stats_.rank_correct[rank] += correct;
        }
        layer.prediction_valid = false;
    }
    if (layer.early_valid) {
        for (size_t i = 0; i < layer.early_prediction.size(); ++i) {
            const int rank = std::min<int>(layer.early_rank[i], 15);
            ++stats_.early_rank_predicted[rank];
            stats_.early_rank_correct[rank] += layer.seen[layer.early_prediction[i]] == epoch_;
        }
        layer.early_valid = false;
    }
    if (layer.late_valid) {
        for (size_t i = 0; i < layer.late_prediction.size(); ++i) {
            const int rank = std::min<int>(layer.late_rank[i], 15);
            ++stats_.late_rank_predicted[rank];
            stats_.late_rank_correct[rank] += layer.seen[layer.late_prediction[i]] == epoch_;
        }
        layer.late_valid = false;
    }
    ++clock_;
    stats_.demanded += demand_.size();
    missing_.clear();
    bool queued_reads = false;
    for (const int expert : demand_) {
        auto &entry = layer.entries[expert];
        reap(layer, expert);
        if (entry.pooled) {
            entry.pooled = false; // selected: an ordinary cache entry from now on
            --layer.pooled;
        }
        entry.early = false;
        entry.stamp = clock_;
        if (entry.state == State::resident) {
            ++stats_.hits;
            stats_.prefetch_hits += entry.prefetched;
        } else if (entry.state == State::loading) {
            ++stats_.inflight_hits;
            queued_reads = true;
        } else {
            ++stats_.misses;
            missing_.push_back(expert);
        }
        entry.prefetched = false;
    }
    if (queued_reads && options_.read_priority) {
        // Speculative reads this layer now waits for, and that no worker has started, are read next.
        stats_.promoted_reads += move_queued(layer, queue_demand, nullptr);
    }
    if (layer.pooled > 0) {
        // Predictions this layer did not use; one still being read is dropped by a later call, and one not
        // started yet is never read (read_priority).
        for (int e = 0; e < layer.experts && layer.pooled > 0; ++e) {
            auto &entry = layer.entries[e];
            if (!entry.pooled) {
                continue;
            }
            if (reap(layer, e)) {
                if (entry.state == State::resident) {
                    evict(layer, e);
                }
            } else if (options_.read_priority) {
                cancel_unread(layer, e);
            }
        }
    }
    int excess = layer.resident - layer.pooled + static_cast<int>(missing_.size()) - layer.quota;
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
    // A prompt batch can demand almost every expert of a layer (all 512, ~0.9 GiB, in Qwen3.8-Flash-Next): other
    // layers lend their entries, so the layer's working set fits inside the budget instead of on top of it.
    if (rows > max_lookahead_rows) {
        borrow(layer_index, missing_.size() * layer.entry_bytes);
    }
    const bool decode = decode_phase_;
    if (decode) {
        stats_.decode_demanded += demand_.size();
        stats_.decode_misses += missing_.size();
        stats_.decode_read_bytes += missing_.size() * layer.entry_bytes;
    }
    // Overlapped reads never bridge resident experts: compute may read those while the run is in flight.
    const bool overlap = overlap_reads_ && rows <= max_lookahead_rows;
    load_runs(layer, missing_, true, overlap ? 0 : options_.coalesce_gap);
    if (rows > max_lookahead_rows && options_.prefill_prefetch_mib > 0) {
        if (auto *next = layer_at(layer_index + 1); next && !next->banks.empty()) {
            const double unused = std::pow(1.0 - static_cast<double>(experts_used_) / next->experts, static_cast<double>(rows));
            if (1.0 - unused >= options_.prefill_prefetch_min_coverage) {
                prefetch_whole_layer(*next);
            }
        }
    }
    if (prefetch_layer_ >= 0) {
        if (auto *next = layer_at(prefetch_layer_)) {
            issue_prefetch(*next);
        }
        prefetch_layer_ = -1;
        if (auto *after = layer_at(early_layer_)) {
            issue_reads_ahead(*after, after->early_prediction, after->early_rank, options_.lookahead2_count, true);
        }
        early_layer_ = -1;
    }
    if (overlap) {
        stats_.overlapped_layers += decode && !missing_.empty();
        stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, resident_bytes_);
        return true;
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

void ExpertStore::predictor() {
    for (;;) {
        int layer = -1;
        int64_t rows = 0;
        if (options_.lookahead_spin) {
            // A futex wake takes about 0.1 ms of a ~7 ms decode layer on a little core.
            while (!predict_flag_.load(std::memory_order_acquire) && awake_decode_.load(std::memory_order_relaxed)) {
#if defined(__aarch64__)
                __asm__ __volatile__("yield");
#endif
            }
        }
        {
            std::unique_lock lock(predict_mutex_);
            predict_ready_.wait(lock, [this] { return predict_stopping_ || predict_pending_; });
            if (predict_stopping_) {
                return;
            }
            layer = predict_layer_;
            rows = predict_rows_;
        }
        const size_t width = rows ? predict_input_.size() / static_cast<size_t>(rows) : 0;
        if (!helpers_.empty()) {
            helpers_hot_.store(true, std::memory_order_relaxed);
            wake_helpers(); // they wake during the norms, before the first projection
        }
        // The next layer's reads are queued before the layer after is scored: they are needed first.
        predict(layer, predict_input_.data(), rows, width * sizeof(float));
        if (prefetch_layer_ >= 0 && !cancelled_.load()) {
            if (auto *next = layer_at(prefetch_layer_)) {
                issue_prefetch(*next);
            }
            predict_after(layer, predict_input_.data(), rows, width * sizeof(float));
            if (auto *after = layer_at(early_layer_); after && !cancelled_.load()) {
                issue_reads_ahead(*after, after->early_prediction, after->early_rank, options_.lookahead2_count, true);
            }
        }
        prefetch_layer_ = -1;
        early_layer_ = -1;
        helpers_hot_.store(false, std::memory_order_relaxed);
        {
            std::lock_guard lock(predict_mutex_);
            predict_pending_ = false;
            predict_flag_.store(false, std::memory_order_release);
        }
        predict_done_.notify_all();
    }
}

void ExpertStore::run_rows() {
    for (;;) {
        const size_t begin = helper_next_.fetch_add(helper_chunk_, std::memory_order_relaxed);
        if (begin >= helper_rows_) {
            return;
        }
        helper_fn_(helper_context_, begin, std::min(begin + helper_chunk_, helper_rows_));
    }
}

void ExpertStore::parallel_rows(size_t rows, size_t chunk, void (*fn)(const void *, size_t, size_t), const void *context) {
    if (helpers_.empty() || rows <= chunk) {
        fn(context, 0, rows);
        return;
    }
    // The job's fields are written only after every helper finished the previous job.
    while (helper_done_.load(std::memory_order_acquire) < helper_posted_ * helpers_.size()) {
#if defined(__aarch64__)
        __asm__ __volatile__("yield");
#endif
    }
    helper_fn_ = fn;
    helper_context_ = context;
    helper_rows_ = rows;
    helper_chunk_ = chunk;
    helper_next_.store(0, std::memory_order_relaxed);
    ++helper_posted_;
    helper_epoch_.fetch_add(1, std::memory_order_release);
    helper_epoch_.notify_all(); // no system call unless a helper sleeps
    run_rows();
    while (helper_done_.load(std::memory_order_acquire) < helper_posted_ * helpers_.size()) {
#if defined(__aarch64__)
        __asm__ __volatile__("yield");
#endif
    }
}

// Posts an empty job without waiting for it: sleeping helpers wake and then spin while helpers_hot_ is set.
void ExpertStore::wake_helpers() {
    while (helper_done_.load(std::memory_order_acquire) < helper_posted_ * helpers_.size()) {
#if defined(__aarch64__)
        __asm__ __volatile__("yield");
#endif
    }
    helper_rows_ = 0;
    helper_next_.store(0, std::memory_order_relaxed);
    ++helper_posted_;
    helper_epoch_.fetch_add(1, std::memory_order_release);
    helper_epoch_.notify_all();
}

void ExpertStore::helper() {
    pin_to(little_cpus_);
    uint32_t seen = helper_epoch_.load(std::memory_order_acquire);
    for (;;) {
        uint32_t now;
        while ((now = helper_epoch_.load(std::memory_order_acquire)) == seen) {
            if (helpers_hot_.load(std::memory_order_relaxed)) {
#if defined(__aarch64__)
                __asm__ __volatile__("yield");
#endif
            } else {
                helper_epoch_.wait(seen, std::memory_order_acquire);
            }
        }
        if (helpers_stop_.load()) {
            return;
        }
        // The poster waits for every helper to finish a job before posting the next, so no job is skipped.
        seen = now;
        run_rows();
        helper_done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertStore::wait_predictor() {
    std::unique_lock lock(predict_mutex_);
    predict_done_.wait(lock, [this] { return !predict_pending_; });
}

void ExpertStore::predict(int layer_index, const float *rows, int64_t count_rows, size_t row_bytes) {
    auto *next = layer_at(layer_index + 1);
    if (!next || !next->router || !(next->norm || next->streams) || count_rows <= 0 ||
        row_bytes < lookahead_width(*next) * sizeof(float)) {
        return;
    }
    const auto start = Clock::now();
    // Approximates layer l+1's router input from the post-attention residual of layer l (skipping layer l's
    // FFN and layer l+1's attention): normalized by layer l+1's RMS weight, or mixed by its FFN mixer. A small
    // batch (speculative verification) prefetches the union of its rows. It only drives speculative reads;
    // routing still uses the exact top-k. Candidates are issued in score order, so extra candidates beyond
    // top-k are read last.
    score(*next, rows, count_rows, row_bytes, options_.lookahead_gate, next->prediction, next->prediction_rank);
    next->prediction_valid = true;
    prefetch_layer_ = layer_index + 1;
    stats_.predict_ms += since(start);
}

void ExpertStore::predict_after(int layer_index, const float *rows, int64_t count_rows, size_t row_bytes) {
    // Two layers ahead the same residual also skips layer l+1's FFN and layer l+2's attention.
    auto *next = layer_at(layer_index + 1);
    auto *after = options_.lookahead_layers > 1 && prefetch_layer_ == layer_index + 1 ? layer_at(layer_index + 2) : nullptr;
    if (!after || !after->router || !(after->norm || after->streams) || lookahead_width(*after) != lookahead_width(*next)) {
        return;
    }
    const auto start = Clock::now();
    score(*after, rows, count_rows, row_bytes, false, after->early_prediction, after->early_rank);
    after->early_valid = true;
    early_layer_ = layer_index + 2;
    stats_.predict_ms += since(start);
}

void ExpertStore::score(Layer &target, const float *rows, int64_t count_rows, size_t row_bytes, bool gate,
                        std::vector<int> &prediction, std::vector<uint8_t> &ranks) {
    const size_t width = static_cast<size_t>(target.width);
    const int count = std::min(experts_used_ + options_.prefetch_extra, target.experts);
    prediction.clear();
    ranks.clear();
    for (int64_t row = 0; row < count_rows; ++row) {
        const auto *x = reinterpret_cast<const float *>(reinterpret_cast<const char *>(rows) + row * row_bytes);
        if (target.streams) {
            mix(target, x, scratch_.data(), gate);
        } else {
            const float scale = 1.0f / std::sqrt(simd::sum_squares(x, width) / width + norm_epsilon_);
            simd::scale_mul(x, target.norm, scale, scratch_.data(), width);
        }
        const size_t experts = static_cast<size_t>(target.experts), chunk = row_chunk(experts);
        if (!target.router_q8.empty()) {
            simd::quantize_blocks(scratch_.data(), width, quant_.data(), quant_scales_.data());
            const size_t stride = ggml_row_size(GGML_TYPE_Q8_0, target.width);
            parallel_rows(experts, chunk, [&](size_t begin, size_t end) {
                simd::gemv_q8_0(target.router_q8.data() + begin * stride, stride, quant_.data(), quant_scales_.data(),
                                scores_.data() + begin, end - begin, width);
            });
        } else {
            parallel_rows(experts, chunk, [&](size_t begin, size_t end) {
                simd::gemv_rows(target.router + begin * width, scratch_.data(), scores_.data() + begin, end - begin, width);
            });
        }
        for (int e = 0; e < target.experts; ++e) {
            order_[e] = e;
        }
        std::partial_sort(
            order_.begin(), order_.begin() + count, order_.begin() + target.experts,
            [this](int a, int b) { return scores_[a] > scores_[b] || (scores_[a] == scores_[b] && a < b); });
        for (int i = 0; i < count; ++i) {
            const auto found = std::find(prediction.begin(), prediction.end(), order_[i]);
            if (found == prediction.end()) {
                prediction.push_back(order_[i]);
                ranks.push_back(static_cast<uint8_t>(std::min(i, 255)));
            } else {
                auto &rank = ranks[found - prediction.begin()];
                rank = std::min<uint8_t>(rank, static_cast<uint8_t>(std::min(i, 255)));
            }
        }
    }
}

// llama.cpp's qwen4exp build_hc_mix for one token: per-stream RMS norm times the stream's weights, a low-rank
// gate silu(down(xn) / streams) projected back by `up`, and the mean of the sigmoid-gated streams. The int8
// activation quantization of Q8_0 projections only affects which experts are prefetched.
void ExpertStore::mix(const Layer &layer, const float *x, float *out, bool use_gate) {
    const size_t width = static_cast<size_t>(layer.width), streams = static_cast<size_t>(layer.streams);
    float *normed = mix_normed_.data(), *gate = mix_gate_.data(), *low = mix_low_.data();
    for (size_t s = 0; s < streams; ++s) {
        const float *stream = x + s * width;
        const float scale = 1.0f / std::sqrt(simd::sum_squares(stream, width) / width + norm_epsilon_);
        simd::scale_mul(stream, layer.mixer_norm + s * width, scale, normed + s * width, width);
    }
    const float inverse = 1.0f / static_cast<float>(streams);
    if (!use_gate) {
        // Without the gate every stream weighs the same; the router's top-k does not depend on a common scale.
        for (size_t i = 0; i < width; ++i) {
            float sum = 0;
            for (size_t s = 0; s < streams; ++s) {
                sum += normed[s * width + i];
            }
            out[i] = sum * inverse;
        }
        return;
    }
    project(layer.mixer_down, normed, low);
    for (int64_t j = 0; j < layer.mixer_down->ne[1]; ++j) {
        const float v = low[j] * inverse;
        low[j] = v / (1.0f + std::exp(-v));
    }
    project(layer.mixer_up, low, gate);
    simd::gated_mean(normed, gate, out, width, streams);
}

void ExpertStore::project(const ggml_tensor *weights, const float *x, float *out) {
    const size_t n = static_cast<size_t>(weights->ne[0]), rows = static_cast<size_t>(weights->ne[1]);
    const size_t chunk = row_chunk(rows);
    if (weights->type == GGML_TYPE_F32) {
        const auto *data = static_cast<const float *>(weights->data);
        parallel_rows(rows, chunk, [&](size_t begin, size_t end) { simd::gemv_rows(data + begin * n, x, out + begin, end - begin, n); });
        return;
    }
    simd::quantize_blocks(x, n, quant_.data(), quant_scales_.data());
    const auto *data = static_cast<const uint8_t *>(weights->data);
    const size_t stride = weights->nb[1];
    parallel_rows(rows, chunk, [&](size_t begin, size_t end) {
        simd::gemv_q8_0(data + begin * stride, stride, quant_.data(), quant_scales_.data(), out + begin, end - begin, n);
    });
}

size_t ExpertStore::row_chunk(size_t rows) const {
    // About four chunks per thread, in multiples of the GEMVs' four-row groups.
    const size_t threads = helpers_.size() + 1;
    return std::max<size_t>(4, (rows / (threads * 4) + 3) / 4 * 4);
}

void ExpertStore::borrow(int layer_index, uint64_t bytes) {
    const int count = static_cast<int>(layers_.size());
    for (int step = 1; step < count && resident_bytes_ + bytes > effective_budget_; ++step) {
        Layer *other = layers_[(layer_index - step + count) % count].get();
        if (!other || other->banks.empty()) {
            continue;
        }
        for (int e = 0; e < other->experts; ++e) {
            reap(*other, e);
        }
        while (resident_bytes_ + bytes > effective_budget_) {
            const int victim = least_recent(*other, 0);
            if (victim < 0) {
                break;
            }
            evict(*other, victim);
            ++stats_.borrowed_evictions;
        }
    }
}

void ExpertStore::issue_prefetch(Layer &layer) {
    issue_reads_ahead(layer, layer.prediction, layer.prediction_rank, options_.prefetch_count, false);
}

void ExpertStore::issue_reads_ahead(Layer &layer, const std::vector<int> &prediction, const std::vector<uint8_t> &ranks,
                                    int limit, bool early) {
    if (!early && options_.read_priority && options_.lookahead_layers > 1) {
        // Reads the two-layer lookahead queued for experts this closer prediction also lists are needed sooner.
        stats_.confirmed_reads += move_queued(layer, queue_next, &prediction);
    }
    const Queue queue = early && options_.read_priority ? queue_after : queue_next;
    for (size_t i = 0; i < prediction.size(); ++i) {
        if (limit > 0 && ranks[i] >= limit) {
            continue;
        }
        const int expert = prediction[i];
        reap(layer, expert);
        auto &entry = layer.entries[expert];
        if (!early) {
            entry.early = false;
        }
        if (entry.state != State::absent) {
            continue; // Prediction never refreshes recency of resident entries.
        }
        if (options_.speculative_pool) {
            if (layer.pooled >= experts_used_ + options_.prefetch_extra) {
                return;
            }
            entry.pooled = true;
            ++layer.pooled;
        } else if (layer.resident >= layer.quota) {
            const int victim = least_recent(layer, 0);
            if (victim < 0 || layer.quota == 0) {
                return;
            }
            evict(layer, victim);
        }
        start_load(layer, expert, queue);
        entry.prefetched = true;
        entry.early = early;
        entry.stamp = 0; // Unused speculative entries are the first eviction candidates.
        ++stats_.prefetch_issued;
        stats_.decode_prefetch_issued += decode_phase_;
        stats_.early_prefetch_issued += early;
    }
}

int ExpertStore::move_queued(Layer &layer, Queue to, const std::vector<int> *experts) {
    const auto selected = [&](const Job &job) {
        if (job.layer != &layer || job.count != 1) {
            return false; // multi-expert runs are prompt reads, never reordered
        }
        return experts ? std::find(experts->begin(), experts->end(), job.first) != experts->end()
                       : layer.seen[job.first] == epoch_;
    };
    int moved = 0;
    {
        std::lock_guard lock(mutex_);
        for (int from = to + 1; from < queue_count; ++from) {
            auto &queue = queues_[from];
            for (auto it = queue.begin(); it != queue.end();) {
                if (!selected(*it)) {
                    ++it;
                    continue;
                }
                queues_[to].push_back(*it);
                if (to == queue_demand) {
                    queues_[to].back().queued = Clock::now(); // demand latency counts from here
                }
                it = queue.erase(it);
                ++moved;
            }
        }
    }
    if (moved && to == queue_demand) {
        demand_signal_.fetch_add(1, std::memory_order_release);
        demand_signal_.notify_all();
    }
    if (moved) {
        queue_ready_.notify_all();
    }
    return moved;
}

bool ExpertStore::cancel_unread(Layer &layer, int expert) {
    auto &entry = layer.entries[expert];
    const int banks = static_cast<int>(layer.banks.size());
    if (entry.state != State::loading) {
        return false;
    }
    int queued_jobs = 0;
    {
        std::lock_guard lock(mutex_);
        // Workers pop a job under this lock before reading it, so a job still queued has not started.
        const auto mine = [&](const Job &job) { return job.layer == &layer && job.count == 1 && job.first == expert; };
        int queued_banks = 0;
        for (int q = queue_next; q < queue_count; ++q) {
            for (const Job &job : queues_[q]) {
                if (mine(job)) {
                    queued_banks += job.banks;
                    ++queued_jobs;
                }
            }
        }
        if (queued_banks != banks || entry.pending.load() != banks) {
            return false;
        }
        for (int q = queue_next; q < queue_count; ++q) {
            queues_[q].erase(std::remove_if(queues_[q].begin(), queues_[q].end(), mine), queues_[q].end());
        }
    }
    inflight_jobs_.fetch_sub(queued_jobs);
    entry.pending.store(0);
    for (const auto &bank : layer.banks) {
        stats_.prefetch_read_bytes -= bank.stride; // as begin_load() counted it
    }
    ++stats_.cancelled_prefetches;
    entry.prefetched = false; // not counted as a wasted read: none happened
    evict(layer, expert);
    job_done_.notify_all();
    return true;
}

void ExpertStore::start_load(Layer &layer, int expert, Queue queue) {
    if (begin_load(layer, expert, queue == queue_demand)) {
        queue_reads(layer, expert, 1, 1, queue);
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
    entry.ready.store(0);
    entry.pending.store(static_cast<int>(layer.banks.size()));
    for (const auto &bank : layer.banks) {
        (demand ? stats_.demand_read_bytes : stats_.prefetch_read_bytes) += bank.stride;
    }
    return true;
}

void ExpertStore::queue_reads(Layer &layer, int first, int count, uint64_t members, Queue queue) {
    const int banks = static_cast<int>(layer.banks.size());
    // Runs of banks per job: one bank each from the GGUF; from a pack, whole experts, or for a demanded single
    // expert pack_split parts cut where their sizes come closest to equal.
    int starts[8], widths[8], jobs = 0;
    if (!packed_) {
        for (int bank = 0; bank < banks; ++bank) {
            starts[jobs] = bank, widths[jobs++] = 1;
        }
    } else {
        const int parts = count == 1 && queue == queue_demand ? std::min(options_.pack_split, banks) : 1;
        const auto end_of = [&](int b) { return layer.banks[b].slot_offset + round_up(layer.banks[b].stride, page); };
        const auto distance = [](uint64_t a, uint64_t b) { return a > b ? a - b : b - a; };
        for (int part = 0, bank = 0; part < parts; ++part) {
            const uint64_t target = layer.slot_bytes * static_cast<uint64_t>(part + 1) / parts;
            const int start = bank++;
            while (bank < banks - (parts - 1 - part) &&
                   (part == parts - 1 || distance(end_of(bank), target) < distance(end_of(bank - 1), target))) {
                ++bank;
            }
            starts[jobs] = start, widths[jobs++] = bank - start;
        }
    }
    stats_.read_jobs += jobs;
    stats_.coalesced_reads += count > 1 ? jobs : 0;
    inflight_jobs_.fetch_add(jobs);
    {
        std::lock_guard lock(mutex_);
        for (int j = 0; j < jobs; ++j) {
            queues_[queue].push_back({&layer, starts[j], first, count, members, Clock::now(), widths[j]});
        }
    }
    if (queue == queue_demand) {
        demand_signal_.fetch_add(1, std::memory_order_release);
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
    size_t widest = packed_ ? layer.slot_bytes : 0; // bytes per expert of the widest read in a run
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
        queue_reads(layer, first, count, members, demand ? queue_demand : queue_next);
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
    entry.early = false;
    if (entry.pooled) {
        entry.pooled = false;
        --layer.pooled;
    }
    entry.ready.store(0);
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
    while (layer.resident - layer.pooled > layer.quota) {
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

void ExpertStore::worker(bool demand_only, bool awake) {
    if (awake) {
        pin_to(little_cpus_);
    }
    std::vector<uint8_t> bounce;
    if (file_->direct()) {
        bounce.assign(2 * page, 0);
    }
    uint32_t signal = demand_signal_.load();
    for (;;) {
        Job job{};
        bool demand = false;
        bool taken = false;
        // Awake while decoding: wait for the counter to change without a futex sleep, then try the queue. A job
        // another thread took leaves this one awake.
        if (awake && awake_decode_.load(std::memory_order_relaxed) && awake_allowed_.load(std::memory_order_relaxed) &&
            !awake_stopping_.load(std::memory_order_relaxed)) {
            const uint32_t now = wait_word_change(demand_signal_, signal);
            if (now == signal) {
                continue;
            }
            signal = now;
            std::lock_guard lock(mutex_);
            if (stopping_) {
                return;
            }
            if (queues_[queue_demand].empty()) {
                continue;
            }
            job = queues_[queue_demand].front();
            queues_[queue_demand].pop_front();
            demand = taken = true;
        }
        if (!taken) {
            std::unique_lock lock(mutex_);
            // The cap applies while decoding; a prompt's whole-layer reads keep the device's full queue depth.
            const auto speculative = [this] {
                return (!queues_[queue_next].empty() || !queues_[queue_after].empty()) &&
                       (options_.prefetch_inflight == 0 || !awake_decode_.load(std::memory_order_relaxed) ||
                        speculative_inflight_ < options_.prefetch_inflight);
            };
            queue_ready_.wait(lock, [&] {
                return stopping_ || !queues_[queue_demand].empty() || (!demand_only && speculative());
            });
            if (stopping_) {
                return;
            }
            const int from = !queues_[queue_demand].empty() ? queue_demand
                             : !queues_[queue_next].empty() ? queue_next
                                                            : queue_after;
            demand = from == queue_demand;
            job = queues_[from].front();
            queues_[from].pop_front();
            speculative_inflight_ += !demand;
        }
        const auto read_start = Clock::now();
        {
            std::lock_guard lock(mutex_);
            if (active_reads_++ == 0) {
                busy_since_ = read_start;
            }
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
                entry.ready.fetch_or(((uint32_t{1} << job.banks) - 1) << job.bank);
                entry.ready.notify_all();
                entry.pending.fetch_sub(job.banks);
            }
        }
        inflight_jobs_.fetch_sub(1);
        {
            std::lock_guard lock(mutex_);
            const bool decoding = awake_decode_.load(std::memory_order_relaxed);
            if (decoding) {
                stats_.decode_io_bytes += job_bytes(job);
            }
            if (--active_reads_ == 0 && decoding) {
                stats_.decode_io_busy_ms += since(busy_since_);
            }
            if (demand) {
                // Queue time is measured by the submitter's clock in queue_reads(); this is the read itself.
                stats_.demand_read_ms += since(read_start);
                stats_.demand_queue_ms += std::chrono::duration<double, std::milli>(read_start - job.queued).count();
                ++stats_.demand_jobs;
                if (decoding) {
                    ++stats_.decode_demand_jobs;
                    stats_.decode_demand_job_ms += since(job.queued);
                }
            } else {
                --speculative_inflight_;
            }
        }
        if (!demand && options_.prefetch_inflight > 0) {
            queue_ready_.notify_all(); // a capped speculative read may start now (demand-only workers ignore it)
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
        {"budget_kind", "routed_expert_pages; a prompt batch borrows other layers' entries and exceeds the budget "
                        "only when one layer alone needs more, plus prefill_prefetch_mib"},
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
        {"lookahead_gate", options_.lookahead_gate},
        {"lookahead_late", options_.lookahead_late},
        {"lookahead_threads", static_cast<int>(helpers_.size()) + (predictor_.joinable() ? 1 : 0)},
        {"lookahead_spin", options_.lookahead_spin},
        {"lookahead_late_gate", options_.lookahead_late_gate},
        {"prefetch_count", options_.prefetch_count},
        {"decode_uses_prompt_allowance", options_.decode_uses_prompt_allowance},
        {"decode_budget_bytes", decode_budget_},
        {"bank_spin_us", options_.bank_spin_us},
        {"speculative_pool", options_.speculative_pool},
        {"lookahead_layers", options_.lookahead_layers},
        {"lookahead2_count", options_.lookahead2_count},
        {"read_priority", options_.read_priority},
        {"prefetch_inflight", options_.prefetch_inflight},
        {"lookahead_router_q8", options_.lookahead_router_q8},
        {"demand_io_threads", std::min(options_.demand_threads, options_.io_threads - 1)},
        {"awake_demand_io_threads", little_cpus_.size() >= 2 ? std::min(options_.awake_demand_threads, std::min(options_.demand_threads, options_.io_threads - 1)) : 0},
        {"coalesce_kib", options_.coalesce_kib},
        {"coalesce_gap", options_.coalesce_gap},
        {"prefill_prefetch_mib", options_.prefill_prefetch_mib},
        {"prefill_prefetch_min_coverage", options_.prefill_prefetch_min_coverage},
        {"slots", options_.slots},
        {"expert_pack", packed_},
        {"expert_pack_split", options_.pack_split},
        {"overlap_reads", overlap_reads_},
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
            {"borrowed_evictions", stats_.borrowed_evictions},
            {"io_wait_ms", stats_.io_wait_ms},
            {"decode_demanded_experts", stats_.decode_demanded},
            {"decode_misses", stats_.decode_misses},
            {"decode_read_bytes", stats_.decode_read_bytes},
            {"decode_io_wait_ms", stats_.decode_io_wait_ms},
            {"overlapped_layers", stats_.overlapped_layers},
            {"demand_jobs", stats_.demand_jobs},
            {"demand_job_queue_ms", stats_.demand_queue_ms},
            {"demand_job_read_ms", stats_.demand_read_ms},
            {"compute_wait_thread_ms", compute_wait_ns_.load() / 1e6},
            {"predict_ms", stats_.predict_ms},
            {"predict_wait_ms", stats_.predict_wait_ms},
            {"predicted_by_rank", std::vector<uint64_t>(std::begin(stats_.rank_predicted), std::end(stats_.rank_predicted))},
            {"predicted_correct_by_rank", std::vector<uint64_t>(std::begin(stats_.rank_correct), std::end(stats_.rank_correct))},
            {"decode_prefetch_issued", stats_.decode_prefetch_issued},
            {"early_prefetch_issued", stats_.early_prefetch_issued},
            {"early_predicted_by_rank", std::vector<uint64_t>(std::begin(stats_.early_rank_predicted), std::end(stats_.early_rank_predicted))},
            {"early_predicted_correct_by_rank", std::vector<uint64_t>(std::begin(stats_.early_rank_correct), std::end(stats_.early_rank_correct))},
            {"late_predicted_by_rank", std::vector<uint64_t>(std::begin(stats_.late_rank_predicted), std::end(stats_.late_rank_predicted))},
            {"late_predicted_correct_by_rank", std::vector<uint64_t>(std::begin(stats_.late_rank_correct), std::end(stats_.late_rank_correct))},
            {"late_predict_ms", stats_.late_ms},
            {"decode_io_bytes", stats_.decode_io_bytes},
            {"decode_io_busy_ms", stats_.decode_io_busy_ms},
            {"decode_budget_bytes", decode_budget_},
            {"decode_demand_jobs", stats_.decode_demand_jobs},
            {"decode_demand_job_ms", stats_.decode_demand_job_ms},
            {"promoted_reads", stats_.promoted_reads},
            {"confirmed_reads", stats_.confirmed_reads},
            {"cancelled_prefetches", stats_.cancelled_prefetches},
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
