#include "weight_memory.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace eqt {
namespace {
#ifdef __ANDROID__
bool writable_cpu_mapping(void *data, size_t bytes) {
    const auto begin = reinterpret_cast<uintptr_t>(data);
    if (bytes > UINTPTR_MAX - begin) {
        return false;
    }
    uintptr_t cursor = begin;
    const uintptr_t target = begin + bytes;
    std::FILE *maps = std::fopen("/proc/self/maps", "r");
    if (!maps) {
        return false;
    }
    char line[1024];
    while (cursor < target && std::fgets(line, sizeof(line), maps)) {
        unsigned long long start = 0, end = 0;
        char access[5]{};
        if (std::sscanf(line, "%llx-%llx %4s", &start, &end, access) != 3 || end <= cursor) {
            continue;
        }
        if (start > cursor || access[0] != 'r' || access[1] != 'w') {
            break;
        }
        cursor = std::min<uintptr_t>(end, target);
    }
    std::fclose(maps);
    return cursor == target;
}
#endif
} // namespace

WeightMemory::WeightMemory(size_t bytes, WeightStorage storage) : storage_(storage) {
    if (!bytes || bytes > std::numeric_limits<size_t>::max() - 4095) {
        throw std::invalid_argument("Invalid weight allocation size");
    }
    bytes_ = (bytes + 4095) / 4096 * 4096;
    if (storage == WeightStorage::hardware_buffer) {
#ifdef __ANDROID__
        if (bytes_ > max_hardware_bytes) {
            throw std::invalid_argument("Hardware weight allocation exceeds the per-buffer limit");
        }
        AHardwareBuffer_Desc description{};
        description.width = static_cast<uint32_t>(bytes_);
        description.height = 1;
        description.layers = 1;
        description.format = AHARDWAREBUFFER_FORMAT_BLOB;
        description.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
        AHardwareBuffer *buffer = nullptr;
        const int allocated = AHardwareBuffer_allocate(&description, &buffer);
        if (allocated != 0 || !buffer) {
            throw std::runtime_error("Hardware weight allocation failed: " + std::to_string(allocated));
        }
        const int locked = AHardwareBuffer_lock(buffer, description.usage, -1, nullptr, &data_);
        if (locked != 0 || !data_) {
            AHardwareBuffer_release(buffer);
            throw std::runtime_error("Hardware weight lock failed: " + std::to_string(locked));
        }
        handle_ = buffer;
        if (reinterpret_cast<uintptr_t>(data_) % 64 != 0 || !writable_cpu_mapping(data_, bytes_)) {
            AHardwareBuffer_unlock(buffer, nullptr);
            AHardwareBuffer_release(buffer);
            data_ = handle_ = nullptr;
            throw std::runtime_error("Hardware weight buffer lacks a complete aligned writable CPU mapping");
        }
#else
        throw std::invalid_argument("Hardware weight buffers require Android");
#endif
    } else {
#ifdef _WIN32
        data_ = VirtualAlloc(nullptr, bytes_, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        data_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
        }
#endif
        if (!data_) {
            throw std::runtime_error("Anonymous weight allocation failed");
        }
    }
}

WeightMemory::~WeightMemory() {
#ifdef __ANDROID__
    if (handle_) {
        auto *buffer = static_cast<AHardwareBuffer *>(handle_);
        AHardwareBuffer_unlock(buffer, nullptr);
        AHardwareBuffer_release(buffer);
        return;
    }
#endif
    if (data_) {
#ifdef _WIN32
        VirtualFree(data_, 0, MEM_RELEASE);
#else
        munmap(data_, bytes_);
#endif
    }
}

ResidencySample sample_residency(const void *data, size_t bytes, size_t max_pages) {
    ResidencySample result;
#ifndef _WIN32
    const long queried = sysconf(_SC_PAGESIZE);
    if (!data || queried <= 0 || !max_pages) {
        return result;
    }
    const auto page = static_cast<uintptr_t>(queried);
    const auto address = reinterpret_cast<uintptr_t>(data);
    if (bytes > std::numeric_limits<uintptr_t>::max() - address || address > UINTPTR_MAX - page + 1) {
        return result;
    }
    const uintptr_t begin = (address + page - 1) / page * page;
    const uintptr_t end = (address + bytes) / page * page;
    if (end <= begin) {
        return result;
    }
    const size_t count = (end - begin) / page;
    const size_t samples = std::min(max_pages, count);
    for (size_t i = 0; i < samples; ++i) {
        const size_t index = samples == 1 ? 0 : i * (count - 1) / (samples - 1);
        unsigned char flags = 0;
        if (mincore(reinterpret_cast<void *>(begin + index * page), page, &flags) != 0) {
            return {}; // partial samples cannot establish residency of the whole range
        }
        ++result.sampled_pages;
        result.resident_pages += flags & 1;
    }
    result.available = true;
#else
    (void)data;
    (void)bytes;
    (void)max_pages;
#endif
    return result;
}
} // namespace eqt
