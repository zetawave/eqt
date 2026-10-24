#pragma once

#include <cstddef>
#include <cstdint>

namespace eqt {
enum class WeightStorage { anonymous, hardware_buffer };

struct ResidencySample {
    uint64_t sampled_pages = 0;
    uint64_t resident_pages = 0;
    bool available = false;
};

// Allocations are created/freed only by the serialized load/unload path. A locked hardware buffer
// stays CPU-accessible for its entire lifetime; it is never copied into a second resident allocation.
class WeightMemory {
  public:
    WeightMemory(size_t bytes, WeightStorage storage);
    ~WeightMemory();
    WeightMemory(const WeightMemory &) = delete;
    WeightMemory &operator=(const WeightMemory &) = delete;

    void *data() const {
        return data_;
    }
    size_t bytes() const {
        return bytes_;
    }
    WeightStorage storage() const {
        return storage_;
    }
    static constexpr size_t max_hardware_bytes = 0x7ffff000u;

  private:
    WeightStorage storage_;
    size_t bytes_ = 0;
    void *data_ = nullptr;
    void *handle_ = nullptr;
};

// Samples complete pages evenly across a range without faulting them in. An unsupported mapping,
// denied syscall, or unreadable page status is unknown, never an assumed fully resident result.
ResidencySample sample_residency(const void *data, size_t bytes, size_t max_pages = 256);
} // namespace eqt
