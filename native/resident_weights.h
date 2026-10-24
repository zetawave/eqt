#pragma once

#include "engine.h"
#include "ggml-backend.h"
#include "weight_memory.h"
#include <memory>
#include <string>
#include <vector>

namespace eqt {
// Owns the buffer type; the model owns its buffers and must be destroyed before this object.
class ResidentWeights {
  public:
    explicit ResidentWeights(WeightStorage storage);
    ~ResidentWeights();
    ResidentWeights(const ResidentWeights &) = delete;
    ResidentWeights &operator=(const ResidentWeights &) = delete;

    ggml_backend_buffer_type_t buffer_type() const {
        return type_.get();
    }
    Json statistics(bool sample) const;
    const std::string &error() const {
        return error_;
    }

  private:
    // Conservative chunking: the Galaxy rejects 512 MiB blobs; its 384 MiB full-buffer probe passes.
    // Mapping validation remains authoritative on other devices. A larger single tensor is rejected.
    static constexpr size_t max_buffer_bytes = (384u << 20) - 4096;
    struct Block;
    static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type, size_t size);
    WeightStorage storage_;
    std::unique_ptr<ggml_backend_buffer_type> type_;
    std::vector<WeightMemory *> blocks_;
    std::string error_;
};
} // namespace eqt
