#include "resident_weights.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace eqt {
struct ResidentWeights::Block {
    ResidentWeights *owner = nullptr;
    std::unique_ptr<WeightMemory> memory;
    ~Block() {
        if (owner) {
            auto &blocks = owner->blocks_;
            blocks.erase(std::remove(blocks.begin(), blocks.end(), memory.get()), blocks.end());
        }
    }
};

ResidentWeights::ResidentWeights(WeightStorage storage) : storage_(storage) {
    auto *device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!device) {
        throw std::runtime_error("No CPU backend for resident weights");
    }
    type_.reset(new ggml_backend_buffer_type{
        {
            /* .get_name */ [](ggml_backend_buffer_type_t type) {
                const auto *owner = static_cast<const ResidentWeights *>(type->context);
                return owner->storage_ == WeightStorage::hardware_buffer ? "EQT_Weights_AHWB"
                                                                         : "EQT_Weights_Anon";
            },
            /* .alloc_buffer */ &ResidentWeights::allocate,
            /* .alloc_buffer_n */ nullptr,
            /* .get_alignment */
            [](ggml_backend_buffer_type_t) -> size_t {
                return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
            },
            // Keep buffer splitting identical in both modes and leave a page for readable SIMD tail slack.
            /* .get_max_size */
            [](ggml_backend_buffer_type_t) -> size_t { return max_buffer_bytes; },
            /* .get_alloc_size */ nullptr,
            /* .get_alloc_size_n */ nullptr,
            /* .is_host */ [](ggml_backend_buffer_type_t) { return true; },
        },
        device,
        this});
}

ResidentWeights::~ResidentWeights() = default;

ggml_backend_buffer_t ResidentWeights::allocate(ggml_backend_buffer_type_t type, size_t size) {
    auto &owner = *static_cast<ResidentWeights *>(type->context);
    try {
        if (size > max_buffer_bytes) {
            throw std::invalid_argument("Resident weight buffer exceeds the advertised allocation limit");
        }
        auto block = std::make_unique<Block>();
        block->owner = &owner;
        block->memory = std::make_unique<WeightMemory>(size + 64, owner.storage_);
        owner.blocks_.push_back(block->memory.get());
        const ggml_backend_buffer_i operations = {
            /* .free_buffer */ [](ggml_backend_buffer_t buffer) {
                delete static_cast<Block *>(buffer->context);
            },
            /* .get_base */
            [](ggml_backend_buffer_t buffer) -> void * {
                return static_cast<Block *>(buffer->context)->memory->data();
            },
            /* .init_tensor */ nullptr,
            /* .memset_tensor */
            [](ggml_backend_buffer_t, ggml_tensor *tensor, uint8_t value, size_t offset, size_t count) {
                std::memset(static_cast<char *>(tensor->data) + offset, value, count);
            },
            /* .set_tensor */
            [](ggml_backend_buffer_t, ggml_tensor *tensor, const void *data, size_t offset, size_t count) {
                std::memcpy(static_cast<char *>(tensor->data) + offset, data, count);
            },
            /* .get_tensor */
            [](ggml_backend_buffer_t, const ggml_tensor *tensor, void *data, size_t offset, size_t count) {
                std::memcpy(data, static_cast<const char *>(tensor->data) + offset, count);
            },
            /* .set_tensor_2d */ nullptr,
            /* .get_tensor_2d */ nullptr,
            /* .cpy_tensor */ nullptr,
            /* .clear */
            [](ggml_backend_buffer_t buffer, uint8_t value) {
                std::memset(static_cast<Block *>(buffer->context)->memory->data(), value, buffer->size);
            },
            /* .reset */ nullptr,
        };
        auto *buffer = ggml_backend_buffer_init(type, operations, block.get(), size);
        if (buffer) {
            block.release();
        }
        return buffer;
    } catch (const std::exception &error) {
        owner.error_ = error.what();
        return nullptr; // strict experimental mode: never silently compare anonymous memory with itself
    }
}

Json ResidentWeights::statistics(bool sample) const {
    uint64_t bytes = 0, sampled = 0, resident = 0;
    bool known = sample && !blocks_.empty();
    for (const auto *block : blocks_) {
        bytes += block->bytes();
        if (sample) {
            const auto status = sample_residency(block->data(), block->bytes());
            known &= status.available;
            sampled += status.sampled_pages;
            resident += status.resident_pages;
        }
    }
    return {{"storage", storage_ == WeightStorage::hardware_buffer ? "ahwb" : "anonymous"},
            {"allocated_bytes", bytes},
            {"buffers", blocks_.size()},
            {"strict_no_fallback", true},
            {"sampled_pages", sample ? Json(sampled) : Json(nullptr)},
            {"resident_pages", known ? Json(resident) : Json(nullptr)},
            {"resident_fraction", known && sampled ? Json(double(resident) / sampled) : Json(nullptr)}};
}
} // namespace eqt
