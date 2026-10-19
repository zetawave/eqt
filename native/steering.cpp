#include "steering.h"
#include "gguf.h"
#include "hash/hash.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace eqt {
namespace {
void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::invalid_argument(message);
}
std::string fingerprint(const Json &manifest, const char *key) {
    const std::string value = manifest.at(key).get<std::string>();
    require(value.size() == 64 &&
                std::all_of(value.begin(), value.end(),
                            [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }),
            std::string("Invalid steering ") + key);
    return value;
}
int integer(const Json &manifest, const char *key, int low, int high) {
    const auto &value = manifest.at(key);
    require(value.is_number_integer() && value >= low && value <= high,
            std::string("Invalid steering ") + key);
    return value.get<int>();
}
std::string text(const gguf_context *metadata, const char *key) {
    const auto index = gguf_find_key(metadata, key);
    require(index >= 0 && gguf_get_kv_type(metadata, index) == GGUF_TYPE_STRING,
            std::string("Missing steering GGUF key: ") + key);
    return gguf_get_val_str(metadata, index);
}
} // namespace

SteeringVector read_steering(const std::string &path, const Json &manifest) {
    static_assert(std::endian::native == std::endian::little);
    const auto model_hash = fingerprint(manifest, "model_sha256");
    const auto vector_hash = fingerprint(manifest, "vector_sha256");
    const int layer = integer(manifest, "layer_index", 1, 1023);
    const int embedding = integer(manifest, "embedding_size", 1, 65536);
    // One direction for its block, or one per block 1..layer_count (block 0 cannot hold a control vector).
    const int layer_count = manifest.contains("layer_count") ? integer(manifest, "layer_count", 1, 1023) : 1;
    require(layer_count == 1 || layer <= layer_count, "Steering layer is outside the vector's blocks");
    const std::string architecture = manifest.at("architecture").get<std::string>();
    require(!architecture.empty() && architecture.size() <= 128, "Invalid steering architecture");
    const std::string name = "direction." + std::to_string(layer);
    require(manifest.at("tensor_name") == name, "Steering tensor name does not match its layer");
    const auto size = std::filesystem::file_size(path);
    require(size >= 24 && size <= 8 * 1024 * 1024, "Steering GGUF must be at most 8 MiB");
    std::vector<char> bytes(static_cast<size_t>(size));
    std::ifstream input(path, std::ios::binary);
    require(bool(input.read(bytes.data(), bytes.size())), "Cannot read steering GGUF");
    require(hash_sha256_hex(bytes.data(), bytes.size()) == vector_hash, "Steering vector SHA-256 mismatch");
    uint64_t tensors = 0, keys = 0;
    std::memcpy(&tensors, bytes.data() + 8, sizeof(tensors));
    std::memcpy(&keys, bytes.data() + 16, sizeof(keys));
    require(tensors == static_cast<uint64_t>(layer_count) && keys <= 128,
            "Steering GGUF tensor count does not match the manifest");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_buffer(bytes.data(), bytes.size(), {true, nullptr}), gguf_free);
    require(metadata != nullptr, "Cannot parse steering GGUF");
    require(text(metadata.get(), "general.architecture") == "controlvector", "Not a control-vector GGUF");
    require(text(metadata.get(), "controlvector.model_hint") == architecture,
            "Steering architecture mismatch");
    std::vector<int> blocks;
    std::vector<float> values(static_cast<size_t>(layer_count) * embedding);
    const size_t payload = static_cast<size_t>(embedding) * sizeof(float);
    double squared_norm = 0;
    for (int index = 0; index < layer_count; ++index) {
        const int block = layer_count == 1 ? layer : index + 1;
        require("direction." + std::to_string(block) == gguf_get_tensor_name(metadata.get(), index) &&
                    gguf_get_tensor_type(metadata.get(), index) == GGML_TYPE_F32,
                "Invalid steering tensor name or type");
        const auto *shape = gguf_get_tensor_ne(metadata.get(), index);
        require(shape[0] == embedding && shape[1] == 1 && shape[2] == 1 && shape[3] == 1,
                "Steering dimensions do not match the manifest");
        const size_t offset =
            gguf_get_data_offset(metadata.get()) + gguf_get_tensor_offset(metadata.get(), index);
        require(offset <= bytes.size() && payload <= bytes.size() - offset, "Truncated steering tensor");
        float *row = values.data() + static_cast<size_t>(index) * embedding;
        std::memcpy(row, bytes.data() + offset, payload);
        double squared = 0;
        for (int i = 0; i < embedding; ++i) {
            require(std::isfinite(row[i]), "Steering vector contains non-finite values");
            squared += static_cast<double>(row[i]) * row[i];
        }
        require(squared > 1e-24, "Steering vector is zero");
        if (block == layer) {
            squared_norm = squared;
        }
        blocks.push_back(block);
    }
    Json info = {{"model_sha256", model_hash},
                 {"vector_sha256", vector_hash},
                 {"architecture", architecture},
                 {"layer_index", layer},
                 {"embedding_size", embedding},
                 {"tensor_name", name},
                 {"layer_count", layer_count},
                 {"direction", manifest.value("direction", "positive_minus_negative")},
                 {"normalized", manifest.at("normalized").get<bool>()},
                 {"l2_norm", std::sqrt(squared_norm)},
                 {"file_bytes", size}};
    return {std::move(info), layer, embedding, std::move(blocks), std::move(values)};
}

const float *SteeringVector::row(int block) const {
    const auto found = std::lower_bound(blocks.begin(), blocks.end(), block);
    if (found == blocks.end() || *found != block) {
        return nullptr;
    }
    return values.data() + static_cast<size_t>(found - blocks.begin()) * embedding;
}
} // namespace eqt
