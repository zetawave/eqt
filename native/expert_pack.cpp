#include "expert_pack.h"

#include "hash/xxhash/xxhash.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace eqt {
namespace {
using Json = nlohmann::ordered_json;
constexpr char pack_magic[8] = {'E', 'Q', 'T', 'P', 'A', 'C', 'K', '1'};

uint64_t round_up(uint64_t value, uint64_t unit) {
    return (value + unit - 1) / unit * unit;
}

int bank_rank(const std::string &tensor) {
    static constexpr const char *order[] = {".ffn_gate_up_exps.weight", ".ffn_gate_exps.weight", ".ffn_up_exps.weight",
                                            ".ffn_down_exps.weight"};
    for (int i = 0; i < 4; ++i) {
        const size_t length = std::strlen(order[i]);
        if (tensor.size() > length && tensor.compare(tensor.size() - length, length, order[i]) == 0) {
            return i;
        }
    }
    return 4;
}

std::string hex(uint64_t value) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}
} // namespace

void plan_pack(PackHeader &header) {
    std::sort(header.layers.begin(), header.layers.end(), [](const PackLayer &a, const PackLayer &b) { return a.layer < b.layer; });
    uint64_t at = pack_header_bytes;
    for (auto &layer : header.layers) {
        std::stable_sort(layer.banks.begin(), layer.banks.end(),
                         [](const PackBank &a, const PackBank &b) { return bank_rank(a.tensor) < bank_rank(b.tensor); });
        layer.expert_bytes = 0;
        for (auto &bank : layer.banks) {
            bank.offset = layer.expert_bytes;
            layer.expert_bytes += round_up(bank.stride, pack_page_bytes);
        }
        layer.data_offset = at;
        at += layer.expert_bytes * static_cast<uint64_t>(layer.experts);
    }
    header.file_bytes = at;
}

std::vector<uint8_t> encode_pack_header(const PackHeader &header) {
    Json layers = Json::array();
    for (const auto &layer : header.layers) {
        Json banks = Json::array();
        for (const auto &bank : layer.banks) {
            banks.push_back({{"tensor", bank.tensor},
                             {"type", bank.type},
                             {"ne", {bank.ne[0], bank.ne[1], bank.ne[2]}},
                             {"source_offset", bank.source_offset},
                             {"stride", bank.stride},
                             {"offset", bank.offset}});
        }
        layers.push_back({{"layer", layer.layer},
                          {"experts", layer.experts},
                          {"expert_bytes", layer.expert_bytes},
                          {"data_offset", layer.data_offset},
                          {"checksum_xxh3", hex(layer.checksum)},
                          {"banks", banks}});
    }
    const Json json = {{"format", "equity-expert-pack"},
                       {"version", pack_version},
                       {"complete", header.complete},
                       {"source_moved", header.source_moved},
                       {"page_bytes", pack_page_bytes},
                       {"header_bytes", pack_header_bytes},
                       {"file_bytes", header.file_bytes},
                       {"source", {{"name", header.source_name}, {"size", header.source_size}}},
                       {"layers", layers}};
    const std::string text = json.dump();
    if (16 + text.size() > pack_header_bytes) {
        throw std::runtime_error("Expert pack header exceeds its reserved size");
    }
    std::vector<uint8_t> bytes(pack_header_bytes, 0);
    std::memcpy(bytes.data(), pack_magic, sizeof(pack_magic));
    const uint64_t length = text.size();
    for (int i = 0; i < 8; ++i) {
        bytes[8 + i] = static_cast<uint8_t>(length >> (8 * i));
    }
    std::memcpy(bytes.data() + 16, text.data(), text.size());
    return bytes;
}

PackHeader read_pack_header(const std::string &path, bool whole_file) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open expert pack: " + path);
    }
    file.seekg(0, std::ios::end);
    const uint64_t size = static_cast<uint64_t>(file.tellg());
    file.seekg(0);
    uint8_t prefix[16];
    if (size < pack_header_bytes || !file.read(reinterpret_cast<char *>(prefix), sizeof(prefix)) ||
        std::memcmp(prefix, pack_magic, sizeof(pack_magic)) != 0) {
        throw std::runtime_error("Not an expert pack: " + path);
    }
    uint64_t length = 0;
    for (int i = 0; i < 8; ++i) {
        length |= static_cast<uint64_t>(prefix[8 + i]) << (8 * i);
    }
    if (length == 0 || 16 + length > pack_header_bytes) {
        throw std::runtime_error("Expert pack header has an invalid length");
    }
    std::string text(length, '\0');
    if (!file.read(text.data(), static_cast<std::streamsize>(length))) {
        throw std::runtime_error("Truncated expert pack header");
    }
    const Json json = Json::parse(text);
    if (json.at("format") != "equity-expert-pack" || json.at("version") != pack_version ||
        json.at("page_bytes") != pack_page_bytes || json.at("header_bytes") != pack_header_bytes) {
        throw std::runtime_error("Unsupported expert pack format");
    }
    PackHeader header;
    header.complete = json.at("complete").get<bool>();
    if (!header.complete) {
        throw std::runtime_error("Expert pack is incomplete (its conversion did not finish)");
    }
    header.file_bytes = json.at("file_bytes").get<uint64_t>();
    header.source_name = json.at("source").at("name").get<std::string>();
    header.source_size = json.at("source").at("size").get<uint64_t>();
    header.source_moved = json.value("source_moved", false);
    if (whole_file ? header.file_bytes != size : header.file_bytes < size) {
        throw std::runtime_error("Expert pack size differs from its header");
    }
    const uint64_t limit = header.file_bytes; // equals the file size for a whole pack
    uint64_t previous_end = pack_header_bytes;
    for (const auto &entry : json.at("layers")) {
        PackLayer layer;
        layer.layer = entry.at("layer").get<int>();
        layer.experts = entry.at("experts").get<int>();
        layer.expert_bytes = entry.at("expert_bytes").get<uint64_t>();
        layer.data_offset = entry.at("data_offset").get<uint64_t>();
        layer.checksum = std::stoull(entry.at("checksum_xxh3").get<std::string>(), nullptr, 16);
        if (layer.experts < 1 || layer.experts > 65536 || layer.expert_bytes == 0 ||
            layer.expert_bytes % pack_page_bytes || layer.data_offset % pack_page_bytes ||
            layer.data_offset < previous_end || layer.data_offset > limit ||
            layer.expert_bytes > (limit - layer.data_offset) / static_cast<uint64_t>(layer.experts)) {
            throw std::runtime_error("Expert pack layer lies outside the file");
        }
        previous_end = layer.data_offset + layer.expert_bytes * static_cast<uint64_t>(layer.experts);
        uint64_t end = 0;
        for (const auto &item : entry.at("banks")) {
            PackBank bank;
            bank.tensor = item.at("tensor").get<std::string>();
            bank.type = item.at("type").get<int>();
            for (int i = 0; i < 3; ++i) {
                bank.ne[i] = item.at("ne").at(i).get<int64_t>();
            }
            bank.source_offset = item.at("source_offset").get<uint64_t>();
            bank.stride = item.at("stride").get<uint64_t>();
            bank.offset = item.at("offset").get<uint64_t>();
            if (bank.offset != end || bank.stride == 0 || bank.ne[2] != layer.experts) {
                throw std::runtime_error("Expert pack bank layout is inconsistent: " + bank.tensor);
            }
            end += round_up(bank.stride, pack_page_bytes);
            layer.banks.push_back(std::move(bank));
        }
        if (end != layer.expert_bytes || layer.banks.empty()) {
            throw std::runtime_error("Expert pack expert size is inconsistent");
        }
        header.layers.push_back(std::move(layer));
    }
    return header;
}

void check_pack_matches(const PackHeader &pack, const PackHeader &expected) {
    if (pack.source_size != expected.source_size) {
        throw std::runtime_error("Expert pack was built from another file (size " + std::to_string(pack.source_size) +
                                 ", model " + std::to_string(expected.source_size) + ")");
    }
    if (pack.layers.size() != expected.layers.size()) {
        throw std::runtime_error("Expert pack layer count differs from the model");
    }
    for (size_t i = 0; i < pack.layers.size(); ++i) {
        const auto &a = pack.layers[i], &b = expected.layers[i];
        if (a.layer != b.layer || a.experts != b.experts || a.expert_bytes != b.expert_bytes ||
            a.data_offset != b.data_offset || a.banks.size() != b.banks.size()) {
            throw std::runtime_error("Expert pack layer " + std::to_string(b.layer) + " differs from the model");
        }
        for (size_t j = 0; j < a.banks.size(); ++j) {
            const auto &x = a.banks[j], &y = b.banks[j];
            if (x.tensor != y.tensor || x.type != y.type || x.ne[0] != y.ne[0] || x.ne[1] != y.ne[1] ||
                x.ne[2] != y.ne[2] || x.source_offset != y.source_offset || x.stride != y.stride || x.offset != y.offset) {
                throw std::runtime_error("Expert pack bank differs from the model: " + y.tensor);
            }
        }
    }
}

bool has_holes(const std::string &path, const std::vector<std::pair<uint64_t, uint64_t>> &ranges) {
#ifdef _WIN32
    (void)path, (void)ranges;
    return false;
#else
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    bool found = false;
    for (const auto &[offset, size] : ranges) {
        const off_t hole = lseek(fd, static_cast<off_t>(offset), SEEK_HOLE);
        if (hole >= 0 && static_cast<uint64_t>(hole) < offset + size) {
            found = true;
            break;
        }
    }
    close(fd);
    return found;
#endif
}

PackHasher::PackHasher() : state_(XXH3_createState()) {
    if (!state_ || XXH3_64bits_reset(static_cast<XXH3_state_t *>(state_)) != XXH_OK) {
        throw std::runtime_error("Cannot initialize XXH3");
    }
}

PackHasher::~PackHasher() {
    XXH3_freeState(static_cast<XXH3_state_t *>(state_));
}

void PackHasher::update(const void *data, size_t size) {
    XXH3_64bits_update(static_cast<XXH3_state_t *>(state_), data, size);
}

uint64_t PackHasher::digest() const {
    return XXH3_64bits_digest(static_cast<const XXH3_state_t *>(state_));
}
} // namespace eqt
