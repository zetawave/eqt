#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace eqt {
// Routed expert banks of the supported GGUF families, one tensor per layer and projection.
inline constexpr const char *routed_bank_pattern = "^blk\\.[0-9]+\\.ffn_(gate|up|down|gate_up)_exps\\.weight$";

// An expert pack holds a GGUF model's routed experts expert-major: within a layer, the banks of expert e (gate
// and up, or gate_up, then down) follow each other, each padded to a 4 KiB page, so one contiguous, page-aligned
// read fetches a whole expert instead of one small read per bank. The GGUF stays the source of every other
// tensor and of the bank shapes; the pack records the GGUF offsets it was built from and is rejected for any
// other file.
//
// File layout: "EQTPACK1", the JSON header's length (u64 LE), the JSON header, zero padding to header_bytes,
// then the layers in order, each at a page multiple.
inline constexpr uint64_t pack_page_bytes = 4096;
inline constexpr uint64_t pack_header_bytes = 1 << 20;
inline constexpr int pack_version = 1;

struct PackBank {
    std::string tensor;          // GGUF tensor name
    int type = 0;                // ggml_type
    int64_t ne[3] = {};          // GGUF shape, experts last
    uint64_t source_offset = 0;  // absolute file offset of expert 0 in the GGUF
    uint64_t stride = 0;         // bytes of one expert in the GGUF
    uint64_t offset = 0;         // offset of this bank inside a packed expert, a page multiple
};

struct PackLayer {
    int layer = 0;
    int experts = 0;
    uint64_t expert_bytes = 0; // one packed expert: every bank padded to a page
    uint64_t data_offset = 0;  // absolute pack offset of expert 0, a page multiple
    uint64_t checksum = 0;     // XXH3-64 of the layer's packed bytes, padding included
    std::vector<PackBank> banks;
};

struct PackHeader {
    uint64_t source_size = 0;
    std::string source_name;
    uint64_t file_bytes = 0;
    bool complete = false;
    // The experts were moved out of the GGUF (eqt-pack move): its routed ranges are holes, readable only here.
    bool source_moved = false;
    std::vector<PackLayer> layers;
};

// Orders a layer's banks (gate_up, gate, up, down: the order compute reads them) and assigns page-aligned
// offsets inside the packed expert and, across layers in ascending order, inside the file.
void plan_pack(PackHeader &header);
std::vector<uint8_t> encode_pack_header(const PackHeader &header);
// Reads and validates a pack header: magic, version, completeness and that every region lies inside the file.
// `whole_file` false accepts a pack truncated by an interrupted unpack (eqt-pack), whose last layers are gone.
PackHeader read_pack_header(const std::string &path, bool whole_file = true);
// Throws unless `pack` holds exactly the banks of `expected` (planned from the GGUF being loaded): the same source
// size, layers, tensors, types, shapes, GGUF offsets and packed placement.
void check_pack_matches(const PackHeader &pack, const PackHeader &expected);

// True when any of the file's [offset, offset + size) ranges contains deallocated blocks (a hole). eqt-pack move
// leaves the routed ranges of a GGUF as holes; such a file must never be read without its pack. False where the
// file system cannot tell.
bool has_holes(const std::string &path, const std::vector<std::pair<uint64_t, uint64_t>> &ranges);

// Streaming XXH3-64.
class PackHasher {
  public:
    PackHasher();
    ~PackHasher();
    PackHasher(const PackHasher &) = delete;
    PackHasher &operator=(const PackHasher &) = delete;
    void update(const void *data, size_t size);
    uint64_t digest() const;

  private:
    void *state_;
};
} // namespace eqt
