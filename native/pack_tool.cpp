// eqt-pack: expert packs (native/expert_pack.h) for GGUF models.
//
//   eqt-pack build  MODEL PACK  copies the experts into a new pack; MODEL is untouched (needs the pack's size free).
//   eqt-pack move   MODEL PACK  moves them layer by layer: a layer is written, re-read from storage and checked
//                               against both the packed and the source bytes, recorded in PACK.move, and only then are
//                               its ranges in MODEL deallocated (hole punch, whole blocks inside each range). Needs
//                               about one layer of free space; an interrupted move resumes. Linux and Android only.
//   eqt-pack unpack MODEL PACK  writes the experts back into MODEL, last layer first, truncating the pack behind it,
//                               and removes the pack: MODEL is then byte-identical to the file before the move.
//                               Resumable, also after an interrupted move.
//   eqt-pack verify MODEL PACK  checks the pack against MODEL's layout and every layer's checksum.
//   eqt-pack sha256 MODEL PACK  SHA-256 of MODEL as it was before the move, with its experts read from the pack.
#include "expert_pack.h"

#include "ggml.h"
#include "gguf.h"

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <linux/falloc.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
using Json = nlohmann::ordered_json;
using eqt::pack_page_bytes;
constexpr int chunk_experts = 32;
constexpr uint64_t free_margin = 256ull << 20;

// Test hook: EQT_PACK_STOP_AFTER=n stops move or unpack after its n-th layer is recorded and before that layer's
// space is released, the point where an interruption is hardest to recover from. The process exits with 3.
struct Interrupted {};
void maybe_stop(int layers_done) {
    const char *value = std::getenv("EQT_PACK_STOP_AFTER");
    if (value && std::atoi(value) == layers_done) {
        throw Interrupted{};
    }
}

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
uint64_t round_up(uint64_t value, uint64_t unit) {
    return (value + unit - 1) / unit * unit;
}
uint64_t round_down(uint64_t value, uint64_t unit) {
    return value / unit * unit;
}
std::string hex(uint64_t value) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

// Page-aligned, so the same buffer serves direct reads.
class Buffer {
  public:
    explicit Buffer(size_t size) : size_(size) {
#ifdef _WIN32
        data_ = static_cast<uint8_t *>(_aligned_malloc(size, pack_page_bytes));
#else
        void *data = nullptr;
        data_ = posix_memalign(&data, pack_page_bytes, size) == 0 ? static_cast<uint8_t *>(data) : nullptr;
#endif
        if (!data_) {
            throw std::bad_alloc();
        }
    }
    ~Buffer() {
#ifdef _WIN32
        _aligned_free(data_);
#else
        std::free(data_);
#endif
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    uint8_t *data() const { return data_; }
    size_t size() const { return size_; }

  private:
    uint8_t *data_;
    size_t size_;
};

class File {
  public:
    enum class Mode { read, read_direct, create, update };
    File(const std::string &path, Mode mode) {
#ifdef _WIN32
        const int length = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring wide(length > 0 ? length : 1, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), length);
        const bool writable = mode == Mode::create || mode == Mode::update;
        handle_ = CreateFileW(wide.c_str(), writable ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              nullptr, mode == Mode::create ? CREATE_NEW : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Cannot open " + path);
        }
#else
        const int flags = mode == Mode::create ? O_RDWR | O_CREAT | O_EXCL : mode == Mode::update ? O_RDWR : O_RDONLY;
        fd_ = open(path.c_str(), flags | O_CLOEXEC | (mode == Mode::read_direct ? O_DIRECT : 0), 0600);
        direct_ = mode == Mode::read_direct;
        if (fd_ < 0 && direct_ && errno == EINVAL) {
            fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC); // a file system without O_DIRECT
            direct_ = false;
        }
        if (fd_ < 0) {
            throw std::runtime_error("Cannot open " + path + ": " + std::strerror(errno));
        }
#endif
    }
    ~File() {
#ifdef _WIN32
        CloseHandle(handle_);
#else
        close(fd_);
#endif
    }
    File(const File &) = delete;
    File &operator=(const File &) = delete;
    bool direct() const { return direct_; }

    uint64_t size() const {
#ifdef _WIN32
        LARGE_INTEGER size{};
        GetFileSizeEx(handle_, &size);
        return static_cast<uint64_t>(size.QuadPart);
#else
        struct stat info{};
        if (fstat(fd_, &info) != 0) {
            throw std::runtime_error("Cannot stat file");
        }
        return static_cast<uint64_t>(info.st_size);
#endif
    }
    void read(uint8_t *data, size_t size, uint64_t offset) const {
        for (size_t done = 0; done < size;) {
#ifdef _WIN32
            OVERLAPPED at{};
            at.Offset = static_cast<DWORD>(offset + done);
            at.OffsetHigh = static_cast<DWORD>((offset + done) >> 32);
            DWORD got = 0;
            if (!ReadFile(handle_, data + done, static_cast<DWORD>(std::min<size_t>(size - done, 1u << 30)), &got, &at)) {
                throw std::runtime_error("Read failed");
            }
#else
            const ssize_t got = pread(fd_, data + done, size - done, static_cast<off_t>(offset + done));
            if (got < 0 && errno == EINTR) {
                continue;
            }
            if (got < 0) {
                throw std::runtime_error(std::string("Read failed: ") + std::strerror(errno));
            }
#endif
            if (got == 0) {
                throw std::runtime_error("Unexpected end of file");
            }
            done += static_cast<size_t>(got);
        }
    }
    void write(const uint8_t *data, size_t size, uint64_t offset) const {
        for (size_t done = 0; done < size;) {
#ifdef _WIN32
            OVERLAPPED at{};
            at.Offset = static_cast<DWORD>(offset + done);
            at.OffsetHigh = static_cast<DWORD>((offset + done) >> 32);
            DWORD put = 0;
            if (!WriteFile(handle_, data + done, static_cast<DWORD>(std::min<size_t>(size - done, 1u << 30)), &put, &at) || !put) {
                throw std::runtime_error("Write failed");
            }
#else
            const ssize_t put = pwrite(fd_, data + done, size - done, static_cast<off_t>(offset + done));
            if (put < 0 && errno == EINTR) {
                continue;
            }
            if (put <= 0) {
                throw std::runtime_error(std::string("Write failed: ") + std::strerror(errno));
            }
#endif
            done += static_cast<size_t>(put);
        }
    }
    void sync() const {
#ifdef _WIN32
        if (!FlushFileBuffers(handle_)) {
            throw std::runtime_error("Flush failed");
        }
#else
        if (fdatasync(fd_) != 0) {
            throw std::runtime_error(std::string("fdatasync failed: ") + std::strerror(errno));
        }
#endif
    }
    void truncate(uint64_t size) const {
#ifdef _WIN32
        LARGE_INTEGER end{};
        end.QuadPart = static_cast<LONGLONG>(size);
        if (!SetFilePointerEx(handle_, end, nullptr, FILE_BEGIN) || !SetEndOfFile(handle_)) {
            throw std::runtime_error("Cannot truncate");
        }
#else
        if (ftruncate(fd_, static_cast<off_t>(size)) != 0) {
            throw std::runtime_error(std::string("Cannot truncate: ") + std::strerror(errno));
        }
#endif
    }
    // Drops clean cached pages of a range already read or synced: the conversion streams tens of GB once.
    void forget(uint64_t offset, uint64_t size) const {
#ifndef _WIN32
        posix_fadvise(fd_, static_cast<off_t>(offset), static_cast<off_t>(size), POSIX_FADV_DONTNEED);
#else
        (void)offset, (void)size;
#endif
    }
    // Deallocates the whole blocks inside [offset, offset + size); the partial blocks at its ends keep their bytes,
    // so no byte outside the range ever changes.
    void punch(uint64_t offset, uint64_t size) const {
#ifdef _WIN32
        (void)offset, (void)size;
        throw std::runtime_error("move is supported on Linux and Android only");
#else
        const uint64_t begin = round_up(offset, pack_page_bytes), end = round_down(offset + size, pack_page_bytes);
        if (begin < end && fallocate(fd_, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, static_cast<off_t>(begin),
                                     static_cast<off_t>(end - begin)) != 0) {
            throw std::runtime_error(std::string("Cannot deallocate model blocks (hole punch): ") + std::strerror(errno));
        }
#endif
    }
    // True when [offset, offset + size) contains a deallocated range.
    bool has_hole(uint64_t offset, uint64_t size) const {
#ifdef _WIN32
        (void)offset, (void)size;
        return false;
#else
        const off_t hole = lseek(fd_, static_cast<off_t>(offset), SEEK_HOLE);
        return hole >= 0 && static_cast<uint64_t>(hole) < offset + size;
#endif
    }

  private:
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
    bool direct_ = false;
};

std::filesystem::path directory_of(const std::string &path) {
    const std::filesystem::path file(path);
    return file.has_parent_path() ? file.parent_path() : std::filesystem::path(".");
}

void require_free(const std::string &path, uint64_t bytes) {
    const uint64_t available = std::filesystem::space(directory_of(path)).available;
    if (available < bytes + free_margin) {
        throw std::runtime_error("Not enough free space: " + std::to_string(bytes) + " bytes plus 256 MiB needed, " +
                                 std::to_string(available) + " free");
    }
}

void sync_directory(const std::string &path) {
#ifndef _WIN32
    const int dir = open(directory_of(path).c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir >= 0) {
        fsync(dir);
        close(dir);
    }
#else
    (void)path;
#endif
}

// Replaces a small state file atomically: a crash leaves either the previous or the new state.
void write_state(const std::string &path, const Json &state) {
    const std::string temporary = path + ".tmp";
    {
        File file(temporary, File::Mode::create);
        const std::string text = state.dump();
        file.write(reinterpret_cast<const uint8_t *>(text.data()), text.size(), 0);
        file.sync();
    }
    std::filesystem::rename(temporary, path);
    sync_directory(path);
}

Json read_state(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    return Json::parse(file);
}

// The pack layout the engine expects for this GGUF: its routed banks, planned.
eqt::PackHeader layout_of(const std::string &model) {
    ggml_context *shapes = nullptr;
    gguf_context *meta = gguf_init_from_file(model.c_str(), {/*no_alloc =*/true, /*ctx =*/&shapes});
    if (!meta || !shapes) {
        throw std::runtime_error("Cannot parse GGUF metadata: " + model);
    }
    eqt::PackHeader header;
    header.source_size = File(model, File::Mode::read).size();
    header.source_name = std::filesystem::path(model).filename().string();
    const std::regex routed(eqt::routed_bank_pattern);
    for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
        const char *name = gguf_get_tensor_name(meta, i);
        if (!std::regex_search(name, routed)) {
            continue;
        }
        const ggml_tensor *tensor = ggml_get_tensor(shapes, name);
        if (!tensor || tensor->ne[2] < 1 || tensor->ne[3] != 1 || tensor->nb[2] * tensor->ne[2] != ggml_nbytes(tensor)) {
            throw std::runtime_error(std::string("Unsupported routed bank layout: ") + name);
        }
        eqt::PackBank bank;
        bank.tensor = name;
        bank.type = static_cast<int>(tensor->type);
        for (int d = 0; d < 3; ++d) {
            bank.ne[d] = tensor->ne[d];
        }
        bank.stride = tensor->nb[2];
        bank.source_offset = gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, i);
        if (bank.source_offset + ggml_nbytes(tensor) > header.source_size) {
            throw std::runtime_error(std::string("Routed bank exceeds the file: ") + name);
        }
        const int index = static_cast<int>(std::strtol(name + 4, nullptr, 10));
        auto layer = std::find_if(header.layers.begin(), header.layers.end(), [&](const auto &l) { return l.layer == index; });
        if (layer == header.layers.end()) {
            header.layers.push_back({});
            layer = header.layers.end() - 1;
            layer->layer = index;
            layer->experts = static_cast<int>(tensor->ne[2]);
        }
        if (layer->experts != tensor->ne[2]) {
            throw std::runtime_error(std::string("Inconsistent expert count in ") + name);
        }
        layer->banks.push_back(std::move(bank));
    }
    gguf_free(meta);
    ggml_free(shapes);
    if (header.layers.empty()) {
        throw std::runtime_error("The model has no routed expert banks");
    }
    eqt::plan_pack(header);
    return header;
}

// Identifies a planned layout independently of checksums and flags.
std::string layout_id(eqt::PackHeader header) {
    header.complete = header.source_moved = false;
    for (auto &layer : header.layers) {
        layer.checksum = 0;
    }
    const auto bytes = eqt::encode_pack_header(header);
    eqt::PackHasher hasher;
    hasher.update(bytes.data(), bytes.size());
    return hex(hasher.digest());
}

uint64_t layer_bytes(const eqt::PackLayer &layer) {
    return layer.expert_bytes * static_cast<uint64_t>(layer.experts);
}

uint64_t source_bytes(const eqt::PackLayer &layer) {
    uint64_t total = 0;
    for (const auto &bank : layer.banks) {
        total += bank.stride * static_cast<uint64_t>(layer.experts);
    }
    return total;
}

struct Buffers {
    explicit Buffers(const eqt::PackHeader &header) : packed(widest(header, true)), source(widest(header, false)) {}
    static size_t widest(const eqt::PackHeader &header, bool packed) {
        uint64_t bytes = 0;
        for (const auto &layer : header.layers) {
            bytes = std::max(bytes, packed ? layer.expert_bytes : 0);
            for (const auto &bank : layer.banks) {
                bytes = std::max(bytes, packed ? 0 : bank.stride);
            }
        }
        return static_cast<size_t>(bytes) * chunk_experts;
    }
    Buffer packed, source;
};

struct LayerHashes {
    uint64_t packed = 0; // padded packed bytes, the header's checksum
    uint64_t source = 0; // the GGUF bytes, chunk by chunk and bank by bank
};

LayerHashes write_layer(const File &model, const File &pack, const eqt::PackLayer &layer, Buffers &buffers) {
    eqt::PackHasher packed_hash, source_hash;
    for (int first = 0; first < layer.experts; first += chunk_experts) {
        const int count = std::min(chunk_experts, layer.experts - first);
        const size_t bytes = static_cast<size_t>(layer.expert_bytes) * count;
        std::memset(buffers.packed.data(), 0, bytes);
        for (const auto &bank : layer.banks) {
            const uint64_t from = bank.source_offset + bank.stride * static_cast<uint64_t>(first);
            const size_t size = static_cast<size_t>(bank.stride) * count;
            model.read(buffers.source.data(), size, from);
            model.forget(from, size);
            source_hash.update(buffers.source.data(), size);
            for (int e = 0; e < count; ++e) {
                std::memcpy(buffers.packed.data() + layer.expert_bytes * e + bank.offset, buffers.source.data() + bank.stride * e,
                            static_cast<size_t>(bank.stride));
            }
        }
        packed_hash.update(buffers.packed.data(), bytes);
        pack.write(buffers.packed.data(), bytes, layer.data_offset + layer.expert_bytes * static_cast<uint64_t>(first));
    }
    return {packed_hash.digest(), source_hash.digest()};
}

// Re-reads a layer from storage (direct I/O where the file system has it); `each_chunk` sees every chunk.
template <class Visit>
LayerHashes read_layer(const File &pack, const eqt::PackLayer &layer, Buffers &buffers, Visit each_chunk) {
    eqt::PackHasher packed_hash, source_hash;
    for (int first = 0; first < layer.experts; first += chunk_experts) {
        const int count = std::min(chunk_experts, layer.experts - first);
        const size_t bytes = static_cast<size_t>(layer.expert_bytes) * count;
        const uint64_t at = layer.data_offset + layer.expert_bytes * static_cast<uint64_t>(first);
        pack.read(buffers.packed.data(), bytes, at);
        pack.forget(at, bytes);
        packed_hash.update(buffers.packed.data(), bytes);
        for (const auto &bank : layer.banks) {
            for (int e = 0; e < count; ++e) {
                source_hash.update(buffers.packed.data() + layer.expert_bytes * e + bank.offset, static_cast<size_t>(bank.stride));
            }
        }
        each_chunk(first, count);
    }
    return {packed_hash.digest(), source_hash.digest()};
}

LayerHashes check_layer(const File &pack, const eqt::PackLayer &layer, Buffers &buffers) {
    return read_layer(pack, layer, buffers, [](int, int) {});
}

Json build(const std::string &model, const std::string &pack) {
    const auto start = Clock::now();
    eqt::PackHeader header = layout_of(model);
    const std::string partial = pack + ".partial";
    if (std::filesystem::exists(pack) || std::filesystem::exists(pack + ".move")) {
        throw std::runtime_error("A pack or an interrupted move already exists: " + pack);
    }
    std::filesystem::remove(partial);
    require_free(pack, header.file_bytes);
    Buffers buffers(header);
    File input(model, File::Mode::read);
    {
        File output(partial, File::Mode::create);
        const auto incomplete = eqt::encode_pack_header(header);
        output.write(incomplete.data(), incomplete.size(), 0);
        File check(partial, File::Mode::read_direct);
        for (auto &layer : header.layers) {
            const LayerHashes written = write_layer(input, output, layer, buffers);
            output.sync();
            output.forget(layer.data_offset, layer_bytes(layer));
            const LayerHashes read = check_layer(check, layer, buffers);
            if (read.packed != written.packed || read.source != written.source) {
                throw std::runtime_error("Pack layer " + std::to_string(layer.layer) + " differs from what was written");
            }
            layer.checksum = written.packed;
        }
        header.complete = true;
        const auto complete = eqt::encode_pack_header(header);
        output.write(complete.data(), complete.size(), 0);
        output.sync();
    }
    std::filesystem::rename(partial, pack);
    sync_directory(pack);
    return {{"operation", "build"}, {"pack", pack}, {"layers", header.layers.size()}, {"pack_bytes", header.file_bytes},
            {"seconds", seconds(start)}};
}

Json move(const std::string &model, const std::string &pack) {
    const auto start = Clock::now();
    eqt::PackHeader header = layout_of(model);
    const std::string partial = pack + ".partial", state_path = pack + ".move";
    if (std::filesystem::exists(pack)) {
        throw std::runtime_error("The pack already exists: " + pack);
    }
    const std::string id = layout_id(header);
    Json state;
    const bool resuming = std::filesystem::exists(state_path);
    if (resuming) {
        state = read_state(state_path);
        if (state.at("layout") != id || !std::filesystem::exists(partial)) {
            throw std::runtime_error("The interrupted move does not belong to this model; run unpack with the model "
                                     "it was started from");
        }
    } else {
        std::filesystem::remove(partial); // a copy (build) interrupted earlier; the model was not touched
        state = {{"operation", "move"}, {"layout", id}, {"layers", Json::object()}};
        File output(partial, File::Mode::create);
        const auto incomplete = eqt::encode_pack_header(header);
        output.write(incomplete.data(), incomplete.size(), 0);
        output.sync();
        write_state(state_path, state);
    }
    Buffers buffers(header);
    File source(model, File::Mode::update);
    File output(partial, File::Mode::update);
    File check(partial, File::Mode::read_direct);
    int moved = 0;
    for (auto &layer : header.layers) {
        const std::string key = std::to_string(layer.layer);
        if (state["layers"].contains(key)) {
            layer.checksum = std::stoull(state["layers"][key].get<std::string>(), nullptr, 16);
        } else {
            for (const auto &bank : layer.banks) {
                if (source.has_hole(bank.source_offset, bank.stride * static_cast<uint64_t>(layer.experts))) {
                    throw std::runtime_error("Layer " + key + " has deallocated model blocks but is not in the pack; "
                                             "the model is damaged");
                }
            }
            require_free(pack, layer_bytes(layer));
            const LayerHashes written = write_layer(source, output, layer, buffers);
            output.sync();
            output.forget(layer.data_offset, layer_bytes(layer));
            const LayerHashes read = check_layer(check, layer, buffers);
            if (read.packed != written.packed || read.source != written.source) {
                throw std::runtime_error("Pack layer " + key + " differs from the model; nothing was deallocated");
            }
            layer.checksum = written.packed;
            state["layers"][key] = hex(layer.checksum);
            write_state(state_path, state);
            maybe_stop(++moved);
        }
        // Recorded first, deallocated second: a crash in between re-punches on resume.
        for (const auto &bank : layer.banks) {
            source.punch(bank.source_offset, bank.stride * static_cast<uint64_t>(layer.experts));
        }
        source.sync();
    }
    header.complete = header.source_moved = true;
    const auto complete = eqt::encode_pack_header(header);
    output.write(complete.data(), complete.size(), 0);
    output.sync();
    std::filesystem::rename(partial, pack);
    sync_directory(pack);
    std::filesystem::remove(state_path);
    sync_directory(pack);
    uint64_t routed = 0, banks = 0;
    for (const auto &layer : header.layers) {
        routed += source_bytes(layer);
        banks += layer.banks.size();
    }
    return {{"operation", "move"}, {"pack", pack}, {"layers", header.layers.size()}, {"layers_moved_now", moved},
            {"resumed", resuming}, {"routed_bytes", routed}, {"banks", banks}, {"pack_bytes", header.file_bytes},
            {"seconds", seconds(start)}};
}

Json unpack(const std::string &model, const std::string &pack) {
    const auto start = Clock::now();
    const eqt::PackHeader layout = layout_of(model);
    const std::string partial = pack + ".partial", move_state = pack + ".move", state_path = pack + ".unpack";
    eqt::PackHeader header = layout;
    std::string source_path;
    std::vector<int> present; // layers whose experts are in the pack
    if (std::filesystem::exists(move_state) && std::filesystem::exists(partial)) {
        // An interrupted move: only the recorded layers left the model.
        const Json state = read_state(move_state);
        if (state.at("layout") != layout_id(layout)) {
            throw std::runtime_error("The interrupted move does not belong to this model");
        }
        for (auto &layer : header.layers) {
            const std::string key = std::to_string(layer.layer);
            if (state["layers"].contains(key)) {
                layer.checksum = std::stoull(state["layers"][key].get<std::string>(), nullptr, 16);
                present.push_back(layer.layer);
            }
        }
        source_path = partial;
    } else if (std::filesystem::exists(pack)) {
        header = eqt::read_pack_header(pack, /*whole_file =*/false);
        eqt::check_pack_matches(header, layout);
        for (const auto &layer : header.layers) {
            present.push_back(layer.layer);
        }
        source_path = pack;
    } else {
        throw std::runtime_error("No pack to unpack: " + pack);
    }
    Json state = std::filesystem::exists(state_path) ? read_state(state_path)
                                                     : Json{{"operation", "unpack"}, {"layout", layout_id(layout)}, {"layers", Json::array()}};
    if (state.at("layout") != layout_id(layout)) {
        throw std::runtime_error("The interrupted unpack does not belong to this model");
    }
    Buffers buffers(header);
    File target(model, File::Mode::update);
    File writable(source_path, File::Mode::update);
    File direct(source_path, File::Mode::read_direct);
    int restored = 0;
    // Last layer first, so each restored layer is the pack's tail and truncating frees its space.
    for (auto it = header.layers.rbegin(); it != header.layers.rend(); ++it) {
        const auto &layer = *it;
        if (std::find(present.begin(), present.end(), layer.layer) == present.end()) {
            continue;
        }
        const bool done = std::find(state["layers"].begin(), state["layers"].end(), layer.layer) != state["layers"].end();
        if (!done) {
            if (writable.size() < layer.data_offset + layer_bytes(layer)) {
                throw std::runtime_error("Layer " + std::to_string(layer.layer) + " is missing from the pack");
            }
            require_free(model, source_bytes(layer));
            if (check_layer(direct, layer, buffers).packed != layer.checksum) {
                throw std::runtime_error("Pack layer " + std::to_string(layer.layer) + " fails its checksum; nothing was written");
            }
            read_layer(direct, layer, buffers, [&](int first, int count) {
                for (const auto &bank : layer.banks) {
                    for (int e = 0; e < count; ++e) {
                        std::memcpy(buffers.source.data() + bank.stride * e,
                                    buffers.packed.data() + layer.expert_bytes * e + bank.offset, static_cast<size_t>(bank.stride));
                    }
                    target.write(buffers.source.data(), static_cast<size_t>(bank.stride) * count,
                                 bank.source_offset + bank.stride * static_cast<uint64_t>(first));
                }
            });
            target.sync();
            state["layers"].push_back(layer.layer);
            write_state(state_path, state);
            maybe_stop(++restored);
        }
        writable.truncate(layer.data_offset);
        writable.sync();
    }
    std::filesystem::remove(source_path);
    std::filesystem::remove(move_state);
    std::filesystem::remove(state_path);
    sync_directory(pack);
    return {{"operation", "unpack"}, {"model", model}, {"layers_restored_now", restored}, {"seconds", seconds(start)}};
}

Json verify(const std::string &model, const std::string &pack) {
    const auto start = Clock::now();
    const eqt::PackHeader header = eqt::read_pack_header(pack);
    eqt::check_pack_matches(header, layout_of(model));
    Buffers buffers(header);
    File file(pack, File::Mode::read_direct);
    for (const auto &layer : header.layers) {
        if (check_layer(file, layer, buffers).packed != layer.checksum) {
            throw std::runtime_error("Expert pack checksum mismatch in layer " + std::to_string(layer.layer));
        }
    }
    return {{"operation", "verify"}, {"pack", pack}, {"layers", header.layers.size()}, {"pack_bytes", header.file_bytes},
            {"source_moved", header.source_moved}, {"direct_io", file.direct()}, {"matches_model", true},
            {"seconds", seconds(start)}};
}

Json sha256(const std::string &model, const std::string &pack) {
    const auto start = Clock::now();
    const eqt::PackHeader header = eqt::read_pack_header(pack);
    eqt::check_pack_matches(header, layout_of(model));
    struct Range {
        const eqt::PackLayer *layer;
        const eqt::PackBank *bank;
    };
    std::vector<Range> ranges;
    for (const auto &layer : header.layers) {
        for (const auto &bank : layer.banks) {
            ranges.push_back({&layer, &bank});
        }
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range &a, const Range &b) { return a.bank->source_offset < b.bank->source_offset; });
    File gguf(model, File::Mode::read), packed(pack, File::Mode::read_direct);
    Buffer buffer(64ull << 20);
    sha256_t context;
    sha256_init(&context);
    uint64_t cursor = 0;
    const auto feed_model = [&](uint64_t end) {
        while (cursor < end) {
            const size_t size = static_cast<size_t>(std::min<uint64_t>(buffer.size(), end - cursor));
            gguf.read(buffer.data(), size, cursor);
            gguf.forget(cursor, size);
            sha256_update(&context, buffer.data(), size);
            cursor += size;
        }
    };
    for (const auto &range : ranges) {
        feed_model(range.bank->source_offset);
        const size_t slice = static_cast<size_t>(round_up(range.bank->stride, pack_page_bytes));
        for (int e = 0; e < range.layer->experts; ++e) {
            const uint64_t at = range.layer->data_offset + range.layer->expert_bytes * static_cast<uint64_t>(e) + range.bank->offset;
            packed.read(buffer.data(), slice, at);
            sha256_update(&context, buffer.data(), static_cast<size_t>(range.bank->stride));
        }
        cursor += range.bank->stride * static_cast<uint64_t>(range.layer->experts);
    }
    feed_model(header.source_size);
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&context, digest);
    std::string text;
    for (const unsigned char byte : digest) {
        char pair[3];
        std::snprintf(pair, sizeof(pair), "%02x", byte);
        text += pair;
    }
    return {{"operation", "sha256"}, {"model", model}, {"sha256", text}, {"seconds", seconds(start)}};
}
} // namespace

int main(int argc, char **argv) {
    const char *commands[] = {"build", "move", "unpack", "verify", "sha256"};
    if (argc != 4 || std::none_of(std::begin(commands), std::end(commands), [&](const char *c) { return std::strcmp(argv[1], c) == 0; })) {
        std::cerr << "Usage: eqt-pack build|move|unpack|verify|sha256 MODEL.gguf PACK\n";
        return 2;
    }
    try {
        const std::string command = argv[1], model = argv[2], pack = argv[3];
        const Json result = command == "build"    ? build(model, pack)
                            : command == "move"   ? move(model, pack)
                            : command == "unpack" ? unpack(model, pack)
                            : command == "verify" ? verify(model, pack)
                                                  : sha256(model, pack);
        std::cout << result.dump() << '\n';
    } catch (const Interrupted &) {
        std::cerr << "eqt-pack: stopped by EQT_PACK_STOP_AFTER" << std::endl;
        return 3;
    } catch (const std::exception &error) {
        std::cerr << "eqt-pack: " << error.what() << '\n';
        return 1;
    }
}
