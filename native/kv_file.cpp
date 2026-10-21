#include "kv_file.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <linux/falloc.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace eqt {
namespace {
struct Mapping {
    uint8_t *base = nullptr;
    size_t size = 0;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE view = nullptr;
#else
    int fd = -1;
#endif
};

size_t page_size() {
#ifdef _WIN32
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return info.dwAllocationGranularity;
#else
    return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
}

void unmap(Mapping *mapping) {
#ifdef _WIN32
    if (mapping->base) {
        UnmapViewOfFile(mapping->base);
    }
    if (mapping->view) {
        CloseHandle(mapping->view);
    }
    if (mapping->file != INVALID_HANDLE_VALUE) {
        CloseHandle(mapping->file); // FILE_FLAG_DELETE_ON_CLOSE removes it
    }
#else
    if (mapping->base) {
        munmap(mapping->base, mapping->size);
    }
    if (mapping->fd >= 0) {
        close(mapping->fd); // already unlinked: the storage is released here
    }
#endif
    delete mapping;
}

// A file of `size` bytes in `directory`, reserved and mapped shared; nullptr (and errno-style text in `error`) when the
// file cannot be created or the device lacks the space.
Mapping *map_file(const std::string &directory, size_t size, std::string &error) {
    auto *mapping = new Mapping;
    mapping->size = size;
#ifdef _WIN32
    wchar_t folder[MAX_PATH], name[MAX_PATH];
    if (!MultiByteToWideChar(CP_UTF8, 0, directory.c_str(), -1, folder, MAX_PATH) ||
        !GetTempFileNameW(folder, L"eqk", 0, name)) {
        error = "cannot create a file in " + directory;
        unmap(mapping);
        return nullptr;
    }
    mapping->file = CreateFileW(name, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    LARGE_INTEGER end{};
    end.QuadPart = static_cast<LONGLONG>(size);
    if (mapping->file == INVALID_HANDLE_VALUE || !SetFilePointerEx(mapping->file, end, nullptr, FILE_BEGIN) ||
        !SetEndOfFile(mapping->file)) {
        error = "cannot reserve " + std::to_string(size) + " bytes in " + directory;
        unmap(mapping);
        return nullptr;
    }
    mapping->view = CreateFileMappingW(mapping->file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    mapping->base = mapping->view ? static_cast<uint8_t *>(MapViewOfFile(mapping->view, FILE_MAP_ALL_ACCESS, 0, 0, size)) : nullptr;
    if (!mapping->base) {
        error = "cannot map the context file";
        unmap(mapping);
        return nullptr;
    }
#else
    std::string path = directory + "/eqt-kv-XXXXXX";
    mapping->fd = mkostemp(path.data(), O_CLOEXEC);
    if (mapping->fd < 0) {
        error = "cannot create a file in " + directory + ": " + std::strerror(errno);
        unmap(mapping);
        return nullptr;
    }
    unlink(path.c_str());
    if (const int failure = posix_fallocate(mapping->fd, 0, static_cast<off_t>(size)); failure != 0) {
        error = "cannot reserve " + std::to_string(size >> 20) + " MiB in " + directory + ": " + std::strerror(failure);
        unmap(mapping);
        return nullptr;
    }
    void *base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, mapping->fd, 0);
    if (base == MAP_FAILED) {
        error = std::string("cannot map the context file: ") + std::strerror(errno);
        unmap(mapping);
        return nullptr;
    }
    mapping->base = static_cast<uint8_t *>(base);
    // Attention reads selected cells, not neighbouring pages.
    madvise(base, size, MADV_RANDOM);
#endif
    return mapping;
}

// Zeroes the whole file. Releasing its blocks and reserving them again leaves zeros without writing them; where the
// file system cannot, the bytes are written.
void zero(Mapping *mapping) {
#ifndef _WIN32
    if (fallocate(mapping->fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, 0, static_cast<off_t>(mapping->size)) == 0 &&
        posix_fallocate(mapping->fd, 0, static_cast<off_t>(mapping->size)) == 0) {
        return;
    }
#endif
    std::memset(mapping->base, 0, mapping->size);
}

const ggml_backend_buffer_i buffer_interface = {
    /* .free_buffer   = */ [](ggml_backend_buffer_t buffer) { unmap(static_cast<Mapping *>(buffer->context)); },
    /* .get_base      = */ [](ggml_backend_buffer_t buffer) -> void * { return static_cast<Mapping *>(buffer->context)->base; },
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */
    [](ggml_backend_buffer_t, ggml_tensor *tensor, uint8_t value, size_t offset, size_t size) {
        std::memset(static_cast<uint8_t *>(tensor->data) + offset, value, size);
    },
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
    /* .clear         = */
    [](ggml_backend_buffer_t buffer, uint8_t value) {
        auto *mapping = static_cast<Mapping *>(buffer->context);
        if (!mapping->base) {
            return;
        }
        if (value == 0) {
            zero(mapping);
        } else {
            std::memset(mapping->base, value, mapping->size);
        }
    },
    /* .reset         = */ nullptr,
};
} // namespace

struct KvFileAccess {
    static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type, size_t size) {
        auto &owner = *static_cast<KvFile *>(type->context);
        Mapping *mapping = nullptr;
        if (size) {
            const size_t page = page_size();
            std::string error;
            mapping = map_file(owner.directory_, (size + page - 1) / page * page, error);
            if (!mapping) {
                GGML_LOG_ERROR("eqt: context keys and values on storage: %s\n", error.c_str());
                owner.error_ = std::move(error);
                return nullptr;
            }
            owner.reserved_ += mapping->size;
        } else {
            mapping = new Mapping;
        }
        return ggml_backend_buffer_init(type, buffer_interface, mapping, size);
    }
};

namespace {
const ggml_backend_buffer_type_i buffer_type_interface = {
    /* .get_name         = */ [](ggml_backend_buffer_type_t) { return "EQT_KVFile"; },
    /* .alloc_buffer     = */ &KvFileAccess::allocate,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ [](ggml_backend_buffer_type_t) -> size_t { return 64; },
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ nullptr,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ [](ggml_backend_buffer_type_t) { return true; },
};
} // namespace

KvFile::KvFile(std::string directory) : directory_(std::move(directory)) {
    if (directory_.empty()) {
        throw std::invalid_argument("kv_file_dir must name a directory");
    }
    std::error_code error;
    const auto path = std::filesystem::path(reinterpret_cast<const char8_t *>(directory_.c_str()));
    std::filesystem::create_directories(path, error);
    if (!std::filesystem::is_directory(path, error)) {
        throw std::invalid_argument("kv_file_dir cannot be created: " + directory_);
    }
    auto *cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        throw std::runtime_error("No CPU backend registered");
    }
    type_.reset(new ggml_backend_buffer_type{buffer_type_interface, cpu, this});
}

KvFile::~KvFile() = default;

ggml_backend_buffer_type *KvFile::buffer_type() {
    return type_.get();
}
} // namespace eqt
