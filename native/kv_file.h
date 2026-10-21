#pragma once

#include <memory>
#include <string>

struct ggml_backend_buffer_type;

namespace eqt {
// Host buffers for KV cache layers backed by a file on storage (kv_file_dir, D034). Each buffer is a shared mapping of
// a file that is deleted as soon as it is opened, so its storage is released when the buffer is freed or the process
// ends, also after a crash. The whole file is reserved when the buffer is allocated: a full device fails the load
// instead of a later write into the mapping. Pages that attention reads stay in the page cache like any file data;
// sparse attention (Qwen3.8's QSA) reads only the cells its indexer selects, so a long context keeps its keys and
// values on storage and only the selected blocks in memory. The data are the bytes llama.cpp would hold in RAM.
class KvFile {
  public:
    explicit KvFile(std::string directory);
    ~KvFile();
    KvFile(const KvFile &) = delete;
    KvFile &operator=(const KvFile &) = delete;
    ggml_backend_buffer_type *buffer_type();
    const std::string &directory() const { return directory_; }
    // Bytes of all buffers allocated so far (the reserved file sizes).
    unsigned long long reserved_bytes() const { return reserved_; }
    // Why the last allocation failed; empty when none did.
    const std::string &error() const { return error_; }

  private:
    friend struct KvFileAccess;
    std::string directory_;
    std::string error_;
    std::unique_ptr<ggml_backend_buffer_type> type_;
    unsigned long long reserved_ = 0;
};
} // namespace eqt
