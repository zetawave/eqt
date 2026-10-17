#include "engine.h"
#include "expert_store.h"
#include "gpu_offload.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "jinja/parser.h"
#include "llama-ext.h"
#include "jinja/runtime.h"
#include "json.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <vector>
#ifdef __ANDROID__
#include <android/log.h>
#endif
#ifndef _WIN32
#include <sys/resource.h>
#endif
#ifdef __linux__
#include <sched.h>
#endif
#if defined(__aarch64__) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace eqt {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
int integer(const Json &j, const char *key, int fallback, int low, int high) {
    const Json value = j.contains(key) ? j.at(key) : Json(fallback);
    if (!value.is_number_integer() || value < low || value > high) {
        throw std::invalid_argument(std::string(key) + " outside supported range");
    }
    return value.get<int>();
}
std::string metadata(llama_model *model, const char *key) {
    const int count = llama_model_meta_val_str(model, key, nullptr, 0);
    if (count < 0) {
        return {};
    }
    std::string result(count + 1, '\0');
    llama_model_meta_val_str(model, key, result.data(), result.size());
    result.resize(count);
    return result;
}
std::string piece(const llama_vocab *vocab, llama_token token) {
    if (token == LLAMA_TOKEN_NULL) {
        return {};
    }
    char buffer[256];
    int count = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, true);
    if (count >= 0) {
        return {buffer, static_cast<size_t>(count)};
    }
    std::string result(-count, '\0');
    count = llama_token_to_piece(vocab, token, result.data(), static_cast<int>(result.size()), 0, true);
    if (count < 0) {
        throw std::runtime_error("Cannot decode token");
    }
    result.resize(count);
    return result;
}
// Routed banks streamed by ExpertStore; everything else is read into ordinary CPU buffers at load.
constexpr const char *routed_bank_pattern = "^blk\\.[0-9]+\\.ffn_(gate|up|down|gate_up)_exps\\.weight$";

struct CpuFeatures {
    bool known = false, dotprod = false, fp16 = false, sve = false, i8mm = false, sve2 = false, sme = false;
};
CpuFeatures cpu_features() {
    CpuFeatures f;
#if defined(__aarch64__) && defined(__linux__)
    const unsigned long hw = getauxval(AT_HWCAP), hw2 = getauxval(AT_HWCAP2);
    f.known = true;
    f.dotprod = hw & HWCAP_ASIMDDP;
    f.fp16 = (hw & HWCAP_FPHP) && (hw & HWCAP_ASIMDHP);
    f.sve = hw & HWCAP_SVE;
    f.i8mm = hw2 & HWCAP2_I8MM;
    f.sve2 = hw2 & HWCAP2_SVE2;
    f.sme = hw2 & HWCAP2_SME;
#endif
    return f;
}
Json features_json(const CpuFeatures &f) {
    if (!f.known) {
        return nullptr;
    }
    return {{"dotprod", f.dotprod}, {"fp16", f.fp16}, {"sve", f.sve},
            {"i8mm", f.i8mm},       {"sve2", f.sve2}, {"sme", f.sme}};
}

std::string cpu_variant = "static";
std::string backend_error;
#ifdef GGML_BACKEND_DL
// GGML's Android variant list (ggml/src/CMakeLists.txt), ordered by its aarch64 score. A variant is
// eligible only when HWCAP reports every feature it was compiled with.
struct Variant {
    const char *name;
    bool dotprod, fp16, sve, i8mm, sve2, sme;
};
constexpr Variant variants[] = {
    {"android_armv9.2_2", true, true, true, true, true, true},
    {"android_armv9.2_1", true, true, true, true, false, true},
    {"android_armv9.0_1", true, true, false, true, true, false},
    {"android_armv8.6_1", true, true, false, true, false, false},
    {"android_armv8.2_2", true, true, false, false, false, false},
    {"android_armv8.2_1", true, false, false, false, false, false},
    {"android_armv8.0_1", false, false, false, false, false, false},
};
bool supported(const Variant &v, const CpuFeatures &f) {
    return (!v.dotprod || f.dotprod) && (!v.fp16 || f.fp16) && (!v.sve || f.sve) && (!v.i8mm || f.i8mm) &&
           (!v.sve2 || f.sve2) && (!v.sme || f.sme);
}
void load_cpu_backend() {
    const auto features = cpu_features();
    const char *requested = std::getenv("EQT_CPU_VARIANT");
    const std::string wanted = requested && *requested ? requested : "auto";
    for (const auto &variant : variants) {
        if ((wanted == "auto" && supported(variant, features)) || wanted == variant.name) {
            if (!supported(variant, features)) {
                backend_error = "CPU lacks features required by " + wanted;
                return;
            }
            // A bare soname resolves through LD_LIBRARY_PATH (CLI) or the APK linker namespace (app).
            const std::string library = std::string("libggml-cpu-") + variant.name + ".so";
            if (!ggml_backend_load(library.c_str())) {
                backend_error = "Cannot load " + library;
                return;
            }
            cpu_variant = variant.name;
            return;
        }
    }
    backend_error = "Unknown EQT_CPU_VARIANT: " + wanted;
}
#endif

// Without OpenMP, a context with no attached pool makes GGML spawn and join worker threads for every
// graph compute; expert streaming splits each token's graph at every routed layer.
struct ThreadpoolApi {
    decltype(&ggml_threadpool_new) create = nullptr;
    decltype(&ggml_threadpool_free) destroy = nullptr;
};
ThreadpoolApi threadpool_api() {
    // The CPU backend registry resolves both static builds and the dynamically loaded ISA variant.
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    if (!reg) {
        return {};
    }
    return {reinterpret_cast<decltype(&ggml_threadpool_new)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new")),
            reinterpret_cast<decltype(&ggml_threadpool_free)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"))};
}

// CPUs this process may run on now; 0 when unknown. Android moves a backgrounded app, even one with a foreground
// service, into a smaller cpuset (on the Galaxy S24+, the four little cores), which narrows every thread's affinity.
int allowed_cpus() {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    return sched_getaffinity(0, sizeof(set), &set) == 0 ? CPU_COUNT(&set) : 0;
#else
    return 0;
#endif
}

// Supplies tensor bytes for llama_model_init_from_user: routed banks are registered with the store,
// all other tensors are read once into the CPU buffers llama allocated.
struct TensorLoader {
    gguf_context *meta;
    ggml_context *shapes;
    std::ifstream file;
    ExpertStore *store;
    const std::atomic_bool &cancelled;
    const Emit &progress;
    uint64_t total = 0, done = 0;
    int last = -1;
    std::string error{};

    static void set(ggml_tensor *tensor, void *opaque) {
        auto &loader = *static_cast<TensorLoader *>(opaque);
        if (!loader.error.empty() || loader.cancelled.load()) {
            return;
        }
        try {
            loader.load(tensor);
        } catch (const std::exception &e) {
            loader.error = e.what();
        }
    }
    void load(ggml_tensor *tensor) {
        const char *name = ggml_get_name(tensor);
        const int64_t id = gguf_find_tensor(meta, name);
        const ggml_tensor *stored = ggml_get_tensor(shapes, name);
        if (id < 0 || !stored) {
            throw std::runtime_error(
                std::string("Tensor is not stored in the file (virtual tensors are unsupported with "
                            "expert streaming): ") +
                name);
        }
        if (stored->type != tensor->type || !ggml_are_same_shape(stored, tensor) ||
            ggml_nbytes(stored) != ggml_nbytes(tensor)) {
            throw std::runtime_error(std::string("Tensor type or shape differs from the file: ") + name);
        }
        const uint64_t offset = gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, id);
        const size_t size = ggml_nbytes(tensor);
        if (store->maps(tensor)) {
            return; // file-backed, loaded on use
        }
        if (store->owns(tensor)) {
            store->add_bank(tensor, offset);
            return;
        }
        file.seekg(static_cast<std::streamoff>(offset));
        if (ggml_backend_buffer_is_host(tensor->buffer)) {
            file.read(static_cast<char *>(tensor->data), static_cast<std::streamsize>(size));
        } else {
            std::vector<char> staging(size);
            file.read(staging.data(), static_cast<std::streamsize>(size));
            ggml_backend_tensor_set(tensor, staging.data(), 0, size);
        }
        if (!file) {
            throw std::runtime_error(std::string("Short read for tensor ") + name);
        }
        store->add_router_tensor(tensor);
        done += size;
        const int percent = total ? static_cast<int>(done * 100 / total) : 100;
        if (progress && percent != last) {
            last = percent;
            progress(std::to_string(percent));
        }
    }
};

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>(std::ceil(q * values.size())) - 1];
}
// GGUF integer metadata: a scalar, or the largest element of a per-layer array; -1 when absent.
int64_t gguf_integer(const gguf_context *meta, const std::string &key) {
    const int64_t id = gguf_find_key(meta, key.c_str());
    if (id < 0) {
        return -1;
    }
    const auto element = [](gguf_type type, const void *data, size_t i) -> int64_t {
        switch (type) {
        case GGUF_TYPE_UINT8: return static_cast<const uint8_t *>(data)[i];
        case GGUF_TYPE_INT8: return static_cast<const int8_t *>(data)[i];
        case GGUF_TYPE_UINT16: return static_cast<const uint16_t *>(data)[i];
        case GGUF_TYPE_INT16: return static_cast<const int16_t *>(data)[i];
        case GGUF_TYPE_UINT32: return static_cast<const uint32_t *>(data)[i];
        case GGUF_TYPE_INT32: return static_cast<const int32_t *>(data)[i];
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(static_cast<const uint64_t *>(data)[i]);
        case GGUF_TYPE_INT64: return static_cast<const int64_t *>(data)[i];
        default: return -1;
        }
    };
    if (gguf_get_kv_type(meta, id) != GGUF_TYPE_ARRAY) {
        return element(gguf_get_kv_type(meta, id), gguf_get_val_data(meta, id), 0);
    }
    if (gguf_get_arr_type(meta, id) == GGUF_TYPE_STRING) {
        return -1;
    }
    int64_t widest = -1;
    for (size_t i = 0; i < gguf_get_arr_n(meta, id); ++i) {
        widest = std::max(widest, element(gguf_get_arr_type(meta, id), gguf_get_arr_data(meta, id), i));
    }
    return widest;
}

struct ContextCost {
    int64_t trained = -1;            // <arch>.context_length; -1 when the file does not say
    uint64_t kv_bytes_per_token = 0; // F16 keys and values of full-attention layers; 0 when unknown
};

// Hybrid models (<arch>.full_attention_interval) keep fixed-size recurrent state in the other layers, so only
// every interval-th layer grows with context; MTP blocks counted in block_count are included, which slightly
// overestimates. Without attention metadata there is no estimate.
ContextCost context_cost(const gguf_context *meta) {
    ContextCost cost;
    const int64_t id = gguf_find_key(meta, "general.architecture");
    if (id < 0 || gguf_get_kv_type(meta, id) != GGUF_TYPE_STRING) {
        return cost;
    }
    const std::string arch = gguf_get_val_str(meta, id);
    const auto value = [&](const char *suffix) { return gguf_integer(meta, arch + suffix); };
    cost.trained = value(".context_length");
    const int64_t layers = value(".block_count"), heads = value(".attention.head_count");
    const int64_t embedding = value(".embedding_length"), interval = value(".full_attention_interval");
    int64_t kv_heads = value(".attention.head_count_kv"), key = value(".attention.key_length");
    if (kv_heads <= 0) {
        kv_heads = heads;
    }
    if (key <= 0 && heads > 0 && embedding > 0) {
        key = embedding / heads;
    }
    const int64_t val = value(".attention.value_length") > 0 ? value(".attention.value_length") : key;
    if (layers <= 0 || kv_heads <= 0 || key <= 0) {
        return cost;
    }
    const int64_t attention_layers = interval > 1 ? (layers + interval - 1) / interval : layers;
    cost.kv_bytes_per_token = static_cast<uint64_t>(attention_layers * kv_heads * (key + val) * 2);
    return cost;
}

// Whole-process CPU accounting (all threads, including expert I/O workers); null where unavailable.
Json process_cpu() {
    Json result = {{"user_ms", nullptr},          {"system_ms", nullptr},
                   {"minor_faults", nullptr},     {"major_faults", nullptr},
                   {"voluntary_switches", nullptr}, {"involuntary_switches", nullptr}};
#ifndef _WIN32
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        result["user_ms"] = usage.ru_utime.tv_sec * 1e3 + usage.ru_utime.tv_usec / 1e3;
        result["system_ms"] = usage.ru_stime.tv_sec * 1e3 + usage.ru_stime.tv_usec / 1e3;
        result["minor_faults"] = usage.ru_minflt;
        result["major_faults"] = usage.ru_majflt;
        result["voluntary_switches"] = usage.ru_nvcsw;
        result["involuntary_switches"] = usage.ru_nivcsw;
    }
#endif
    return result;
}

Json counter_delta(const Json &before, const Json &after) {
    Json delta = Json::object();
    for (const auto &[key, value] : after.items()) {
        delta[key] = value.is_null() || before.at(key).is_null() ? Json(nullptr)
                                                                  : Json(value.get<double>() - before.at(key).get<double>());
    }
    return delta;
}

void log_line(ggml_log_level level, const char *text, void *) {
    if (level == GGML_LOG_LEVEL_DEBUG) {
        return;
    }
#ifdef __ANDROID__
    __android_log_write(level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, "Equity", text);
#else
    std::fputs(text, stderr);
#endif
}

constexpr size_t max_images_per_request = 8;

// Chat messages for the template: content parts are flattened to one string in which every image is the media
// marker that mtmd_tokenize later replaces with that image's embeddings, in order.
struct ChatInput {
    Json messages = Json::array();
    std::vector<std::string> images;
};
ChatInput chat_input(const Json &messages) {
    if (!messages.is_array() || messages.empty() || messages.size() > 128) {
        throw std::invalid_argument("messages must contain 1..128 messages");
    }
    ChatInput input;
    for (const auto &message : messages) {
        const auto role = message.at("role").get<std::string>();
        if (role != "system" && role != "user" && role != "assistant") {
            throw std::invalid_argument("Only system/user/assistant roles are supported");
        }
        const auto &content = message.at("content");
        std::string text;
        if (content.is_string()) {
            text = content.get<std::string>();
        } else if (content.is_array()) {
            for (const auto &part : content) {
                const auto type = part.at("type").get<std::string>();
                if (type == "text") {
                    text += part.at("text").get<std::string>();
                } else if (type == "image") {
                    if (role != "user") {
                        throw std::invalid_argument("Images are supported in user messages only");
                    }
                    if (input.images.size() == max_images_per_request) {
                        throw std::invalid_argument("At most " + std::to_string(max_images_per_request) +
                                                    " images per request");
                    }
                    input.images.push_back(part.at("path").get<std::string>());
                    text += mtmd_default_marker();
                } else {
                    throw std::invalid_argument("Unsupported content part: " + type);
                }
            }
        } else {
            throw std::invalid_argument("content must be a string or an array of parts");
        }
        if (text.size() > 1024 * 1024) {
            throw std::invalid_argument("Message exceeds 1 MiB");
        }
        input.messages.push_back({{"role", role}, {"content", std::move(text)}});
    }
    return input;
}

using MediaContext = std::unique_ptr<mtmd_context, decltype(&mtmd_free)>;
using MediaChunks = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;

} // namespace

Json process_memory() {
    Json result = {{"rss_bytes", nullptr},    {"peak_rss_bytes", nullptr}, {"pss_bytes", nullptr},
                   {"minor_faults", nullptr}, {"major_faults", nullptr},   {"storage_read_bytes", nullptr}};
#ifndef _WIN32
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        for (const auto &[prefix, key] : {std::pair{"VmRSS:", "rss_bytes"}, {"VmHWM:", "peak_rss_bytes"}}) {
            if (line.starts_with(prefix)) {
                result[key] = std::stoull(line.substr(std::char_traits<char>::length(prefix))) * 1024;
            }
        }
    }
    std::ifstream smaps("/proc/self/smaps_rollup");
    while (std::getline(smaps, line)) {
        if (line.starts_with("Pss:")) {
            result["pss_bytes"] = std::stoull(line.substr(4)) * 1024;
        }
    }
    std::ifstream io("/proc/self/io");
    while (std::getline(io, line)) {
        if (line.starts_with("read_bytes:")) {
            result["storage_read_bytes"] = std::stoull(line.substr(11));
        }
    }
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        result["minor_faults"] = usage.ru_minflt;
        result["major_faults"] = usage.ru_majflt;
    }
#endif
    return result;
}

std::string render_chat(const std::string &source, const Json &messages, bool thinking,
                        const std::string &bos, const std::string &eos) {
    if (source.empty()) {
        throw std::invalid_argument("Model has no embedded chat template");
    }
    jinja::lexer lexer;
    auto tokens = lexer.tokenize(source);
    auto program = jinja::parse_from_tokens(tokens);
    jinja::context context(source);
    context.current_time = 0;
    const Json vars = {{"messages", messages},
                       {"add_generation_prompt", true},
                       {"enable_thinking", thinking},
                       {"bos_token", bos},
                       {"eos_token", eos}};
    jinja::global_from_json(context, common_json::parse(vars.dump()), true);
    jinja::runtime runtime(context);
    return jinja::render_string_parts(runtime.gather_string_parts(runtime.execute(program)));
}

// A graph stops when the user cancels or when a compute thread found a failed expert read.
struct AbortSource {
    const std::atomic_bool *cancelled = nullptr;
    const ExpertStore *store = nullptr;
    const GpuOffload *gpu = nullptr;
};

struct Engine::State {
    // Members are destroyed in reverse order: context and model release expert buffers before the store
    // joins its I/O workers and frees the buffer type those buffers reference.
    std::unique_ptr<ExpertStore> store;
    // Uses the store's experts; removed before it.
    std::unique_ptr<GpuOffload> gpu;
    bool gpu_vision_flash_attn = false;
    AbortSource abort;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta{nullptr, gguf_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> shapes{nullptr, ggml_free};
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<ggml_threadpool, void (*)(ggml_threadpool *)> threadpool{nullptr, nullptr};
    // Smaller pools used while Android allows fewer CPUs, by thread count. GGML's CPU backends keep a pointer to
    // the last pool they ran and pause it when another one is attached, so pools live until the contexts are freed.
    std::vector<std::pair<int, std::unique_ptr<ggml_threadpool, void (*)(ggml_threadpool *)>>> reduced_pools;
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch{nullptr, llama_batch_ext_free};
    // Native MTP head of the same model, drafting tokens that the target verifies in one batch.
    std::unique_ptr<llama_context, decltype(&llama_free)> draft{nullptr, llama_free};
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> draft_batch{nullptr, llama_batch_ext_free};
    Json info;
    int context_size = 0;
    int batch_size = 0;
    int draft_max = 0;
    double draft_p_min = 0;
    size_t n_embd = 0;
    std::vector<float> pending_h, verify_h, draft_h;
    std::string chat_template;
    // Vision projector (mmproj GGUF). It is loaded only for requests with images and released as soon as they
    // are encoded, so text chats, prompt evaluation and decoding never hold its weights in memory.
    std::string projector_path;
    int image_max_tokens = 0;
    int threads = 0;
    // Persistent pool sized to min(configured threads, allowed CPUs).
    ThreadpoolApi pool_api;
    ggml_threadpool_params pool_params{};
    bool adaptive_threads = false;
    int pool_threads = 0;
    int pool_changes = 0;

    // Spinning workers that outnumber the CPUs Android allows stall one another at every barrier (measured:
    // 0.2 tokens/s with six threads on four cores), so the pool follows the allowed CPU count, never above the
    // configured threads. Called before each graph evaluation; it only rebuilds the pool when the count changes.
    void fit_threads() {
        if (!adaptive_threads) {
            return;
        }
        const int allowed = allowed_cpus();
        const int wanted = allowed > 0 ? std::min(threads, allowed) : threads;
        // Awake I/O threads need CPUs of their own next to the compute threads.
        if (store) {
            store->set_awake_allowed(allowed <= 0 || allowed >= threads + 2);
        }
        if (wanted == pool_threads) {
            return;
        }
        ggml_threadpool *pool = wanted == threads ? threadpool.get() : nullptr;
        for (const auto &[size, reduced] : reduced_pools) {
            if (size == wanted) {
                pool = reduced.get();
            }
        }
        if (!pool) {
            auto params = pool_params;
            params.n_threads = wanted;
            std::unique_ptr<ggml_threadpool, void (*)(ggml_threadpool *)> created{pool_api.create(&params), pool_api.destroy};
            if (!created) {
                return;
            }
            pool = created.get();
            reduced_pools.emplace_back(wanted, std::move(created));
        }
        for (llama_context *target : {context.get(), draft.get()}) {
            if (target) {
                llama_attach_threadpool(target, pool, pool);
                llama_set_n_threads(target, wanted, wanted);
            }
        }
        pool_threads = wanted;
        ++pool_changes;
    }
};

Engine::Engine() {
    static std::once_flag init;
    std::call_once(init, [] {
        llama_log_set(log_line, nullptr);
        mtmd_helper_log_set(log_line, nullptr);
#ifdef GGML_BACKEND_DL
        load_cpu_backend();
#endif
        llama_backend_init();
    });
}
Engine::~Engine() = default;
void Engine::prepare() {
    cancelled_.store(false);
}
void Engine::cancel() {
    cancelled_.store(true);
}
void Engine::unload() {
    state_.reset();
}

Json Engine::load(const std::string &path, const Json &options, const Emit &progress) {
    unload();
    const auto start = Clock::now();
    const int context_size = integer(options, "context", 2048, 128, 1 << 20);
    const int batch_size = integer(options, "batch", 128, 1, 2048);
    const int threads = integer(options, "threads", 4, 1, 16);
    const int budget = integer(options, "memory_budget_mib", 2048, 768, 16384);
    if (!backend_error.empty()) {
        throw std::runtime_error("CPU backend unavailable: " + backend_error);
    }
    // GGML reads its tiled-GEMM switch once per process, at the first prompt-sized matrix product.
    if (options.contains("tiled_mm")) {
        const char *value = options.at("tiled_mm").get<bool>() ? "1" : "0";
#ifdef _WIN32
        _putenv_s("GGML_CPU_TILED_MM", value);
#else
        setenv("GGML_CPU_TILED_MM", value, 1);
#endif
    }
    const bool streaming = options.value("expert_streaming", false);
    const std::string projector = options.value("mmproj", std::string());
    const int image_max_tokens = integer(options, "image_max_tokens", 1024, 64, 4096);
    if (!projector.empty() && !mtmd_get_cap_from_file(projector.c_str()).inp_vision) {
        throw std::runtime_error("The vision projector cannot be read or has no image encoder");
    }
    const auto size = std::filesystem::file_size(path);
    {
        std::ifstream probe(path, std::ios::binary);
        if (!probe) {
            throw std::runtime_error("Cannot read model; check the file provider's access permissions");
        }
        char magic[4]{};
        probe.read(magic, sizeof(magic));
        if (probe.gcount() != 4 || std::string_view(magic, 4) != "GGUF") {
            throw std::runtime_error("Expected a GGUF weight file");
        }
    }
    auto state = std::make_unique<State>();
    std::unique_ptr<gguf_context, decltype(&gguf_free)> header{nullptr, gguf_free};
    uint64_t resident_bytes = size;
    ExpertOptions expert_options;
    std::unordered_map<std::string, uint64_t> routed_offsets;
    uint64_t mapped_bytes = 0; // file-backed tensors, never resident as a whole
    const bool gpu = options.value("gpu", false);
    GpuOptions gpu_options;
    gpu_options.min_rows = integer(options, "gpu_min_rows", gpu_options.min_rows, 1, 1 << 20);
    gpu_options.min_expert_rows = integer(options, "gpu_min_expert_rows", gpu_options.min_expert_rows, 1, 1 << 20);
    gpu_options.copy_threads = integer(options, "gpu_copy_threads", gpu_options.copy_threads, 0, 16);
    if (options.contains("gpu_wait")) {
        const std::string wait = options.at("gpu_wait").get<std::string>();
        if (wait != "sleep" && wait != "spin" && wait != "wfe") {
            throw std::invalid_argument("gpu_wait must be sleep, spin or wfe");
        }
        gpu_options.wait = wait == "sleep" ? GpuOptions::Wait::sleep
                           : wait == "spin" ? GpuOptions::Wait::spin
                                            : GpuOptions::Wait::wfe;
    }
    uint64_t largest_weight = 0;
    auto note_weight = [&](const gguf_context *meta) {
        for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
            // The embedding is read by row and the output head runs on prompts' last token only: neither is staged.
            const char *name = gguf_get_tensor_name(meta, i);
            if (std::strcmp(name, "token_embd.weight") != 0 && std::strcmp(name, "output.weight") != 0) {
                largest_weight = std::max<uint64_t>(largest_weight, gguf_get_tensor_size(meta, i));
            }
        }
    };
    if (streaming) {
        ggml_context *shapes = nullptr;
        state->meta.reset(gguf_init_from_file(path.c_str(), {/*no_alloc =*/true, /*ctx =*/&shapes}));
        state->shapes.reset(shapes);
        if (!state->meta || !state->shapes) {
            throw std::runtime_error("Cannot parse GGUF metadata");
        }
        const std::regex routed(routed_bank_pattern);
        uint64_t routed_bytes = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(state->meta.get()); ++i) {
            const char *name = gguf_get_tensor_name(state->meta.get(), i);
            if (std::regex_search(name, routed)) {
                routed_bytes += gguf_get_tensor_size(state->meta.get(), i);
                routed_offsets[name] = gguf_get_data_offset(state->meta.get()) + gguf_get_tensor_offset(state->meta.get(), i);
            } else if (std::strcmp(name, "token_embd.weight") == 0 && options.value("map_embeddings", true)) {
                routed_offsets[name] = gguf_get_data_offset(state->meta.get()) + gguf_get_tensor_offset(state->meta.get(), i);
                mapped_bytes = gguf_get_tensor_size(state->meta.get(), i);
            }
        }
        if (!routed_bytes) {
            throw std::runtime_error("expert_streaming requires routed expert banks");
        }
        expert_options.budget_bytes =
            static_cast<uint64_t>(integer(options, "expert_cache_mib", 1024, 1, 16384)) << 20;
        expert_options.io_threads = integer(options, "io_threads", 4, 1, 16);
        expert_options.demand_threads = integer(options, "demand_io_threads", 4, 0, 15);
        expert_options.awake_demand_threads = integer(options, "awake_demand_io_threads", 0, 0, 4);
        expert_options.direct_io = options.value("direct_io", false);
        expert_options.prefetch = options.value("prefetch", false);
        expert_options.prefetch_extra = integer(options, "prefetch_extra", 0, 0, 32);
        expert_options.coalesce_kib = integer(options, "coalesce_kib", 0, 0, 65536);
        expert_options.coalesce_gap = integer(options, "coalesce_gap", 1, 0, 8);
        expert_options.prefill_prefetch_mib = integer(options, "prefill_prefetch_mib", 0, 0, 4096);
        expert_options.slots = options.value("expert_slots", true);
        expert_options.overlap_reads = options.value("overlap_reads", true);
        expert_options.trace_path = options.value("expert_trace", std::string());
        if (options.contains("expert_layer_weights")) {
            expert_options.layer_weights = options.at("expert_layer_weights").get<std::vector<double>>();
        }
        resident_bytes = size - routed_bytes - mapped_bytes + expert_options.budget_bytes +
                         (static_cast<uint64_t>(expert_options.prefill_prefetch_mib) << 20);
    } else if (options.contains("expert_cache_mib") || options.value("prefetch", false)) {
        throw std::invalid_argument("Expert cache options require expert_streaming=true");
    }
    // The context is bounded by the model's trained context, not by an engine constant; its KV cache counts
    // toward admission.
    if (!streaming) {
        header.reset(gguf_init_from_file(path.c_str(), {/*no_alloc =*/true, /*ctx =*/nullptr}));
        if (!header) {
            throw std::runtime_error("Cannot parse GGUF metadata");
        }
    }
    const ContextCost context_cost_estimate = context_cost(streaming ? state->meta.get() : header.get());
    if (gpu) {
        // Staging for one operation at a time; the GPU never holds a copy of the model.
        const gguf_context *meta = streaming ? state->meta.get() : header.get();
        note_weight(meta);
        const int64_t arch_key = gguf_find_key(meta, "general.architecture");
        const std::string arch_name =
            arch_key >= 0 && gguf_get_kv_type(meta, arch_key) == GGUF_TYPE_STRING ? gguf_get_val_str(meta, arch_key) : "";
        auto value = [&](const char *suffix, int64_t fallback) {
            const int64_t found = arch_name.empty() ? -1 : gguf_integer(meta, arch_name + suffix);
            return found > 0 ? found : fallback;
        };
        resident_bytes += GpuOffload::staging_estimate(largest_weight, batch_size, value(".embedding_length", 4096),
                                                       static_cast<int>(value(".expert_used_count", 1)));
    }
    header.reset();
    if (context_cost_estimate.trained > 0 && context_size > context_cost_estimate.trained) {
        throw std::invalid_argument("context exceeds the model's trained context of " +
                                    std::to_string(context_cost_estimate.trained) + " tokens");
    }
    const uint64_t kv_bytes = static_cast<uint64_t>(context_size) * context_cost_estimate.kv_bytes_per_token;
    resident_bytes += kv_bytes;
    // Admission is conservative, not an allocator-enforced RSS limit.
    if (resident_bytes > static_cast<uint64_t>(budget - 512) * 1024 * 1024) {
        throw std::runtime_error(streaming ? "Shared weights, expert cache and KV cache exceed the admission budget "
                                             "(512 MiB reserved)"
                                           : "Model and KV cache exceed the resident admission budget (512 MiB "
                                             "reserved); enable expert_streaming for routed MoE models");
    }
    struct Progress {
        std::atomic_bool &cancelled;
        const Emit &emit;
        int last = -1;
    } progress_state{cancelled_, progress};
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    // The model and its contexts use the CPU backend only, also after a GPU-mode load registered the Vulkan
    // backend in this process; the GPU path computes products through the CPU backend's hook instead.
    static ggml_backend_dev_t no_devices[] = {nullptr};
    model_params.devices = no_devices;
    model_params.load_mode = options.value("mmap", true) ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    model_params.lazy_mode = LLAMA_LAZY_MODE_OFF;
    const bool mtp = options.value("mtp", false);
    const int draft_max = integer(options, "draft_max", 2, 1, 8);
    const double draft_p_min = options.value("draft_p_min", 0.0);
    if (!std::isfinite(draft_p_min) || draft_p_min < 0 || draft_p_min > 1) {
        throw std::invalid_argument("draft_p_min must be in [0, 1]");
    }
    model_params.load_mtp = mtp;
    // GGML weight repacking (interleaved blocks for dotprod/i8mm GEMV) applies to resident tensors only;
    // routed banks keep the GGUF layout of the expert store.
    const bool repack = options.value("repack", false);
    model_params.use_extra_bufts = repack;
    model_params.progress_callback_user_data = &progress_state;
    model_params.progress_callback = [](float value, void *opaque) {
        auto &p = *static_cast<Progress *>(opaque);
        const int percent = static_cast<int>(value * 100);
        if (p.emit && percent != p.last) {
            p.last = percent;
            p.emit(std::to_string(percent));
        }
        return !p.cancelled.load();
    };
    if (streaming) {
        state->store = std::make_unique<ExpertStore>(path, expert_options, cancelled_);
        state->store->set_source_offsets(std::move(routed_offsets));
        const llama_model_tensor_buft_override overrides[] = {
            {routed_bank_pattern, state->store->buffer_type()},
            {mapped_bytes ? "^token_embd\\.weight$" : nullptr, state->store->mapped_buffer_type()},
            {nullptr, nullptr}};
        // Metadata mode: no file mapping and no MAP_POPULATE of the 20+ GB artifact.
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;
        model_params.tensor_buft_overrides = overrides;
        TensorLoader loader{state->meta.get(),  state->shapes.get(), std::ifstream(path, std::ios::binary),
                            state->store.get(), cancelled_,          progress};
        loader.total = resident_bytes - expert_options.budget_bytes;
        if (!loader.file) {
            throw std::runtime_error("Cannot read model; check the file provider's access permissions");
        }
        state->model.reset(
            llama_model_init_from_user(state->meta.get(), &TensorLoader::set, &loader, model_params));
        if (cancelled_) {
            throw std::runtime_error("Loading cancelled");
        }
        if (!loader.error.empty()) {
            throw std::runtime_error(loader.error);
        }
    } else {
        state->model.reset(llama_model_load_from_file(path.c_str(), model_params));
    }
    if (!state->model) {
        throw std::runtime_error(cancelled_ ? "Loading cancelled" : "GGUF load failed; see native log");
    }
    const auto arch = metadata(state->model.get(), "general.architecture");
    if (arch != "qwen3" && arch != "qwen35" && arch != "qwen35moe") {
        throw std::runtime_error("Unsupported architecture: " + arch + "; expected Qwen3/Qwen3.5 family");
    }
    if (cancelled_) {
        throw std::runtime_error("Loading cancelled");
    }
    if (state->store) {
        const auto used = metadata(state->model.get(), (arch + ".expert_used_count").c_str());
        const auto epsilon =
            metadata(state->model.get(), (arch + ".attention.layer_norm_rms_epsilon").c_str());
        state->store->finalize(used.empty() ? 0 : std::stoi(used),
                               epsilon.empty() ? 1e-6f : std::stof(epsilon));
    }
    auto params = llama_context_default_params();
    params.n_ctx = context_size;
    params.n_batch = batch_size;
    params.n_ubatch = batch_size;
    params.n_seq_max = 1;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.offload_kqv = false;
    params.op_offload = false;
    params.no_perf = false;
    if (gpu) {
        state->gpu = std::make_unique<GpuOffload>(gpu_options, state->store.get());
        state->gpu_vision_flash_attn = options.value("gpu_vision_flash_attn", false);
    }
    state->abort = {&cancelled_, state->store.get(), state->gpu.get()};
    params.abort_callback = [](void *opaque) {
        const auto *abort = static_cast<const AbortSource *>(opaque);
        return abort->cancelled->load() || (abort->store && abort->store->aborted()) ||
               (abort->gpu && abort->gpu->failed());
    };
    params.abort_callback_data = &state->abort;
    if (mtp) {
        if (llama_model_n_layer_nextn(state->model.get()) <= 0) {
            throw std::runtime_error("Model has no MTP (nextn) layer; use an artifact that includes it");
        }
        // Recurrent-state snapshots let the target roll back rejected draft positions without re-evaluation.
        params.n_rs_seq = static_cast<uint32_t>(draft_max + 1);
    }
    if (state->store) {
        params.cb_eval = &ExpertStore::observe;
        params.cb_eval_user_data = state->store.get();
    }
    state->context.reset(llama_init_from_model(state->model.get(), params));
    if (!state->context) {
        throw std::runtime_error("Context allocation failed; lower context or memory pressure");
    }
    Json pool_info = {{"persistent", false}};
    if (options.value("threadpool", true)) {
        const auto api = threadpool_api();
        if (!api.create || !api.destroy) {
            throw std::runtime_error("CPU backend does not export a threadpool");
        }
        auto pool_params = ggml_threadpool_params_default(threads);
        pool_params.poll = static_cast<uint32_t>(integer(options, "poll", 50, 0, 100));
        const std::string mask = options.value("cpu_mask", std::string());
        if (!mask.empty()) {
            // Hex CPU-id bitmask; all workers share the mask (strict_cpu=false) and the OS places them.
            const unsigned long long bits = std::stoull(mask, nullptr, 16);
            if (!bits) {
                throw std::invalid_argument("cpu_mask selects no CPU");
            }
            for (int cpu = 0; cpu < 64; ++cpu) {
                pool_params.cpumask[cpu] = (bits >> cpu) & 1;
            }
        }
        state->threadpool = {api.create(&pool_params), api.destroy};
        if (!state->threadpool) {
            throw std::runtime_error("Threadpool creation failed");
        }
        llama_attach_threadpool(state->context.get(), state->threadpool.get(), state->threadpool.get());
        state->pool_api = api;
        state->pool_params = pool_params;
        state->pool_threads = threads;
        state->adaptive_threads = mask.empty() && options.value("adaptive_threads", true);
        pool_info = {{"persistent", true},
                     {"adaptive", state->adaptive_threads},
                     {"poll", pool_params.poll},
                     {"cpu_mask", mask.empty() ? Json(nullptr) : Json(mask)}};
    }
    if (mtp) {
        auto draft_params = params;
        draft_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        draft_params.n_rs_seq = 0;
        state->draft.reset(llama_init_from_model(state->model.get(), draft_params));
        if (!state->draft) {
            throw std::runtime_error("MTP draft context allocation failed");
        }
        if (state->threadpool) {
            llama_attach_threadpool(state->draft.get(), state->threadpool.get(), state->threadpool.get());
        }
        // The target exports its next-token hidden state for every row; the head pairs it with the next token.
        llama_set_embeddings_nextn(state->context.get(), true, false);
        llama_set_embeddings_nextn(state->draft.get(), true, true);
        state->draft_batch.reset(llama_batch_ext_init(state->draft.get()));
        state->draft_max = draft_max;
        state->draft_p_min = draft_p_min;
        state->n_embd = static_cast<size_t>(llama_model_n_embd_out(state->model.get()));
        state->pending_h.assign(state->n_embd, 0.0f);
        state->draft_h.assign(state->n_embd, 0.0f);
        if (!state->draft_batch) {
            throw std::runtime_error("MTP batch allocation failed");
        }
    }
    state->batch.reset(llama_batch_ext_init(state->context.get()));
    if (!state->batch) {
        throw std::runtime_error("Batch allocation failed");
    }
    state->context_size = static_cast<int>(llama_n_ctx(state->context.get()));
    state->batch_size = batch_size;
    const auto *tmpl = llama_model_chat_template(state->model.get(), nullptr);
    if (!tmpl) {
        throw std::runtime_error("Missing model chat template");
    }
    state->chat_template = tmpl;
    state->projector_path = projector;
    state->image_max_tokens = image_max_tokens;
    state->threads = threads;
    if (state->gpu) {
        state->gpu->install();
    }
    state->info = {{"backend", state->gpu ? "cpu+vulkan" : "cpu"},
                   {"gpu", state->gpu ? state->gpu->configuration() : Json(nullptr)},
                   {"architecture", arch},
                   {"model_file_bytes", size},
                   {"model_tensor_bytes", llama_model_size(state->model.get())},
                   {"file_type", metadata(state->model.get(), "general.file_type")},
                   {"context", state->context_size},
                   {"tiled_mm", options.contains("tiled_mm") ? options.at("tiled_mm") : Json(nullptr)},
                   {"trained_context", context_cost_estimate.trained > 0 ? Json(context_cost_estimate.trained) : Json(nullptr)},
                   {"kv_bytes_per_token_estimate", context_cost_estimate.kv_bytes_per_token
                                                       ? Json(context_cost_estimate.kv_bytes_per_token) : Json(nullptr)},
                   {"batch", batch_size},
                   {"threads", threads},
                   {"memory_budget_mib", budget},
                   {"memory_budget_kind", "admission_only"},
                   {"mmap", options.value("mmap", true)},
                   {"mapped_tensor_bytes", mapped_bytes},
                   {"expert_cache_bytes",
                    state->store ? state->store->configuration()["effective_budget_bytes"] : Json(nullptr)},
                   {"expert_streaming", state->store ? state->store->configuration() : Json(nullptr)},
                   {"threadpool", pool_info},
                   {"cpu_variant", cpu_variant},
                   {"cpu_features", features_json(cpu_features())},
                   {"mtp", mtp},
                   {"repack", repack},
                   {"speculative", mtp ? Json{{"type", "mtp_self_draft"}, {"draft_max", draft_max},
                                              {"draft_p_min", draft_p_min}, {"sampling", "greedy only"}}
                                       : Json(nullptr)},
                   {"steering", false},
                   {"vision", projector.empty() ? Json(nullptr)
                                                : Json{{"projector_bytes", std::filesystem::file_size(projector)},
                                                       {"image_max_tokens", image_max_tokens},
                                                       {"residency", "per_request"}}},
                   {"load_ms", milliseconds(start)},
                   {"eqt_revision", EQT_REV},
                   {"compiler", EQT_COMPILER},
                   {"llama_revision", "6c59c40076c00eab49754dc955d7652d93f9e125"},
                   {"system_info", llama_print_system_info()},
                   {"memory_after_load", process_memory()}};
    state_ = std::move(state);
    return state_->info;
}

Json Engine::generate(const Json &request, const Emit &emit) {
    if (!state_) {
        throw std::runtime_error("Load a model first");
    }
    auto &s = *state_;
    const auto start = Clock::now();
    const auto memory_before = process_memory();
    const int pool_changes_before = s.pool_changes;
    const ChatInput input = chat_input(request.at("messages"));
    const bool has_media = !input.images.empty();
    if (has_media && s.projector_path.empty()) {
        throw std::invalid_argument("This model was loaded without image support");
    }
    // Bounded by the context window, checked with the prompt length below.
    const int maximum = integer(request, "max_tokens", 128, 1, 1 << 20);
    const int seed = integer(request, "seed", 42, 0, 2147483647);
    const double temperature = request.value("temperature", 0.0);
    if (!std::isfinite(temperature) || temperature < 0 || temperature > 2) {
        throw std::invalid_argument("temperature must be in [0, 2]");
    }
    const bool thinking = request.value("thinking", false);
    const bool ignore_eos = request.value("ignore_eos", false);
    const auto *vocab = llama_model_get_vocab(s.model.get());
    const auto prompt = render_chat(s.chat_template, input.messages, thinking,
                                    piece(vocab, llama_vocab_bos(vocab)), piece(vocab, llama_vocab_eos(vocab)));
    // A text prompt is one token sequence. With images, mtmd splits it into text and image chunks; an image
    // fills its token count of KV cells but advances M-RoPE positions by its own n_pos, so `positions` (the next
    // free position) differs from `count` (cells used).
    std::vector<llama_token> tokens;
    MediaContext media{nullptr, mtmd_free};
    MediaChunks chunks{nullptr, mtmd_input_chunks_free};
    int count = 0;
    llama_pos positions = 0;
    double media_load_ms = 0;
    if (!has_media) {
        count = -llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()), nullptr, 0, true, true);
        if (count > 0 && count + maximum <= s.context_size) {
            tokens.resize(count);
            if (llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()), tokens.data(), count, true,
                               true) != count) {
                throw std::runtime_error("Tokenization failed");
            }
        }
        positions = count;
    } else {
        const auto media_start = Clock::now();
        auto media_params = mtmd_context_params_default();
        media_params.use_gpu = s.gpu != nullptr;
        if (s.gpu && !s.gpu_vision_flash_attn) {
            // Without matrix cores the Vulkan flash-attention shader is the slow path on mobile GPUs (measured on
            // the Xclipse 940 for the decoder); the encoder then uses plain matrix products for attention.
            media_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        media_params.print_timings = false;
        // The encoder spins at barriers like the decoder; it uses at most the CPUs Android allows right now.
        const int allowed = allowed_cpus();
        media_params.n_threads = allowed > 0 ? std::min(s.threads, allowed) : s.threads;
        media_params.warmup = false;
        media_params.image_max_tokens = s.image_max_tokens;
        media.reset(mtmd_init_from_file(s.projector_path.c_str(), s.model.get(), media_params));
        if (!media) {
            throw std::runtime_error("Cannot load the vision projector; see native log");
        }
        std::vector<std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>> bitmaps;
        std::vector<const mtmd_bitmap *> views;
        for (const auto &path : input.images) {
            const auto wrapper =
                mtmd_helper_bitmap_init_from_file(media.get(), path.c_str(), false, mtmd_helper_init_opt_default());
            if (wrapper.video_ctx) {
                mtmd_helper_video_free(wrapper.video_ctx);
            }
            bitmaps.emplace_back(wrapper.bitmap, mtmd_bitmap_free);
            if (!wrapper.bitmap || mtmd_bitmap_is_audio(wrapper.bitmap)) {
                throw std::invalid_argument("Cannot decode image " + std::filesystem::path(path).filename().string());
            }
            views.push_back(wrapper.bitmap);
        }
        chunks.reset(mtmd_input_chunks_init());
        const mtmd_input_text text{prompt.c_str(), prompt.size(), true, true};
        if (mtmd_tokenize(media.get(), chunks.get(), &text, views.data(), views.size()) != 0) {
            throw std::runtime_error("Cannot tokenize the prompt with its images");
        }
        const size_t chunk_count = mtmd_input_chunks_size(chunks.get());
        if (!chunk_count ||
            mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks.get(), chunk_count - 1)) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
            throw std::runtime_error("The prompt must end with text after its images");
        }
        count = static_cast<int>(mtmd_helper_get_n_tokens(chunks.get()));
        positions = mtmd_helper_get_n_pos(chunks.get());
        media_load_ms = milliseconds(media_start);
    }
    if (count <= 0) {
        throw std::invalid_argument("The prompt contains no tokens");
    }
    if (count + maximum > s.context_size) {
        throw ContextOverflowError(count, maximum, s.context_size);
    }
    // Every image is encoded before the prompt is evaluated and the projector is released at once: its weights
    // and encoder buffers are never resident together with the prompt's expert prefetch or during decoding.
    std::vector<std::vector<float>> image_embeddings;
    bool use_mrope = false;
    double encode_ms = 0;
    Json encode_memory = nullptr;
    if (has_media) {
        const auto encode_start = Clock::now();
        use_mrope = mtmd_decode_use_mrope(media.get());
        const size_t width = static_cast<size_t>(llama_model_n_embd_inp(s.model.get()));
        for (size_t c = 0; c < mtmd_input_chunks_size(chunks.get()); ++c) {
            const auto *chunk = mtmd_input_chunks_get(chunks.get(), c);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                continue;
            }
            if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE ||
                mtmd_decode_use_non_causal(media.get(), chunk)) {
                throw std::runtime_error("This projector needs a decoding mode Equity does not support");
            }
            if (cancelled_) {
                break;
            }
            if (mtmd_encode_chunk(media.get(), chunk) != 0) {
                throw std::runtime_error("Image encoding failed; see native log");
            }
            const float *embedding = mtmd_get_output_embd(media.get());
            image_embeddings.emplace_back(embedding, embedding + mtmd_input_chunk_get_n_tokens(chunk) * width);
        }
        // Peak of the image path: projector weights and encoder buffers on top of everything else.
        encode_memory = process_memory();
        media.reset();
        encode_ms = milliseconds(encode_start);
    }
    // Rebuild the entire prefix each request: safe for hybrid recurrent state and cancellation.
    llama_memory_clear(llama_get_memory(s.context.get()), true);
    // The expert cache stays warm across requests; per-request counters reset here and in-flight
    // speculative reads are drained on every exit path.
    struct RequestScope {
        ExpertStore *store;
        explicit RequestScope(ExpertStore *active) : store(active) {
            if (store) {
                store->begin_request();
            }
        }
        ~RequestScope() {
            if (store) {
                store->end_request();
            }
        }
    } request_scope(s.store.get());
    if (s.gpu) {
        s.gpu->begin_request();
    }
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    if (temperature == 0) {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(20));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(static_cast<float>(temperature)));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(seed));
    }
    auto run = [&](llama_context *context, llama_batch_ext *batch) {
        if (cancelled_) {
            return false;
        }
        s.fit_threads();
        const int code = llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch);
        if (code != 0 && !cancelled_) {
            const auto failure = s.store ? s.store->error() : std::string();
            throw std::runtime_error(failure.empty() ? "Decode failed with code " + std::to_string(code)
                                                     : "Expert streaming failed: " + failure);
        }
        return code == 0 && !cancelled_;
    };
    auto add = [](llama_batch_ext *batch, llama_token token, llama_pos pos, bool output) {
        const int idx = llama_batch_ext_add_token(batch, 0, token);
        if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, &pos) ||
            (output && !llama_batch_ext_set_output_logits(batch, idx, true))) {
            throw std::runtime_error("Invalid native batch");
        }
        return idx;
    };
    auto evaluate = [&](const llama_token *ids, int length, int position) {
        llama_batch_ext_clear(s.batch.get());
        for (int i = 0; i < length; ++i) {
            add(s.batch.get(), ids[i], position + i, i == length - 1);
        }
        return run(s.context.get(), s.batch.get());
    };
    // MTP pairing follows llama.cpp draft-mtp: the head sees token p with the target hidden state of
    // position p-1; pending_h carries the last hidden row across batches.
    double draft_ms = 0, mtp_ms = 0;
    auto mtp_process = [&](const llama_token *ids, int length, int position) {
        const auto begin = Clock::now();
        const float *h = llama_get_embeddings_nextn(s.context.get());
        if (!h) {
            throw std::runtime_error("Target produced no MTP hidden states");
        }
        s.verify_h.assign(h, h + length * s.n_embd);
        llama_batch_ext_clear(s.draft_batch.get());
        for (int i = 0; i < length; ++i) {
            const int idx = add(s.draft_batch.get(), ids[i], position + i, false);
            const float *row = i == 0 ? s.pending_h.data() : s.verify_h.data() + (i - 1) * s.n_embd;
            llama_batch_ext_set_embd_token(s.draft_batch.get(), idx, {row, 1, s.n_embd});
        }
        const bool ok = run(s.draft.get(), s.draft_batch.get());
        std::copy(s.verify_h.end() - s.n_embd, s.verify_h.end(), s.pending_h.begin());
        mtp_ms += milliseconds(begin);
        return ok;
    };
    // Greedy autoregressive drafting on the MTP head; its own entries are removed before verification.
    auto draft_tokens = [&](llama_token token, int position, int limit, std::vector<llama_token> &out) {
        const auto begin = Clock::now();
        out.clear();
        std::copy(s.pending_h.begin(), s.pending_h.end(), s.draft_h.begin());
        const int vocab_size = llama_vocab_n_tokens(vocab);
        bool ok = true;
        for (int i = 0; i < limit; ++i) {
            llama_batch_ext_clear(s.draft_batch.get());
            const int idx = add(s.draft_batch.get(), token, position + i, true);
            llama_batch_ext_set_embd_token(s.draft_batch.get(), idx, {s.draft_h.data(), 1, s.n_embd});
            if (!(ok = run(s.draft.get(), s.draft_batch.get()))) {
                break;
            }
            const float *logits = llama_get_logits_ith(s.draft.get(), idx);
            const int best = static_cast<int>(std::max_element(logits, logits + vocab_size) - logits);
            if (s.draft_p_min > 0) {
                double sum = 0;
                for (int v = 0; v < vocab_size; ++v) {
                    sum += std::exp(static_cast<double>(logits[v] - logits[best]));
                }
                if (1.0 / sum < s.draft_p_min) {
                    break;
                }
            }
            out.push_back(best);
            const float *h = llama_get_embeddings_nextn_ith(s.draft.get(), idx);
            std::copy(h, h + s.n_embd, s.draft_h.begin());
            token = best;
        }
        llama_memory_seq_rm(llama_get_memory(s.draft.get()), 0, position, -1);
        draft_ms += milliseconds(begin);
        return ok;
    };
    // The MTP head pairs every prompt token with a target hidden state; image embeddings have no token to pair,
    // so prompts with images decode without drafting.
    const bool use_draft = s.draft && !has_media;
    const bool speculative = use_draft && temperature == 0 && count >= 2;
    if (use_draft) {
        llama_memory_clear(llama_get_memory(s.draft.get()), true);
        std::fill(s.pending_h.begin(), s.pending_h.end(), 0.0f);
    }
    // With MTP the last prompt token is evaluated by the first verification batch, as in llama.cpp.
    const int prefill = speculative ? count - 1 : count;
    const auto prefill_start = Clock::now();
    bool ready = true;
    double image_ms = 0;
    if (!has_media) {
        for (int i = 0; i < prefill && ready; i += s.batch_size) {
            const int length = std::min(s.batch_size, prefill - i);
            ready = evaluate(tokens.data() + i, length, i) && (!use_draft || mtp_process(tokens.data() + i, length, i));
        }
    } else {
        const size_t width = static_cast<size_t>(llama_model_n_embd_inp(s.model.get()));
        ready = !cancelled_ && image_embeddings.size() == input.images.size();
        size_t image = 0;
        llama_pos position = 0;
        std::vector<mtmd_decoder_pos> relative;
        std::vector<llama_pos> rows;
        // GPU mode reads the whole prompt in shared batches of text tokens and image rows: every batch is one pass
        // over the streamed experts, and the short text chunks around an image cost a pass each otherwise.
        struct MixedEntry {
            llama_token token;
            const float *embedding;
            llama_pos positions[4];
        };
        std::vector<MixedEntry> mixed;
        const bool mix = s.gpu != nullptr;
        const auto mixed_start = Clock::now();
        for (size_t c = 0; c < mtmd_input_chunks_size(chunks.get()) && ready; ++c) {
            const auto *chunk = mtmd_input_chunks_get(chunks.get(), c);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t length = 0;
                const llama_token *ids = mtmd_input_chunk_get_tokens_text(chunk, &length);
                if (mix) {
                    for (size_t i = 0; i < length; ++i) {
                        mixed.push_back({ids[i], nullptr, {position + static_cast<llama_pos>(i), 0, 0, 0}});
                    }
                }
                for (size_t i = 0; i < length && ready && !mix; i += s.batch_size) {
                    const int part = static_cast<int>(std::min<size_t>(s.batch_size, length - i));
                    ready = evaluate(ids + i, part, position + static_cast<llama_pos>(i));
                }
                position += static_cast<llama_pos>(length);
                continue;
            }
            const auto image_start = Clock::now();
            const auto &embedding = image_embeddings[image++];
            const int length = static_cast<int>(mtmd_input_chunk_get_n_tokens(chunk));
            // M-RoPE: each embedding row has (t, y, x, z) positions from the image grid; otherwise positions are
            // consecutive. Same layout as mtmd-helper's decode_embd_batch.
            constexpr int sections = 4;
            rows.assign(static_cast<size_t>(length) * sections, 0);
            if (use_mrope) {
                relative.resize(length);
                mtmd_helper_image_get_decoder_pos(mtmd_input_chunk_get_tokens_image(chunk), position, relative.data());
                for (int r = 0; r < length; ++r) {
                    llama_pos *row = rows.data() + static_cast<size_t>(r) * sections;
                    row[0] = relative[r].t;
                    row[1] = relative[r].y;
                    row[2] = relative[r].x;
                    row[3] = relative[r].z;
                }
            } else {
                for (int r = 0; r < length; ++r) {
                    rows[static_cast<size_t>(r) * sections] = position + r;
                }
            }
            if (mix) {
                for (int r = 0; r < length; ++r) {
                    const llama_pos *row = rows.data() + static_cast<size_t>(r) * sections;
                    mixed.push_back({LLAMA_TOKEN_NULL, embedding.data() + r * width, {row[0], row[1], row[2], row[3]}});
                }
            }
            for (int first = 0; first < length && ready && !mix; first += s.batch_size) {
                llama_batch_ext_clear(s.batch.get());
                for (int r = first; r < std::min(length, first + s.batch_size); ++r) {
                    const int idx = llama_batch_ext_add_embd(s.batch.get(), 0, {embedding.data() + r * width, 1, width});
                    if (idx < 0 || !llama_batch_ext_set_pos(s.batch.get(), idx, rows.data() + static_cast<size_t>(r) * sections)) {
                        throw std::runtime_error("Invalid native image batch");
                    }
                }
                ready = run(s.context.get(), s.batch.get());
            }
            position += mtmd_input_chunk_get_n_pos(chunk);
            image_ms += milliseconds(image_start);
        }
        const int entries = static_cast<int>(mixed.size());
        for (int first = 0; first < entries && ready; first += s.batch_size) {
            llama_batch_ext_clear(s.batch.get());
            for (int e = first; e < std::min(entries, first + s.batch_size); ++e) {
                const MixedEntry &entry = mixed[e];
                if (entry.embedding) {
                    const int idx = llama_batch_ext_add_embd(s.batch.get(), 0, {entry.embedding, 1, width});
                    if (idx < 0 || !llama_batch_ext_set_pos(s.batch.get(), idx, entry.positions)) {
                        throw std::runtime_error("Invalid native image batch");
                    }
                } else {
                    add(s.batch.get(), entry.token, entry.positions[0], e == entries - 1);
                }
            }
            ready = run(s.context.get(), s.batch.get());
        }
        if (mix) {
            image_ms = milliseconds(mixed_start);
        }
        if (ready && position != positions) {
            throw std::runtime_error("Image prompt positions disagree with mtmd");
        }
        image_embeddings.clear();
        chunks.reset();
    }
    const double prefill_ms = milliseconds(prefill_start);
    const bool prefill_complete = ready;
    if (s.gpu) {
        // Decoding stays on the CPU: the staging memory goes back to the expert cache's neighbours.
        s.gpu->release_staging();
    }
    if (s.store) {
        s.store->set_decode_phase(true);
    }
    const auto decode_cpu_start = process_cpu();
    const auto decode_start = Clock::now();
    auto last_token = decode_start;
    std::vector<double> latencies;
    std::vector<llama_token> generated;
    generated.reserve(maximum);
    latencies.reserve(maximum);
    const bool capture = request.value("capture_logits", false);
    Json first_logits = nullptr;
    Json ttft = nullptr;
    Json first_visible = nullptr;
    std::string text;
    std::string reason = ready ? "length" : "cancelled";
    // Records one sampled token; false when EOS or the output limit ends generation.
    auto push = [&](llama_token token) {
        if (!ignore_eos && llama_vocab_is_eog(vocab, token)) {
            reason = "eos";
            return false;
        }
        if (generated.empty()) {
            ttft = milliseconds(start);
        }
        const auto chunk = piece(vocab, token);
        if (first_visible.is_null() && chunk.find_first_not_of(" \n\r\t") != std::string::npos) {
            first_visible = milliseconds(start);
        }
        latencies.push_back(milliseconds(last_token));
        last_token = Clock::now();
        text += chunk;
        generated.push_back(token);
        if (emit) {
            emit(chunk);
        }
        return static_cast<int>(generated.size()) < maximum;
    };
    int steps = 0, drafted = 0, accepted_total = 0;
    if (!speculative) {
        if (ready && capture) {
            const float *logits = llama_get_logits_ith(s.context.get(), -1);
            first_logits = std::vector<float>(logits, logits + llama_vocab_n_tokens(vocab));
        }
        for (int i = 0; ready; ++i) {
            if (cancelled_) {
                reason = "cancelled";
                break;
            }
            const auto token = llama_sampler_sample(sampler.get(), s.context.get(), -1);
            if (!push(token)) {
                break;
            }
            if (!evaluate(&token, 1, positions + i)) {
                reason = "cancelled";
                break;
            }
        }
    } else {
        llama_token last = tokens[count - 1];
        int past = count - 1;
        std::vector<llama_token> draft, inputs;
        while (ready) {
            if (cancelled_) {
                reason = "cancelled";
                break;
            }
            const int room = maximum - static_cast<int>(generated.size());
            draft.clear();
            if (room > 1 && !draft_tokens(last, past, std::min(s.draft_max, room - 1), draft)) {
                reason = "cancelled";
                break;
            }
            inputs.assign(1, last);
            inputs.insert(inputs.end(), draft.begin(), draft.end());
            llama_batch_ext_clear(s.batch.get());
            for (size_t i = 0; i < inputs.size(); ++i) {
                add(s.batch.get(), inputs[i], past + static_cast<int>(i), true);
            }
            if (!run(s.context.get(), s.batch.get()) ||
                !mtp_process(inputs.data(), static_cast<int>(inputs.size()), past)) {
                reason = "cancelled";
                break;
            }
            if (capture && first_logits.is_null()) {
                const float *logits = llama_get_logits_ith(s.context.get(), 0);
                first_logits = std::vector<float>(logits, logits + llama_vocab_n_tokens(vocab));
            }
            // Row i holds the target prediction after input i; drafts are kept while they match it.
            int accepted = 0;
            bool more = true;
            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto token = llama_sampler_sample(sampler.get(), s.context.get(), static_cast<int>(i));
                more = push(token);
                last = token;
                if (!more || i == draft.size() || token != draft[i]) {
                    break;
                }
                ++accepted;
            }
            ++steps;
            drafted += static_cast<int>(draft.size());
            accepted_total += accepted;
            std::copy(s.verify_h.begin() + accepted * s.n_embd, s.verify_h.begin() + (accepted + 1) * s.n_embd,
                      s.pending_h.begin());
            past += accepted + 1;
            if (!llama_memory_seq_rm(llama_get_memory(s.context.get()), 0, past, -1)) {
                throw std::runtime_error("Target memory cannot roll back rejected draft tokens");
            }
            llama_memory_seq_rm(llama_get_memory(s.draft.get()), 0, past, -1);
            if (!more) {
                break;
            }
        }
    }
    if (s.store) {
        s.store->set_decode_phase(false);
    }
    const double decode_ms = milliseconds(decode_start);
    const Json decode_cpu = counter_delta(decode_cpu_start, process_cpu());
    const double total_ms = milliseconds(start);
    Json expert_stats = nullptr;
    if (s.store) {
        s.store->end_request();
        expert_stats = s.store->statistics();
    }
    Json result = {
        {"schema_version", 1},
        {"configuration", s.info},
        {"sampling",
         {{"temperature", temperature},
          {"top_k", temperature > 0 ? 20 : 0},
          {"top_p", temperature > 0 ? 0.95 : 1.0},
          {"seed", seed},
          {"thinking", thinking},
          {"ignore_eos", ignore_eos}}},
        {"prompt_tokens", count},
        {"threads_at_end", s.pool_threads},
        {"thread_pool_rebuilds", s.pool_changes - pool_changes_before},
        {"prompt_images", input.images.size()},
        {"image_projector_load_ms", has_media ? Json(media_load_ms) : Json(nullptr)},
        {"image_encode_ms", has_media ? Json(encode_ms) : Json(nullptr)},
        {"image_prefill_ms", has_media ? Json(image_ms) : Json(nullptr)},
        {"memory_after_image_encode", encode_memory},
        {"generated_tokens", generated.size()},
        {"prefill_ms", prefill_ms},
        {"prefill_tokens_per_second", prefill_complete ? Json(count * 1000 / prefill_ms) : Json(nullptr)},
        {"ttft_ms", ttft},
        {"first_visible_piece_ms", first_visible},
        {"first_answer_content_ms", nullptr},
        {"decode_ms", decode_ms},
        {"total_ms", total_ms},
        {"decode_tokens_per_second", generated.size() * 1000.0 / decode_ms},
        {"token_latency_ms", latencies},
        {"token_latency_p50_ms", latencies.empty() ? Json(nullptr) : Json(percentile(latencies, 0.50))},
        {"token_latency_p95_ms", latencies.empty() ? Json(nullptr) : Json(percentile(latencies, 0.95))},
        {"token_latency_p99_ms", latencies.empty() ? Json(nullptr) : Json(percentile(latencies, 0.99))},
        {"termination", reason},
        {"memory_before", memory_before},
        {"memory_after", process_memory()},
        {"decode_process_cpu", decode_cpu},
        {"reasoning_tokens", nullptr},
        {"answer_tokens", nullptr},
        {"expert_cache_hits", expert_stats.is_null() ? Json(nullptr) : expert_stats["hits"]},
        {"expert_cache_misses", expert_stats.is_null() ? Json(nullptr) : expert_stats["misses"]},
        {"expert_io_wait_ms", expert_stats.is_null() ? Json(nullptr) : expert_stats["io_wait_ms"]},
        {"expert_cache", expert_stats},
        {"gpu_offload", s.gpu ? s.gpu->statistics() : Json(nullptr)},
        {"speculative", speculative ? Json{{"verification_steps", steps},
                                           {"drafted_tokens", drafted},
                                           {"accepted_tokens", accepted_total},
                                           {"acceptance_rate", drafted ? Json(double(accepted_total) / drafted) : Json(nullptr)},
                                           {"tokens_per_step", steps ? Json(double(generated.size()) / steps) : Json(nullptr)},
                                           {"draft_ms", draft_ms},
                                           {"mtp_process_ms", mtp_ms}}
                                    : Json(nullptr)},
        {"energy_joules", nullptr},
        {"text", text},
        {"token_ids", generated},
        {"first_logits", first_logits}};
    llama_memory_clear(llama_get_memory(s.context.get()), true);
    return result;
}
} // namespace eqt
