#include "engine.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "json.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>
#ifdef __ANDROID__
#include <android/log.h>
#endif
#ifndef _WIN32
#include <sys/resource.h>
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
double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>(std::ceil(q * values.size())) - 1];
}
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

struct Engine::State {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch{nullptr, llama_batch_ext_free};
    Json info;
    int context_size = 0;
    int batch_size = 0;
    std::string chat_template;
};

Engine::Engine() {
    static std::once_flag init;
    std::call_once(init, [] {
        llama_log_set(
            [](ggml_log_level level, const char *text, void *) {
                if (level == GGML_LOG_LEVEL_DEBUG) {
                    return;
                }
#ifdef __ANDROID__
                __android_log_write(level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO,
                                    "Equity", text);
#else
            std::fputs(text, stderr);
#endif
            },
            nullptr);
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
    const int context_size = integer(options, "context", 2048, 128, 8192);
    const int batch_size = integer(options, "batch", 128, 1, 512);
    const int threads = integer(options, "threads", 4, 1, 16);
    const int budget = integer(options, "memory_budget_mib", 2048, 768, 8192);
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
    // Admission is conservative, not an allocator-enforced RSS limit or an expert cache.
    if (size > static_cast<uint64_t>(budget - 512) * 1024 * 1024) {
        throw std::runtime_error("Model exceeds resident admission budget (512 MiB reserved). Bounded expert "
                                 "streaming is not implemented.");
    }
    auto state = std::make_unique<State>();
    struct Progress {
        std::atomic_bool &cancelled;
        const Emit &emit;
        int last = -1;
    } progress_state{cancelled_, progress};
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.load_mode = options.value("mmap", true) ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    model_params.lazy_mode = LLAMA_LAZY_MODE_OFF;
    model_params.load_mtp = false;
    model_params.use_extra_bufts = false;
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
    state->model.reset(llama_model_load_from_file(path.c_str(), model_params));
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
    params.abort_callback = [](void *opaque) { return static_cast<std::atomic_bool *>(opaque)->load(); };
    params.abort_callback_data = &cancelled_;
    state->context.reset(llama_init_from_model(state->model.get(), params));
    if (!state->context) {
        throw std::runtime_error("Context allocation failed; lower context or memory pressure");
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
    state->info = {{"backend", "cpu"},
                   {"architecture", arch},
                   {"model_file_bytes", size},
                   {"model_tensor_bytes", llama_model_size(state->model.get())},
                   {"file_type", metadata(state->model.get(), "general.file_type")},
                   {"context", state->context_size},
                   {"batch", batch_size},
                   {"threads", threads},
                   {"memory_budget_mib", budget},
                   {"memory_budget_kind", "admission_only"},
                   {"mmap", options.value("mmap", true)},
                   {"expert_cache_bytes", nullptr},
                   {"mtp", false},
                   {"steering", false},
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
    const auto &messages = request.at("messages");
    if (!messages.is_array() || messages.empty() || messages.size() > 128) {
        throw std::invalid_argument("messages must contain 1..128 text messages");
    }
    for (const auto &message : messages) {
        const auto role = message.at("role").get<std::string>();
        if (role != "system" && role != "user" && role != "assistant") {
            throw std::invalid_argument("Only text system/user/assistant roles are supported");
        }
        if (message.at("content").get_ref<const std::string &>().size() > 1024 * 1024) {
            throw std::invalid_argument("Message exceeds 1 MiB");
        }
    }
    const int maximum = integer(request, "max_tokens", 128, 1, 2048);
    const int seed = integer(request, "seed", 42, 0, 2147483647);
    const double temperature = request.value("temperature", 0.0);
    if (!std::isfinite(temperature) || temperature < 0 || temperature > 2) {
        throw std::invalid_argument("temperature must be in [0, 2]");
    }
    const bool thinking = request.value("thinking", false);
    const bool ignore_eos = request.value("ignore_eos", false);
    const auto *vocab = llama_model_get_vocab(s.model.get());
    const auto prompt = render_chat(s.chat_template, messages, thinking, piece(vocab, llama_vocab_bos(vocab)),
                                    piece(vocab, llama_vocab_eos(vocab)));
    const int count =
        -llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()), nullptr, 0, true, true);
    if (count <= 0 || count + maximum > s.context_size) {
        throw std::invalid_argument(
            "Prompt plus output exceeds context; start a new chat or lower output limit");
    }
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()), tokens.data(), count, true,
                       true) != count) {
        throw std::runtime_error("Tokenization failed");
    }
    // Rebuild the entire prefix each request: safe for hybrid recurrent state and cancellation.
    llama_memory_clear(llama_get_memory(s.context.get()), true);
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
    auto evaluate = [&](const llama_token *ids, int length, int position) {
        if (cancelled_) {
            return false;
        }
        llama_batch_ext_clear(s.batch.get());
        for (int i = 0; i < length; ++i) {
            const int idx = llama_batch_ext_add_token(s.batch.get(), 0, ids[i]);
            const llama_pos pos = position + i;
            if (idx < 0 || !llama_batch_ext_set_pos(s.batch.get(), idx, &pos)) {
                throw std::runtime_error("Invalid native batch");
            }
        }
        llama_batch_ext_set_output_logits(s.batch.get(), length - 1, true);
        const int code = llama_process(s.context.get(), LLAMA_PROCESS_TYPE_DECODE, s.batch.get());
        if (code != 0 && !cancelled_) {
            throw std::runtime_error("Decode failed with code " + std::to_string(code));
        }
        return code == 0 && !cancelled_;
    };
    const auto prefill_start = Clock::now();
    bool ready = true;
    for (int i = 0; i < count && ready; i += s.batch_size) {
        ready = evaluate(tokens.data() + i, std::min(s.batch_size, count - i), i);
    }
    const double prefill_ms = milliseconds(prefill_start);
    const bool prefill_complete = ready;
    const auto decode_start = Clock::now();
    auto last_token = decode_start;
    std::vector<double> latencies;
    std::vector<llama_token> generated;
    generated.reserve(maximum);
    latencies.reserve(maximum);
    Json first_logits = nullptr;
    if (ready && request.value("capture_logits", false)) {
        const float *logits = llama_get_logits_ith(s.context.get(), -1);
        first_logits = std::vector<float>(logits, logits + llama_vocab_n_tokens(vocab));
    }
    Json ttft = nullptr;
    Json first_visible = nullptr;
    std::string text;
    std::string reason = ready ? "length" : "cancelled";
    for (int i = 0; ready && i < maximum; ++i) {
        if (cancelled_) {
            reason = "cancelled";
            break;
        }
        const auto token = llama_sampler_sample(sampler.get(), s.context.get(), -1);
        if (!ignore_eos && llama_vocab_is_eog(vocab, token)) {
            reason = "eos";
            break;
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
        if (i + 1 < maximum && !evaluate(&token, 1, count + i)) {
            reason = "cancelled";
            break;
        }
    }
    const double decode_ms = milliseconds(decode_start);
    const double total_ms = milliseconds(start);
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
        {"reasoning_tokens", nullptr},
        {"answer_tokens", nullptr},
        {"expert_cache_hits", nullptr},
        {"expert_cache_misses", nullptr},
        {"expert_io_wait_ms", nullptr},
        {"energy_joules", nullptr},
        {"text", text},
        {"token_ids", generated},
        {"first_logits", first_logits}};
    llama_memory_clear(llama_get_memory(s.context.get()), true);
    return result;
}
} // namespace eqt
