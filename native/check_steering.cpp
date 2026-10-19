#include "engine.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "hash/hash.h"
#include "steering.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

namespace {
void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
std::string hash_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(input), {}};
    return hash_sha256_hex(bytes.data(), bytes.size());
}
double delta(const eqt::Json &a, const eqt::Json &b) {
    const auto &x = a.at("first_logits"), &y = b.at("first_logits");
    require(!x.empty() && x.size() == y.size(), "Missing logits");
    double maximum = 0;
    for (size_t i = 0; i < x.size(); ++i)
        maximum = std::max(maximum, std::abs(x[i].get<double>() - y[i].get<double>()));
    return maximum;
}
void same(const eqt::Json &reference, const eqt::Json &actual) {
    require(delta(reference, actual) == 0 && reference.at("token_ids") == actual.at("token_ids"),
            "Steering OFF did not recover the exact baseline");
}

struct Capture {
    std::vector<std::vector<float>> blocks;
    int embedding = 0;
};

bool capture_blocks(ggml_tensor *tensor, bool ask, void *opaque) {
    auto &capture = *static_cast<Capture *>(opaque);
    // Block outputs are l_out, or l_last in hyper-connection graphs, where each row of n_embd is one stream. The
    // name must match exactly: views of a block output carry suffixes such as " (reshaped)".
    int block = -1, end = 0;
    const char *name = ggml_get_name(tensor);
    const bool named = (std::sscanf(name, "l_out-%d%n", &block, &end) == 1 && name[end] == '\0') ||
                       (std::sscanf(name, "l_last-%d%n", &block, &end) == 1 && name[end] == '\0');
    if (!named || block < 0 || block >= static_cast<int>(capture.blocks.size()))
        return !ask;
    if (ask)
        return true;
    require(tensor->type == GGML_TYPE_F32 && tensor->ne[0] == capture.embedding && ggml_is_contiguous(tensor),
            "Unexpected block output layout");
    capture.blocks[block].resize(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, capture.blocks[block].data(), 0, ggml_nbytes(tensor));
    return true;
}

// Largest |d.h| / |h| over the token rows of one captured block output.
double relative_projection(const std::vector<float> &rows, const std::vector<float> &unit) {
    double worst = 0;
    for (size_t row = 0; row < rows.size(); row += unit.size()) {
        double projection = 0, squared = 0;
        for (size_t i = 0; i < unit.size(); ++i) {
            projection += double(rows[row + i]) * unit[i];
            squared += double(rows[row + i]) * rows[row + i];
        }
        worst = std::max(worst, std::abs(projection) / std::sqrt(squared));
    }
    return worst;
}

// Checks the graph-level ablation against block outputs captured from the same prompt.
eqt::Json check_ablation(const std::string &model_path) {
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.load_mtp = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path.c_str(), model_params), llama_model_free);
    require(model != nullptr, "Cannot load the ablation fixture");
    Capture capture;
    capture.embedding = llama_model_n_embd(model.get());
    const int blocks = llama_model_n_layer(model.get());
    capture.blocks.resize(blocks);
    auto params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = params.n_ubatch = 32;
    params.n_threads = params.n_threads_batch = 2;
    params.cb_eval = capture_blocks;
    params.cb_eval_user_data = &capture;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), params),
                                                                  llama_free);
    require(context != nullptr, "Cannot create the ablation context");
    std::vector<llama_token> tokens(12);
    const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    for (size_t i = 0; i < tokens.size(); ++i)
        tokens[i] = static_cast<llama_token>((i * 7919 + 11) % vocabulary);
    auto run = [&] {
        llama_memory_clear(llama_get_memory(context.get()), true);
        require(llama_decode(context.get(), llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()))) == 0,
                "Ablation fixture decode failed");
        llama_synchronize(context.get());
        return capture.blocks;
    };
    const auto reference = run();
    // A direction aligned with real activations: the unit mean of block 0's rows.
    std::vector<float> unit(capture.embedding, 0.0f);
    for (size_t row = 0; row < reference[0].size(); row += unit.size())
        for (size_t i = 0; i < unit.size(); ++i)
            unit[i] += reference[0][row + i];
    double norm = 0;
    for (float value : unit)
        norm += double(value) * value;
    for (float &value : unit)
        value = static_cast<float>(value / std::sqrt(norm));
    require(relative_projection(reference[0], unit) > 0.1, "Fixture direction does not overlap activations");

    require(llama_set_adapter_cvec_ablation(context.get(), unit.data(), unit.size(), capture.embedding, 0.5f, 0, 0) == 0,
            "Cannot set partial ablation");
    const auto half = run();
    double half_error = 0;
    for (size_t row = 0; row < half[0].size(); row += unit.size()) {
        double before = 0, after = 0;
        for (size_t i = 0; i < unit.size(); ++i) {
            before += double(reference[0][row + i]) * unit[i];
            after += double(half[0][row + i]) * unit[i];
        }
        half_error = std::max(half_error, std::abs(after - 0.5 * before) / std::max(1e-6, std::abs(before)));
    }
    require(half_error < 1e-4, "Partial ablation did not scale the block 0 projection by 0.5");

    std::vector<float> rows;
    for (int block = 0; block < blocks; ++block)
        rows.insert(rows.end(), unit.begin(), unit.end());
    require(llama_set_adapter_cvec_ablation(context.get(), rows.data(), rows.size(), capture.embedding, 1.0f, 0,
                                            blocks - 1) == 0,
            "Cannot set full ablation");
    const auto removed = run();
    double residual = 0;
    for (const auto &block : removed)
        residual = std::max(residual, relative_projection(block, unit));
    require(residual < 1e-5, "Full ablation left a component along the direction");

    require(llama_set_adapter_cvec_ablation(context.get(), nullptr, 0, 0, 0.0f, -1, -1) == 0, "Cannot clear ablation");
    const auto restored = run();
    for (int block = 0; block < blocks; ++block)
        require(restored[block] == reference[block], "Clearing the ablation did not restore block outputs bitwise");
    return {{"blocks", blocks}, {"partial_relative_error", half_error}, {"full_max_relative_projection", residual},
            {"off_recovery_bitwise", true}};
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--inspect") {
            std::ifstream input(argv[3]);
            std::cout << eqt::read_steering(argv[2], eqt::Json::parse(input)).metadata.dump(2) << '\n';
            return 0;
        }
        require(argc >= 3 && argc <= 5, "Usage: eqt-steering-check MODEL.gguf WORK_DIR [--stream] [--mtp]");
        bool stream = false, mtp = false;
        for (int i = 3; i < argc; ++i) {
            const std::string option = argv[i];
            require(option == "--stream" || option == "--mtp", "Unknown check option");
            stream |= option == "--stream";
            mtp |= option == "--mtp";
        }
        const auto folder = std::filesystem::path(argv[2]);
        std::filesystem::create_directories(folder);
        eqt::Engine engine;
        eqt::Json options = {{"context", 512},
                             {"batch", 32},
                             {"threads", 2},
                             {"memory_budget_mib", 2048},
                             {"model_sha256", std::string(64, 'a')}};
        if (stream)
            options.update({{"expert_streaming", true}, {"expert_cache_mib", 32}, {"prefetch", true}});
        if (mtp)
            options.update({{"mtp", true}, {"draft_max", 2}});
        const auto ablation = check_ablation(argv[1]);
        const auto info = engine.load(argv[1], options);
        const int embedding = info.at("embedding_size");
        const int layer = std::min(20, info.at("decoder_blocks").get<int>() - 1);
        // Single-block fixture, as the app's synthetic package, and one direction per block 1..blocks-1.
        auto write_vector = [&](const std::string &name, int first, int last) {
            std::vector<std::vector<float>> rows;
            std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(
                ggml_init({size_t(last - first + 1) * ggml_tensor_overhead(), nullptr, true}), ggml_free);
            std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(gguf_init_empty(), gguf_free);
            gguf_set_val_str(gguf.get(), "general.architecture", "controlvector");
            gguf_set_val_str(gguf.get(), "controlvector.model_hint",
                             info.at("architecture").get<std::string>().c_str());
            for (int block = first; block <= last; ++block) {
                auto &values = rows.emplace_back(embedding);
                for (int i = 0; i < embedding; ++i)
                    values[i] = float((i + block) % 13 - 6) * 0.05f;
                auto *tensor = ggml_new_tensor_1d(tensors.get(), GGML_TYPE_F32, embedding);
                ggml_format_name(tensor, "direction.%d", block);
                tensor->data = values.data();
                gguf_add_tensor(gguf.get(), tensor);
            }
            const auto path = (folder / name).string();
            require(gguf_write_to_file(gguf.get(), path.c_str(), false), "Cannot write fixture");
            eqt::Json manifest = {{"model_sha256", std::string(64, 'a')},
                                  {"vector_sha256", hash_file(path)},
                                  {"architecture", info.at("architecture")},
                                  {"layer_index", layer},
                                  {"embedding_size", embedding},
                                  {"tensor_name", "direction." + std::to_string(layer)},
                                  {"normalized", false}};
            if (last > first)
                manifest["layer_count"] = last;
            std::ofstream(folder / (name + ".json")) << manifest.dump(2) << '\n';
            return std::pair{path, manifest};
        };
        const auto [path, manifest] = write_vector("test-vector.gguf", layer, layer);
        const auto [layered_path, layered_manifest] =
            write_vector("test-vector-blocks.gguf", 1, info.at("decoder_blocks").get<int>() - 1);
        eqt::Json request = {
            {"messages", eqt::Json::array({{{"role", "user"}, {"content", "Explain a cache briefly."}}})},
            {"max_tokens", 4},
            {"temperature", 0},
            {"ignore_eos", true},
            {"capture_logits", true}};
        const auto baseline = engine.generate(request);
        auto enabled = request;
        enabled["model_sha256"] = std::string(64, 'a');
        enabled["steering"] = {{"path", path}, {"metadata", manifest}, {"strength", 1.0}};
        const auto positive = engine.generate(enabled);
        require(!mtp || (!baseline.at("speculative").is_null() && positive.at("speculative").is_null()),
                "MTP did not fall back to ordinary decoding under steering");
        require(delta(baseline, positive) > 1e-7 && !positive.at("steering").is_null(),
                "Steering had no numerical effect");
        same(baseline, engine.generate(request));
        enabled["steering"]["strength"] = -1.0;
        const auto negative = engine.generate(enabled);
        require(delta(positive, negative) > 1e-7 && negative.at("steering").at("mode") == "ablation" &&
                    positive.at("steering").at("mode") == "addition",
                "Steering sign had no numerical effect");
        same(baseline, engine.generate(request));
        auto layered = enabled;
        layered["steering"] = {{"path", layered_path}, {"metadata", layered_manifest}, {"strength", 1.0}};
        const auto layered_positive = engine.generate(layered);
        require(layered_positive.at("steering").at("layer_count") == info.at("decoder_blocks").get<int>() - 1 &&
                    delta(baseline, layered_positive) > 1e-7,
                "Per-block vector had no numerical effect");
        layered["steering"]["strength"] = -1.0;
        const auto layered_negative = engine.generate(layered);
        require(delta(negative, layered_negative) > 1e-7, "Per-block ablation matched the single direction");
        same(baseline, engine.generate(request));
        int rejected = 0;
        auto reject = [&](eqt::Json invalid) {
            bool failed = false;
            try {
                engine.generate(invalid);
            } catch (const std::exception &) {
                failed = true;
            }
            require(failed, "Invalid vector request was accepted");
            same(baseline, engine.generate(request));
            ++rejected;
        };
        for (double strength : {0.0, -1.5, 2.5, std::numeric_limits<double>::quiet_NaN()}) {
            auto invalid = enabled;
            invalid["steering"]["strength"] = strength;
            reject(invalid);
        }
        auto invalid = enabled;
        invalid["steering"]["metadata"]["vector_sha256"] = std::string(64, '0');
        reject(invalid);
        invalid = enabled;
        invalid["model_sha256"] = std::string(64, 'b');
        reject(invalid);
        invalid = enabled;
        invalid["steering"]["metadata"]["embedding_size"] = embedding + 1;
        reject(invalid);
        invalid = layered;
        invalid["steering"]["metadata"]["layer_count"] = info.at("decoder_blocks").get<int>() - 2;
        reject(invalid);
        auto cancelled = engine.generate(enabled, [&](const std::string &) { engine.cancel(); });
        require(cancelled.at("termination") == "cancelled", "Steered generation did not cancel");
        engine.prepare();
        same(baseline, engine.generate(request));
        engine.unload();
        auto unbound = options;
        unbound.erase("model_sha256");
        engine.load(argv[1], unbound);
        reject(enabled);
        engine.unload();
        engine.load(argv[1], options);
        same(baseline, engine.generate(request));
        const eqt::Json result = {{"passed", true},
                                  {"streaming", stream},
                                  {"mtp_fallback_checked", mtp},
                                  {"rejected_invalid_requests", rejected},
                                  {"positive_max_logit_delta", delta(baseline, positive)},
                                  {"sign_max_logit_delta", delta(positive, negative)},
                                  {"per_block_ablation_max_logit_delta", delta(negative, layered_negative)},
                                  {"graph_ablation", ablation},
                                  {"off_recovery_bitwise", true},
                                  {"cancellation_recovery", true},
                                  {"reload_recovery", true}};
        std::ofstream(folder / "result.json") << result.dump(2) << '\n';
        std::cout << result.dump(2) << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
