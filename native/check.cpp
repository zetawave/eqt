#include "engine.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
eqt::Json compare(const std::string &name, const eqt::Json &reference, const eqt::Json &actual) {
    const auto &expected = reference.at("first_logits");
    const auto &observed = actual.at("first_logits");
    require(expected.is_array() && !expected.empty() && expected.size() == observed.size(),
            name + ": invalid logit dimensions");
    double maximum = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double a = expected[i].get<double>(), b = observed[i].get<double>();
        require(std::isfinite(a) && std::isfinite(b), name + ": non-finite logits");
        maximum = std::max(maximum, std::abs(a - b));
    }
    const bool match = reference.at("token_ids") == actual.at("token_ids");
    require(match && maximum <= 0.002, name + ": output changed after state reset");
    return {{"case", name}, {"max_abs_logit_delta", maximum}, {"token_ids_equal", match}, {"passed", true}};
}
// Distance between two first-token distributions: largest logit difference, KL(p_reference || p_actual), and
// agreement of the top-1 and top-10 tokens.
eqt::Json divergence(const eqt::Json &reference, const eqt::Json &actual) {
    const auto a = reference.get<std::vector<double>>(), b = actual.get<std::vector<double>>();
    require(a.size() == b.size() && !a.empty(), "invalid logit dimensions");
    const auto log_softmax = [](const std::vector<double> &x) {
        const double peak = *std::max_element(x.begin(), x.end());
        double sum = 0;
        for (const double v : x) {
            sum += std::exp(v - peak);
        }
        std::vector<double> out(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            out[i] = x[i] - peak - std::log(sum);
        }
        return out;
    };
    const auto la = log_softmax(a), lb = log_softmax(b);
    double kl = 0, maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        kl += std::exp(la[i]) * (la[i] - lb[i]);
        maximum = std::max(maximum, std::abs(a[i] - b[i]));
    }
    const auto top = [](const std::vector<double> &x, size_t k) {
        std::vector<size_t> order(x.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](size_t l, size_t r) { return x[l] > x[r]; });
        order.resize(k);
        return order;
    };
    const auto ta = top(a, 10), tb = top(b, 10);
    size_t shared = 0;
    for (const size_t t : ta) {
        shared += std::find(tb.begin(), tb.end(), t) != tb.end();
    }
    return {{"max_abs_logit_delta", maximum}, {"kl_divergence", kl}, {"top1_equal", ta[0] == tb[0]}, {"top10_shared", shared}};
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: eqt-check MODEL.gguf REQUEST.json RESULT.json\n";
        return 2;
    }
    try {
        std::ifstream input(argv[2]);
        auto request = eqt::Json::parse(input);
        request["capture_logits"] = true;
        request["max_tokens"] = 24;
        request["temperature"] = 0;
        request["ignore_eos"] = false;
        const auto options = request.value("load", eqt::Json::object());
        eqt::Engine engine;
        engine.prepare();
        const auto configuration = engine.load(argv[1], options);
        const auto reference = engine.generate(request);
        require(reference.at("generated_tokens").get<int>() > 0, "Empty fixture generation");
        auto outcomes = eqt::Json::array();
        outcomes.push_back(compare("repeat_request", reference, engine.generate(request)));

        auto conversation = request;
        conversation["messages"].push_back({{"role", "assistant"}, {"content", reference.at("text")}});
        conversation["messages"].push_back(
            {{"role", "user"}, {"content", "Summarize that answer in English."}});
        const auto conversation_reference = engine.generate(conversation);
        require(conversation_reference.at("generated_tokens").get<int>() > 0, "Empty conversation response");
        outcomes.push_back(compare("new_chat_after_conversation", reference, engine.generate(request)));
        outcomes.push_back(
            compare("conversation_prefix_replay", conversation_reference, engine.generate(conversation)));
        // The conversation extends the request, so its prompt reuses the request's state; rebuilt from scratch it
        // is the same tokens evaluated in other batch shapes, which may differ by float rounding only.
        auto rebuilt_conversation = conversation;
        rebuilt_conversation["prompt_cache"] = false;
        const auto rebuilt = engine.generate(rebuilt_conversation);
        auto distance = divergence(rebuilt.at("first_logits"), conversation_reference.at("first_logits"));
        require(rebuilt.at("reused_prompt_tokens") == 0, "A request without the prompt cache reused state");
        require(std::isfinite(distance.at("kl_divergence").get<double>()) &&
                    rebuilt.at("token_ids") == conversation_reference.at("token_ids"),
                "prompt_cache_rebuild: tokens differ from the reused-state evaluation");
        distance.update({{"case", "prompt_cache_rebuild"},
                         {"reused_prompt_tokens", conversation_reference.at("reused_prompt_tokens")},
                         {"token_ids_equal", true},
                         {"passed", true}});
        outcomes.push_back(distance);

        auto long_request = request;
        long_request["ignore_eos"] = true;
        int pieces = 0;
        const auto interrupted = engine.generate(long_request, [&](const std::string &) {
            if (++pieces == 3) {
                engine.cancel();
            }
        });
        require(interrupted.at("termination") == "cancelled" && interrupted.at("generated_tokens") == 3,
                "Cancellation did not stop at the requested decode boundary");
        engine.prepare();
        outcomes.push_back(compare("recovery_after_decode_cancel", reference, engine.generate(request)));

        engine.cancel();
        const auto pre_cancelled = engine.generate(request);
        require(pre_cancelled.at("termination") == "cancelled" && pre_cancelled.at("generated_tokens") == 0,
                "Cancellation before prefill emitted tokens");
        engine.prepare();
        outcomes.push_back(compare("recovery_after_pre_cancel", reference, engine.generate(request)));

        auto sampled = request;
        sampled["temperature"] = 0.7;
        sampled["seed"] = 42;
        const auto sample_reference = engine.generate(sampled);
        outcomes.push_back(compare("seeded_sampling_repeat", sample_reference, engine.generate(sampled)));
        engine.unload();
        engine.prepare();
        engine.load(argv[1], options);
        outcomes.push_back(compare("reload", reference, engine.generate(request)));

        // Saved sessions. A turn is saved only when the next turn will reuse its state; saved to disk and restored
        // into a freshly loaded context it continues exactly as the state kept in memory. A session of another model
        // is kept and not read; one the prompt does not extend is deleted, and the prompt is rebuilt. The
        // random-weight fixtures' answers do not re-tokenize to their tokens: their state is written anyway (force)
        // to exercise those paths.
        const std::string session_path = std::string(argv[3]) + ".session";
        std::filesystem::remove(session_path);
        std::filesystem::remove(session_path + ".json");
        const auto with_model = [&](char digit) {
            auto loaded = options;
            loaded["model_sha256"] = std::string(64, digit);
            engine.unload();
            engine.prepare();
            engine.load(argv[1], loaded);
        };
        auto first_turn = request;
        first_turn["session"] = {{"path", session_path}};
        auto next_turn = conversation;
        next_turn["session"] = {{"path", session_path}, {"save", false}};
        with_model('a');
        auto saved = engine.generate(first_turn);
        const bool extendable = saved.at("session").contains("saved_bytes");
        if (!extendable) {
            require(saved.at("session").at("save_skipped") == "not_extendable" && !std::filesystem::exists(session_path),
                    "session_save: a skipped state was written");
            first_turn["session"]["force"] = true;
            saved = engine.generate(first_turn);
        }
        require(saved.at("session").contains("saved_bytes") && std::filesystem::exists(session_path),
                "session_save: no session was written");
        with_model('b');
        const auto other = engine.generate(next_turn);
        require(other.at("session").at("status") == "other_model" && other.at("reused_prompt_tokens") == 0 &&
                    std::filesystem::exists(session_path),
                "session_other_model: a session of another model was read or deleted");
        with_model('a');
        engine.generate(request);
        const auto in_memory = engine.generate(conversation);
        with_model('a');
        const auto restored = engine.generate(next_turn);
        const auto reused = in_memory.at("reused_prompt_tokens").get<size_t>();
        require(extendable == (reused > 0), "session_save: the saved state's reuse was mispredicted");
        if (reused > 0) {
            require(restored.at("session").at("status") == "restored" && restored.at("reused_prompt_tokens") == reused &&
                        restored.at("session").at("restored_tokens") == reused,
                    "session_restore: the saved state was not used");
            require(restored.at("first_logits") == in_memory.at("first_logits") &&
                        restored.at("token_ids") == in_memory.at("token_ids"),
                    "session_restore: output differs from the in-memory state");
        } else {
            require(restored.at("session").at("status") == "mismatch" && !std::filesystem::exists(session_path) &&
                        restored.at("first_logits") == rebuilt.at("first_logits") &&
                        restored.at("token_ids") == rebuilt.at("token_ids"),
                    "session_mismatch: an unusable session was kept or the rebuild changed");
        }
        outcomes.push_back({{"case", "session_save_restore"},
                            {"saved_bytes", saved.at("session").at("saved_bytes")},
                            {"extendable", extendable},
                            {"restore_status", restored.at("session").at("status")},
                            {"reused_prompt_tokens", restored.at("reused_prompt_tokens")},
                            {"bitwise_equal", true},
                            {"passed", true}});
        std::filesystem::remove(session_path);
        std::filesystem::remove(session_path + ".json");

        std::ofstream output(argv[3]);
        output << eqt::Json({{"configuration", configuration},
                             {"tolerance_absolute", 0.002},
                             {"outcomes", outcomes},
                             {"scope", "State reset, full-prefix replay, prompt-cache reuse against a rebuild, saved sessions, "
                                       "cancellation recovery and reload; no rollback"}})
                      .dump(2)
               << '\n';
        output.close();
        require(static_cast<bool>(output), "Cannot write check result");
        std::cout << "PASS: " << outcomes.size() << " state checks\n";
    } catch (const std::exception &error) {
        std::cerr << "eqt-check: " << error.what() << '\n';
        return 1;
    }
}
