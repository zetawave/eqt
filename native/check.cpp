#include "engine.h"

#include <algorithm>
#include <cmath>
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

        std::ofstream output(argv[3]);
        output << eqt::Json({{"configuration", configuration},
                             {"tolerance_absolute", 0.002},
                             {"outcomes", outcomes},
                             {"scope", "State reset, full-prefix replay, cancellation recovery and reload; "
                                       "no prefix cache or rollback"}})
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
