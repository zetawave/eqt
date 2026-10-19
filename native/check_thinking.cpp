#include "engine.h"
#include <fstream>
#include <iostream>

namespace {
void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}

constexpr std::string_view think_end = "</think>";

// Qwen's recommended thinking temperature with a fixed seed: deterministic, and greedy reasoning can loop.
eqt::Json ask(eqt::Engine &engine, bool thinking, int max_tokens, const eqt::Emit &emit = {}) {
    const eqt::Json request = {
        {"messages", eqt::Json::array({{{"role", "user"}, {"content", "What is 17 + 25? Reply with the number."}}})},
        {"max_tokens", max_tokens},
        {"temperature", thinking ? 0.6 : 0.0},
        {"seed", 42},
        {"thinking", thinking}};
    return engine.generate(request, emit);
}

// Reasoning fields must agree with the text: tokens through the closing tag are reasoning, the rest is the answer.
void check_split(const eqt::Json &result) {
    const std::string text = result.at("text");
    const int generated = result.at("generated_tokens");
    const int reasoning = result.at("reasoning_tokens");
    const bool complete = result.at("reasoning_complete");
    require(reasoning + result.at("answer_tokens").get<int>() == generated, "Reasoning and answer tokens do not add up");
    require(complete == (text.find(think_end) != std::string::npos), "reasoning_complete disagrees with the text");
    require(result.at("reasoning_ms").get<double>() <= result.at("decode_ms").get<double>(), "Reasoning outlasted decoding");
    if (!complete) {
        require(reasoning == generated && result.at("first_answer_content_ms").is_null(),
                "Unfinished reasoning counted answer tokens");
    }
}
} // namespace

// Usage: eqt-thinking-check MODEL.gguf RESULT.json. Uses a model whose template has `enable_thinking`.
int main(int argc, char **argv) {
    try {
        require(argc == 3, "Usage: eqt-thinking-check MODEL.gguf RESULT.json");
        eqt::Engine engine;
        const auto info = engine.load(argv[1], {{"context", 2048}, {"batch", 64}, {"threads", 4}});
        require(info.at("thinking_supported") == true, "Fixture template has no thinking mode");

        const auto plain = ask(engine, false, 32);
        require(plain.at("reasoning_tokens").is_null() && plain.at("reasoning_ms").is_null() &&
                    plain.at("text").get<std::string>().find(think_end) == std::string::npos,
                "Thinking off reported reasoning");

        const auto full = ask(engine, true, 1536);
        check_split(full);
        require(full.at("reasoning_complete") == true && full.at("answer_tokens").get<int>() > 0 &&
                    !full.at("first_answer_content_ms").is_null(),
                "Reasoning did not finish into an answer within 1,536 tokens; it ended with: " +
                    full.at("text").get<std::string>().substr(full.at("text").get<std::string>().size() -
                                                              std::min<size_t>(300, full.at("text").get<std::string>().size())));
        const std::string text = full.at("text");
        require(text.find("42", text.find(think_end)) != std::string::npos, "The answer after reasoning lacks 42");
        require(ask(engine, true, 1536).at("text") == text, "Seeded thinking output changed between runs");

        const auto limited = ask(engine, true, 4);
        check_split(limited);
        require(limited.at("reasoning_complete") == false, "Four tokens closed the reasoning");

        int seen = 0;
        const auto stopped = ask(engine, true, 1536, [&](const std::string &) {
            if (++seen == 3)
                engine.cancel();
        });
        check_split(stopped);
        require(stopped.at("termination") == "cancelled" && stopped.at("reasoning_complete") == false,
                "Cancelled reasoning was reported complete");
        engine.prepare();
        require(ask(engine, true, 1536).at("text") == text, "Cancellation changed the next thinking answer");

        const eqt::Json result = {{"passed", true},
                                  {"reasoning_tokens", full.at("reasoning_tokens")},
                                  {"answer_tokens", full.at("answer_tokens")},
                                  {"reasoning_ms", full.at("reasoning_ms")},
                                  {"first_answer_content_ms", full.at("first_answer_content_ms")}};
        std::ofstream(argv[2]) << result.dump(2) << '\n';
        std::cout << result.dump(2) << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
