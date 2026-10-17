#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace eqt {
using Json = nlohmann::ordered_json;
using Emit = std::function<void(const std::string &)>;

class ContextOverflowError : public std::invalid_argument {
  public:
    ContextOverflowError(int prompt, int output, int capacity)
        : std::invalid_argument("Prompt plus output exceeds context: the message needs " + std::to_string(prompt) +
                                " input tokens and " + std::to_string(output) + " output tokens, but the loaded context holds " +
                                std::to_string(capacity) + ". Increase the context or use a smaller attachment."),
          prompt_tokens(prompt), output_tokens(output), context_tokens(capacity) {}
    const int prompt_tokens;
    const int output_tokens;
    const int context_tokens;
};

class Engine {
  public:
    Engine();
    ~Engine();
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    void prepare();
    void cancel();
    Json load(const std::string &path, const Json &options, const Emit &progress = {});
    Json generate(const Json &request, const Emit &emit = {});
    void unload();

  private:
    struct State;
    std::unique_ptr<State> state_;
    std::atomic_bool cancelled_{false};
};

Json process_memory();
std::string render_chat(const std::string &source, const Json &messages, bool thinking,
                        const std::string &bos, const std::string &eos);
} // namespace eqt
