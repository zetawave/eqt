#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace eqt {
using Json = nlohmann::ordered_json;
using Emit = std::function<void(const std::string &)>;

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
