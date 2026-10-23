#include "engine.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: eqt-bench MODEL.gguf REQUEST.json RESULT.json\n";
        return 2;
    }
    try {
        std::ifstream input(argv[2]);
        if (!input) {
            throw std::runtime_error("Cannot open request");
        }
        const auto request = eqt::Json::parse(input);
        const double sustained_seconds = request.value("sustained_seconds", 0.0);
        if (!std::isfinite(sustained_seconds) || sustained_seconds < 0 || sustained_seconds > 86400) {
            throw std::invalid_argument("sustained_seconds must be in 0..86400");
        }
        if (sustained_seconds > 0) {
            if (!request.contains("sequence") || !request.at("sequence").is_array() || request.at("sequence").empty()) {
                throw std::invalid_argument("Sustained runs require a nonempty sequence");
            }
            for (const auto &step : request.at("sequence")) {
                if (step.value("ignore_eos", false)) {
                    throw std::invalid_argument("Sustained runs require natural EOS");
                }
                if (step.value("max_tokens", 0) <= 0) {
                    throw std::invalid_argument("Sustained requests require a positive max_tokens safety limit");
                }
            }
        }
        eqt::Engine engine;
        engine.prepare();
        engine.load(argv[1], request.value("load", eqt::Json::object()));
        const auto emit = [](const std::string &piece) { std::cout << piece << std::flush; };
        // "sequence": requests run one after another on the same loaded model; the result holds them in order.
        eqt::Json result;
        if (request.contains("sequence")) {
            result = {{"results", eqt::Json::array()}};
            const auto start = std::chrono::steady_clock::now();
            double elapsed = 0;
            size_t index = 0;
            do {
                for (const auto &step : request.at("sequence")) {
                    auto entry = engine.generate(step, emit);
                    entry["sequence_index"] = index++ % request.at("sequence").size();
                    elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                    entry["sequence_elapsed_seconds"] = elapsed;
                    result["results"].push_back(std::move(entry));
                    std::cout << '\n';
                    if (sustained_seconds > 0 && elapsed >= sustained_seconds) {
                        break;
                    }
                }
            } while (sustained_seconds > 0 && elapsed < sustained_seconds);
            result["sequence_wall_seconds"] = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            result["sustained_seconds_requested"] = sustained_seconds;
            result["configuration"] = result["results"].empty() ? eqt::Json(nullptr)
                : result["results"].front().at("configuration");
        } else {
            result = engine.generate(request, emit);
        }
        std::ofstream output(argv[3]);
        if (!output) {
            throw std::runtime_error("Cannot open result path");
        }
        output << result.dump(2, ' ', false, eqt::Json::error_handler_t::replace) << '\n';
        output.close();
        if (!output) {
            throw std::runtime_error("Result write failed");
        }
        std::cout << '\n';
    } catch (const std::exception &error) {
        std::cerr << "eqt: " << error.what() << '\n';
        return 1;
    }
}
