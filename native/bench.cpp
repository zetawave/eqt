#include "engine.h"
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
        eqt::Engine engine;
        engine.prepare();
        engine.load(argv[1], request.value("load", eqt::Json::object()));
        auto result =
            engine.generate(request, [](const std::string &piece) { std::cout << piece << std::flush; });
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
