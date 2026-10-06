#include "engine.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct File {
    int fd = -1;
    ~File() {
        if (fd >= 0)
            close(fd);
    }
};
double quantile(std::vector<double> values, double probability) {
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>(std::ceil(probability * values.size())) - 1];
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: eqt-io-bench WEIGHT_FILE REQUEST.json RESULT.json\n";
        return 2;
    }
    try {
        std::ifstream input(argv[2]);
        const auto request = eqt::Json::parse(input);
        const auto read_size = request.value("read_bytes", 2 * 1024 * 1024);
        const int count = request.value("reads", 128);
        const bool direct = request.value("direct", false);
        const unsigned seed = request.value("seed", 42u);
        if (read_size < 4096 || read_size > 16 * 1024 * 1024 || read_size % 4096 || count < 1 ||
            count > 4096) {
            throw std::invalid_argument("Reads must be 4 KiB aligned, 4 KiB..16 MiB; count 1..4096");
        }
        File file;
        file.fd = open(argv[1], O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
        if (file.fd < 0) {
            throw std::runtime_error(std::string("open failed (no silent fallback): ") +
                                     std::strerror(errno));
        }
        struct stat stat{};
        if (fstat(file.fd, &stat) || !S_ISREG(stat.st_mode) || stat.st_size < read_size) {
            throw std::runtime_error("Expected a regular file at least as large as one read");
        }
        void *allocation = nullptr;
        if (posix_memalign(&allocation, 4096, read_size) != 0) {
            throw std::bad_alloc();
        }
        std::unique_ptr<void, decltype(&std::free)> buffer(allocation, std::free);
        std::mt19937 random(seed);
        std::uniform_int_distribution<int64_t> blocks(0, (stat.st_size - read_size) / 4096);
        std::vector<int64_t> offsets(count);
        for (auto &offset : offsets)
            offset = blocks(random) * 4096;
        std::vector<double> latencies;
        latencies.reserve(count);
        int partial_reads = 0;
        uint64_t check = 0;
        const auto before = eqt::process_memory();
        const auto start = Clock::now();
        for (const auto offset : offsets) {
            size_t done = 0;
            const auto read_start = Clock::now();
            while (done < static_cast<size_t>(read_size)) {
                const auto got =
                    pread(file.fd, static_cast<char *>(buffer.get()) + done, read_size - done, offset + done);
                if (got < 0 && errno == EINTR)
                    continue;
                if (got <= 0) {
                    throw std::runtime_error(got == 0 ? "Unexpected EOF" : std::strerror(errno));
                }
                done += got;
                if (done < static_cast<size_t>(read_size)) {
                    ++partial_reads;
                    // Direct reads must retain alignment after a short read; reject unaligned recovery.
                    if (direct && done % 4096)
                        throw std::runtime_error("Unaligned short direct read");
                }
            }
            latencies.push_back(elapsed(read_start));
            const auto *data = static_cast<const unsigned char *>(buffer.get());
            check += data[0] + data[read_size - 1];
        }
        const auto total_ms = elapsed(start);
        const uint64_t total_bytes = static_cast<uint64_t>(read_size) * count;
        eqt::Json result = {
            {"schema_version", 1},
            {"kind", "random_read_microbenchmark"},
            {"request", request},
            {"file_bytes", stat.st_size},
            {"read_bytes", read_size},
            {"reads", count},
            {"seed", seed},
            {"direct_requested", direct},
            {"direct_flag_accepted", (fcntl(file.fd, F_GETFL) & O_DIRECT) != 0},
            {"cache_state", "uncontrolled; buffered reads may hit page cache"},
            {"total_bytes", total_bytes},
            {"total_ms", total_ms},
            {"bytes_per_second", total_bytes * 1000.0 / total_ms},
            {"read_p50_ms", quantile(latencies, 0.50)},
            {"read_p95_ms", quantile(latencies, 0.95)},
            {"read_p99_ms", quantile(latencies, 0.99)},
            {"read_latency_ms", latencies},
            {"offsets", offsets},
            {"partial_reads", partial_reads},
            {"edge_byte_checksum", check},
            {"memory_before", before},
            {"memory_after", eqt::process_memory()},
            {"scope", "Read-only CLI proxy; not expert traces, app-private storage or model throughput"}};
        std::ofstream output(argv[3]);
        output << result.dump(2) << '\n';
        output.close();
        if (!output)
            throw std::runtime_error("Cannot write result");
        std::cout << result["bytes_per_second"] << " bytes/s\n";
    } catch (const std::exception &error) {
        std::cerr << "eqt-io: " << error.what() << '\n';
        return 1;
    }
}
