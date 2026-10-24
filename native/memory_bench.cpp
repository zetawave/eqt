// Read-only bandwidth screening. Payload bandwidth is an achieved rate, not a DRAM-controller counter.
#include "weight_memory.h"
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;

int number(const std::string &value, int low, int high) {
    size_t end = 0;
    const int n = std::stoi(value, &end);
    if (end != value.size() || n < low || n > high) {
        throw std::invalid_argument("Argument outside its supported range");
    }
    return n;
}

// Separate accumulators avoid a single dependency chain. Every loaded word affects a checked result.
uint64_t sum(const uint64_t *data, size_t count) {
    size_t i = 0;
    uint64_t result = 0;
#if defined(__aarch64__)
    uint64x2_t a = vdupq_n_u64(0), b = a, c = a, d = a, e = a, f = a, g = a, h = a;
    for (; i + 16 <= count; i += 16) {
        a = vaddq_u64(a, vld1q_u64(data + i));
        b = vaddq_u64(b, vld1q_u64(data + i + 2));
        c = vaddq_u64(c, vld1q_u64(data + i + 4));
        d = vaddq_u64(d, vld1q_u64(data + i + 6));
        e = vaddq_u64(e, vld1q_u64(data + i + 8));
        f = vaddq_u64(f, vld1q_u64(data + i + 10));
        g = vaddq_u64(g, vld1q_u64(data + i + 12));
        h = vaddq_u64(h, vld1q_u64(data + i + 14));
    }
    result = vaddvq_u64(vaddq_u64(vaddq_u64(a, b), vaddq_u64(c, d))) +
             vaddvq_u64(vaddq_u64(vaddq_u64(e, f), vaddq_u64(g, h)));
#else
    uint64_t a = 0, b = 0, c = 0, d = 0;
    for (; i + 4 <= count; i += 4) {
        a += data[i];
        b += data[i + 1];
        c += data[i + 2];
        d += data[i + 3];
    }
    result = a + b + c + d;
#endif
    for (; i < count; ++i) {
        result += data[i];
    }
    return result;
}

#if defined(__aarch64__)
void load_only(const uint64_t *data, size_t count) {
    const uint64_t *end = data + count;
    for (; end - data >= 16; data += 16) {
        // Volatile assembly keeps every read without adding an arithmetic reduction dependency.
        asm volatile("ldp q0, q1, [%0]\n\tldp q2, q3, [%0, #32]\n\t"
                     "ldp q4, q5, [%0, #64]\n\tldp q6, q7, [%0, #96]"
                     :
                     : "r"(data)
                     : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
    }
    for (; data < end; ++data) {
        asm volatile("ldr x9, [%0]" : : "r"(data) : "x9", "memory");
    }
}
#endif

std::string mhz(int cpu) {
    std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/scaling_cur_freq");
    long khz = 0;
    return file >> khz ? std::to_string(khz / 1000) : "null";
}

struct alignas(64) Worker {
    uint64_t checksum = 0, expected = 0;
    size_t words = 0;
    bool affinity_ok = false, cpu_ok = true;
};
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 5 || argc > 7) {
            throw std::invalid_argument("Usage: eqt-memory-bench CPUS|host:THREADS MIB SECONDS PASSES "
                                        "[sum|load|sum-dynamic|load-dynamic] [anonymous|ahwb]");
        }
        const std::string mode = argc >= 6 ? argv[5] : "sum";
        const std::string storage = argc == 7 ? argv[6] : "anonymous";
        if (storage != "anonymous" && storage != "ahwb") {
            throw std::invalid_argument("Weight storage must be anonymous or ahwb");
        }
        const bool dynamic = mode == "sum-dynamic" || mode == "load-dynamic";
        const bool loads = mode == "load" || mode == "load-dynamic";
        if (mode != "sum" && !loads && !dynamic) {
            throw std::invalid_argument("Unknown read mode");
        }
#if !defined(__aarch64__)
        if (loads) {
            throw std::invalid_argument("Load-only mode requires AArch64");
        }
#endif
        std::vector<int> cpus;
        const std::string cpu_arg(argv[1]);
        const bool pinned = cpu_arg.rfind("host:", 0) != 0;
        if (pinned) {
            for (size_t begin = 0; begin < cpu_arg.size();) {
                const auto end = cpu_arg.find(',', begin);
                cpus.push_back(number(cpu_arg.substr(begin, end - begin), 0, 63));
                if (end == std::string::npos) {
                    break;
                }
                begin = end + 1;
                if (begin == cpu_arg.size()) {
                    throw std::invalid_argument("Empty CPU number");
                }
            }
            auto unique = cpus;
            std::sort(unique.begin(), unique.end());
            if (std::adjacent_find(unique.begin(), unique.end()) != unique.end()) {
                throw std::invalid_argument("Duplicate CPU number");
            }
        } else {
            cpus.resize(number(cpu_arg.substr(5), 1, 16), -1);
        }
        if (cpus.empty() || cpus.size() > 16) {
            throw std::invalid_argument("Need 1..16 CPUs");
        }
        const int mib = number(argv[2], 64, 1024), seconds = number(argv[3], 1, 300),
                  passes = number(argv[4], 1, 64);
        eqt::WeightMemory memory(static_cast<size_t>(mib) << 20, storage == "ahwb"
            ? eqt::WeightStorage::hardware_buffer : eqt::WeightStorage::anonymous);
        std::span<uint64_t> buffer(static_cast<uint64_t *>(memory.data()), memory.bytes() / sizeof(uint64_t));
        for (size_t i = 0; i < buffer.size(); ++i) {
            buffer[i] = i % 251 + 1;
        }
        std::vector<Worker> states(cpus.size());
        std::barrier start(static_cast<std::ptrdiff_t>(cpus.size() + 1));
        std::barrier done(static_cast<std::ptrdiff_t>(cpus.size() + 1));
        bool stop = false;
        std::atomic<size_t> next_chunk{0};
        constexpr size_t chunk_words = (1u << 20) / sizeof(uint64_t);
        const size_t chunks = buffer.size() / chunk_words;
        std::vector<std::thread> workers;
        for (size_t t = 0; t < cpus.size(); ++t) {
            // Partition in cache-line units; include the remainder exactly once.
            const size_t begin = (buffer.size() / 8 * t / cpus.size()) * 8;
            const size_t end = (buffer.size() / 8 * (t + 1) / cpus.size()) * 8;
            states[t].expected =
                std::accumulate(buffer.begin() + begin, buffer.begin() + end, uint64_t{0}) * passes;
            try {
                workers.emplace_back([&, t, begin, end] {
                    auto &state = states[t];
                    state.affinity_ok = !pinned;
#if defined(__linux__)
                    if (pinned) {
                        cpu_set_t mask, actual;
                        CPU_ZERO(&mask);
                        CPU_SET(cpus[t], &mask);
                        CPU_ZERO(&actual);
                        state.affinity_ok = sched_setaffinity(0, sizeof(mask), &mask) == 0 &&
                                            sched_getaffinity(0, sizeof(actual), &actual) == 0 &&
                                            CPU_COUNT(&actual) == 1 && CPU_ISSET(cpus[t], &actual);
                    }
#endif
                    const uint64_t initial_checksum = sum(buffer.data() + begin, end - begin) * passes;
                    for (;;) {
                        start.arrive_and_wait();
                        if (stop) {
                            break;
                        }
                        uint64_t checksum = 0;
                        state.words = 0;
                        const auto read_chunk = [&](size_t offset, size_t count) {
#if defined(__aarch64__)
                            if (loads) {
                                load_only(buffer.data() + offset, count);
                            } else
#endif
                            {
                                checksum += sum(buffer.data() + offset, count);
                            }
                            state.words += count;
                            // Prevent loop-invariant hoisting of repeated read passes in optimized builds.
#if defined(__GNUC__) || defined(__clang__)
                            asm volatile("" : : "r"(checksum) : "memory");
#else
                            std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
#if defined(__linux__)
                            state.cpu_ok &= !pinned || sched_getcpu() == cpus[t];
#endif
                        };
                        if (dynamic) {
                            for (;;) {
                                const size_t chunk = next_chunk.fetch_add(1, std::memory_order_relaxed);
                                if (chunk >= chunks * passes) {
                                    break;
                                }
                                read_chunk((chunk % chunks) * chunk_words, chunk_words);
                            }
                        } else {
                            for (int p = 0; p < passes; ++p) {
                                read_chunk(begin, end - begin);
                            }
                        }
                        state.checksum = loads ? initial_checksum : checksum;
                        done.arrive_and_wait();
                    }
                });
            } catch (...) {
                for (size_t missing = workers.size(); missing < cpus.size(); ++missing) {
                    start.arrive_and_drop();
                }
                stop = true;
                start.arrive_and_wait();
                for (auto &worker : workers) {
                    worker.join();
                }
                throw;
            }
        }
        bool valid = true;
        const auto measure = [&] {
            next_chunk.store(0, std::memory_order_relaxed);
            const auto begin = Clock::now();
            start.arrive_and_wait();
            done.arrive_and_wait();
            const double elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
            uint64_t actual = 0, expected = 0;
            size_t words = 0;
            for (const auto &state : states) {
                valid &= state.affinity_ok && state.cpu_ok;
                actual += state.checksum;
                expected += state.expected;
                words += state.words;
            }
            valid &= actual == expected && words == buffer.size() * passes;
            return elapsed;
        };
        measure(); // Untimed warm-up includes waking and pinning all persistent workers.
        std::printf("{\"kind\":\"read_bandwidth_screening\",\"storage\":\"%s\",\"mode\":\"%s\",\"checksum_scope\":\"%s\","
                    "\"pinned\":%s,\"buffer_bytes\":%zu,\"passes\":%d,\"cpus\":[",
                    storage.c_str(), mode.c_str(), loads ? "before_timing" : "every_pass", pinned ? "true" : "false",
                    buffer.size() * 8, passes);
        for (size_t t = 0; t < cpus.size(); ++t) {
            std::printf("%s%d", t ? "," : "", cpus[t]);
        }
        std::printf("],\"clock_sampling\":\"after_window; workers may be asleep\",\"samples\":[");
        const auto begin = Clock::now();
        bool first = true;
        while (valid && std::chrono::duration<double>(Clock::now() - begin).count() < seconds) {
            double elapsed = 0;
            int rounds = 0;
            while (elapsed < 0.5) {
                elapsed += measure();
                ++rounds;
            }
            std::printf("%s{\"t\":%.3f,\"gb_s\":%.3f,\"seconds\":%.6f,\"bytes\":%llu,\"mhz\":[",
                        first ? "" : ",", std::chrono::duration<double>(Clock::now() - begin).count(),
                        double(buffer.size() * 8) * passes * rounds / elapsed / 1e9, elapsed,
                        static_cast<unsigned long long>(buffer.size() * 8 * passes * rounds));
            for (size_t t = 0; t < cpus.size(); ++t) {
                std::printf("%s%s", t ? "," : "", mhz(cpus[t]).c_str());
            }
            std::printf("]}");
            std::fflush(stdout);
            first = false;
        }
        stop = true;
        start.arrive_and_wait();
        for (auto &worker : workers) {
            worker.join();
        }
        const auto residency = eqt::sample_residency(memory.data(), memory.bytes());
        std::printf(
            "],\"checksum\":%llu,\"checksum_and_affinity_valid\":%s,\"dram_controller_bytes\":null,"
            "\"sampled_pages\":%llu,\"resident_pages\":%s}\n",
            static_cast<unsigned long long>(states[0].checksum), valid ? "true" : "false",
            static_cast<unsigned long long>(residency.sampled_pages),
            residency.available ? std::to_string(residency.resident_pages).c_str() : "null");
        return valid ? 0 : 2;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
