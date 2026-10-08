#pragma once

#include "engine.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct ggml_tensor;
struct ggml_backend_buffer_type;

namespace eqt {
struct ExpertOptions {
    uint64_t budget_bytes = 0;
    std::vector<double> layer_weights; // relative per-layer cache shares; empty means uniform
    int io_threads = 4;
    bool direct_io = false;
    bool prefetch = false;
    int prefetch_extra = 0; // predicted candidates prefetched beyond the routed top-k, per row
    // Demand misses of adjacent experts merge into one read per bank of at most coalesce_kib (0: one read
    // per expert and bank), bridging gaps of at most coalesce_gap resident experts.
    int coalesce_kib = 0;
    int coalesce_gap = 1;
    // Transient allowance above the cache for reading the next layer during batched (prompt) evaluation;
    // 0 disables. Decode and small speculative batches use the router lookahead instead.
    int prefill_prefetch_mib = 0;
    // Experts live in persistent, reused page-aligned slots that GGML's MUL_MAT_ID resolves through
    // ggml_cpu_set_expert_address, instead of in a GGUF-shaped reserved range released on eviction: reloads
    // take no page faults or zero-fill, and direct reads land whole in the slot.
    bool slots = false;
    std::string trace_path;
};

class SourceFile;

// Exact, bounded routed-expert cache. Without slots, expert banks keep their GGUF tensor layout inside a
// reserved virtual range and only selected (layer, expert) slices are populated; with slots, each resident
// expert occupies a reused slot that MUL_MAT_ID addresses through a GGML hook. Either way, selected experts are
// resident before GGML's MUL_MAT_ID reads them. Routing is never changed: a miss always blocks until the exact
// bytes are resident.
class ExpertStore {
  public:
    ExpertStore(const std::string &path, ExpertOptions options, const std::atomic_bool &cancelled);
    ~ExpertStore();
    ExpertStore(const ExpertStore &) = delete;
    ExpertStore &operator=(const ExpertStore &) = delete;

    ggml_backend_buffer_type *buffer_type();
    bool owns(const ggml_tensor *tensor) const;
    // Load phase (single thread): register routed banks and optional F32 router tensors.
    // File offsets of routed banks, known before allocation so banks can be placed page-congruently.
    void set_source_offsets(std::unordered_map<std::string, uint64_t> offsets);
    uint64_t source_offset(const char *name) const;
    void add_bank(ggml_tensor *tensor, uint64_t file_offset);
    void add_router_tensor(ggml_tensor *tensor);
    void finalize(int experts_used, float norm_epsilon);
    // Inference phase: GGML scheduler evaluation callback, invoked on the decode thread only.
    static bool observe(ggml_tensor *tensor, bool ask, void *store);
    void begin_request();
    // Decode-phase evaluations (single tokens, speculative verification, MTP drafting) feed decode_* counters.
    void set_decode_phase(bool decode);
    void end_request();
    Json configuration() const;
    Json statistics() const;
    std::string error() const;

  private:
    enum class State : uint8_t { absent, loading, resident };
    struct Bank {
        uint8_t *data;
        size_t stride;
        uint64_t file_offset;
        const ggml_tensor *tensor;
        size_t slot_offset = 0; // this bank's region inside an expert slot
    };
    struct Entry {
        State state = State::absent;
        bool prefetched = false;
        uint8_t *slot = nullptr; // slot mode: page-aligned base of this expert's banks
        uint64_t stamp = 0;
        std::atomic<int> pending{0};
        std::atomic<bool> failed{false};
    };
    struct Layer {
        std::vector<Bank> banks;
        int experts = 0;
        int quota = 0;
        int resident = 0;
        uint64_t entry_bytes = 0;
        size_t slot_bytes = 0; // slot mode: all banks of one expert, each page-rounded plus one page
        int slot_class = -1;
        std::unique_ptr<Entry[]> entries;
        std::vector<uint32_t> seen;
        const float *router = nullptr; // [experts][width], F32
        const float *norm = nullptr;   // [width], F32
        int width = 0;
        int router_rows = 0;
        std::vector<int> prediction;
        bool prediction_valid = false;
    };
    // One contiguous read of experts [first, first + count) of one bank; bit i of `members` marks expert
    // first + i as loaded by this read (other experts in the range are resident and re-read unchanged).
    struct Job {
        Layer *layer;
        int bank;
        int first;
        int count;
        uint64_t members;
    };
    struct Stats {
        uint64_t ensure_calls = 0, demanded = 0, hits = 0, prefetch_hits = 0, inflight_hits = 0, misses = 0;
        uint64_t demand_read_bytes = 0, prefetch_read_bytes = 0, prefetch_issued = 0, prefetch_wasted = 0;
        uint64_t evictions = 0, overflow_events = 0, predicted = 0, predicted_correct = 0;
        uint64_t decode_demanded = 0, decode_misses = 0, decode_read_bytes = 0;
        uint64_t read_jobs = 0, coalesced_reads = 0, bridged_experts = 0, slot_cold_reuses = 0;
        double io_wait_ms = 0, decode_io_wait_ms = 0, predict_ms = 0, evict_ms = 0;
        uint64_t peak_resident_bytes = 0;
    };

    // Slots of one size; warm ones keep their pages, cold ones were released and fault once when reused.
    struct SlotClass {
        size_t bytes;
        std::vector<uint8_t *> warm, cold;
    };
    struct BankRef {
        const ggml_tensor *tensor;
        Layer *layer;
        int bank;
    };

    static const void *expert_address(const ggml_tensor *weights, int64_t expert, void *store);
    uint8_t *data_of(const Layer &layer, int bank, int expert) const;
    bool assign_slot(Layer &layer, int expert);
    void release_slot(Layer &layer, int expert);
    void read_slots(const Job &job) const;
    void setup_slots();
    bool ensure(int layer_index, const ggml_tensor *ids);
    void predict(int layer_index, const ggml_tensor *hidden);
    void issue_prefetch(Layer &layer);
    void start_load(Layer &layer, int expert, bool demand);
    bool begin_load(Layer &layer, int expert, bool demand);
    void queue_reads(Layer &layer, int first, int count, uint64_t members, bool demand);
    void load_runs(Layer &layer, std::vector<int> &experts, bool demand, int max_gap);
    void prefetch_whole_layer(Layer &layer);
    bool reap(Layer &layer, int expert);
    void evict(Layer &layer, int expert);
    int least_recent(const Layer &layer, uint32_t epoch) const;
    void trim(Layer &layer);
    bool wait(Layer &layer, int expert);
    void worker();
    void fail(const std::string &message);
    Layer *layer_at(int index) const;

    ExpertOptions options_;
    const std::atomic_bool &cancelled_;
    std::unique_ptr<SourceFile> file_;
    std::unique_ptr<ggml_backend_buffer_type> buffer_type_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::vector<std::thread> workers_;
    std::deque<Job> demand_queue_, prefetch_queue_;
    mutable std::mutex mutex_;
    std::condition_variable queue_ready_, job_done_;
    std::atomic<int> inflight_jobs_{0};
    bool stopping_ = false;
    bool decode_phase_ = false;
    std::string error_;
    std::ofstream trace_;
    Stats stats_;
    uint64_t clock_ = 0, resident_bytes_ = 0, routed_bytes_ = 0, effective_budget_ = 0;
    uint32_t epoch_ = 0;
    uint64_t requests_ = 0;
    int experts_used_ = 0, overflow_layer_ = -1, prefetch_layer_ = -1, predictable_layers_ = 0;
    float norm_epsilon_ = 1e-6f;
    size_t max_stride_ = 0;
    std::vector<int> demand_, missing_, order_, whole_layer_;
    std::vector<float> scratch_, scores_;
    std::unordered_map<std::string, uint64_t> source_offsets_;
    std::vector<const ggml_tensor *> residuals_; // per-layer lookahead input of the current graph
    uint8_t *arena_ = nullptr;
    size_t arena_size_ = 0, arena_used_ = 0;
    std::vector<SlotClass> slot_classes_;
    std::vector<BankRef> bank_index_; // sorted by tensor; read-only once finalized
    void (*set_expert_address_)(const void *(*)(const ggml_tensor *, int64_t, void *), void *) = nullptr;
};
} // namespace eqt
