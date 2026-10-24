#pragma once

#include "engine.h"

#include <atomic>
#include <chrono>
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
    // I/O threads that never take speculative reads, so a demanded expert starts reading at once instead of
    // queueing behind prefetches (measured: 0.57 ms average queueing with all threads shared).
    int demand_threads = 4;
    // Of those, threads that stay awake while decoding (waiting on the queue counter with WFE), pinned to the
    // lowest-capacity CPUs: a sleeping thread took ~0.45 ms to start a demanded read (measured). Off until the
    // change itself is measured on the device.
    int awake_demand_threads = 0;
    bool direct_io = false;
    bool prefetch = false;
    int prefetch_extra = 0; // predicted candidates prefetched beyond the routed top-k, per row
    // Only the first `prefetch_count` predicted candidates of each row are read ahead (0: all of them). The ranks
    // below are still predicted and scored, so precision by rank stays comparable across settings.
    int prefetch_count = 0;
    // The prompt-time next-layer allowance (prefill_prefetch_mib) is cache while decoding: memory the plan reserved
    // anyway, returned to prompts by the borrowing in ensure().
    bool decode_uses_prompt_allowance = true;
    // How long a compute thread spins on a pending expert bank before it sleeps on the bank's word.
    int bank_spin_us = 20000;
    // Lookahead reads go to a pool beside the layer's quota instead of evicting cached experts: a predicted expert
    // joins the cache only when the router selects it, and the rest are dropped once their layer has run. The pool
    // holds at most one prediction per layer (about 10 experts) for the one or two layers ahead.
    bool speculative_pool = false;
    // With 2, the lookahead also scores the layer after next from the same residual (without the mixer's gate,
    // so the predictor thread keeps pace) and starts reading its `lookahead2_count` best candidates a layer
    // earlier; the next layer's own prediction then adds the rest.
    int lookahead_layers = 1;
    int lookahead2_count = 5;
    // Reads are served demand first, then the next layer's predictions, then the layer after's. A queued
    // speculative read moves to the demand queue when its layer routes to the expert, and is dropped unread when
    // its layer has run without it; reads of the layer after that the next prediction confirms move up a queue.
    bool read_priority = false;
    // While decoding, at most this many speculative reads in flight (0: no limit), so a demanded read queues
    // behind fewer of them inside the storage device.
    int prefetch_inflight = 0;
    // The lookahead scores routers from a Q8_0 copy instead of the F32 tensor: about a quarter of the bytes for
    // the predictor thread, which usually runs on a small core. Only speculative reads depend on it.
    bool lookahead_router_q8 = false;
    // Hyper-connection lookahead: apply the mixer's low-rank gate (two Q8_0 GEMVs per layer) or average the normed
    // streams without it, which costs only the router product.
    bool lookahead_gate = true;
    // Late lookahead (hyper-connection models): at layer l's output, before layer l+1's attention, layer l+1's
    // routing is scored again from the exact streams. 1 records its precision only (scored on the decode thread,
    // for measurement); lookahead_late_gate applies the mixer's gate as the early lookahead does.
    int lookahead_late = 0;
    bool lookahead_late_gate = true;
    // Threads that score a lookahead together (1: the predictor alone). The others are helpers pinned to the
    // lowest-capacity CPUs; each projection is split by rows, in groups of four so every row is computed the same
    // way whatever the thread count. Helpers spin only while a lookahead runs.
    int lookahead_threads = 1;
    // While decoding, the predictor polls for the next lookahead instead of sleeping on a condition variable.
    bool lookahead_spin = false;
    // Demand misses of adjacent experts merge into one read per bank of at most coalesce_kib (0: one read
    // per expert and bank), bridging gaps of at most coalesce_gap resident experts.
    int coalesce_kib = 0;
    int coalesce_gap = 1;
    // A decoded token's top-k is handled inside the graph (the CPU backend's node hook, compute_hook) instead of
    // at a graph split per layer: compute thread 0 sorts the router's single row itself, exactly as the backend
    // would, then requests the layer's experts and starts the lookahead before the node ends. A failure stops the
    // graph at that node's end, before anything reads an expert.
    bool topk_in_graph = false;
    // Transient allowance above the cache for reading the next layer during batched (prompt) evaluation;
    // 0 disables. Decode and small speculative batches use the router lookahead instead.
    int prefill_prefetch_mib = 0;
    // The next layer is read ahead only when a batch of that many rows is expected to use at least this share of
    // its experts, 1 - (1 - k/E)^rows for top-k of E (with 512 experts, 52 prompt tokens use about 64%); below it
    // the reads are left to demand. 0 always reads ahead.
    double prefill_prefetch_min_coverage = 0;
    // Experts live in persistent, reused page-aligned slots that GGML's MUL_MAT_ID resolves through
    // ggml_cpu_set_expert_address, instead of in a GGUF-shaped reserved range released on eviction: reloads
    // take no page faults or zero-fill, and direct reads land whole in the slot.
    bool slots = false;
    // With slots: decode does not wait for missing experts before MUL_MAT_ID (see overlap_reads_).
    bool overlap_reads = true;
    // Expert pack (expert_pack.h): routed experts are read from this file, one contiguous page-aligned read per
    // expert, instead of one read per bank from the GGUF. Requires slots.
    std::string pack_path;
    // With a pack, a demanded single expert is read as this many concurrent parts (whole banks, split by size);
    // speculative reads always read whole experts.
    int pack_split = 1;
    std::string trace_path;
    // Diagnostic only: recorded native routes supply future reads; routing still comes from the model.
    std::string oracle_path;
    int oracle_layers = 2;
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
    // Tensors read a few rows per token (the token embedding, the PLE n-gram table) are mapped from the file
    // instead of copied into anonymous memory: their pages are loaded on use and reclaimable, which leaves that
    // RAM to the expert cache. Mapped tensors need no loading.
    ggml_backend_buffer_type *mapped_buffer_type();
    bool maps(const ggml_tensor *tensor) const;
    // Maps a tensor's file range at its (page-congruent) address; called from the buffer's init_tensor.
    bool map_tensor(ggml_tensor *tensor) const;
    // Load phase (single thread): register routed banks and optional F32 router tensors.
    // File offsets of routed banks, known before allocation so banks can be placed page-congruently.
    void set_source_offsets(std::unordered_map<std::string, uint64_t> offsets);
    uint64_t source_offset(const char *name) const;
    void add_bank(ggml_tensor *tensor, uint64_t file_offset);
    void add_router_tensor(ggml_tensor *tensor);
    void finalize(int experts_used, float norm_epsilon);
    // Inference phase: GGML scheduler evaluation callback, invoked on the decode thread only.
    static bool observe(ggml_tensor *tensor, bool ask, void *store);
    // CPU backend node hook (topk_in_graph): computes a pending top-k's argsort on compute thread 0 and runs the
    // layer's store work; true when it computed the node. Enabled once the engine has installed it.
    static bool compute_hook(ggml_tensor *node, int ith, void *store);
    void enable_topk_in_graph() { topk_in_graph_ = options_.topk_in_graph; }
    void begin_request();
    // Decode-phase evaluations (single tokens, speculative verification, MTP drafting) feed decode_* counters.
    void set_decode_phase(bool decode);
    // More cache while decoding: compute-buffer memory a prompt needs and decoding leaves untouched (the engine
    // releases those pages after the prompt and measures what decoding uses). Applies to the current decode; the
    // next prompt gets the memory back when decoding ends, as set_decode_phase(false) trims to the prompt quota
    // and returns the freed slots' pages at once.
    void set_decode_extra(uint64_t bytes);
    void end_request();
    // Between requests: evicts every cached expert and returns their pages, so another user of the memory (a vision
    // encoder) fits beside the shared weights. Exact: the next prompt reads its experts again. Returns the bytes freed.
    uint64_t release_cache();
    Json configuration() const;
    Json statistics() const;
    std::string error() const;
    // True after a failed read was observed by a compute thread; the engine's abort callback stops the graph.
    bool aborted() const { return aborted_.load(); }
    bool diagnostic_oracle() const { return !oracle_routes_.empty(); }
    // Address of resident expert `expert` of a streamed bank, waiting for its read if needed; nullptr when the
    // tensor is not a streamed bank. For copying experts elsewhere (GPU staging) from inside a graph compute.
    const void *expert_data(const ggml_tensor *bank, int64_t expert) const {
        return expert_address(bank, expert, const_cast<ExpertStore *>(this));
    }

  private:
    enum class State : uint8_t { absent, loading, resident };
    struct Bank {
        uint8_t *data;
        size_t stride;
        uint64_t file_offset;
        const ggml_tensor *tensor;
        size_t slot_offset = 0; // this bank's region inside an expert slot
        int component = 3; // gate, up, down, other
    };
    struct Entry {
        State state = State::absent;
        bool prefetched = false;
        bool pooled = false; // a lookahead read held beside the quota (speculative_pool)
        bool early = false;  // issued from the two-layer lookahead, not yet confirmed by the next layer's
        uint8_t *slot = nullptr; // slot mode: page-aligned base of this expert's banks
        uint64_t stamp = 0;
        std::atomic<int> pending{0};
        std::atomic<bool> failed{false};
        // Bit b set once bank b's bytes are in the slot (or its read failed); compute threads wait per bank.
        std::atomic<uint32_t> ready{0};
    };
    struct Layer {
        std::vector<Bank> banks;
        int experts = 0;
        int quota = 0;
        int prompt_quota = 0, decode_quota = 0; // quota during prompts and while decoding
        double share = 0;                       // this layer's fraction of the budget
        int pooled = 0;                         // entries held in the speculative pool
        int resident = 0;
        uint64_t entry_bytes = 0;
        size_t slot_bytes = 0; // slot mode: all banks of one expert, each page-rounded plus one page
        int slot_class = -1;
        std::unique_ptr<Entry[]> entries;
        std::vector<uint32_t> seen;
        const float *router = nullptr; // [experts][width], F32
        std::vector<uint8_t> router_q8; // lookahead_router_q8: the router in GGML Q8_0 blocks, one row per expert
        const float *norm = nullptr;   // [width], F32
        int width = 0;
        int router_rows = 0;
        // Hyper-connection models (qwen4exp) route the FFN mixer's output instead of a normed residual: the mean of
        // `streams` residual streams, each RMS-normed with its own weights and gated elementwise through a
        // low-rank projection. The lookahead recomputes it with these weights (Q8_0 or F32, as stored).
        const float *mixer_norm = nullptr;       // [streams][width], F32
        const ggml_tensor *mixer_down = nullptr; // streams * width -> rank
        const ggml_tensor *mixer_up = nullptr;   // rank -> streams * width
        int streams = 0;
        std::vector<int> prediction;
        std::vector<uint8_t> prediction_rank; // best rank of each predicted expert over the batch rows
        bool prediction_valid = false;
        std::vector<int> early_prediction; // two layers ahead (lookahead_layers = 2)
        std::vector<uint8_t> early_rank;
        bool early_valid = false;
        std::vector<int> late_prediction; // from the previous layer's output (lookahead_late)
        std::vector<uint8_t> late_rank;
        bool late_valid = false;
        uint64_t pack_base = 0; // pack offset of expert 0; experts are slot_bytes apart
    };
    // One contiguous read of experts [first, first + count) of banks [bank, bank + banks); bit i of `members`
    // marks expert first + i as loaded by this read (other experts in the range are resident and re-read
    // unchanged). Without a pack a job covers one bank; with one, a job covers whole experts, or for a single
    // expert a run of adjacent banks.
    struct Job {
        Layer *layer;
        int bank;
        int first;
        int count;
        uint64_t members;
        std::chrono::steady_clock::time_point queued;
        int banks = 1;
        bool decode = false; // submission phase, retained through promotion and final draining
    };
    struct Stats {
        uint64_t ensure_calls = 0, demanded = 0, hits = 0, prefetch_hits = 0, inflight_hits = 0, misses = 0;
        uint64_t demand_read_bytes = 0, prefetch_read_bytes = 0, prefetch_issued = 0, prefetch_wasted = 0;
        uint64_t evictions = 0, overflow_events = 0, borrowed_evictions = 0, predicted = 0, predicted_correct = 0;
        uint64_t decode_demanded = 0, decode_misses = 0, decode_read_bytes = 0;
        uint64_t decode_hits = 0, decode_prefetch_hits = 0, decode_inflight_hits = 0;
        uint64_t read_jobs = 0, coalesced_reads = 0, bridged_experts = 0, slot_cold_reuses = 0;
        double io_wait_ms = 0, decode_io_wait_ms = 0, predict_ms = 0, predict_wait_ms = 0, evict_ms = 0;
        uint64_t overlapped_layers = 0; // decode layers whose misses compute threads waited for per bank
        uint64_t demand_jobs = 0;
        uint64_t decode_prefetch_issued = 0, decode_io_bytes = 0;
        uint64_t decode_bank_bytes[2][4] = {}; // dispatch class (demand/speculative), then component
        uint64_t decode_failed_jobs = 0;
        double decode_io_busy_ms = 0; // decoding wall time with at least one expert read in flight
        uint64_t rank_predicted[16] = {}, rank_correct[16] = {};
        uint64_t early_rank_predicted[16] = {}, early_rank_correct[16] = {}, early_prefetch_issued = 0;
        uint64_t late_rank_predicted[16] = {}, late_rank_correct[16] = {};
        double late_ms = 0;
        double demand_queue_ms = 0, demand_read_ms = 0;
        uint64_t decode_demand_jobs = 0, promoted_reads = 0, confirmed_reads = 0, cancelled_prefetches = 0;
        double decode_demand_job_ms = 0; // decoding: from demand (or promotion) to the bank's bytes in place
        uint64_t peak_resident_bytes = 0;
    };
    // Read queues in service order (read_priority); without it speculative reads all use queue_next.
    enum Queue { queue_demand, queue_next, queue_after, queue_count };

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
    static bool expert_ready(const ggml_tensor *weights, int64_t expert, void *store);
    const BankRef *bank_of(const ggml_tensor *weights) const;
    bool layer_ready(const BankRef &ref, int64_t expert) const;
    void wait_bank(const BankRef &ref, int expert) const;
    uint8_t *data_of(const Layer &layer, int bank, int expert) const;
    bool assign_slot(Layer &layer, int expert);
    void release_slot(Layer &layer, int expert);
    void read_slots(const Job &job) const;
    uint64_t job_bytes(const Job &job) const;
    void setup_slots();
    // Validates the pack against the registered banks and places each bank at its packed offset in the slot.
    void attach_pack();
    bool ensure(int layer_index, const ggml_tensor *ids);
    bool check_oracle(int layer_index, const ggml_tensor *ids);
    void prefetch_oracle(int current_layer);
    // A layer's routing is known: requests its experts and starts the next lookahead.
    bool on_topk(int layer, const ggml_tensor *tensor);
    // Predicts layer_index + 1 (predict) and, with lookahead_layers 2, layer_index + 2 (predict_after) from the
    // same rows; the predictor thread queues the first prediction's reads before computing the second.
    void predict(int layer_index, const float *rows, int64_t count, size_t row_bytes);
    void predict_after(int layer_index, const float *rows, int64_t count, size_t row_bytes);
    // Scores `target`'s router on the lookahead rows; candidates in rank order, deduplicated over rows.
    void score(Layer &target, const float *rows, int64_t count_rows, size_t row_bytes, bool gate,
               std::vector<int> &prediction, std::vector<uint8_t> &ranks);
    void issue_reads_ahead(Layer &layer, const std::vector<int> &prediction, const std::vector<uint8_t> &ranks, int limit,
                           bool early);
    // Floats per lookahead input row for the layer whose experts are predicted (all streams with a mixer).
    static size_t lookahead_width(const Layer &layer);
    // The router input of `layer` computed from one row of residual streams.
    void mix(const Layer &layer, const float *streams, float *out, bool use_gate);
    void project(const ggml_tensor *weights, const float *x, float *out);
    // Prompt batches: makes room for `bytes` more by evicting other layers' resident entries, those whose next
    // use in a layer-ordered sweep lies furthest ahead first, so a layer's whole working set fits the budget.
    void borrow(int layer_index, uint64_t bytes);
    // Lookahead off the decode thread: the residual is copied, the decode thread returns to the graph, and the
    // predictor thread scores the next layer's router and queues its prefetch. Every decode-thread entry point
    // that touches cache state first waits for it, so the cache keeps a single writer at any time.
    void predictor();
    void wait_predictor();
    // Runs fn(context, begin, end) over [0, rows) in chunks of `chunk` rows (a multiple of four) on the predictor
    // and its helpers; returns when every chunk is done. Called by one thread at a time (the predictor, or the
    // decode thread while the predictor is idle).
    void parallel_rows(size_t rows, size_t chunk, void (*fn)(const void *, size_t, size_t), const void *context);
    template <class F>
    void parallel_rows(size_t rows, size_t chunk, const F &fn) {
        parallel_rows(rows, chunk, [](const void *f, size_t b, size_t e) { (*static_cast<const F *>(f))(b, e); }, &fn);
    }
    void run_rows();
    void wake_helpers();
    void helper();
    size_t row_chunk(size_t rows) const;
    void issue_prefetch(Layer &layer);
    void start_load(Layer &layer, int expert, Queue queue);
    bool begin_load(Layer &layer, int expert, bool demand);
    void queue_reads(Layer &layer, int first, int count, uint64_t members, Queue queue);
    // read_priority: moves the queued single-expert reads of `layer` that `layer.seen` marks for this epoch (or,
    // for the layer after, that its new prediction lists) from the speculative queues to `to`; returns the count.
    int move_queued(Layer &layer, Queue to, const std::vector<int> *experts);
    // Drops a speculative read of `expert` that no worker has started (all bank jobs still queued) and releases
    // the entry; a read with a bank started is left to complete.
    bool cancel_unread(Layer &layer, int expert);
    void load_runs(Layer &layer, std::vector<int> &experts, bool demand, int max_gap);
    void prefetch_whole_layer(Layer &layer);
    bool reap(Layer &layer, int expert);
    void evict(Layer &layer, int expert);
    int least_recent(const Layer &layer, uint32_t epoch) const;
    void trim(Layer &layer);
    bool wait(Layer &layer, int expert);
    void worker(bool demand_only, bool awake);
    // Whether awake demand threads may run: off when Android allows few CPUs (a backgrounded app), where they
    // would compete with compute threads. Safe from any thread.
public:
    void set_awake_allowed(bool allowed);
private:
    void fail(const std::string &message);
    Layer *layer_at(int index) const;

    ExpertOptions options_;
    const std::atomic_bool &cancelled_;
    std::unique_ptr<SourceFile> file_; // the expert source: the GGUF, or the pack
    uint64_t model_size_ = 0;
    bool packed_ = false;
    std::unique_ptr<ggml_backend_buffer_type> buffer_type_;
    std::unique_ptr<ggml_backend_buffer_type> mapped_type_;
    std::string path_;
    int map_fd_ = -1;
#ifdef _WIN32
    void *map_handle_ = nullptr; // file-mapping object of the mapped tensors
#endif
    std::vector<std::unique_ptr<Layer>> layers_;
    std::vector<std::thread> workers_;
    std::deque<Job> queues_[queue_count]; // guarded by mutex_
    int speculative_inflight_ = 0;        // guarded by mutex_
    mutable std::mutex mutex_;
    std::condition_variable queue_ready_, job_done_;
    std::thread predictor_;
    std::vector<std::thread> helpers_;
    std::atomic<uint32_t> helper_epoch_{0};  // one increment per posted job, the helpers' wake word
    std::atomic<uint64_t> helper_done_{0};   // jobs finished, summed over helpers
    uint64_t helper_posted_ = 0;              // jobs posted (written by the posting thread only)
    std::atomic<bool> helpers_hot_{false};   // a lookahead runs: helpers spin between its jobs
    std::atomic<bool> helpers_stop_{false};
    void (*helper_fn_)(const void *, size_t, size_t) = nullptr;
    const void *helper_context_ = nullptr;
    size_t helper_rows_ = 0, helper_chunk_ = 0;
    std::atomic<size_t> helper_next_{0};
    std::atomic<bool> predict_flag_{false}; // predict_pending_, readable without the mutex (lookahead_spin)
    std::mutex predict_mutex_;
    std::condition_variable predict_ready_, predict_done_;
    bool predict_pending_ = false, predict_stopping_ = false; // guarded by predict_mutex_
    int predict_layer_ = -1;
    int64_t predict_rows_ = 0;
    std::vector<float> predict_input_;
    std::atomic<int> inflight_jobs_{0};
    bool stopping_ = false;
    std::atomic<uint32_t> demand_signal_{0};   // incremented per queued demand job, the awake threads' wake word
    std::atomic<bool> awake_decode_{false};    // decoding with awake demand threads enabled
    std::atomic<bool> awake_allowed_{true};
    std::atomic<bool> awake_stopping_{false};
    std::vector<int> little_cpus_;
    bool decode_phase_ = false;
    std::string error_;
    std::ofstream trace_;
    struct OracleRoute {
        int layer = 0, rows = 0;
        bool decode = false;
        std::vector<int> experts;
        std::vector<uint8_t> ranks;
    };
    std::vector<OracleRoute> oracle_routes_;
    size_t oracle_cursor_ = 0;
    Stats stats_;
    uint64_t clock_ = 0, resident_bytes_ = 0, routed_bytes_ = 0, effective_budget_ = 0;
    uint64_t prompt_budget_ = 0, decode_budget_ = 0, decode_extra_ = 0;
    void plan_decode_quotas();
    int active_reads_ = 0;                    // decode reads only; guarded by mutex_
    std::chrono::steady_clock::time_point busy_since_;
    uint32_t epoch_ = 0;
    uint64_t requests_ = 0;
    int experts_used_ = 0, overflow_layer_ = -1, prefetch_layer_ = -1, early_layer_ = -1, predictable_layers_ = 0;
    float norm_epsilon_ = 1e-6f;
    size_t max_stride_ = 0;
    std::vector<int> demand_, missing_, order_, whole_layer_;
    std::vector<float> scratch_, scores_;
    std::vector<float> mix_normed_, mix_gate_, mix_low_, quant_scales_; // mixer lookahead scratch
    std::vector<int8_t> quant_;
    std::unordered_map<std::string, uint64_t> source_offsets_;
    std::vector<const ggml_tensor *> residuals_; // per-layer lookahead input of the current graph
    // Single-token top-k tensors of the current graph not handled yet (topk_in_graph), per layer.
    std::vector<const ggml_tensor *> graph_topk_;
    bool topk_in_graph_ = false;
    uint8_t *arena_ = nullptr;
    size_t arena_size_ = 0, arena_used_ = 0;
    std::vector<SlotClass> slot_classes_;
    std::vector<BankRef> bank_index_; // sorted by tensor; read-only once finalized
    void (*set_expert_address_)(const void *(*)(const ggml_tensor *, int64_t, void *), void *) = nullptr;
    void (*set_expert_ready_)(bool (*)(const ggml_tensor *, int64_t, void *), void *) = nullptr;
    // Small batches do not wait for missing experts in ensure(): MUL_MAT_ID computes ready experts first and its
    // compute threads wait only for the banks they read, so reads overlap with the layer's other work.
    bool overlap_reads_ = false;
    mutable std::atomic<bool> aborted_{false};
    mutable std::atomic<uint64_t> compute_wait_ns_{0};
};
} // namespace eqt
