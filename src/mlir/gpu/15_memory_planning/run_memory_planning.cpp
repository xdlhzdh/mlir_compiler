// run_memory_planning.cpp — Memory Planning & Optimization 7-Stage Pipeline
//
// Optimizes memory allocation for inference/training graph execution:
//
//   Stage 0: Graph Setup            — build operator graph with tensor shapes
//   Stage 1: Liveness Analysis      — compute live intervals [first_use, last_use] per tensor
//   Stage 2: Interference Graph     — build conflict graph (overlapping lifetimes)
//   Stage 3: Greedy Offset Planning — assign offsets using best-fit-decreasing
//   Stage 4: Buffer Reuse           — share memory between non-overlapping tensors
//   Stage 5: In-place Optimization  — eliminate copies by aliasing input/output buffers
//   Stage 6: Summary                — peak memory, reuse ratio, fragmentation
//
// Test case: ResNet-like block (Conv → BN → ReLU → Conv → Add → ReLU)
//
// Pure C++17, header-only IR, no external dependencies.

#include "memory_ir.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using namespace mem_ir;

static void sep(const char *title) {
  std::cout << "\n  ──── " << title << " ────\n\n";
}

// =====================================================================
// Stage 0: Graph Setup
// =====================================================================

static ExecGraph stage0_setup() {
  sep("Stage 0: Graph Setup — ResNet Residual Block");

  ExecGraph g;

  int b_input    = g.add_buffer("input",       1*64*56*56*4,  true);
  int b_w1       = g.add_buffer("conv1.weight", 64*64*3*3*4,  false, false, true);
  int b_conv1    = g.add_buffer("conv1_out",   1*64*56*56*4);
  int b_bn_w     = g.add_buffer("bn1.params",  64*4*4,        false, false, true);
  int b_bn1      = g.add_buffer("bn1_out",     1*64*56*56*4);
  int b_relu1    = g.add_buffer("relu1_out",   1*64*56*56*4);
  int b_w2       = g.add_buffer("conv2.weight", 64*64*3*3*4,  false, false, true);
  int b_conv2    = g.add_buffer("conv2_out",   1*64*56*56*4);
  int b_add      = g.add_buffer("add_out",     1*64*56*56*4);
  int b_relu2    = g.add_buffer("relu2_out",   1*64*56*56*4,  false, true);

  auto &op0 = g.add_op("conv1");
  op0.input_bufs = {b_input, b_w1};
  op0.output_bufs = {b_conv1};
  op0.compute_cost = 2LL * 64 * 64 * 3 * 3 * 56 * 56;

  auto &op1 = g.add_op("bn1");
  op1.input_bufs = {b_conv1, b_bn_w};
  op1.output_bufs = {b_bn1};

  auto &op2 = g.add_op("relu1");
  op2.input_bufs = {b_bn1};
  op2.output_bufs = {b_relu1};

  auto &op3 = g.add_op("conv2");
  op3.input_bufs = {b_relu1, b_w2};
  op3.output_bufs = {b_conv2};
  op3.compute_cost = 2LL * 64 * 64 * 3 * 3 * 56 * 56;

  auto &op4 = g.add_op("add (residual)");
  op4.input_bufs = {b_conv2, b_input};
  op4.output_bufs = {b_add};

  auto &op5 = g.add_op("relu2");
  op5.input_bufs = {b_add};
  op5.output_bufs = {b_relu2};

  std::cout << "  Operators (" << g.ops.size() << "):\n";
  for (auto &op : g.ops) {
    std::cout << "    [" << op.id << "] " << op.name;
    std::cout << "  in={";
    for (size_t i = 0; i < op.input_bufs.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << g.buffers[op.input_bufs[i]].name;
    }
    std::cout << "} out={";
    for (size_t i = 0; i < op.output_bufs.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << g.buffers[op.output_bufs[i]].name;
    }
    std::cout << "}\n";
  }

  std::cout << "\n  Buffers (" << g.buffers.size() << "):\n";
  int64_t total = 0;
  for (auto &b : g.buffers) {
    std::cout << "    [" << b.id << "] " << std::left << std::setw(16) << b.name
              << std::right << std::setw(10) << b.size_bytes << " bytes ("
              << b.size_bytes / 1024 << " KB)";
    if (b.is_input) std::cout << " [INPUT]";
    if (b.is_output) std::cout << " [OUTPUT]";
    if (b.is_weight) std::cout << " [WEIGHT]";
    std::cout << "\n";
    total += b.size_bytes;
  }
  std::cout << "\n  Naive total (no reuse): " << total << " bytes ("
            << total / (1024*1024) << " MB)\n";

  return g;
}

// =====================================================================
// Stage 1: Liveness Analysis
// =====================================================================

static std::vector<LiveInterval> stage1_liveness(const ExecGraph &g) {
  sep("Stage 1: Liveness Analysis");

  std::vector<LiveInterval> intervals(g.buffers.size());
  for (size_t i = 0; i < g.buffers.size(); ++i) {
    intervals[i].buf_id = static_cast<int>(i);
    intervals[i].first_use = -1;
    intervals[i].last_use = -1;
  }

  for (auto &op : g.ops) {
    for (int bid : op.input_bufs) {
      if (intervals[bid].first_use < 0)
        intervals[bid].first_use = op.id;
      intervals[bid].last_use = std::max(intervals[bid].last_use, op.id);
    }
    for (int bid : op.output_bufs) {
      if (intervals[bid].first_use < 0)
        intervals[bid].first_use = op.id;
      intervals[bid].last_use = std::max(intervals[bid].last_use, op.id);
    }
  }

  for (auto &b : g.buffers) {
    if (b.is_input) intervals[b.id].first_use = 0;
    if (b.is_output) intervals[b.id].last_use = static_cast<int>(g.ops.size()) - 1;
    if (b.is_weight) {
      intervals[b.id].first_use = 0;
      intervals[b.id].last_use = static_cast<int>(g.ops.size()) - 1;
    }
  }

  std::cout << "  Live intervals (op index range):\n\n";
  int max_op = static_cast<int>(g.ops.size());
  std::cout << "  " << std::left << std::setw(18) << "Buffer"
            << std::setw(10) << "[first"
            << std::setw(8) << "last]"
            << "Timeline\n";
  std::cout << "  " << std::string(60, '-') << "\n";

  for (size_t i = 0; i < intervals.size(); ++i) {
    auto &iv = intervals[i];
    std::cout << "  " << std::left << std::setw(18) << g.buffers[i].name
              << "[" << std::setw(4) << iv.first_use << " → "
              << std::setw(4) << iv.last_use << "]  ";

    for (int t = 0; t < max_op; ++t) {
      if (t >= iv.first_use && t <= iv.last_use)
        std::cout << "█";
      else
        std::cout << "·";
    }
    std::cout << "\n";
  }

  return intervals;
}

// =====================================================================
// Stage 2: Interference Graph
// =====================================================================

static std::vector<std::vector<bool>>
stage2_interference(const ExecGraph &g, const std::vector<LiveInterval> &intervals) {
  sep("Stage 2: Interference Graph");

  int n = static_cast<int>(intervals.size());
  std::vector<std::vector<bool>> conflict(n, std::vector<bool>(n, false));

  int edge_count = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (g.buffers[i].is_weight || g.buffers[j].is_weight) continue;
      if (intervals[i].overlaps(intervals[j])) {
        conflict[i][j] = conflict[j][i] = true;
        ++edge_count;
      }
    }
  }

  std::cout << "  Conflict matrix (activation buffers only, excluding weights):\n\n";
  std::cout << "  " << std::setw(18) << " ";
  for (auto &b : g.buffers)
    if (!b.is_weight)
      std::cout << std::setw(4) << b.id;
  std::cout << "\n";

  for (int i = 0; i < n; ++i) {
    if (g.buffers[i].is_weight) continue;
    std::cout << "  " << std::left << std::setw(18) << g.buffers[i].name;
    for (int j = 0; j < n; ++j) {
      if (g.buffers[j].is_weight) continue;
      if (i == j) std::cout << std::setw(4) << "-";
      else std::cout << std::setw(4) << (conflict[i][j] ? "X" : ".");
    }
    std::cout << "\n";
  }
  std::cout << "\n  Edges (conflicts): " << edge_count << "\n";
  std::cout << "  Non-overlapping buffer pairs can share memory!\n";

  return conflict;
}

// =====================================================================
// Stage 3: Greedy Offset Planning (Best-Fit Decreasing)
// =====================================================================

static MemPool stage3_offset_planning(
    const ExecGraph &g,
    const std::vector<LiveInterval> &intervals,
    const std::vector<std::vector<bool>> &conflict) {
  sep("Stage 3: Greedy Offset Planning (Best-Fit Decreasing)");

  std::vector<int> activation_ids;
  for (auto &b : g.buffers)
    if (!b.is_weight && !b.is_input && !b.is_output)
      activation_ids.push_back(b.id);

  std::sort(activation_ids.begin(), activation_ids.end(),
            [&](int a, int b) {
              return g.buffers[a].aligned_size() > g.buffers[b].aligned_size();
            });

  std::cout << "  Allocation order (by size, descending):\n";
  for (int id : activation_ids) {
    std::cout << "    " << g.buffers[id].name << ": "
              << g.buffers[id].aligned_size() << " bytes\n";
  }
  std::cout << "\n";

  MemPool pool;
  pool.name = "activation_pool";

  struct FreeGap {
    int64_t offset;
    int64_t size;
  };

  for (int id : activation_ids) {
    int64_t needed = g.buffers[id].aligned_size();

    int64_t best_offset = pool.total_size;

    for (auto &existing : pool.allocs) {
      auto &eb = g.buffers[existing.buf_id];
      if (!conflict[id][existing.buf_id]) {
        best_offset = std::min(best_offset, existing.offset);
        break;
      }
    }

    int64_t chosen_offset = -1;
    for (auto &existing : pool.allocs) {
      if (conflict[id][existing.buf_id]) continue;
      if (existing.size >= needed) {
        chosen_offset = existing.offset;
        break;
      }
    }

    if (chosen_offset < 0) {
      chosen_offset = pool.total_size;
    }

    pool.allocs.push_back({id, chosen_offset, needed});
    int64_t end = chosen_offset + needed;
    pool.total_size = std::max(pool.total_size, end);

    std::cout << "  Assign " << std::left << std::setw(14) << g.buffers[id].name
              << " → offset " << std::setw(8) << chosen_offset
              << " size " << std::setw(8) << needed
              << " [" << chosen_offset << ", " << end << ")\n";
  }

  std::cout << "\n  Pool total size: " << pool.total_size << " bytes ("
            << pool.total_size / 1024 << " KB)\n";

  return pool;
}

// =====================================================================
// Stage 4: Buffer Reuse Analysis
// =====================================================================

static void stage4_buffer_reuse(
    const ExecGraph &g,
    const std::vector<LiveInterval> &intervals,
    const std::vector<std::vector<bool>> &conflict) {
  sep("Stage 4: Buffer Reuse Analysis");

  std::cout << "  Non-conflicting buffer pairs (can share memory):\n\n";

  int reuse_count = 0;
  int64_t saved_bytes = 0;

  for (size_t i = 0; i < g.buffers.size(); ++i) {
    if (g.buffers[i].is_weight) continue;
    for (size_t j = i + 1; j < g.buffers.size(); ++j) {
      if (g.buffers[j].is_weight) continue;
      if (!conflict[i][j]) {
        int64_t save = std::min(g.buffers[i].size_bytes, g.buffers[j].size_bytes);
        std::cout << "    " << g.buffers[i].name << " ↔ " << g.buffers[j].name
                  << " (save " << save / 1024 << " KB)\n";
        saved_bytes += save;
        ++reuse_count;
      }
    }
  }

  std::cout << "\n  Reusable pairs: " << reuse_count << "\n";
  std::cout << "  Potential savings: " << saved_bytes / 1024 << " KB\n\n";

  std::cout << "  Reuse strategies:\n";
  std::cout << "    1. Best-fit: reuse the smallest buffer that fits\n";
  std::cout << "    2. First-fit: reuse the first freed buffer that fits\n";
  std::cout << "    3. Offset-based: all activations share one arena, non-overlapping offsets\n";
}

// =====================================================================
// Stage 5: In-place Optimization
// =====================================================================

static void stage5_inplace(const ExecGraph &g) {
  sep("Stage 5: In-place Optimization");

  std::cout << "  In-place candidates (output aliases input):\n\n";

  struct InplaceCandidate {
    std::string op_name;
    std::string input;
    std::string output;
    bool safe;
    std::string reason;
  };

  std::vector<InplaceCandidate> candidates = {
    {"relu1",          "bn1_out",   "relu1_out",  true,
     "elementwise, single consumer, same shape → safe in-place"},
    {"bn1",            "conv1_out", "bn1_out",    true,
     "elementwise (inference mode), single consumer → safe in-place"},
    {"relu2",          "add_out",   "relu2_out",  true,
     "elementwise, single consumer, same shape → safe in-place"},
    {"add (residual)", "conv2_out", "add_out",    true,
     "addf is elementwise, conv2_out not used after → safe"},
    {"add (residual)", "input",     "add_out",    false,
     "UNSAFE: input is also used by conv1 (WAR conflict)"},
  };

  for (auto &c : candidates) {
    std::cout << "  " << (c.safe ? "✓" : "✗") << " " << c.op_name
              << ": " << c.input << " → " << c.output << "\n";
    std::cout << "    " << c.reason << "\n\n";
  }

  std::cout << "  In-place safety conditions:\n";
  std::cout << "    1. Output has same shape and dtype as input\n";
  std::cout << "    2. Input has no other live consumers after this op\n";
  std::cout << "    3. No write-after-read conflict (other ops reading input concurrently)\n";
  std::cout << "    4. Op semantics allow in-place (elementwise, reduction, etc.)\n\n";

  int64_t inplace_saved = 0;
  int inplace_count = 0;
  for (auto &c : candidates) {
    if (c.safe) {
      inplace_saved += 1 * 64 * 56 * 56 * 4;
      ++inplace_count;
    }
  }
  std::cout << "  In-place rewrites: " << inplace_count << "\n";
  std::cout << "  Memory saved: " << inplace_saved / 1024 << " KB\n";
}

// =====================================================================
// Stage 6: Summary
// =====================================================================

static void stage6_summary(const ExecGraph &g, const MemPool &pool) {
  sep("Stage 6: Summary — Memory Efficiency Report");

  int64_t total_naive = 0;
  int64_t weight_bytes = 0;
  int64_t activation_bytes = 0;
  int64_t io_bytes = 0;

  for (auto &b : g.buffers) {
    total_naive += b.size_bytes;
    if (b.is_weight) weight_bytes += b.size_bytes;
    else if (b.is_input || b.is_output) io_bytes += b.size_bytes;
    else activation_bytes += b.size_bytes;
  }

  int64_t optimized_activation = pool.total_size;
  int64_t total_optimized = weight_bytes + io_bytes + optimized_activation;

  std::cout << "  Memory breakdown:\n";
  std::cout << "    Weights:      " << std::setw(10) << weight_bytes << " bytes ("
            << weight_bytes / 1024 << " KB) — not optimized (persistent)\n";
  std::cout << "    I/O buffers:  " << std::setw(10) << io_bytes << " bytes ("
            << io_bytes / 1024 << " KB) — not optimized (external)\n";
  std::cout << "    Activations:  " << std::setw(10) << activation_bytes << " bytes ("
            << activation_bytes / 1024 << " KB) — naive\n";
  std::cout << "    Optimized:    " << std::setw(10) << optimized_activation << " bytes ("
            << optimized_activation / 1024 << " KB) — after planning\n\n";

  std::cout << "  Summary:\n";
  std::cout << "    Naive total:     " << total_naive / 1024 << " KB\n";
  std::cout << "    Optimized total: " << total_optimized / 1024 << " KB\n";
  float ratio = static_cast<float>(total_optimized) / total_naive;
  std::cout << "    Reduction:       " << static_cast<int>((1.0f - ratio) * 100) << "%\n";
  std::cout << "    Peak activation: " << optimized_activation / 1024 << " KB\n\n";

  std::cout << "  Fragmentation:\n";
  int64_t used_in_pool = 0;
  for (auto &a : pool.allocs)
    used_in_pool = std::max(used_in_pool, a.offset + a.size);
  int64_t actual_used = 0;
  for (auto &a : pool.allocs)
    actual_used += a.size;
  float frag = 1.0f - static_cast<float>(actual_used) / used_in_pool;
  if (used_in_pool == 0) frag = 0;
  std::cout << "    Internal fragmentation: " << static_cast<int>(frag * 100) << "%\n\n";

  std::cout << "  Advanced techniques (not implemented, for reference):\n";
  std::cout << "    • Memory-aware scheduling: reorder ops to minimize peak memory\n";
  std::cout << "    • Gradient checkpointing: trade compute for memory in training\n";
  std::cout << "    • Defragmentation: compact memory pool periodically\n";
  std::cout << "    • Unified memory pool: share between CPU/GPU with async prefetch\n";
  std::cout << "    • Tensor swapping: offload to CPU memory when GPU is full\n";
}

// =====================================================================
// main
// =====================================================================

int main() {
  std::cout << "========================================================\n";
  std::cout << " Stage 15: Memory Planning & Optimization Pipeline\n";
  std::cout << "========================================================\n";

  auto g = stage0_setup();
  auto intervals = stage1_liveness(g);
  auto conflict = stage2_interference(g, intervals);
  auto pool = stage3_offset_planning(g, intervals, conflict);
  stage4_buffer_reuse(g, intervals, conflict);
  stage5_inplace(g);
  stage6_summary(g, pool);

  std::cout << "\n========================================================\n";
  std::cout << " Memory Planning Pipeline Complete!\n";
  std::cout << "========================================================\n";
  return 0;
}
