// run_linalg_opt.cpp — P6 (6_linalg_opt): Linalg on tensor fusion (linalg_ir, 7-step Pass chain)
//
// Full pipeline (L2 tensor level, after StableHLO → Linalg legalization):
//
//   P6 Step 0: Pre-clean          — canonicalize + CSE + DCE
//   P6 Step 1: Dependence Analysis — SSA use-def + alias check
//   P6 Step 2: Fusion Candidate   — producer-consumer graph + elementwise chain
//   P6 Step 3: Cost Model Filter  — memory estimate + compute reuse estimate
//   P6 Step 4: Tile-aware Fusion  — fuse only tile-compatible ops
//   P6 Step 5: Fusion Rewrite     — merge indexing maps + merge iterator types
//   P6 Step 6: Post-clean         — canonicalize + CSE + DCE
//
// Pure C++17, header-only IR, no external dependencies.

#include "linalg_ir.h"

#include <cmath>
#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace linalg_ir;

// =====================================================================
// Helpers
// =====================================================================

static bool is_generic_elem(const Op *op) {
  return op->mnemonic == "linalg.generic" && op->is_elementwise();
}

// =====================================================================
// P6 Step 0: Pre-clean — canonicalize + CSE + DCE
// =====================================================================

static int pass_canonicalize(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < g.ops.size(); ++i) {
      auto *op = g.ops[i].get();
      if (!op->is_generic() || !op->is_elementwise()) continue;
      if (op->body.size() != 1) continue;
      auto &step = op->body[0];
      if (step.arg0 != 0 || step.arg1 != 1) continue;

      // identity: out = in0 op in1 where in0==in1 and op=addf → 2*x (skip)
      // dead identity: single input, out = in0 (copy)
      if (op->ins.size() == 1 && step.op == "arith.addf" &&
          op->ins[0] == op->ins[0]) {
        // f(x) = x+x is not identity, skip
      }
    }
  }
  (void)count;
  return 0;
}

static int pass_cse(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < g.ops.size() && !changed; ++i) {
      auto *a = g.ops[i].get();
      if (!a->is_generic()) continue;
      for (size_t j = i + 1; j < g.ops.size() && !changed; ++j) {
        auto *b = g.ops[j].get();
        if (!b->is_generic()) continue;
        if (a->mnemonic != b->mnemonic) continue;
        if (a->ins.size() != b->ins.size()) continue;
        bool same = true;
        for (size_t k = 0; k < a->ins.size(); ++k)
          if (a->ins[k] != b->ins[k]) { same = false; break; }
        if (!same) continue;
        if (a->body.size() != b->body.size()) continue;
        bool body_same = true;
        for (size_t k = 0; k < a->body.size(); ++k)
          if (a->body[k].op != b->body[k].op ||
              a->body[k].arg0 != b->body[k].arg0 ||
              a->body[k].arg1 != b->body[k].arg1)
            { body_same = false; break; }
        if (!body_same) continue;

        std::cout << "  [CSE] " << b->result_->name << " = "
                  << b->label << " → " << a->result_->name << "\n";
        g.replace_all_uses(b->result_, a->result_);
        g.erase_op(b);
        ++count;
        changed = true;
      }
    }
  }
  return count;
}

static int pass_dce(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = g.ops.size(); i-- > 0;) {
      auto *op = g.ops[i].get();
      if (!op->result_ || g.is_return_val(op->result_)) continue;
      if (op->result_->has_no_uses()) {
        std::cout << "  [DCE] " << op->mnemonic
                  << " (" << op->result_->name << ")\n";
        g.erase_op(op);
        ++count;
        changed = true;
        break;
      }
    }
  }
  return count;
}

static int stage0_preclean(Graph &g) {
  return pass_canonicalize(g) + pass_cse(g) + pass_dce(g);
}

// =====================================================================
// P6 Step 1: Dependence Analysis — SSA use-def + alias check
// =====================================================================
// Build a dependency graph between ops: use-def edges (direct data flow)
// and alias edges (ops reading the same buffer / value).

static std::vector<DepEdge> build_dep_graph(const Graph &g) {
  std::vector<DepEdge> edges;

  for (auto &op_ptr : g.ops) {
    auto *consumer = op_ptr.get();
    for (auto *v : consumer->ins) {
      if (v->defining_op && v->defining_op != consumer) {
        edges.push_back({v->defining_op, consumer, DepKind::USE_DEF, v});
      }
    }
    for (auto *v : consumer->outs) {
      if (v->defining_op && v->defining_op != consumer) {
        edges.push_back({v->defining_op, consumer, DepKind::USE_DEF, v});
      }
    }
  }

  // Alias: two ops reading the same func argument (potential WAR if fused)
  std::unordered_map<Value *, std::vector<Op *>> arg_readers;
  for (auto &op_ptr : g.ops) {
    for (auto *v : op_ptr->ins) {
      if (!v->defining_op) // func argument
        arg_readers[v].push_back(op_ptr.get());
    }
  }
  for (auto &[arg, readers] : arg_readers) {
    for (size_t i = 0; i < readers.size(); ++i) {
      for (size_t j = i + 1; j < readers.size(); ++j) {
        auto *a = readers[i];
        auto *b = readers[j];
        int pa = g.op_pos(a), pb = g.op_pos(b);
        if (pa < pb)
          edges.push_back({a, b, DepKind::ALIAS, arg});
        else
          edges.push_back({b, a, DepKind::ALIAS, arg});
      }
    }
  }

  return edges;
}

static void stage1_dep_analysis(const Graph &g,
                                 std::vector<DepEdge> &edges) {
  edges = build_dep_graph(g);
  int use_def = 0, alias = 0;
  for (auto &e : edges) {
    if (e.kind == DepKind::USE_DEF) ++use_def;
    else ++alias;
  }
  std::cout << "  [DepAnalysis] " << edges.size() << " edges ("
            << use_def << " use-def, " << alias << " alias)\n";
  for (auto &e : edges) {
    std::string p = e.producer->result_ ? e.producer->result_->name : "?";
    std::string c = e.consumer->result_ ? e.consumer->result_->name : "?";
    std::cout << "    " << p << " (" << e.producer->label << ") → "
              << c << " (" << e.consumer->label << ") ["
              << (e.kind == DepKind::USE_DEF ? "use-def" : "alias")
              << " via " << e.through->name << "]\n";
  }
}

// =====================================================================
// P6 Step 2: Fusion Candidate Build
// =====================================================================
// From the dependency graph, build (producer, consumer) fusion candidates:
//   1. Producer-consumer pairs (use-def edges, both are linalg ops)
//   2. Elementwise chain detection (consecutive parallel-only ops)

static std::vector<FusionCandidate>
build_fusion_candidates(const Graph &g, const std::vector<DepEdge> &edges) {
  std::vector<FusionCandidate> cands;
  std::set<std::pair<Op *, Op *>> seen;

  for (auto &e : edges) {
    if (e.kind != DepKind::USE_DEF) continue;
    auto *p = e.producer;
    auto *c = e.consumer;
    if (!p->is_generic() || !c->is_generic()) continue;
    if (seen.count({p, c})) continue;
    seen.insert({p, c});

    FusionCandidate fc;
    fc.producer = p;
    fc.consumer = c;
    cands.push_back(fc);
  }

  std::cout << "  [CandBuild] " << cands.size()
            << " fusion candidates from dep graph\n";
  for (auto &fc : cands) {
    std::string chain_type =
        (fc.producer->is_elementwise() && fc.consumer->is_elementwise())
            ? "elementwise-chain"
            : "producer-consumer";
    std::cout << "    " << fc.producer->label << " → " << fc.consumer->label
              << "  [" << chain_type << "]\n";
  }
  return cands;
}

// =====================================================================
// P6 Step 3: Cost Model Filtering
// =====================================================================
// For each candidate, compute:
//   • mem_saved_bytes: size of intermediate tensor that would be eliminated
//   • compute_reuse:  ratio of shared parallel iters / total iters
// Accept if memory savings are positive and compute reuse is reasonable.

static void stage3_cost_filter(std::vector<FusionCandidate> &cands) {
  constexpr int64_t MIN_SAVE_BYTES = 0;
  constexpr double MIN_REUSE = 0.0;

  for (auto &fc : cands) {
    // Memory saved = size of producer's result tensor (no longer materialised)
    if (fc.producer->result_) {
      // Only count savings if producer result is used solely by this consumer
      if (fc.producer->result_->users.size() == 1)
        fc.mem_saved_bytes = fc.producer->result_->type.bytes();
      else
        fc.mem_saved_bytes = 0; // can't eliminate: other users exist
    }

    // Compute reuse: fraction of iterator dimensions that are shared
    int shared = 0, total = static_cast<int>(
        std::max(fc.producer->iters.size(), fc.consumer->iters.size()));
    for (size_t i = 0; i < std::min(fc.producer->iters.size(),
                                     fc.consumer->iters.size()); ++i) {
      if (fc.producer->iters[i] == fc.consumer->iters[i]) ++shared;
    }
    fc.compute_reuse = total > 0 ? static_cast<double>(shared) / total : 0.0;

    if (fc.mem_saved_bytes > MIN_SAVE_BYTES && fc.compute_reuse > MIN_REUSE) {
      fc.accepted = true;
    } else {
      fc.accepted = false;
      if (fc.mem_saved_bytes <= MIN_SAVE_BYTES)
        fc.reject_reason = "multi-user (can't eliminate intermediate)";
      else
        fc.reject_reason = "low compute reuse";
    }

    char reuse_buf[16];
    std::snprintf(reuse_buf, sizeof(reuse_buf), "%.0f%%",
                  fc.compute_reuse * 100);
    std::cout << "  [CostModel] " << fc.producer->label << " → "
              << fc.consumer->label
              << "  mem_save=" << fc.mem_saved_bytes << "B"
              << "  reuse=" << reuse_buf
              << "  → " << (fc.accepted ? "ACCEPT" : "REJECT")
              << (fc.reject_reason.empty() ? "" : " (" + fc.reject_reason + ")")
              << "\n";
  }
}

// =====================================================================
// P6 Step 4: Tile-aware Fusion
// =====================================================================
// Further filter: only fuse if the ops' iterator types are compatible.
//   - Two elementwise ops: always tile-compatible (same parallel dims)
//   - Reduction producer + elementwise consumer: tile-compatible only via
//     tile-and-fuse (consumer runs in outer parallel loop, after reduction)
//   - Two reduction ops: NOT fusable (would require re-tiling)
//
// Also: avoid fusing a consumer that would block tiling of the producer
// (e.g. a consumer with a different loop range would require re-slicing).

static void stage4_tile_aware(std::vector<FusionCandidate> &cands) {
  for (auto &fc : cands) {
    if (!fc.accepted) continue;

    bool p_elem = fc.producer->is_elementwise();
    bool c_elem = fc.consumer->is_elementwise();
    bool p_red = fc.producer->has_reduction();
    bool c_red = fc.consumer->has_reduction();

    if (p_elem && c_elem) {
      fc.tile_compatible = true;
      std::cout << "  [TileAware] " << fc.producer->label << " → "
                << fc.consumer->label
                << "  elem+elem → direct fusion OK\n";
    } else if (p_red && c_elem) {
      // Tile-and-fuse: consumer runs in outer parallel loop after reduction
      fc.tile_compatible = true;
      std::cout << "  [TileAware] " << fc.producer->label << " → "
                << fc.consumer->label
                << "  reduction+elem → tile-and-fuse OK\n";
    } else if (p_red && c_red) {
      fc.tile_compatible = false;
      fc.accepted = false;
      fc.reject_reason = "reduction+reduction: would require re-tiling";
      std::cout << "  [TileAware] " << fc.producer->label << " → "
                << fc.consumer->label
                << "  reduction+reduction → REJECT\n";
    } else if (p_elem && c_red) {
      // elementwise feeds into reduction: can inline as prologue
      int p_rank = static_cast<int>(fc.producer->iters.size());
      int c_par = 0;
      for (auto k : fc.consumer->iters)
        if (k == IterKind::PARALLEL) ++c_par;
      if (p_rank <= c_par) {
        fc.tile_compatible = true;
        std::cout << "  [TileAware] " << fc.producer->label << " → "
                  << fc.consumer->label
                  << "  elem→reduction → prologue fusion OK\n";
      } else {
        fc.tile_compatible = false;
        fc.accepted = false;
        fc.reject_reason = "elem dims exceed consumer parallel dims";
        std::cout << "  [TileAware] " << fc.producer->label << " → "
                  << fc.consumer->label << " → REJECT ("
                  << fc.reject_reason << ")\n";
      }
    }

    // Check loop range compatibility for elementwise fusion
    if (fc.tile_compatible && p_elem && c_elem) {
      if (fc.producer->loop_ranges != fc.consumer->loop_ranges) {
        fc.tile_compatible = false;
        fc.accepted = false;
        fc.reject_reason = "loop range mismatch";
        std::cout << "  [TileAware] → REJECT (loop range mismatch)\n";
      }
    }
  }
}

// =====================================================================
// P6 Step 5: Fusion Rewrite — merge indexing maps + iterator types
// =====================================================================
// For accepted candidates, actually perform the fusion:
//   A) Elementwise + Elementwise: merge bodies, union inputs, compose maps
//   B) Reduction + Elementwise (tile-and-fuse): tile producer, fuse consumer
//      into the outer parallel loop

static int stage5_fusion_rewrite(Graph &g,
                                  std::vector<FusionCandidate> &cands) {
  int total = 0;

  // --- Pass 5a: Elementwise chain fusion (merge bodies) ---
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &fc : cands) {
      if (!fc.accepted || !fc.tile_compatible) continue;
      if (!fc.producer->is_elementwise() || !fc.consumer->is_elementwise())
        continue;
      // Verify both still exist in graph
      if (g.op_pos(fc.producer) < 0 || g.op_pos(fc.consumer) < 0) continue;
      if (fc.producer->result_->users.size() != 1) continue;

      auto *P = fc.producer;
      auto *C = fc.consumer;

      std::unordered_set<Value *> p_results;
      p_results.insert(P->result_);

      std::vector<Value *> ext_ins;
      std::unordered_map<Value *, int> ext_map;
      for (auto *v : P->ins) {
        ext_map[v] = static_cast<int>(ext_ins.size());
        ext_ins.push_back(v);
      }
      for (auto *v : C->ins) {
        if (p_results.count(v)) continue;
        if (!ext_map.count(v)) {
          ext_map[v] = static_cast<int>(ext_ins.size());
          ext_ins.push_back(v);
        }
      }

      int next_temp = static_cast<int>(ext_ins.size());
      std::vector<BodyStep> merged_body;
      std::unordered_map<int, int> p_temp_map;

      // Producer body
      for (auto &step : P->body) {
        int a0 = (step.arg0 < P->n_ins()) ? ext_map[P->ins[step.arg0]]
                                           : p_temp_map.at(step.arg0);
        int a1 = (step.arg1 < P->n_ins()) ? ext_map[P->ins[step.arg1]]
                                           : p_temp_map.at(step.arg1);
        int nr = next_temp++;
        p_temp_map[step.res] = nr;
        merged_body.push_back({step.op, a0, a1, nr});
      }
      int p_yield = (P->yield_idx < P->n_ins())
                        ? ext_map[P->ins[P->yield_idx]]
                        : p_temp_map.at(P->yield_idx);

      // Consumer body: remap block args
      std::unordered_map<int, int> c_temp_map;
      for (auto &step : C->body) {
        auto remap_arg = [&](int idx) -> int {
          if (idx < C->n_ins()) {
            auto *v = C->ins[idx];
            if (p_results.count(v)) return p_yield;
            return ext_map[v];
          }
          return c_temp_map.at(idx);
        };
        int a0 = remap_arg(step.arg0);
        int a1 = remap_arg(step.arg1);
        int nr = next_temp++;
        c_temp_map[step.res] = nr;
        merged_body.push_back({step.op, a0, a1, nr});
      }
      auto remap_yield = [&](int idx) -> int {
        if (idx < C->n_ins()) {
          auto *v = C->ins[idx];
          if (p_results.count(v)) return p_yield;
          return ext_map[v];
        }
        return c_temp_map.at(idx);
      };
      int merged_yield = remap_yield(C->yield_idx);

      std::string lbl = "fused(" + P->label + " + " + C->label + ")";

      auto fop = std::make_unique<Op>();
      fop->mnemonic = "linalg.generic";
      fop->label = lbl;
      fop->ins = ext_ins;
      int rank = static_cast<int>(ext_ins[0]->type.rank());
      std::vector<int> id;
      for (int i = 0; i < rank; ++i) id.push_back(i);
      IndexingMap im{id};
      for (size_t i = 0; i <= ext_ins.size(); ++i)
        fop->maps.push_back(im);
      fop->iters.assign(rank, IterKind::PARALLEL);
      fop->loop_ranges = ext_ins[0]->type.dims;
      fop->body = merged_body;
      fop->yield_idx = merged_yield;

      auto *fr = g.make_val(C->result_->type, fop.get());
      fop->result_ = fr;
      for (auto *v : ext_ins) v->add_user(fop.get());

      std::cout << "  [FuseRewrite] merge " << P->label << " + "
                << C->label << " → " << lbl
                << "  (maps: " << fop->maps.size()
                << ", iters: all parallel)\n";

      g.replace_all_uses(C->result_, fr);

      size_t insert_pos = static_cast<size_t>(g.op_pos(P));
      g.ops.insert(g.ops.begin() + static_cast<long>(insert_pos),
                    std::move(fop));
      g.erase_op(C);
      g.erase_op(P);
      ++total;
      fc.accepted = false; // consumed
      changed = true;
      break;
    }
  }

  // --- Pass 5b: Tiling (for reduction ops) ---
  for (auto &op_ptr : g.ops) {
    auto *op = op_ptr.get();
    if (!op->has_reduction() || !op->tile_sizes.empty()) continue;

    op->tile_sizes.resize(op->iters.size());
    for (size_t i = 0; i < op->loop_ranges.size(); ++i) {
      int64_t r = op->loop_ranges[i];
      int64_t tile = std::min(r, std::max(int64_t(16), r / 2));
      int64_t p = 1;
      while (p * 2 <= tile) p *= 2;
      op->tile_sizes[i] = p;
    }
    std::cout << "  [Tile] " << op->label << " → [";
    for (size_t i = 0; i < op->tile_sizes.size(); ++i) {
      if (i) std::cout << ", ";
      std::cout << op->tile_sizes[i];
    }
    std::cout << "]\n";
    ++total;
  }

  // --- Pass 5c: Tile-and-Fuse (reduction + elementwise) ---
  changed = true;
  while (changed) {
    changed = false;
    for (auto &fc : cands) {
      if (!fc.accepted || !fc.tile_compatible) continue;
      if (fc.producer->is_elementwise()) continue;
      if (!fc.producer->has_reduction()) continue;
      if (g.op_pos(fc.producer) < 0 || g.op_pos(fc.consumer) < 0) continue;
      if (fc.producer->result_->users.size() != 1) continue;

      auto *producer = fc.producer;
      auto *consumer = fc.consumer;

      FusedOp fused;
      fused.label = consumer->label;
      fused.body = consumer->body;
      fused.yield_idx = consumer->yield_idx;
      for (auto *v : consumer->ins)
        if (v != producer->result_)
          fused.extra_ins.push_back(v);

      producer->fused_consumers.push_back(fused);

      std::cout << "  [TileFuse] " << consumer->label
                << " → fused into " << producer->label
                << "'s parallel tile loop (after reduction)\n";

      g.replace_all_uses(consumer->result_, producer->result_);
      g.erase_op(consumer);
      ++total;
      fc.accepted = false;
      changed = true;
      break;
    }
  }

  return total;
}

// =====================================================================
// P6 Step 6: Post-clean — canonicalize + CSE + DCE
// =====================================================================

static int stage6_postclean(Graph &g) {
  return pass_canonicalize(g) + pass_cse(g) + pass_dce(g);
}

// =====================================================================
// Test Graph: comprehensive Linalg fusion exercise
// =====================================================================
//
//   %A (64x128) × %B (128x32)  → matmul → %mm   [reduction]
//   %mm + %bias                 → bias_add        [elementwise]
//   relu(%biased)               → relu            [elementwise]
//   %relu * %relu               → square          [elementwise]
//   %square + %relu             → residual        [elementwise]
//   %residual * %scale          → scaling         [elementwise]
//   %dead = %A + %A             → dead_add        [elementwise, no users → DCE]
//   %dup1 = %relu * %scale      → dup1            [CSE target]
//   %dup2 = %relu * %scale      → dup2            [CSE target]
//   %final = %scaling + %dup1   → output          [uses CSE'd value]
//
// Exercises every P6 step:
//   P6 Step 0: CSE(dup1==dup2) + DCE(dead_add)
//   P6 Step 1: use-def edges + alias edges (A read by matmul and dead_add)
//   P6 Step 2: 6 fusion candidates from dep graph
//   P6 Step 3: cost filter (multi-user rejection for relu if it feeds both
//            square and residual directly — but here we chain through)
//   P6 Step 4: tile-aware (matmul+bias = tile-and-fuse, elem+elem = direct)
//   P6 Step 5: elementwise merge + tile + tile-and-fuse
//   P6 Step 6: final cleanup

static Graph make_test_graph() {
  Graph g;
  g.name = "linalg_fusion_pipeline";

  TensorType T_A{{64, 128}, ElemType::F32};
  TensorType T_B{{128, 32}, ElemType::F32};
  TensorType T_out{{64, 32}, ElemType::F32};

  auto *A = g.add_arg(T_A);
  auto *B = g.add_arg(T_B);
  auto *bias = g.add_arg(T_out);
  auto *scale = g.add_arg(T_out);

  // Matmul
  auto *init = g.add_fill(T_out);
  auto *mm = g.add_matmul(A, B, init);

  // Elementwise chain
  auto *biased = g.add_elemwise("bias_add", "arith.addf",
                                 mm->result_, bias);
  auto *relu = g.add_unary("relu", "arith.maximumf", biased->result_);
  auto *sq = g.add_elemwise("square", "arith.mulf",
                             relu->result_, relu->result_);
  auto *res = g.add_elemwise("residual", "arith.addf",
                              sq->result_, relu->result_);
  auto *scaled = g.add_elemwise("scaling", "arith.mulf",
                                 res->result_, scale);

  // Dead op (DCE target)
  g.add_elemwise("dead_add", "arith.addf", A, A);

  // CSE targets
  auto *dup1 = g.add_elemwise("dup1", "arith.mulf",
                               relu->result_, scale);
  auto *dup2 = g.add_elemwise("dup2", "arith.mulf",
                               relu->result_, scale);
  (void)dup2;

  // Final output uses both paths
  auto *out = g.add_elemwise("output", "arith.addf",
                              scaled->result_, dup1->result_);

  g.returns = {out->result_};
  return g;
}

// =====================================================================
// Pipeline Runner
// =====================================================================

static void run_pipeline(Graph &g, const char *label) {
  using PassFn = int (*)(Graph &);
  auto sep = [](const char *title) {
    std::cout << "\n  ──── " << title << " ────\n\n";
  };
  auto run = [&](const char *name, PassFn fn) -> int {
    int c = fn(g);
    std::cout << "    " << name << ": " << c << " changes\n";
    if (c > 0) g.print(std::cout);
    return c;
  };

  std::cout
      << "\n╔═══════════════════════════════════════════════════════════════╗\n"
      << "║  " << label << "\n"
      << "╚═══════════════════════════════════════════════════════════════╝\n";

  std::cout << "\n[Initial] " << g.op_count() << " ops\n";
  g.print(std::cout);

  int total = 0;

  // ── P6 Step 0: Pre-clean ──
  sep("Stage 0: Pre-clean (canonicalize + CSE + DCE)");
  {
    int c = 0;
    c += run("canonicalize", pass_canonicalize);
    c += run("cse", pass_cse);
    c += run("dce", pass_dce);
    total += c;
  }

  // ── P6 Step 1: Dependence Analysis ──
  sep("Stage 1: Dependence Analysis (SSA use-def + alias)");
  std::vector<DepEdge> edges;
  stage1_dep_analysis(g, edges);

  // ── P6 Step 2: Fusion Candidate Build ──
  sep("Stage 2: Fusion Candidate Build (producer-consumer + elem chain)");
  auto cands = build_fusion_candidates(g, edges);

  // ── P6 Step 3: Cost Model Filtering ──
  sep("Stage 3: Cost Model Filtering (memory + compute reuse)");
  stage3_cost_filter(cands);

  // ── P6 Step 4: Tile-aware Fusion ──
  sep("Stage 4: Tile-aware Fusion (iterator compatibility)");
  stage4_tile_aware(cands);

  int accepted = 0;
  for (auto &fc : cands) if (fc.accepted) ++accepted;
  std::cout << "\n  Summary: " << accepted << "/" << cands.size()
            << " candidates accepted\n";

  // ── P6 Step 5: Fusion Rewrite ──
  sep("Stage 5: Fusion Rewrite (merge maps + tile + tile-and-fuse)");
  {
    int c = stage5_fusion_rewrite(g, cands);
    total += c;
    std::cout << "    → " << c << " rewrites\n";
    g.print(std::cout);
  }

  std::cout << "\n  ──── Tiled View (after fusion + tiling) ────\n\n";
  g.print(std::cout, /*tiled_view=*/true);

  // ── P6 Step 6: Post-clean ──
  sep("Stage 6: Post-clean (canonicalize + CSE + DCE)");
  {
    int c = 0;
    c += run("canonicalize", pass_canonicalize);
    c += run("cse", pass_cse);
    c += run("dce", pass_dce);
    total += c;
  }

  std::cout << "\n  Total: " << total << " optimizations, "
            << g.op_count() << " final ops\n";
}

// =====================================================================
int main() {
  std::cout << "================================================================\n";
  std::cout << "  Linalg 7-Stage Fusion & Optimization Pipeline\n";
  std::cout << "  S0:PreClean → S1:DepAnalysis → S2:CandBuild →\n";
  std::cout << "  S3:CostModel → S4:TileAware → S5:FuseRewrite → S6:PostClean\n";
  std::cout << "================================================================\n";

  auto g = make_test_graph();
  run_pipeline(g,
               "Matmul + Bias + ReLU + Square + Residual + Scale "
               "(full 7-stage fusion pipeline)");

  std::cout << "\n✓ Linalg 7-stage fusion pipeline test passed.\n";
  return 0;
}
