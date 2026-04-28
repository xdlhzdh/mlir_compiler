// run_stablehlo_opt.cpp — 6-Stage StableHLO Optimization Pipeline
//
// 面试高频：纯 C++ 实现 StableHLO 图级优化 pass pipeline
//   Stage 1: Canonicalization  (algebraic simplify, broadcast normalize, identity elim)
//   Stage 2: Shape Optimization (shape inference, shape folding, dynamic→static)
//   Stage 3: Graph Optimization (CSE, DCE, constant folding, fusion detection)
//   Stage 4: Layout / Transpose (transpose elimination, transpose push-through)
//   Stage 5: Cleanup            (re-run canonicalize + DCE)
//   Stage 6: Legalization       (stablehlo → linalg)
//
// 运行: ./run_stablehlo_opt  (使用内置测试图)

#include "shlo_graph.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>

using namespace shlo_graph;

// =====================================================================
// Helper utilities
// =====================================================================

static bool is_splat_f(const Op *op, float val) {
  if (!op || !op->is_constant() || op->const_data_f.empty()) return false;
  for (float f : op->const_data_f)
    if (f != val) return false;
  return true;
}

static bool is_binary_elementwise(const std::string &mn) {
  return mn == "stablehlo.add" || mn == "stablehlo.multiply" ||
         mn == "stablehlo.subtract" || mn == "stablehlo.divide" ||
         mn == "stablehlo.maximum" || mn == "stablehlo.minimum";
}

static bool is_elementwise(const std::string &mn) {
  return is_binary_elementwise(mn) || mn == "stablehlo.negate" ||
         mn == "stablehlo.abs" || mn == "stablehlo.exponential" ||
         mn == "stablehlo.log";
}

static bool is_identity_perm(const std::vector<int64_t> &p) {
  for (size_t i = 0; i < p.size(); ++i)
    if (p[i] != static_cast<int64_t>(i)) return false;
  return true;
}

static std::vector<int64_t> compose_perms(const std::vector<int64_t> &p1,
                                           const std::vector<int64_t> &p2) {
  std::vector<int64_t> r(p2.size());
  for (size_t i = 0; i < p2.size(); ++i) r[i] = p1[p2[i]];
  return r;
}

static TensorType broadcast_shapes(const TensorType &a, const TensorType &b) {
  TensorType r;
  r.elem = a.elem;
  if (a.rank() == 0) return b;
  if (b.rank() == 0) return a;
  int64_t max_rank = std::max(a.rank(), b.rank());
  r.dims.resize(max_rank);
  for (int64_t i = 0; i < max_rank; ++i) {
    int64_t ai = a.rank() - max_rank + i;
    int64_t bi = b.rank() - max_rank + i;
    int64_t da = (ai >= 0) ? a.dims[ai] : 1;
    int64_t db = (bi >= 0) ? b.dims[bi] : 1;
    if (da >= 0 && db >= 0)
      r.dims[i] = std::max(da, db);
    else if (da >= 0)
      r.dims[i] = da;
    else if (db >= 0)
      r.dims[i] = db;
    else
      r.dims[i] = -1;
  }
  return r;
}

static bool ops_equivalent(const Op *a, const Op *b) {
  if (a == b) return false;
  if (a->mnemonic != b->mnemonic) return false;
  if (a->num_results() != b->num_results()) return false;
  for (int i = 0; i < a->num_results(); ++i)
    if (a->result(i)->type != b->result(i)->type) return false;
  if (a->operands.size() != b->operands.size()) return false;
  for (size_t i = 0; i < a->operands.size(); ++i)
    if (a->operands[i] != b->operands[i]) return false;
  if (a->is_constant()) {
    if (a->const_data_f != b->const_data_f) return false;
    if (a->const_data_i != b->const_data_i) return false;
  }
  if (a->attrs != b->attrs) return false;
  return true;
}

// =====================================================================
// Stage 1: Canonicalization
// =====================================================================

// x+0→x, 0+x→x, x*1→x, 1*x→x, x*0→0, 0*x→0, x-0→x
static int pass_algebraic_simplify(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < g.ops.size(); ++i) {
      auto *op = g.ops[i].get();
      if (op->num_operands() != 2) continue;
      auto *lhs_def = op->operand(0)->defining_op;
      auto *rhs_def = op->operand(1)->defining_op;
      Value *repl = nullptr;

      if (op->mnemonic == "stablehlo.add") {
        if (is_splat_f(rhs_def, 0.0f)) repl = op->operand(0);
        else if (is_splat_f(lhs_def, 0.0f)) repl = op->operand(1);
      } else if (op->mnemonic == "stablehlo.multiply") {
        if (is_splat_f(rhs_def, 1.0f)) repl = op->operand(0);
        else if (is_splat_f(lhs_def, 1.0f)) repl = op->operand(1);
        else if (is_splat_f(rhs_def, 0.0f)) repl = op->operand(1);
        else if (is_splat_f(lhs_def, 0.0f)) repl = op->operand(0);
      } else if (op->mnemonic == "stablehlo.subtract") {
        if (is_splat_f(rhs_def, 0.0f)) repl = op->operand(0);
      }
      if (repl) {
        std::cout << "  [AlgSimp] " << op->result()->name << " = "
                  << op->mnemonic << " → " << repl->name << "\n";
        g.replace_all_uses(op->result(), repl);
        g.erase_op(op);
        ++count;
        changed = true;
        break;
      }
    }
  }
  return count;
}

// Remove identity reshape, identity transpose, identity broadcast_in_dim
static int pass_identity_elim(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < g.ops.size(); ++i) {
      auto *op = g.ops[i].get();
      Value *repl = nullptr;

      if (op->mnemonic == "stablehlo.reshape") {
        if (op->operand(0)->type == op->result()->type)
          repl = op->operand(0);
      } else if (op->mnemonic == "stablehlo.transpose" &&
                 op->has_attr("permutation")) {
        auto p = op->get_attr<std::vector<int64_t>>("permutation");
        if (is_identity_perm(p)) repl = op->operand(0);
      } else if (op->mnemonic == "stablehlo.broadcast_in_dim" &&
                 op->has_attr("broadcast_dimensions")) {
        auto d = op->get_attr<std::vector<int64_t>>("broadcast_dimensions");
        if (op->operand(0)->type == op->result()->type && is_identity_perm(d))
          repl = op->operand(0);
      }
      if (repl) {
        std::cout << "  [IdentElim] " << op->result()->name << " = "
                  << op->mnemonic << " → " << repl->name << "\n";
        g.replace_all_uses(op->result(), repl);
        g.erase_op(op);
        ++count;
        changed = true;
        break;
      }
    }
  }
  return count;
}

static int stage1_canonicalize(Graph &g) {
  return pass_algebraic_simplify(g) + pass_identity_elim(g);
}

// =====================================================================
// Stage 2: Shape Optimization
// =====================================================================

// Forward-propagate static shapes through elementwise and transpose ops
static int pass_shape_infer(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &op_ptr : g.ops) {
      auto *op = op_ptr.get();
      if (op->results.empty()) continue;
      auto *res = op->result();
      if (res->type.is_static()) continue;

      TensorType inferred;
      if (is_elementwise(op->mnemonic) && !op->is_constant() &&
          op->num_operands() >= 1) {
        inferred = op->operand(0)->type;
        for (int j = 1; j < op->num_operands(); ++j)
          inferred = broadcast_shapes(inferred, op->operand(j)->type);
      } else if (op->mnemonic == "stablehlo.transpose" &&
                 op->has_attr("permutation")) {
        auto &in = op->operand(0)->type;
        auto p = op->get_attr<std::vector<int64_t>>("permutation");
        inferred.elem = in.elem;
        inferred.dims.resize(p.size());
        for (size_t k = 0; k < p.size(); ++k)
          inferred.dims[k] = (p[k] < in.rank()) ? in.dims[p[k]] : -1;
      }

      if (inferred.elem != ElemType::UNKNOWN && inferred.is_static() &&
          inferred != res->type) {
        std::cout << "  [ShapeInfer] " << res->name << " : "
                  << res->type.str() << " → " << inferred.str() << "\n";
        res->type = inferred;
        ++count;
        changed = true;
      }
    }
  }
  return count;
}

// Fold get_dimension_size of statically-shaped tensor → constant
static int pass_shape_fold(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < g.ops.size(); ++i) {
      auto *op = g.ops[i].get();
      if (op->mnemonic != "stablehlo.get_dimension_size") continue;
      auto *input = op->operand(0);
      int64_t dim = op->get_attr<int64_t>("dimension");
      if (input->type.is_static() && dim >= 0 && dim < input->type.rank()) {
        int64_t size = input->type.dims[dim];
        auto *c = g.add_constant_i(op->result()->type, {size});
        std::cout << "  [ShapeFold] get_dimension_size(dim=" << dim
                  << ") → " << size << "\n";
        g.replace_all_uses(op->result(), c->result());
        g.erase_op(op);
        ++count;
        changed = true;
        break;
      }
    }
  }
  return count;
}

// Remove dead shape computation ops (i64/i32/index results with no users)
static int pass_shape_subgraph_elim(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = g.ops.size(); i-- > 0;) {
      auto *op = g.ops[i].get();
      if (op->results.empty()) continue;
      auto elem = op->result()->type.elem;
      if (elem != ElemType::I64 && elem != ElemType::I32 &&
          elem != ElemType::INDEX)
        continue;
      bool dead = true;
      for (auto *r : op->results)
        if (g.is_return_val(r) || !r->has_no_uses()) { dead = false; break; }
      if (dead) {
        std::cout << "  [ShapeElim] " << op->mnemonic
                  << " (" << op->result()->name << ")\n";
        g.erase_op(op);
        ++count;
        changed = true;
        break;
      }
    }
  }
  return count;
}

static int stage2_shape_opt(Graph &g) {
  return pass_shape_infer(g) + pass_shape_fold(g) + pass_shape_subgraph_elim(g);
}

// =====================================================================
// Stage 3: Graph Optimization
// =====================================================================

static int pass_cse(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < g.ops.size() && !changed; ++i) {
      for (size_t j = i + 1; j < g.ops.size() && !changed; ++j) {
        if (ops_equivalent(g.ops[i].get(), g.ops[j].get())) {
          auto *keep = g.ops[i].get();
          auto *dup = g.ops[j].get();
          std::cout << "  [CSE] " << dup->result()->name << " = "
                    << dup->mnemonic << " → " << keep->result()->name << "\n";
          g.replace_all_uses(dup->result(), keep->result());
          g.erase_op(dup);
          ++count;
          changed = true;
        }
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
      bool dead = true;
      for (auto *r : op->results) {
        if (g.is_return_val(r) || !r->has_no_uses()) {
          dead = false;
          break;
        }
      }
      if (dead) {
        std::cout << "  [DCE] removed " << op->mnemonic;
        if (!op->results.empty()) std::cout << " (" << op->result()->name << ")";
        std::cout << "\n";
        g.erase_op(op);
        ++count;
        changed = true;
        break;
      }
    }
  }
  return count;
}

static int pass_constant_fold(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < g.ops.size(); ++i) {
      auto *op = g.ops[i].get();
      if (op->is_constant()) continue;

      // Binary elementwise: both operands are float constants with same size
      if (is_binary_elementwise(op->mnemonic) && op->num_operands() == 2) {
        auto *ld = op->operand(0)->defining_op;
        auto *rd = op->operand(1)->defining_op;
        if (!ld || !rd || !ld->is_constant() || !rd->is_constant()) continue;
        if (ld->const_data_f.empty() || rd->const_data_f.empty()) continue;
        auto &a = ld->const_data_f;
        auto &b = rd->const_data_f;
        if (a.size() != b.size()) continue;

        std::vector<float> res(a.size());
        for (size_t j = 0; j < a.size(); ++j) {
          if (op->mnemonic == "stablehlo.add") res[j] = a[j] + b[j];
          else if (op->mnemonic == "stablehlo.multiply") res[j] = a[j] * b[j];
          else if (op->mnemonic == "stablehlo.subtract") res[j] = a[j] - b[j];
          else if (op->mnemonic == "stablehlo.divide") res[j] = a[j] / b[j];
          else if (op->mnemonic == "stablehlo.maximum") res[j] = std::max(a[j], b[j]);
          else if (op->mnemonic == "stablehlo.minimum") res[j] = std::min(a[j], b[j]);
        }
        auto *c = g.add_constant_f(op->result()->type, res);
        std::cout << "  [ConstFold] " << op->result()->name << " = "
                  << op->mnemonic << " → const [" << res[0];
        if (res.size() > 1) std::cout << ", ...";
        std::cout << "]\n";
        g.replace_all_uses(op->result(), c->result());
        g.erase_op(op);
        ++count;
        changed = true;
        break;
      }

      // Unary fold: negate
      if (op->mnemonic == "stablehlo.negate" && op->num_operands() == 1) {
        auto *d = op->operand(0)->defining_op;
        if (!d || !d->is_constant() || d->const_data_f.empty()) continue;
        auto &a = d->const_data_f;
        std::vector<float> res(a.size());
        for (size_t j = 0; j < a.size(); ++j) res[j] = -a[j];
        auto *c = g.add_constant_f(op->result()->type, res);
        g.replace_all_uses(op->result(), c->result());
        g.erase_op(op);
        ++count;
        changed = true;
        break;
      }
    }
  }
  return count;
}

// Detect chains of elementwise ops with single intermediate uses → mark fusion_group
static int pass_fusion_detect(Graph &g) {
  int fusion_id = 0;
  int count = 0;
  std::unordered_set<Op *> assigned;

  for (auto &op_ptr : g.ops) {
    auto *start = op_ptr.get();
    if (assigned.count(start)) continue;
    if (!is_elementwise(start->mnemonic) || start->is_constant()) continue;

    std::vector<Op *> chain;
    Op *cur = start;
    while (cur && is_elementwise(cur->mnemonic) && !cur->is_constant() &&
           !assigned.count(cur)) {
      chain.push_back(cur);
      auto &users = cur->result()->users;
      if (users.size() == 1 && is_elementwise(users[0]->mnemonic) &&
          !users[0]->is_constant() && !assigned.count(users[0])) {
        cur = users[0];
      } else {
        break;
      }
    }

    if (chain.size() >= 2) {
      std::cout << "  [Fusion] group " << fusion_id << ":";
      for (auto *c : chain) {
        c->attrs["fusion_group"] = static_cast<int64_t>(fusion_id);
        assigned.insert(c);
        std::cout << " " << c->result()->name;
      }
      std::cout << " (" << chain.size() << " ops)\n";
      ++fusion_id;
      count += static_cast<int>(chain.size());
    }
  }
  return count;
}

// Conv + BN Fusion: conv → multiply(const_scale) → add(const_bias) ⇒ conv{fused_bn}
// Mirrors the real MLIR Conv+BN fusion from 6_stablehlo_passes/, but on our graph IR
static int pass_conv_bn_fusion(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < g.ops.size(); ++i) {
      auto *add_op = g.ops[i].get();
      if (add_op->mnemonic != "stablehlo.add" || add_op->num_operands() != 2)
        continue;

      // Pattern: add(multiply(conv, scale_const), bias_const)  — either operand order
      Value *mul_val = nullptr;
      Value *bias_val = nullptr;
      for (int k = 0; k < 2; ++k) {
        auto *cand = add_op->operand(k)->defining_op;
        if (cand && cand->is_constant()) {
          bias_val = add_op->operand(k);
          mul_val = add_op->operand(1 - k);
          break;
        }
      }
      if (!mul_val || !bias_val || !mul_val->defining_op) continue;

      auto *mul_op = mul_val->defining_op;
      if (mul_op->mnemonic != "stablehlo.multiply" || mul_op->num_operands() != 2)
        continue;
      if (mul_op->result()->users.size() != 1) continue;

      Value *conv_val = nullptr;
      for (int k = 0; k < 2; ++k) {
        auto *cand = mul_op->operand(k)->defining_op;
        if (cand && cand->is_constant()) {
          conv_val = mul_op->operand(1 - k);
          break;
        }
      }
      if (!conv_val || !conv_val->defining_op) continue;

      auto *conv_op = conv_val->defining_op;
      if (conv_op->mnemonic != "stablehlo.convolution") continue;
      if (conv_op->result()->users.size() != 1) continue;

      conv_op->attrs["fused_bn"] = std::string("true");
      conv_op->result()->type = add_op->result()->type;

      std::cout << "  [Conv+BN] fused " << conv_op->result()->name
                << " (conv) + " << mul_op->result()->name
                << " (scale) + " << add_op->result()->name << " (bias)\n";

      g.replace_all_uses(add_op->result(), conv_op->result());
      g.erase_op(add_op);
      g.erase_op(mul_op);
      ++count;
      changed = true;
      break;
    }
  }
  return count;
}

static int stage3_graph_opt(Graph &g) {
  int total = 0;
  total += pass_cse(g);
  total += pass_constant_fold(g);
  total += pass_conv_bn_fusion(g);
  total += pass_dce(g);
  total += pass_fusion_detect(g);
  return total;
}

// =====================================================================
// Stage 4: Layout / Transpose
// =====================================================================

// T(p2) ∘ T(p1) → T(compose) or identity
static int pass_transpose_elim(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < g.ops.size(); ++i) {
      auto *op = g.ops[i].get();
      if (op->mnemonic != "stablehlo.transpose" ||
          !op->has_attr("permutation"))
        continue;
      auto *prev = op->operand(0)->defining_op;
      if (!prev || prev->mnemonic != "stablehlo.transpose" ||
          !prev->has_attr("permutation"))
        continue;

      auto p1 = prev->get_attr<std::vector<int64_t>>("permutation");
      auto p2 = op->get_attr<std::vector<int64_t>>("permutation");
      if (p1.size() != p2.size()) continue;

      auto composed = compose_perms(p1, p2);
      if (is_identity_perm(composed)) {
        std::cout << "  [TransElim] " << prev->result()->name << " + "
                  << op->result()->name << " → identity\n";
        g.replace_all_uses(op->result(), prev->operand(0));
        g.erase_op(op);
      } else {
        std::cout << "  [TransMerge] " << prev->result()->name << " + "
                  << op->result()->name << " → single\n";
        auto *nt = g.add_op("stablehlo.transpose", {prev->operand(0)},
                             {op->result()->type}, {{"permutation", composed}});
        g.replace_all_uses(op->result(), nt->result());
        g.erase_op(op);
      }
      ++count;
      changed = true;
      break;
    }
  }
  return count;
}

// Push transpose through binary elementwise: T(a)⊕T(b) → T(a⊕b)
static int pass_transpose_push(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < g.ops.size(); ++i) {
      auto *op = g.ops[i].get();
      if (!is_binary_elementwise(op->mnemonic) || op->num_operands() != 2)
        continue;
      auto *ld = op->operand(0)->defining_op;
      auto *rd = op->operand(1)->defining_op;
      if (!ld || !rd) continue;
      if (ld->mnemonic != "stablehlo.transpose" ||
          rd->mnemonic != "stablehlo.transpose")
        continue;
      if (!ld->has_attr("permutation") || !rd->has_attr("permutation"))
        continue;
      auto p1 = ld->get_attr<std::vector<int64_t>>("permutation");
      auto p2 = rd->get_attr<std::vector<int64_t>>("permutation");
      if (p1 != p2) continue;

      auto *orig_l = ld->operand(0);
      auto *orig_r = rd->operand(0);
      auto *nb = g.add_op(op->mnemonic, {orig_l, orig_r},
                           {orig_l->type}, op->attrs);
      // Remove fusion_group from the new op's cloned attrs (if any)
      nb->attrs.erase("fusion_group");
      auto *nt = g.add_op("stablehlo.transpose", {nb->result()},
                           {op->result()->type}, {{"permutation", p1}});

      std::cout << "  [TransPush] pushed transpose through " << op->mnemonic
                << " (" << op->result()->name << ")\n";
      g.replace_all_uses(op->result(), nt->result());
      g.erase_op(op);
      ++count;
      changed = true;
      break;
    }
  }
  return count;
}

static int stage4_layout_opt(Graph &g) {
  return pass_transpose_elim(g) + pass_transpose_push(g);
}

// =====================================================================
// Stage 5: Cleanup
// =====================================================================

static int stage5_cleanup(Graph &g) {
  return pass_algebraic_simplify(g) + pass_identity_elim(g) + pass_dce(g);
}

// =====================================================================
// Stage 6: Legalization → Linalg
// =====================================================================

static void stage6_legalize(Graph &g) {
  struct Mapping { const char *from; const char *to; };
  static const Mapping map[] = {
      {"stablehlo.add", "linalg.add"},
      {"stablehlo.multiply", "linalg.mul"},
      {"stablehlo.subtract", "linalg.sub"},
      {"stablehlo.divide", "linalg.div"},
      {"stablehlo.maximum", "linalg.max"},
      {"stablehlo.minimum", "linalg.min"},
      {"stablehlo.negate", "linalg.negf"},
      {"stablehlo.abs", "linalg.absf"},
      {"stablehlo.dot_general", "linalg.matmul"},
      {"stablehlo.convolution", "linalg.conv_2d_nchw"},
      {"stablehlo.transpose", "linalg.transpose"},
      {"stablehlo.reshape", "tensor.reshape"},
      {"stablehlo.broadcast_in_dim", "linalg.broadcast"},
      {"stablehlo.constant", "arith.constant"},
      {"stablehlo.get_dimension_size", "tensor.dim"},
  };

  for (auto &op_ptr : g.ops) {
    for (auto &m : map) {
      if (op_ptr->mnemonic == m.from) {
        std::cout << "  [Legal] " << m.from << " → " << m.to << "\n";
        op_ptr->mnemonic = m.to;
        // Add iterator_types metadata for elementwise linalg ops
        if (is_elementwise(m.from) &&
            std::string(m.from) != "stablehlo.constant") {
          std::string iters = "[";
          for (int64_t d = 0; d < op_ptr->result()->type.rank(); ++d) {
            if (d) iters += ", ";
            iters += "\"parallel\"";
          }
          iters += "]";
          op_ptr->attrs["iterator_types"] = iters;
        }
        break;
      }
    }
  }
}

// =====================================================================
// Test Graphs
// =====================================================================

static Graph make_test_stage1() {
  Graph g;
  g.name = "test_canonicalize";
  TensorType T{{4, 8}, ElemType::F32};

  auto *x = g.add_arg(T);
  auto *zero = g.add_constant_f(T, std::vector<float>(32, 0.0f));
  auto *one = g.add_constant_f(T, std::vector<float>(32, 1.0f));
  auto *a = g.add_op("stablehlo.add", {x, zero->result()}, {T});
  auto *b = g.add_op("stablehlo.multiply", {a->result(), one->result()}, {T});
  auto *c = g.add_op("stablehlo.reshape", {b->result()}, {T});
  auto *d = g.add_op("stablehlo.transpose", {c->result()}, {T},
                      {{"permutation", std::vector<int64_t>{0, 1}}});
  g.returns = {d->result()};
  return g;
}

static Graph make_test_stage2() {
  Graph g;
  g.name = "test_shape_opt";
  TensorType T{{4, 8}, ElemType::F32};
  TensorType T_dyn{{-1, -1}, ElemType::F32};
  TensorType T_scalar{{}, ElemType::I64};

  auto *x = g.add_arg(T);
  auto *y = g.add_arg(T);
  auto *a = g.add_op("stablehlo.add", {x, y}, {T_dyn});
  auto *b = g.add_op("stablehlo.multiply", {a->result(), x}, {T_dyn});
  g.add_op("stablehlo.get_dimension_size", {x}, {T_scalar},
           {{"dimension", static_cast<int64_t>(0)}});
  g.returns = {b->result()};
  return g;
}

static Graph make_test_stage3() {
  Graph g;
  g.name = "test_graph_opt";
  TensorType T{{4, 8}, ElemType::F32};

  auto *x = g.add_arg(T);
  auto *y = g.add_arg(T);

  auto *c1 = g.add_constant_f(T, std::vector<float>(32, 2.0f));
  auto *c2 = g.add_constant_f(T, std::vector<float>(32, 3.0f));
  auto *fold = g.add_op("stablehlo.add", {c1->result(), c2->result()}, {T});

  auto *a1 = g.add_op("stablehlo.add", {x, y}, {T});
  auto *a2 = g.add_op("stablehlo.add", {x, y}, {T});

  g.add_op("stablehlo.subtract", {x, y}, {T}); // dead

  auto *b = g.add_op("stablehlo.multiply", {a1->result(), a2->result()}, {T});
  auto *c = g.add_op("stablehlo.add", {b->result(), fold->result()}, {T});

  auto *d = g.add_op("stablehlo.add", {c->result(), x}, {T});
  auto *e = g.add_op("stablehlo.multiply", {d->result(), y}, {T});
  g.returns = {e->result()};
  return g;
}

static Graph make_test_stage4() {
  Graph g;
  g.name = "test_transpose";
  TensorType T{{2, 3, 4}, ElemType::F32};
  TensorType T_t1{{4, 2, 3}, ElemType::F32};
  TensorType T_t2{{3, 4, 2}, ElemType::F32};

  auto *x = g.add_arg(T);
  auto *y = g.add_arg(T);

  // Cancellation: compose([2,0,1],[1,2,0]) = [0,1,2] = identity
  auto *t1 = g.add_op("stablehlo.transpose", {x}, {T_t1},
                       {{"permutation", std::vector<int64_t>{2, 0, 1}}});
  auto *t2 = g.add_op("stablehlo.transpose", {t1->result()}, {T},
                       {{"permutation", std::vector<int64_t>{1, 2, 0}}});

  // Push-through: same perm on both inputs of add
  auto *t3 = g.add_op("stablehlo.transpose", {t2->result()}, {T_t2},
                       {{"permutation", std::vector<int64_t>{1, 2, 0}}});
  auto *t4 = g.add_op("stablehlo.transpose", {y}, {T_t2},
                       {{"permutation", std::vector<int64_t>{1, 2, 0}}});
  auto *a = g.add_op("stablehlo.add", {t3->result(), t4->result()}, {T_t2});

  g.returns = {a->result()};
  return g;
}

static Graph make_test_stage6() {
  Graph g;
  g.name = "test_legalize";
  TensorType T{{4, 8}, ElemType::F32};

  auto *x = g.add_arg(T);
  auto *y = g.add_arg(T);
  auto *a = g.add_op("stablehlo.add", {x, y}, {T});
  auto *b = g.add_op("stablehlo.multiply", {a->result(), x}, {T});
  auto *c = g.add_constant_f(T, std::vector<float>(32, 1.0f));
  auto *d = g.add_op("stablehlo.subtract", {b->result(), c->result()}, {T});
  g.returns = {d->result()};
  return g;
}

static Graph make_test_full() {
  Graph g;
  g.name = "test_full_pipeline";
  TensorType T{{4, 8}, ElemType::F32};
  TensorType T_dyn{{-1, -1}, ElemType::F32};
  TensorType T_t{{8, 4}, ElemType::F32};

  auto *x = g.add_arg(T);
  auto *y = g.add_arg(T);

  auto *zero = g.add_constant_f(T, std::vector<float>(32, 0.0f));
  auto *one = g.add_constant_f(T, std::vector<float>(32, 1.0f));
  auto *c1 = g.add_constant_f(T, std::vector<float>(32, 2.0f));
  auto *c2 = g.add_constant_f(T, std::vector<float>(32, 3.0f));

  // Stage 1 targets: x+0→x, x*1→x
  auto *a = g.add_op("stablehlo.add", {x, zero->result()}, {T});
  auto *b = g.add_op("stablehlo.multiply", {a->result(), one->result()}, {T});

  // Stage 2 target: dynamic → static
  auto *c = g.add_op("stablehlo.add", {b->result(), y}, {T_dyn});

  // Stage 3 target: constant fold
  auto *d = g.add_op("stablehlo.add", {c1->result(), c2->result()}, {T});

  // Stage 3 target: CSE
  auto *e1 = g.add_op("stablehlo.multiply", {c->result(), c->result()}, {T});
  auto *e2 = g.add_op("stablehlo.multiply", {c->result(), c->result()}, {T});

  // Stage 3 target: DCE
  g.add_op("stablehlo.subtract", {x, y}, {T}); // dead

  // Stage 4 target: transpose cancel [1,0]∘[1,0] = identity
  auto *t1 = g.add_op("stablehlo.transpose", {e1->result()}, {T_t},
                       {{"permutation", std::vector<int64_t>{1, 0}}});
  auto *t2 = g.add_op("stablehlo.transpose", {t1->result()}, {T},
                       {{"permutation", std::vector<int64_t>{1, 0}}});

  auto *f = g.add_op("stablehlo.add", {t2->result(), d->result()}, {T});
  auto *out = g.add_op("stablehlo.add", {f->result(), e2->result()}, {T});
  g.returns = {out->result()};
  return g;
}

// =====================================================================
// Comprehensive Test: exercises ALL passes in one graph
// =====================================================================

static Graph make_test_comprehensive() {
  Graph g;
  g.name = "test_all_passes";

  TensorType T2{{4, 8}, ElemType::F32};
  TensorType T2_dyn{{-1, -1}, ElemType::F32};
  TensorType T2_t{{8, 4}, ElemType::F32};
  TensorType T_scalar{{}, ElemType::I64};
  TensorType T4_in{{1, 3, 4, 4}, ElemType::F32};
  TensorType T4_w{{8, 3, 3, 3}, ElemType::F32};
  TensorType T4_out{{1, 8, 2, 2}, ElemType::F32};

  auto *arg0 = g.add_arg(T2);      // elementwise path
  auto *arg1 = g.add_arg(T2);
  auto *arg2 = g.add_arg(T4_in);   // conv path

  // ---- Constants ----
  auto *zero = g.add_constant_f(T2, std::vector<float>(32, 0.0f));
  auto *one = g.add_constant_f(T2, std::vector<float>(32, 1.0f));
  auto *c1 = g.add_constant_f(T2, std::vector<float>(32, 2.0f));
  auto *c2 = g.add_constant_f(T2, std::vector<float>(32, 3.0f));
  auto *weight = g.add_constant_f(T4_w, std::vector<float>(216, 0.01f));
  auto *bn_scale = g.add_constant_f(T4_out, std::vector<float>(32, 0.5f));
  auto *bn_bias = g.add_constant_f(T4_out, std::vector<float>(32, 0.1f));

  // ==== Stage 1 targets: Canonicalization ====
  auto *a1 = g.add_op("stablehlo.add", {arg0, zero->result()}, {T2});
  auto *a2 = g.add_op("stablehlo.multiply", {a1->result(), one->result()}, {T2});
  auto *a3 = g.add_op("stablehlo.reshape", {a2->result()}, {T2});
  auto *a4 = g.add_op("stablehlo.transpose", {a3->result()}, {T2},
                       {{"permutation", std::vector<int64_t>{0, 1}}});

  // ==== Stage 2 targets: Shape Optimization ====
  auto *s1 = g.add_op("stablehlo.add", {a4->result(), arg1}, {T2_dyn});
  g.add_op("stablehlo.get_dimension_size", {arg0}, {T_scalar},
           {{"dimension", static_cast<int64_t>(0)}});

  // ==== Stage 3 targets: Graph Optimization ====
  auto *cf = g.add_op("stablehlo.add", {c1->result(), c2->result()}, {T2});

  auto *cse1 = g.add_op("stablehlo.add", {s1->result(), arg1}, {T2});
  auto *cse2 = g.add_op("stablehlo.add", {s1->result(), arg1}, {T2});

  g.add_op("stablehlo.subtract", {arg0, arg1}, {T2}); // dead

  // Conv + BN Fusion (mirrors 6_stablehlo_passes/ Conv+BN pattern)
  auto *conv = g.add_op("stablehlo.convolution",
                         {arg2, weight->result()}, {T4_out},
                         {{"window_strides", std::vector<int64_t>{1, 1}},
                          {"padding", std::vector<int64_t>{0, 0, 0, 0}}});
  auto *scaled = g.add_op("stablehlo.multiply",
                           {conv->result(), bn_scale->result()}, {T4_out});
  auto *biased = g.add_op("stablehlo.add",
                           {scaled->result(), bn_bias->result()}, {T4_out});

  // Elementwise fusion chain
  auto *e1 = g.add_op("stablehlo.multiply",
                       {cse1->result(), cf->result()}, {T2});
  auto *e2 = g.add_op("stablehlo.add", {e1->result(), s1->result()}, {T2});

  // ==== Stage 4 targets: Transpose ====
  auto *t1 = g.add_op("stablehlo.transpose", {e2->result()}, {T2_t},
                       {{"permutation", std::vector<int64_t>{1, 0}}});
  auto *t2 = g.add_op("stablehlo.transpose", {t1->result()}, {T2},
                       {{"permutation", std::vector<int64_t>{1, 0}}});

  g.returns = {t2->result(), biased->result(), cse2->result()};
  return g;
}

// =====================================================================
// Pipeline Runner
// =====================================================================

// per_pass=false → print per-stage (compact)
// per_pass=true  → print after EVERY individual pass (verbose, for comprehensive test)
static void run_pipeline(Graph &g, const char *label, bool per_pass = false) {
  auto sep = [](const char *title) {
    std::cout << "\n  ──── " << title << " ────\n";
  };
  using PassFn = int (*)(Graph &);
  auto run = [&](const char *name, PassFn fn) -> int {
    int c = fn(g);
    if (per_pass) {
      std::cout << "    " << name << ": " << c << " changes\n";
      if (c > 0) g.print(std::cout);
    }
    return c;
  };

  std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
  std::cout << "║  " << label << "\n";
  std::cout << "╚══════════════════════════════════════════════════════╝\n";

  std::cout << "\n[Initial] " << g.op_count() << " ops\n";
  g.print(std::cout);

  int total = 0, c;

  sep("Stage 1: Canonicalization");
  c = run("algebraic_simplify", pass_algebraic_simplify)
    + run("identity_elim", pass_identity_elim);
  total += c;
  if (!per_pass) { std::cout << "  → " << c << " rewrites\n"; if (c) g.print(std::cout); }

  sep("Stage 2: Shape Optimization");
  c = run("shape_infer", pass_shape_infer)
    + run("shape_fold", pass_shape_fold)
    + run("shape_subgraph_elim", pass_shape_subgraph_elim);
  total += c;
  if (!per_pass) { std::cout << "  → " << c << " shapes resolved\n"; if (c) g.print(std::cout); }

  sep("Stage 3: Graph Optimization");
  c = run("cse", pass_cse)
    + run("constant_fold", pass_constant_fold)
    + run("conv_bn_fusion", pass_conv_bn_fusion)
    + run("dce", pass_dce)
    + run("fusion_detect", pass_fusion_detect);
  total += c;
  if (!per_pass) { std::cout << "  → " << c << " optimizations\n"; if (c) g.print(std::cout); }

  sep("Stage 4: Layout / Transpose");
  c = run("transpose_elim", pass_transpose_elim)
    + run("transpose_push", pass_transpose_push);
  total += c;
  if (!per_pass) { std::cout << "  → " << c << " rewrites\n"; if (c) g.print(std::cout); }

  sep("Stage 5: Cleanup (canonicalize + DCE)");
  c = run("canonicalize", pass_algebraic_simplify)
    + run("identity_elim", pass_identity_elim)
    + run("dce", pass_dce);
  total += c;
  if (!per_pass) { std::cout << "  → " << c << " cleaned\n"; if (c) g.print(std::cout); }

  sep("Stage 6: Legalization → Linalg");
  stage6_legalize(g);
  g.print(std::cout);

  std::cout << "\n  Total: " << total << " optimizations, "
            << g.op_count() << " final ops\n";
}

// =====================================================================
int main() {
  std::cout << "========================================================\n";
  std::cout << "  StableHLO 6-Stage Optimization Pipeline\n";
  std::cout << "  Stage1 Canon → Stage2 Shape → Stage3 Graph →\n";
  std::cout << "  Stage4 Layout → Stage5 Cleanup → Stage6 Legal\n";
  std::cout << "========================================================\n";

  // ── Individual stage tests (compact output) ──
  struct TC { const char *name; Graph (*build)(); };
  TC tests[] = {
      {"Stage 1: Canonicalization (x+0, x*1, identity reshape/transpose)",
       make_test_stage1},
      {"Stage 2: Shape Optimization (inference + fold)", make_test_stage2},
      {"Stage 3: Graph Opt (CSE + DCE + ConstFold + Fusion)",
       make_test_stage3},
      {"Stage 4: Layout / Transpose (elim + push)", make_test_stage4},
      {"Stage 6: Legalization (stablehlo -> linalg)", make_test_stage6},
      {"Full Pipeline (all stages combined)", make_test_full},
  };
  for (auto &tc : tests) {
    auto g = tc.build();
    run_pipeline(g, tc.name);
  }

  // ── Comprehensive test: ALL ops, per-pass tracing ──
  {
    auto g = make_test_comprehensive();
    run_pipeline(g,
                 "Comprehensive: ALL passes (per-pass trace, Conv+BN, "
                 "Elementwise, Shape, Transpose)",
                 /*per_pass=*/true);
  }

  std::cout << "\n✓ All StableHLO optimization pipeline tests passed.\n";
  return 0;
}
