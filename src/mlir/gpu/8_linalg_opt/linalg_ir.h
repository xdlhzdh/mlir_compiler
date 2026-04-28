#pragma once
// linalg_ir.h — Simplified Linalg tensor-level IR for fusion pipeline.
//
// Key Linalg concepts modelled:
//   • IndexingMap  — affine map from iteration domain to tensor dimensions
//   • IterKind     — parallel vs reduction iterator
//   • BodyStep     — scalar operation inside a linalg.generic region
//   • DepEdge      — use-def / alias dependency between ops
//   • FusionCandidate — pair of ops eligible for fusion + cost info
//   • FusedOp      — consumer fused into a tiled producer (tile-and-fuse)
//
// Header-only, pure C++17, no external dependencies.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace linalg_ir {

// ======================== Element Type ========================

enum class ElemType { F32, F64, I32, I64 };

inline const char *elem_str(ElemType t) {
  switch (t) {
  case ElemType::F32: return "f32";
  case ElemType::F64: return "f64";
  case ElemType::I32: return "i32";
  case ElemType::I64: return "i64";
  }
  return "?";
}

inline int64_t elem_bytes(ElemType t) {
  switch (t) {
  case ElemType::F32: case ElemType::I32: return 4;
  case ElemType::F64: case ElemType::I64: return 8;
  }
  return 4;
}

// ======================== Tensor Type ========================

struct TensorType {
  std::vector<int64_t> dims;
  ElemType elem = ElemType::F32;
  int64_t rank() const { return static_cast<int64_t>(dims.size()); }
  int64_t num_elements() const {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    return n;
  }
  int64_t bytes() const { return num_elements() * elem_bytes(elem); }
  std::string str() const {
    std::ostringstream os;
    os << "tensor<";
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i) os << "x";
      os << dims[i];
    }
    if (!dims.empty()) os << "x";
    os << elem_str(elem) << ">";
    return os.str();
  }
  bool operator==(const TensorType &o) const {
    return dims == o.dims && elem == o.elem;
  }
};

// ======================== Indexing Map ========================

struct IndexingMap {
  std::vector<int> dim_pos;
  std::string str(const std::vector<std::string> &dnames) const {
    std::ostringstream os;
    os << "(";
    for (size_t i = 0; i < dnames.size(); ++i) {
      if (i) os << ", ";
      os << dnames[i];
    }
    os << ") -> (";
    for (size_t i = 0; i < dim_pos.size(); ++i) {
      if (i) os << ", ";
      os << dnames[dim_pos[i]];
    }
    os << ")";
    return os.str();
  }
  bool operator==(const IndexingMap &o) const { return dim_pos == o.dim_pos; }
};

// ======================== Iterator Kind ========================

enum class IterKind { PARALLEL, REDUCTION };

inline const char *iter_str(IterKind k) {
  return k == IterKind::PARALLEL ? "parallel" : "reduction";
}

// ======================== Forward ========================

struct Op;

// ======================== SSA Value ========================

struct Value {
  int id = 0;
  std::string name;
  TensorType type;
  Op *defining_op = nullptr;
  std::vector<Op *> users;

  void add_user(Op *op) {
    if (std::find(users.begin(), users.end(), op) == users.end())
      users.push_back(op);
  }
  void remove_user(Op *op) {
    users.erase(std::remove(users.begin(), users.end(), op), users.end());
  }
  bool has_no_uses() const { return users.empty(); }
};

// ======================== Body Step ========================

struct BodyStep {
  std::string op;  // "arith.mulf", "arith.addf", "arith.subf", ...
  int arg0, arg1;
  int res;
};

// ======================== Dependency Edge ========================

enum class DepKind { USE_DEF, ALIAS };

struct DepEdge {
  Op *producer;
  Op *consumer;
  DepKind kind;
  Value *through;
};

// ======================== Fusion Candidate ========================

struct FusionCandidate {
  Op *producer;
  Op *consumer;
  int64_t mem_saved_bytes = 0;       // intermediate tensor eliminated
  double compute_reuse = 0.0;        // ratio: shared_iters / total_iters
  bool tile_compatible = false;      // iterator types match for fusion
  bool accepted = false;
  std::string reject_reason;
};

// ======================== Fused Consumer (tile-and-fuse) ========================

struct FusedOp {
  std::string label;
  std::vector<Value *> extra_ins;
  std::vector<BodyStep> body;
  int yield_idx = -1;
};

// ======================== Operation ========================

struct Op {
  std::string mnemonic;
  std::string label;

  std::vector<Value *> ins;
  std::vector<Value *> outs;
  Value *result_ = nullptr;

  std::vector<IndexingMap> maps;
  std::vector<IterKind> iters;
  std::vector<int64_t> loop_ranges;
  std::vector<BodyStep> body;
  int yield_idx = -1;

  std::vector<int64_t> tile_sizes;
  int cluster_id = -1;
  std::vector<FusedOp> fused_consumers;

  int n_ins() const { return static_cast<int>(ins.size()); }
  int n_outs() const { return static_cast<int>(outs.size()); }
  int n_block_args() const { return n_ins() + n_outs(); }

  bool is_elementwise() const {
    for (auto k : iters)
      if (k == IterKind::REDUCTION) return false;
    return !iters.empty();
  }
  bool has_reduction() const {
    for (auto k : iters)
      if (k == IterKind::REDUCTION) return true;
    return false;
  }
  bool is_fill() const { return mnemonic == "linalg.fill"; }
  bool is_generic() const {
    return mnemonic == "linalg.generic" || mnemonic == "linalg.matmul";
  }

  std::string ba_name(int idx) const {
    if (idx < n_ins()) return "%in" + std::to_string(idx);
    int oi = idx - n_ins();
    if (oi < n_outs()) return "%acc";
    return "%t" + std::to_string(idx - n_block_args());
  }

  std::vector<std::string> dim_names() const {
    if (iters.size() == 3 && iters[2] == IterKind::REDUCTION)
      return {"m", "n", "k"};
    std::vector<std::string> v;
    for (size_t i = 0; i < iters.size(); ++i)
      v.push_back("d" + std::to_string(i));
    return v;
  }

  void print(std::ostream &os) const {
    if (is_fill()) {
      os << "    " << result_->name << " = linalg.fill(0.0) : "
         << result_->type.str() << "\n";
      return;
    }
    auto dn = dim_names();
    os << "    " << result_->name << " = " << mnemonic;
    if (!label.empty()) os << "  // " << label;
    os << "\n";
    os << "      ins(";
    for (size_t i = 0; i < ins.size(); ++i) {
      if (i) os << ", ";
      os << ins[i]->name << ": " << ins[i]->type.str();
    }
    os << ")";
    if (!outs.empty()) {
      os << " outs(";
      for (size_t i = 0; i < outs.size(); ++i) {
        if (i) os << ", ";
        os << outs[i]->name << ": " << outs[i]->type.str();
      }
      os << ")";
    }
    os << "\n      maps: [";
    for (size_t i = 0; i < maps.size(); ++i) {
      if (i) os << ", ";
      os << maps[i].str(dn);
    }
    os << "]\n      iters: [";
    for (size_t i = 0; i < iters.size(); ++i) {
      if (i) os << ", ";
      os << "\"" << iter_str(iters[i]) << "\"";
    }
    os << "]  ranges: [";
    for (size_t i = 0; i < loop_ranges.size(); ++i) {
      if (i) os << ", ";
      os << loop_ranges[i];
    }
    os << "]\n      body {\n";
    for (auto &s : body) {
      os << "        " << ba_name(s.res) << " = " << s.op << " "
         << ba_name(s.arg0) << ", " << ba_name(s.arg1) << "\n";
    }
    os << "        linalg.yield " << ba_name(yield_idx) << "\n";
    os << "      }";
    if (!tile_sizes.empty()) {
      os << "  tile: [";
      for (size_t i = 0; i < tile_sizes.size(); ++i) {
        if (i) os << ", ";
        os << tile_sizes[i];
      }
      os << "]";
    }
    if (cluster_id >= 0) os << "  cluster=" << cluster_id;
    os << "\n      : " << result_->type.str() << "\n";
  }

  void print_tiled(std::ostream &os) const {
    if (tile_sizes.empty()) { print(os); return; }
    auto dn = dim_names();
    os << "    // === Tiled: " << label << " [";
    for (size_t i = 0; i < tile_sizes.size(); ++i) {
      if (i) os << "x";
      os << tile_sizes[i];
    }
    os << "] ===\n";

    std::vector<int> par, red;
    for (size_t i = 0; i < iters.size(); ++i) {
      if (iters[i] == IterKind::PARALLEL) par.push_back(i);
      else red.push_back(i);
    }

    std::string ind = "    ";
    for (int d : par) {
      os << ind << "scf.for %" << dn[d] << " = 0 to " << loop_ranges[d]
         << " step " << tile_sizes[d] << " {  // " << iter_str(iters[d]) << "\n";
      ind += "  ";
    }
    if (!red.empty()) {
      os << ind << "%acc = tensor.empty<";
      for (size_t i = 0; i < par.size(); ++i) {
        if (i) os << "x";
        os << tile_sizes[par[i]];
      }
      os << "x" << elem_str(result_->type.elem) << ">\n";
    }
    for (int d : red) {
      os << ind << "scf.for %" << dn[d] << " = 0 to " << loop_ranges[d]
         << " step " << tile_sizes[d] << " {  // " << iter_str(iters[d]) << "\n";
      ind += "  ";
    }
    for (size_t i = 0; i < ins.size(); ++i) {
      auto &m = maps[i];
      os << ind << "%t_" << ins[i]->name.substr(1) << " = tensor.extract_slice "
         << ins[i]->name << "[";
      for (size_t j = 0; j < m.dim_pos.size(); ++j) {
        if (j) os << ", ";
        os << "%" << dn[m.dim_pos[j]];
      }
      os << "][";
      for (size_t j = 0; j < m.dim_pos.size(); ++j) {
        if (j) os << ", ";
        os << tile_sizes[m.dim_pos[j]];
      }
      os << "]\n";
    }
    os << ind << "%acc = " << mnemonic << " ins(...) outs(%acc) { " << label
       << " body }\n";
    for (size_t i = 0; i < red.size(); ++i) {
      ind.resize(ind.size() - 2);
      os << ind << "}\n";
    }
    for (auto &fc : fused_consumers) {
      os << ind << "// ── fused: " << fc.label << " ──\n";
      for (auto *v : fc.extra_ins) {
        os << ind << "%t_" << v->name.substr(1)
           << " = tensor.extract_slice " << v->name << "[";
        for (size_t j = 0; j < par.size(); ++j) {
          if (j) os << ", ";
          os << "%" << dn[par[j]];
        }
        os << "][";
        for (size_t j = 0; j < par.size(); ++j) {
          if (j) os << ", ";
          os << tile_sizes[par[j]];
        }
        os << "]\n";
      }
      os << ind << "%result = linalg.generic { ";
      for (size_t i = 0; i < fc.body.size(); ++i) {
        if (i) os << ", ";
        os << fc.body[i].op;
      }
      os << " }\n";
    }
    os << ind << "tensor.insert_slice → " << result_->name << "[...]\n";
    for (size_t i = 0; i < par.size(); ++i) {
      ind.resize(ind.size() - 2);
      os << ind << "}\n";
    }
  }
};

// ======================== Graph ========================

class Graph {
  int val_cnt_ = 0;
  std::vector<std::unique_ptr<Value>> vals_;

public:
  std::string name = "main";
  std::vector<Value *> args;
  std::vector<std::unique_ptr<Op>> ops;
  std::vector<Value *> returns;

  Value *add_arg(const TensorType &t) {
    auto v = std::make_unique<Value>();
    v->id = val_cnt_++;
    v->name = "%arg" + std::to_string(args.size());
    v->type = t;
    auto *ptr = v.get();
    vals_.push_back(std::move(v));
    args.push_back(ptr);
    return ptr;
  }

  Value *make_val(const TensorType &t, Op *def = nullptr) {
    auto v = std::make_unique<Value>();
    v->id = val_cnt_++;
    v->name = "%" + std::to_string(v->id);
    v->type = t;
    v->defining_op = def;
    auto *ptr = v.get();
    vals_.push_back(std::move(v));
    return ptr;
  }

  Value *add_fill(const TensorType &t) {
    auto op = std::make_unique<Op>();
    op->mnemonic = "linalg.fill";
    op->label = "zero_init";
    auto *r = make_val(t, op.get());
    op->result_ = r;
    ops.push_back(std::move(op));
    return r;
  }

  Op *add_matmul(Value *A, Value *B, Value *init) {
    int64_t M = A->type.dims[0], K = A->type.dims[1], N = B->type.dims[1];
    auto op = std::make_unique<Op>();
    op->mnemonic = "linalg.matmul";
    op->label = "matmul";
    op->ins = {A, B};
    op->outs = {init};
    op->maps = {IndexingMap{{0, 2}}, IndexingMap{{2, 1}}, IndexingMap{{0, 1}}};
    op->iters = {IterKind::PARALLEL, IterKind::PARALLEL, IterKind::REDUCTION};
    op->loop_ranges = {M, N, K};
    op->body = {{"arith.mulf", 0, 1, 3}, {"arith.addf", 3, 2, 4}};
    op->yield_idx = 4;
    auto *r = make_val(init->type, op.get());
    op->result_ = r;
    for (auto *v : op->ins) v->add_user(op.get());
    init->add_user(op.get());
    auto *ptr = op.get();
    ops.push_back(std::move(op));
    return ptr;
  }

  Op *add_elemwise(const std::string &lbl, const std::string &scalar_op,
                   Value *a, Value *b) {
    int rank = static_cast<int>(a->type.rank());
    std::vector<int> id;
    for (int i = 0; i < rank; ++i) id.push_back(i);
    IndexingMap im{id};
    std::vector<IterKind> it(rank, IterKind::PARALLEL);

    auto op = std::make_unique<Op>();
    op->mnemonic = "linalg.generic";
    op->label = lbl;
    op->ins = {a, b};
    op->maps = {im, im, im};
    op->iters = it;
    op->loop_ranges = a->type.dims;
    op->body = {{scalar_op, 0, 1, 2}};
    op->yield_idx = 2;
    auto *r = make_val(a->type, op.get());
    op->result_ = r;
    for (auto *v : op->ins) v->add_user(op.get());
    auto *ptr = op.get();
    ops.push_back(std::move(op));
    return ptr;
  }

  Op *add_unary(const std::string &lbl, const std::string &scalar_op,
                Value *a) {
    int rank = static_cast<int>(a->type.rank());
    std::vector<int> id;
    for (int i = 0; i < rank; ++i) id.push_back(i);
    IndexingMap im{id};
    std::vector<IterKind> it(rank, IterKind::PARALLEL);

    auto op = std::make_unique<Op>();
    op->mnemonic = "linalg.generic";
    op->label = lbl;
    op->ins = {a};
    op->maps = {im, im};
    op->iters = it;
    op->loop_ranges = a->type.dims;
    op->body = {{scalar_op, 0, 0, 1}};
    op->yield_idx = 1;
    auto *r = make_val(a->type, op.get());
    op->result_ = r;
    a->add_user(op.get());
    auto *ptr = op.get();
    ops.push_back(std::move(op));
    return ptr;
  }

  void replace_all_uses(Value *old_v, Value *new_v) {
    if (old_v == new_v) return;
    for (auto &op : ops) {
      bool changed = false;
      for (auto &v : op->ins)
        if (v == old_v) { v = new_v; changed = true; }
      for (auto &v : op->outs)
        if (v == old_v) { v = new_v; changed = true; }
      if (changed) {
        old_v->remove_user(op.get());
        new_v->add_user(op.get());
      }
    }
    for (auto &r : returns)
      if (r == old_v) r = new_v;
  }

  void erase_op(Op *target) {
    for (auto *v : target->ins) v->remove_user(target);
    for (auto *v : target->outs) v->remove_user(target);
    if (target->result_) target->result_->defining_op = nullptr;
    ops.erase(
        std::remove_if(ops.begin(), ops.end(),
                        [target](const auto &p) { return p.get() == target; }),
        ops.end());
  }

  bool is_return_val(Value *v) const {
    for (auto *r : returns)
      if (r == v) return true;
    return false;
  }

  int op_pos(const Op *target) const {
    for (size_t i = 0; i < ops.size(); ++i)
      if (ops[i].get() == target) return static_cast<int>(i);
    return -1;
  }

  void print(std::ostream &os, bool tiled_view = false) const {
    os << "module {\n  func.func @" << name << "(";
    for (size_t i = 0; i < args.size(); ++i) {
      if (i) os << ",\n      ";
      os << args[i]->name << ": " << args[i]->type.str();
    }
    os << ")";
    if (!returns.empty()) {
      os << " -> ";
      if (returns.size() == 1)
        os << returns[0]->type.str();
      else {
        os << "(";
        for (size_t i = 0; i < returns.size(); ++i) {
          if (i) os << ", ";
          os << returns[i]->type.str();
        }
        os << ")";
      }
    }
    os << " {\n";
    for (auto &op : ops) {
      if (tiled_view && !op->tile_sizes.empty())
        op->print_tiled(os);
      else
        op->print(os);
    }
    os << "    return";
    for (size_t i = 0; i < returns.size(); ++i)
      os << (i == 0 ? " " : ", ") << returns[i]->name;
    if (!returns.empty()) {
      os << " : ";
      for (size_t i = 0; i < returns.size(); ++i) {
        if (i) os << ", ";
        os << returns[i]->type.str();
      }
    }
    os << "\n  }\n}\n";
  }

  int op_count() const { return static_cast<int>(ops.size()); }
};

} // namespace linalg_ir
