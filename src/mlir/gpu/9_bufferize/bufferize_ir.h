#pragma once
// bufferize_ir.h — Simplified IR for One-Shot Bufferization (OSB) pipeline.
//
// Models tensor → memref transition:
//   • TensorType / MemRefType — value-semantic vs buffer-semantic types
//   • BufDecision             — per-op in-place / out-of-place annotation
//   • WriteConflict           — WAR / RAW conflict between ops
//   • AliasGroup              — values potentially sharing one buffer
//   • OwnershipInfo           — who must deallocate each buffer
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

namespace bufferize_ir {

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

// ======================== MemRef Type ========================

struct MemRefType {
  std::vector<int64_t> dims;
  ElemType elem = ElemType::F32;

  static MemRefType from_tensor(const TensorType &t) {
    return {t.dims, t.elem};
  }
  int64_t bytes() const {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    return n * elem_bytes(elem);
  }
  std::string str() const {
    std::ostringstream os;
    os << "memref<";
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i) os << "x";
      os << dims[i];
    }
    if (!dims.empty()) os << "x";
    os << elem_str(elem) << ">";
    return os.str();
  }
};

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

// ======================== Buffer Decision ========================

enum class BufDecision { UNDECIDED, INPLACE, OUT_OF_PLACE };

inline const char *decision_str(BufDecision d) {
  switch (d) {
  case BufDecision::UNDECIDED:    return "UNDECIDED";
  case BufDecision::INPLACE:      return "INPLACE";
  case BufDecision::OUT_OF_PLACE: return "OUT_OF_PLACE";
  }
  return "?";
}

struct OpBufInfo {
  BufDecision decision = BufDecision::UNDECIDED;
  Value *aliases_with = nullptr;
  bool needs_copy_before = false;
  std::string reason;
};

// ======================== Write Conflict ========================

enum class ConflictKind { RAW, WAR };

inline const char *conflict_str(ConflictKind k) {
  return k == ConflictKind::RAW ? "RAW" : "WAR";
}

struct WriteConflict {
  Op *writer;
  Op *later_reader;
  Value *shared_value;
  ConflictKind kind;
};

// ======================== Alias Group ========================

struct AliasGroup {
  int id = 0;
  std::vector<Value *> members;
  Value *root = nullptr;
};

// ======================== Ownership Info ========================

struct OwnershipInfo {
  int buf_id = -1;
  bool is_func_arg = false;
  bool needs_dealloc = false;
  int last_use_op_idx = -1;
};

// ======================== Operation ========================

struct Op {
  std::string mnemonic;
  std::string label;

  std::vector<Value *> ins;
  std::vector<Value *> outs;
  Value *result_ = nullptr;

  OpBufInfo buf_info;

  int n_ins() const { return static_cast<int>(ins.size()); }
  bool is_fill() const { return mnemonic == "linalg.fill"; }
  bool is_linalg() const { return mnemonic.rfind("linalg.", 0) == 0; }

  bool reads_from(Value *v) const {
    for (auto *i : ins)
      if (i == v) return true;
    for (auto *o : outs)
      if (o == v) return true;
    return false;
  }

  void print(std::ostream &os) const {
    if (is_fill()) {
      os << "    " << result_->name << " = linalg.fill(0.0) : "
         << result_->type.str() << "\n";
      return;
    }
    os << "    " << result_->name << " = " << mnemonic;
    if (!label.empty()) os << "  // " << label;
    os << "\n      ins(";
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
    os << "\n      : " << result_->type.str();
    if (buf_info.decision != BufDecision::UNDECIDED) {
      os << "  {" << decision_str(buf_info.decision);
      if (buf_info.aliases_with)
        os << " → " << buf_info.aliases_with->name;
      if (buf_info.needs_copy_before) os << ", COPY_BEFORE";
      if (!buf_info.reason.empty()) os << ", " << buf_info.reason;
      os << "}";
    }
    os << "\n";
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
    auto op = std::make_unique<Op>();
    op->mnemonic = "linalg.matmul";
    op->label = "matmul";
    op->ins = {A, B};
    op->outs = {init};
    auto *r = make_val(init->type, op.get());
    op->result_ = r;
    for (auto *v : op->ins) v->add_user(op.get());
    init->add_user(op.get());
    auto *ptr = op.get();
    ops.push_back(std::move(op));
    return ptr;
  }

  Op *add_elemwise(const std::string &lbl, Value *a, Value *b) {
    auto op = std::make_unique<Op>();
    op->mnemonic = "linalg.generic";
    op->label = lbl;
    op->ins = {a, b};
    auto *r = make_val(a->type, op.get());
    op->result_ = r;
    for (auto *v : op->ins) v->add_user(op.get());
    auto *ptr = op.get();
    ops.push_back(std::move(op));
    return ptr;
  }

  Op *add_unary(const std::string &lbl, Value *a) {
    auto op = std::make_unique<Op>();
    op->mnemonic = "linalg.generic";
    op->label = lbl;
    op->ins = {a};
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

  int op_count() const { return static_cast<int>(ops.size()); }

  void print(std::ostream &os) const {
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
    for (auto &op : ops) op->print(os);
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
};

} // namespace bufferize_ir
