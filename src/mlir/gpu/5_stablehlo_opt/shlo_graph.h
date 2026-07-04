#pragma once
// shlo_graph.h — P5 (5_stablehlo_opt): Mutable StableHLO graph IR with use-def chains.
//
// Designed for implementing classic compiler optimization passes:
// canonicalize, CSE, DCE, constant fold, transpose elimination,
// fusion detection, and legalization. Header-only, no external deps.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace shlo_graph {

// ======================== Element Type ========================

enum class ElemType { F32, F64, I32, I64, I1, INDEX, UNKNOWN };

inline const char *elem_str(ElemType t) {
  switch (t) {
  case ElemType::F32:   return "f32";
  case ElemType::F64:   return "f64";
  case ElemType::I32:   return "i32";
  case ElemType::I64:   return "i64";
  case ElemType::I1:    return "i1";
  case ElemType::INDEX: return "index";
  default:              return "unknown";
  }
}

// ======================== Tensor Type ========================

struct TensorType {
  std::vector<int64_t> dims; // -1 = dynamic ('?')
  ElemType elem = ElemType::UNKNOWN;

  int64_t rank() const { return static_cast<int64_t>(dims.size()); }
  bool is_static() const {
    for (auto d : dims)
      if (d < 0) return false;
    return true;
  }
  bool is_dynamic() const { return !is_static(); }
  int64_t num_elements() const {
    if (dims.empty()) return 1;
    int64_t n = 1;
    for (auto d : dims) {
      if (d < 0) return -1;
      n *= d;
    }
    return n;
  }
  std::string str() const {
    std::ostringstream os;
    os << "tensor<";
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i) os << "x";
      if (dims[i] < 0) os << "?";
      else os << dims[i];
    }
    if (!dims.empty()) os << "x";
    os << elem_str(elem) << ">";
    return os.str();
  }
  bool operator==(const TensorType &o) const {
    return dims == o.dims && elem == o.elem;
  }
  bool operator!=(const TensorType &o) const { return !(*this == o); }
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

// ======================== Attribute ========================

using Attr = std::variant<int64_t, float, std::string,
                          std::vector<int64_t>, std::vector<float>>;

inline std::string attr_str(const Attr &v) {
  struct V {
    std::string operator()(int64_t x) const { return std::to_string(x); }
    std::string operator()(float x) const {
      std::ostringstream os;
      os << x;
      return os.str();
    }
    std::string operator()(const std::string &x) const {
      return "\"" + x + "\"";
    }
    std::string operator()(const std::vector<int64_t> &x) const {
      std::ostringstream os;
      os << "[";
      for (size_t i = 0; i < x.size(); ++i) {
        if (i) os << ", ";
        os << x[i];
      }
      os << "]";
      return os.str();
    }
    std::string operator()(const std::vector<float> &x) const {
      std::ostringstream os;
      os << "[";
      for (size_t i = 0; i < x.size(); ++i) {
        if (i) os << ", ";
        os << x[i];
      }
      os << "]";
      return os.str();
    }
  };
  return std::visit(V{}, v);
}

// ======================== Operation ========================

struct Op {
  std::string mnemonic;
  std::vector<Value *> operands;
  std::vector<Value *> results;
  std::unordered_map<std::string, Attr> attrs;
  std::vector<float> const_data_f;
  std::vector<int64_t> const_data_i;

  bool is_constant() const {
    return mnemonic == "stablehlo.constant" ||
           mnemonic == "arith.constant";
  }
  Value *result(int i = 0) const { return results.at(i); }
  Value *operand(int i) const { return operands.at(i); }
  int num_operands() const { return static_cast<int>(operands.size()); }
  int num_results() const { return static_cast<int>(results.size()); }
  bool has_attr(const std::string &key) const { return attrs.count(key); }

  template <typename T>
  T get_attr(const std::string &key) const {
    return std::get<T>(attrs.at(key));
  }
  template <typename T>
  T get_attr(const std::string &key, T def) const {
    auto it = attrs.find(key);
    return (it != attrs.end()) ? std::get<T>(it->second) : def;
  }

  void print(std::ostream &os) const {
    os << "    ";
    if (!results.empty()) os << results[0]->name << " = ";
    os << mnemonic;
    if (is_constant()) {
      os << " dense<";
      if (!const_data_f.empty()) {
        if (const_data_f.size() == 1) {
          os << const_data_f[0];
        } else {
          os << "[";
          for (size_t i = 0; i < const_data_f.size() && i < 8; ++i) {
            if (i) os << ", ";
            os << const_data_f[i];
          }
          if (const_data_f.size() > 8) os << ", ...";
          os << "]";
        }
      } else if (!const_data_i.empty()) {
        if (const_data_i.size() == 1) {
          os << const_data_i[0];
        } else {
          os << "[";
          for (size_t i = 0; i < const_data_i.size() && i < 8; ++i) {
            if (i) os << ", ";
            os << const_data_i[i];
          }
          if (const_data_i.size() > 8) os << ", ...";
          os << "]";
        }
      }
      os << ">";
    } else {
      for (size_t i = 0; i < operands.size(); ++i)
        os << (i == 0 ? " " : ", ") << operands[i]->name;
    }
    if (!attrs.empty()) {
      os << " {";
      bool first = true;
      for (auto &[k, v] : attrs) {
        if (!first) os << ", ";
        os << k << " = " << attr_str(v);
        first = false;
      }
      os << "}";
    }
    if (!results.empty()) os << " : " << results[0]->type.str();
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

  Op *add_op(const std::string &mn, const std::vector<Value *> &ins,
             const std::vector<TensorType> &out_types,
             const std::unordered_map<std::string, Attr> &at = {}) {
    auto op = std::make_unique<Op>();
    op->mnemonic = mn;
    op->operands = ins;
    op->attrs = at;
    for (auto *v : ins) v->add_user(op.get());
    for (auto &t : out_types)
      op->results.push_back(make_val(t, op.get()));
    auto *ptr = op.get();
    ops.push_back(std::move(op));
    return ptr;
  }

  Op *add_constant_f(const TensorType &t, const std::vector<float> &data) {
    auto *op = add_op("stablehlo.constant", {}, {t});
    op->const_data_f = data;
    return op;
  }

  Op *add_constant_i(const TensorType &t, const std::vector<int64_t> &data) {
    auto *op = add_op("stablehlo.constant", {}, {t});
    op->const_data_i = data;
    return op;
  }

  void replace_all_uses(Value *old_v, Value *new_v) {
    if (old_v == new_v) return;
    for (auto &op : ops) {
      bool changed = false;
      for (auto &inp : op->operands)
        if (inp == old_v) { inp = new_v; changed = true; }
      if (changed) {
        old_v->remove_user(op.get());
        new_v->add_user(op.get());
      }
    }
    for (auto &r : returns)
      if (r == old_v) r = new_v;
  }

  void erase_op(Op *target) {
    for (auto *v : target->operands) v->remove_user(target);
    for (auto *r : target->results) r->defining_op = nullptr;
    ops.erase(std::remove_if(ops.begin(), ops.end(),
                              [target](const auto &p) {
                                return p.get() == target;
                              }),
              ops.end());
  }

  bool is_return_val(Value *v) const {
    for (auto *r : returns)
      if (r == v) return true;
    return false;
  }

  void print(std::ostream &os) const {
    os << "module {\n  func.func @" << name << "(";
    for (size_t i = 0; i < args.size(); ++i) {
      if (i) os << ", ";
      os << args[i]->name << ": " << args[i]->type.str();
    }
    os << ")";
    if (!returns.empty()) {
      os << " -> ";
      if (returns.size() == 1) {
        os << returns[0]->type.str();
      } else {
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

  int op_count() const { return static_cast<int>(ops.size()); }
  int count_op(const std::string &mn) const {
    int c = 0;
    for (auto &op : ops)
      if (op->mnemonic == mn) ++c;
    return c;
  }
};

} // namespace shlo_graph
