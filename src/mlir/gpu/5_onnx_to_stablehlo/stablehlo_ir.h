#pragma once
// stablehlo_ir.h — P4: lightweight StableHLO text IR for ONNX→StableHLO lowering exercises.
//
// Provides typed SSA values, operations, functions, modules, and a Builder
// that produces StableHLO-compatible MLIR text output. Designed for learning
// the semantic mapping from ONNX to StableHLO/XLA.

#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace shlo {

// ======================== Element Type ========================

enum class ElemType { F16, F32, F64, I8, I16, I32, I64, I1, INDEX, UNKNOWN };

inline const char *elem_type_str(ElemType t) {
  switch (t) {
  case ElemType::F16:    return "f16";
  case ElemType::F32:    return "f32";
  case ElemType::F64:    return "f64";
  case ElemType::I8:     return "i8";
  case ElemType::I16:    return "i16";
  case ElemType::I32:    return "i32";
  case ElemType::I64:    return "i64";
  case ElemType::I1:     return "i1";
  case ElemType::INDEX:  return "index";
  default:               return "unknown";
  }
}

// ======================== Tensor Type ========================
// dims[i] == -1 ⇒ dynamic dimension (printed as '?')

struct TensorType {
  std::vector<int64_t> dims;
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

  // e.g. "tensor<2x3xf32>" or "tensor<?x3xf32>"
  std::string str() const {
    std::ostringstream os;
    os << "tensor<";
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i) os << "x";
      if (dims[i] < 0)
        os << "?";
      else
        os << dims[i];
    }
    if (!dims.empty()) os << "x";
    os << elem_type_str(elem) << ">";
    return os.str();
  }

  bool operator==(const TensorType &o) const {
    return dims == o.dims && elem == o.elem;
  }
  bool operator!=(const TensorType &o) const { return !(*this == o); }
};

// ======================== SSA Value ========================

struct Value {
  std::string name;  // "%arg0", "%0", …
  TensorType type;
  bool valid() const { return !name.empty(); }
};

// ======================== Operation ========================
// Stores a pre-formatted assembly string for each op.

struct Op {
  std::string mnemonic;
  std::string result_name;
  TensorType result_type;
  std::string assembly;

  void print(std::ostream &os, int indent = 4) const {
    for (int i = 0; i < indent; ++i) os << ' ';
    if (!result_name.empty())
      os << result_name << " = ";
    os << assembly << "\n";
  }
};

// ======================== Function ========================

struct FuncOp {
  std::string name;
  std::vector<Value> args;
  std::vector<Value> results;
  std::vector<Op> body;

  void print(std::ostream &os) const {
    os << "  func.func @" << name << "(";
    for (size_t i = 0; i < args.size(); ++i) {
      if (i) os << ", ";
      os << args[i].name << ": " << args[i].type.str();
    }
    os << ") -> ";
    if (results.size() == 1) {
      os << results[0].type.str();
    } else {
      os << "(";
      for (size_t i = 0; i < results.size(); ++i) {
        if (i) os << ", ";
        os << results[i].type.str();
      }
      os << ")";
    }
    os << " {\n";
    for (auto &op : body)
      op.print(os);
    os << "  }\n";
  }
};

// ======================== Module ========================

struct ModuleOp {
  std::vector<FuncOp> funcs;

  void print(std::ostream &os) const {
    os << "module {\n";
    for (auto &f : funcs)
      f.print(os);
    os << "}\n";
  }
};

// ======================== IR Builder ========================

class Builder {
  int ssa_counter_ = 0;
  FuncOp *func_ = nullptr;

  void append(Op op) {
    assert(func_ && "call set_insertion_point first");
    func_->body.push_back(std::move(op));
  }

public:
  void set_insertion_point(FuncOp &f) {
    func_ = &f;
    ssa_counter_ = 0;
  }

  std::string fresh_ssa() { return "%" + std::to_string(ssa_counter_++); }

  // ---- stablehlo.add / stablehlo.multiply ----

  Value emit_binary(const char *mnemonic, const Value &lhs, const Value &rhs,
                    const TensorType &res_type) {
    std::string name = fresh_ssa();
    std::ostringstream a;
    a << mnemonic << " " << lhs.name << ", " << rhs.name << " : "
      << res_type.str();
    append({mnemonic, name, res_type, a.str()});
    return {name, res_type};
  }

  Value emit_add(const Value &lhs, const Value &rhs) {
    return emit_binary("stablehlo.add", lhs, rhs, lhs.type);
  }
  Value emit_add(const Value &lhs, const Value &rhs, const TensorType &rt) {
    return emit_binary("stablehlo.add", lhs, rhs, rt);
  }
  Value emit_multiply(const Value &lhs, const Value &rhs) {
    return emit_binary("stablehlo.multiply", lhs, rhs, lhs.type);
  }
  Value emit_multiply(const Value &lhs, const Value &rhs, const TensorType &rt) {
    return emit_binary("stablehlo.multiply", lhs, rhs, rt);
  }
  Value emit_subtract(const Value &lhs, const Value &rhs) {
    return emit_binary("stablehlo.subtract", lhs, rhs, lhs.type);
  }
  Value emit_subtract(const Value &lhs, const Value &rhs,
                      const TensorType &rt) {
    return emit_binary("stablehlo.subtract", lhs, rhs, rt);
  }
  Value emit_divide(const Value &lhs, const Value &rhs) {
    return emit_binary("stablehlo.divide", lhs, rhs, lhs.type);
  }
  Value emit_divide(const Value &lhs, const Value &rhs,
                    const TensorType &rt) {
    return emit_binary("stablehlo.divide", lhs, rhs, rt);
  }

  // ---- stablehlo.exponential ----

  Value emit_exponential(const Value &operand) {
    auto name = fresh_ssa();
    std::ostringstream a;
    a << "stablehlo.exponential " << operand.name << " : "
      << operand.type.str();
    append({"stablehlo.exponential", name, operand.type, a.str()});
    return {name, operand.type};
  }

  Value emit_sqrt(const Value &operand) {
    auto name = fresh_ssa();
    std::ostringstream a;
    a << "stablehlo.sqrt " << operand.name << " : " << operand.type.str();
    append({"stablehlo.sqrt", name, operand.type, a.str()});
    return {name, operand.type};
  }

  // ---- stablehlo.reduce ----

  Value emit_reduce(const Value &operand, const Value &init,
                    const char *reducer_mnemonic,
                    const std::vector<int64_t> &dimensions,
                    const TensorType &result_type) {
    auto name = fresh_ssa();
    std::ostringstream a;
    a << "stablehlo.reduce(" << operand.name << " init: " << init.name
      << ") applies " << reducer_mnemonic << " across dimensions = [";
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (i) a << ", ";
      a << dimensions[i];
    }
    a << "] : (" << operand.type.str() << ", " << init.type.str()
      << ") -> " << result_type.str();
    append({"stablehlo.reduce", name, result_type, a.str()});
    return {name, result_type};
  }

  // ---- stablehlo.dot_general ----

  Value emit_dot_general(const Value &lhs, const Value &rhs,
                         const std::vector<int64_t> &lhs_batch,
                         const std::vector<int64_t> &rhs_batch,
                         const std::vector<int64_t> &lhs_contract,
                         const std::vector<int64_t> &rhs_contract,
                         const TensorType &result_type) {
    auto name = fresh_ssa();
    std::ostringstream a;
    a << "stablehlo.dot_general " << lhs.name << ", " << rhs.name
      << ", batching_dims = [";
    for (size_t i = 0; i < lhs_batch.size(); ++i) {
      if (i) a << ", ";
      a << lhs_batch[i];
    }
    a << "] x [";
    for (size_t i = 0; i < rhs_batch.size(); ++i) {
      if (i) a << ", ";
      a << rhs_batch[i];
    }
    a << "], contracting_dims = [";
    for (size_t i = 0; i < lhs_contract.size(); ++i) {
      if (i) a << ", ";
      a << lhs_contract[i];
    }
    a << "] x [";
    for (size_t i = 0; i < rhs_contract.size(); ++i) {
      if (i) a << ", ";
      a << rhs_contract[i];
    }
    a << "] : (" << lhs.type.str() << ", " << rhs.type.str() << ") -> "
      << result_type.str();
    append({"stablehlo.dot_general", name, result_type, a.str()});
    return {name, result_type};
  }

  // ---- stablehlo.convolution ----

  Value emit_convolution(
      const Value &input, const Value &kernel,
      const std::vector<int64_t> &strides,
      const std::vector<std::pair<int64_t, int64_t>> &padding,
      const std::vector<int64_t> &lhs_dilation,
      const std::vector<int64_t> &rhs_dilation, int64_t feature_group_count,
      int64_t batch_group_count, const TensorType &result_type) {
    auto name = fresh_ssa();
    int sp = static_cast<int>(strides.size());
    std::ostringstream a;
    a << "stablehlo.convolution(" << input.name << ", " << kernel.name
      << ")\n";
    // dimension_numbers (NCHW x OIHW -> NCHW for 2-D)
    a << "          dim_numbers = [b, f";
    for (int i = 0; i < sp; ++i) a << ", " << i;
    a << "]x[o, i";
    for (int i = 0; i < sp; ++i) a << ", " << i;
    a << "]->[b, f";
    for (int i = 0; i < sp; ++i) a << ", " << i;
    a << "],\n";
    // window
    a << "          window = {stride = [";
    for (int i = 0; i < sp; ++i) {
      if (i) a << ", ";
      a << strides[i];
    }
    a << "], pad = [";
    for (int i = 0; i < sp; ++i) {
      if (i) a << ", ";
      a << "[" << padding[i].first << ", " << padding[i].second << "]";
    }
    a << "]";
    bool need_rhs_dil = false;
    for (auto d : rhs_dilation)
      if (d != 1) need_rhs_dil = true;
    if (need_rhs_dil) {
      a << ", rhs_dilate = [";
      for (int i = 0; i < sp; ++i) {
        if (i) a << ", ";
        a << rhs_dilation[i];
      }
      a << "]";
    }
    a << "}\n";
    a << "          {feature_group_count = " << feature_group_count
      << " : i64, batch_group_count = " << batch_group_count << " : i64}\n";
    a << "          : (" << input.type.str() << ", " << kernel.type.str()
      << ") -> " << result_type.str();
    append({"stablehlo.convolution", name, result_type, a.str()});
    return {name, result_type};
  }

  // ---- stablehlo.reshape ----

  Value emit_reshape(const Value &operand, const TensorType &result_type) {
    auto name = fresh_ssa();
    std::ostringstream a;
    a << "stablehlo.reshape " << operand.name << " : (" << operand.type.str()
      << ") -> " << result_type.str();
    append({"stablehlo.reshape", name, result_type, a.str()});
    return {name, result_type};
  }

  // ---- stablehlo.transpose ----

  Value emit_transpose(const Value &operand,
                       const std::vector<int64_t> &permutation,
                       const TensorType &result_type) {
    auto name = fresh_ssa();
    std::ostringstream a;
    a << "stablehlo.transpose " << operand.name << ", dims = [";
    for (size_t i = 0; i < permutation.size(); ++i) {
      if (i) a << ", ";
      a << permutation[i];
    }
    a << "] : (" << operand.type.str() << ") -> " << result_type.str();
    append({"stablehlo.transpose", name, result_type, a.str()});
    return {name, result_type};
  }

  // ---- stablehlo.broadcast_in_dim ----

  Value emit_broadcast_in_dim(const Value &operand,
                              const std::vector<int64_t> &broadcast_dims,
                              const TensorType &result_type) {
    auto name = fresh_ssa();
    std::ostringstream a;
    a << "stablehlo.broadcast_in_dim " << operand.name << ", dims = [";
    for (size_t i = 0; i < broadcast_dims.size(); ++i) {
      if (i) a << ", ";
      a << broadcast_dims[i];
    }
    a << "] : (" << operand.type.str() << ") -> " << result_type.str();
    append({"stablehlo.broadcast_in_dim", name, result_type, a.str()});
    return {name, result_type};
  }

  // ---- stablehlo.constant ----

  Value emit_constant(const TensorType &type, const std::string &dense_repr) {
    auto name = fresh_ssa();
    std::ostringstream a;
    a << "stablehlo.constant dense<" << dense_repr << "> : " << type.str();
    append({"stablehlo.constant", name, type, a.str()});
    return {name, type};
  }

  // ---- stablehlo.dynamic_reshape ----

  Value emit_dynamic_reshape(const Value &operand, const Value &shape,
                             const TensorType &result_type) {
    auto name = fresh_ssa();
    std::ostringstream a;
    a << "stablehlo.dynamic_reshape " << operand.name << ", " << shape.name
      << " : (" << operand.type.str() << ", " << shape.type.str() << ") -> "
      << result_type.str();
    append({"stablehlo.dynamic_reshape", name, result_type, a.str()});
    return {name, result_type};
  }

  // ---- return ----

  void emit_return(const std::vector<Value> &values) {
    std::ostringstream a;
    a << "return ";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i) a << ", ";
      a << values[i].name;
    }
    a << " : ";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i) a << ", ";
      a << values[i].type.str();
    }
    append({"func.return", "", {}, a.str()});
  }
};

} // namespace shlo
