#pragma once
// mini_ir.h — Lightweight graph IR for L1 Graph Level exercises.
//
// Modeled after the core abstractions of real compiler IRs (ONNX GraphProto,
// StableHLO, XLA HLO) but kept intentionally minimal for learning purposes.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace mini_ir {

// ---- Tensor type (shape + element type, no data) ----

enum class DType { F32, F64, I32, I64, UNKNOWN };

inline const char *dtype_str(DType d) {
  switch (d) {
  case DType::F32: return "f32";
  case DType::F64: return "f64";
  case DType::I32: return "i32";
  case DType::I64: return "i64";
  default: return "unknown";
  }
}

struct TensorType {
  std::vector<int64_t> shape;
  DType dtype = DType::UNKNOWN;

  int64_t rank() const { return static_cast<int64_t>(shape.size()); }
  int64_t num_elements() const {
    if (shape.empty()) return 1;
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
  std::string str() const {
    std::ostringstream os;
    os << "tensor<";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i) os << "x";
      os << shape[i];
    }
    os << "x" << dtype_str(dtype) << ">";
    return os.str();
  }
};

// ---- Tensor data (for constants / initializers) ----

struct TensorData {
  TensorType type;
  std::vector<float> float_data;

  float at(size_t i) const { return float_data.at(i); }
  std::string summary() const {
    std::ostringstream os;
    os << type.str() << " [";
    size_t limit = std::min<size_t>(float_data.size(), 6);
    for (size_t i = 0; i < limit; ++i) {
      if (i) os << ", ";
      os << float_data[i];
    }
    if (float_data.size() > limit) os << ", ...";
    os << "]";
    return os.str();
  }
};

// ---- Attribute value ----

using AttrValue = std::variant<int64_t, float, std::string,
                               std::vector<int64_t>, std::vector<float>>;

inline std::string attr_str(const AttrValue &v) {
  struct Visitor {
    std::string operator()(int64_t x) const { return std::to_string(x); }
    std::string operator()(float x) const { return std::to_string(x); }
    std::string operator()(const std::string &x) const { return "\"" + x + "\""; }
    std::string operator()(const std::vector<int64_t> &x) const {
      std::ostringstream os;
      os << "[";
      for (size_t i = 0; i < x.size(); ++i) { if (i) os << ","; os << x[i]; }
      os << "]";
      return os.str();
    }
    std::string operator()(const std::vector<float> &x) const {
      std::ostringstream os;
      os << "[";
      for (size_t i = 0; i < x.size(); ++i) { if (i) os << ","; os << x[i]; }
      os << "]";
      return os.str();
    }
  };
  return std::visit(Visitor{}, v);
}

// ---- Op node in graph ----

struct Node {
  std::string name;
  std::string op_type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::unordered_map<std::string, AttrValue> attrs;

  void dump(std::ostream &os) const {
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (i) os << ", ";
      os << "%" << outputs[i];
    }
    os << " = " << op_type << "(";
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (i) os << ", ";
      os << "%" << inputs[i];
    }
    os << ")";
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
    if (!name.empty()) os << "  // " << name;
    os << "\n";
  }
};

// ---- Graph ----

struct Graph {
  std::string name;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::unordered_map<std::string, TensorType> value_types;
  std::unordered_map<std::string, TensorData> initializers;
  std::vector<std::shared_ptr<Node>> nodes;

  // Lookup helpers
  std::shared_ptr<Node> find_producer(const std::string &val) const {
    for (auto &n : nodes)
      for (auto &o : n->outputs)
        if (o == val) return n;
    return nullptr;
  }

  std::vector<std::shared_ptr<Node>> find_consumers(const std::string &val) const {
    std::vector<std::shared_ptr<Node>> res;
    for (auto &n : nodes)
      for (auto &inp : n->inputs)
        if (inp == val) { res.push_back(n); break; }
    return res;
  }

  bool is_initializer(const std::string &name) const {
    return initializers.count(name);
  }

  void dump(std::ostream &os) const {
    os << "graph " << name << " {\n";
    os << "  // inputs: ";
    for (auto &n : input_names) {
      os << "%" << n;
      if (value_types.count(n)) os << " : " << value_types.at(n).str();
      os << "  ";
    }
    os << "\n";
    os << "  // initializers: " << initializers.size() << "\n";
    for (auto &n : nodes) {
      os << "  ";
      n->dump(os);
    }
    os << "  // outputs: ";
    for (auto &n : output_names) os << "%" << n << "  ";
    os << "\n}\n";
  }

  // Remove a node (does NOT fix wiring — caller must handle)
  void erase_node(const std::shared_ptr<Node> &target) {
    nodes.erase(std::remove(nodes.begin(), nodes.end(), target), nodes.end());
  }
};

} // namespace mini_ir
