#pragma once
// onnx_to_shlo_utils.h — P4: shared utilities for ONNX → StableHLO conversion.
//
// Provides: ONNX protobuf ↔ ShLO type conversion, data extraction,
// attribute helpers, dense constant formatting, model loading,
// and the ConversionContext used by P4 tiers 1–3.

#include "stablehlo_ir.h"
#include "onnx-ml.pb.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace onnx2shlo {

// ======================== Type Conversion ========================

inline shlo::ElemType onnx_elem_to_shlo(int onnx_type) {
  switch (onnx_type) {
  case 1:  return shlo::ElemType::F32;   // FLOAT
  case 11: return shlo::ElemType::F64;   // DOUBLE
  case 10: return shlo::ElemType::F16;   // FLOAT16
  case 6:  return shlo::ElemType::I32;   // INT32
  case 7:  return shlo::ElemType::I64;   // INT64
  case 5:  return shlo::ElemType::I16;   // INT16
  case 3:  return shlo::ElemType::I8;    // INT8
  case 9:  return shlo::ElemType::I1;    // BOOL
  default: return shlo::ElemType::UNKNOWN;
  }
}

inline shlo::TensorType type_from_proto(const onnx::TypeProto &tp) {
  shlo::TensorType t;
  if (tp.has_tensor_type()) {
    auto &tt = tp.tensor_type();
    t.elem = onnx_elem_to_shlo(tt.elem_type());
    if (tt.has_shape())
      for (auto &d : tt.shape().dim())
        t.dims.push_back(d.has_dim_value() ? d.dim_value() : -1);
  }
  return t;
}

inline shlo::TensorType type_from_tensor(const onnx::TensorProto &tp) {
  shlo::TensorType t;
  for (auto d : tp.dims()) t.dims.push_back(d);
  t.elem = onnx_elem_to_shlo(tp.data_type());
  return t;
}

// ======================== Data Extraction ========================

inline std::vector<float> extract_floats(const onnx::TensorProto &tp) {
  std::vector<float> data;
  if (!tp.raw_data().empty()) {
    size_t n = tp.raw_data().size() / sizeof(float);
    data.resize(n);
    std::memcpy(data.data(), tp.raw_data().data(), n * sizeof(float));
  } else {
    for (int i = 0; i < tp.float_data_size(); ++i)
      data.push_back(tp.float_data(i));
  }
  return data;
}

inline std::vector<int64_t> extract_int64s(const onnx::TensorProto &tp) {
  std::vector<int64_t> data;
  if (!tp.raw_data().empty()) {
    size_t n = tp.raw_data().size() / sizeof(int64_t);
    data.resize(n);
    std::memcpy(data.data(), tp.raw_data().data(), n * sizeof(int64_t));
  } else {
    for (int i = 0; i < tp.int64_data_size(); ++i)
      data.push_back(tp.int64_data(i));
  }
  return data;
}

// ======================== Attribute Helpers ========================

inline int64_t get_int_attr(const onnx::NodeProto &n, const std::string &key,
                            int64_t def = 0) {
  for (auto &a : n.attribute())
    if (a.name() == key) return a.i();
  return def;
}

inline float get_float_attr(const onnx::NodeProto &n, const std::string &key,
                            float def = 0.f) {
  for (auto &a : n.attribute())
    if (a.name() == key) return a.f();
  return def;
}

inline std::vector<int64_t> get_ints_attr(const onnx::NodeProto &n,
                                          const std::string &key) {
  for (auto &a : n.attribute())
    if (a.name() == key) {
      std::vector<int64_t> v;
      for (auto i : a.ints()) v.push_back(i);
      return v;
    }
  return {};
}

inline std::string get_str_attr(const onnx::NodeProto &n,
                                const std::string &key) {
  for (auto &a : n.attribute())
    if (a.name() == key) return a.s();
  return "";
}

// ======================== Dense Constant Formatting ========================

inline std::string format_dense(const std::vector<float> &data,
                                const shlo::TensorType &type) {
  int64_t n = type.num_elements();
  if (n <= 0 || data.empty()) return "0.0";
  if (n == 1) {
    std::ostringstream os;
    os << data[0];
    return os.str();
  }
  std::ostringstream os;
  if (n <= 16) {
    os << "[";
    for (int64_t i = 0; i < n && i < (int64_t)data.size(); ++i) {
      if (i) os << ", ";
      os << data[i];
    }
    os << "]";
  } else {
    os << "[" << data[0] << ", " << data[1] << ", ..., "
       << data[std::min<size_t>(data.size(), (size_t)n) - 1] << "]";
  }
  return os.str();
}

inline std::string format_dense_int64(const std::vector<int64_t> &data,
                                      const shlo::TensorType &type) {
  int64_t n = type.num_elements();
  if (n <= 0 || data.empty()) return "0";
  if (n == 1) return std::to_string(data[0]);
  std::ostringstream os;
  os << "[";
  for (int64_t i = 0; i < n && i < (int64_t)data.size(); ++i) {
    if (i) os << ", ";
    os << data[i];
  }
  os << "]";
  return os.str();
}

// ======================== Model Loading ========================

inline bool load_model(const std::string &path, onnx::ModelProto &model) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin.is_open()) {
    std::cerr << "Error: cannot open " << path << "\n";
    return false;
  }
  if (!model.ParseFromIstream(&fin)) {
    std::cerr << "Error: failed to parse " << path << "\n";
    return false;
  }
  return true;
}

// ======================== Conversion Context ========================

struct Context {
  const onnx::GraphProto &graph;
  shlo::Builder builder;
  shlo::FuncOp func;

  std::unordered_map<std::string, shlo::Value> value_map;
  std::unordered_map<std::string, shlo::TensorType> type_map;
  std::unordered_set<std::string> initializer_names;
  std::unordered_map<std::string, const onnx::TensorProto *> initializer_data;

  int converted = 0;
  int skipped = 0;
  int errors = 0;
  int warnings = 0;

  explicit Context(const onnx::GraphProto &g) : graph(g) {}

  shlo::Value lookup(const std::string &name) const {
    auto it = value_map.find(name);
    return (it != value_map.end()) ? it->second : shlo::Value{};
  }

  shlo::TensorType get_type(const std::string &name) const {
    auto it = type_map.find(name);
    return (it != type_map.end()) ? it->second : shlo::TensorType{};
  }

  const onnx::TensorProto *get_initializer(const std::string &name) const {
    auto it = initializer_data.find(name);
    return (it != initializer_data.end()) ? it->second : nullptr;
  }

  void error(const std::string &msg) {
    std::cerr << "  [ERROR] " << msg << "\n";
    ++errors;
  }
  void warn(const std::string &msg) {
    std::cerr << "  [WARN]  " << msg << "\n";
    ++warnings;
  }
  void info(const std::string &msg) {
    std::cout << "  [INFO]  " << msg << "\n";
  }

  // Populate type_map and initializer_data from the ONNX graph
  void init() {
    for (auto &init : graph.initializer()) {
      initializer_names.insert(init.name());
      initializer_data[init.name()] = &init;
      type_map[init.name()] = type_from_tensor(init);
    }
    for (auto &vi : graph.value_info())
      type_map[vi.name()] = type_from_proto(vi.type());
    for (auto &inp : graph.input())
      type_map[inp.name()] = type_from_proto(inp.type());
    for (auto &out : graph.output())
      type_map[out.name()] = type_from_proto(out.type());
  }

  // Create func args (non-initializer inputs become %argN)
  void create_func_args() {
    func.name = "main";
    int idx = 0;
    for (auto &inp : graph.input()) {
      if (initializer_names.count(inp.name())) continue;
      std::string n = "%arg" + std::to_string(idx++);
      auto t = type_map[inp.name()];
      func.args.push_back({n, t});
      value_map[inp.name()] = {n, t};
    }
    builder.set_insertion_point(func);
  }

  // Emit each initializer as stablehlo.constant
  void emit_initializers() {
    for (auto &init : graph.initializer()) {
      auto type = type_map[init.name()];
      std::string repr;
      if (type.elem == shlo::ElemType::I64 || type.elem == shlo::ElemType::I32) {
        auto data = extract_int64s(init);
        repr = format_dense_int64(data, type);
      } else {
        auto data = extract_floats(init);
        repr = format_dense(data, type);
      }
      auto val = builder.emit_constant(type, repr);
      value_map[init.name()] = val;
    }
  }

  // Append return op from graph outputs
  void finalize() {
    std::vector<shlo::Value> rets;
    for (auto &out : graph.output()) {
      auto v = lookup(out.name());
      if (!v.valid()) {
        error("Output '" + out.name() + "' was not produced");
        continue;
      }
      rets.push_back(v);
      func.results.push_back(v);
    }
    builder.emit_return(rets);
  }

  void print_stats() const {
    std::cout << "\n--- Statistics ---\n";
    std::cout << "  Converted : " << converted << "\n";
    std::cout << "  Skipped   : " << skipped << "\n";
    std::cout << "  Errors    : " << errors << "\n";
    std::cout << "  Warnings  : " << warnings << "\n";
    std::cout << "  Total ops : " << func.body.size() << "\n";
  }
};

} // namespace onnx2shlo
