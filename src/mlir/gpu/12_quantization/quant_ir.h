#pragma once
// quant_ir.h — P12 (12_quantization): simplified IR for quantization & mixed precision.
//
// Models quantization concepts common in AI compiler inference optimization:
//
//   QuantParam    — per-tensor / per-channel scale + zero_point
//   QuantOp       — quantize / dequantize / qlinear_conv / qlinear_matmul
//   CalibData     — min/max statistics from calibration
//   QuantGraph    — graph of operations with quantization annotations
//
// Header-only, pure C++17, no external dependencies.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace quant_ir {

// ======================== Data Types ========================

enum class DType { FP32, FP16, BF16, INT8, UINT8, INT4 };

inline const char *dtype_str(DType t) {
  switch (t) {
  case DType::FP32:  return "f32";
  case DType::FP16:  return "f16";
  case DType::BF16:  return "bf16";
  case DType::INT8:  return "i8";
  case DType::UINT8: return "ui8";
  case DType::INT4:  return "i4";
  }
  return "?";
}

inline int dtype_bits(DType t) {
  switch (t) {
  case DType::FP32:  return 32;
  case DType::FP16:  return 16;
  case DType::BF16:  return 16;
  case DType::INT8:  return 8;
  case DType::UINT8: return 8;
  case DType::INT4:  return 4;
  }
  return 32;
}

// ======================== Quantization Parameters ========================

enum class QuantScheme {
  NONE,
  PER_TENSOR_SYMMETRIC,
  PER_TENSOR_AFFINE,
  PER_CHANNEL_SYMMETRIC,
  PER_CHANNEL_AFFINE
};

inline const char *scheme_str(QuantScheme s) {
  switch (s) {
  case QuantScheme::NONE:                   return "none";
  case QuantScheme::PER_TENSOR_SYMMETRIC:   return "per_tensor_sym";
  case QuantScheme::PER_TENSOR_AFFINE:      return "per_tensor_affine";
  case QuantScheme::PER_CHANNEL_SYMMETRIC:  return "per_channel_sym";
  case QuantScheme::PER_CHANNEL_AFFINE:     return "per_channel_affine";
  }
  return "?";
}

struct QuantParam {
  QuantScheme scheme = QuantScheme::NONE;
  DType target_dtype = DType::INT8;
  std::vector<float> scales;
  std::vector<int> zero_points;
  int channel_axis = -1;

  float qmin() const {
    if (target_dtype == DType::INT8)  return -128.0f;
    if (target_dtype == DType::UINT8) return 0.0f;
    if (target_dtype == DType::INT4)  return -8.0f;
    return 0.0f;
  }

  float qmax() const {
    if (target_dtype == DType::INT8)  return 127.0f;
    if (target_dtype == DType::UINT8) return 255.0f;
    if (target_dtype == DType::INT4)  return 7.0f;
    return 255.0f;
  }

  void print(std::ostream &os) const {
    os << "    scheme: " << scheme_str(scheme) << "\n";
    os << "    dtype:  " << dtype_str(target_dtype) << "\n";
    if (!scales.empty()) {
      os << "    scale:  [";
      for (size_t i = 0; i < std::min(scales.size(), size_t(4)); ++i) {
        if (i) os << ", ";
        os << scales[i];
      }
      if (scales.size() > 4) os << ", ... (" << scales.size() << " total)";
      os << "]\n";
    }
    if (!zero_points.empty()) {
      os << "    zp:     [";
      for (size_t i = 0; i < std::min(zero_points.size(), size_t(4)); ++i) {
        if (i) os << ", ";
        os << zero_points[i];
      }
      if (zero_points.size() > 4)
        os << ", ... (" << zero_points.size() << " total)";
      os << "]\n";
    }
  }
};

// ======================== Calibration Data ========================

struct CalibData {
  float observed_min = 0.0f;
  float observed_max = 0.0f;
  std::vector<float> per_channel_min;
  std::vector<float> per_channel_max;
  std::vector<float> histogram;
  int histogram_bins = 2048;

  void update(float val) {
    observed_min = std::min(observed_min, val);
    observed_max = std::max(observed_max, val);
  }
};

// ======================== Tensor ========================

struct TensorType {
  std::string name;
  std::vector<int64_t> shape;
  DType dtype = DType::FP32;
  QuantParam qparam;

  int64_t numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }

  int64_t size_bytes() const {
    return numel() * dtype_bits(dtype) / 8;
  }

  std::string str() const {
    std::ostringstream os;
    os << "tensor<";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i) os << "x";
      os << shape[i];
    }
    if (!shape.empty()) os << "x";
    os << dtype_str(dtype) << ">";
    return os.str();
  }
};

// ======================== Operation ========================

enum class OpKind {
  CONV2D,
  MATMUL,
  ADD,
  RELU,
  BATCH_NORM,
  QUANTIZE,
  DEQUANTIZE,
  QLINEAR_CONV,
  QLINEAR_MATMUL,
};

inline const char *opkind_str(OpKind k) {
  switch (k) {
  case OpKind::CONV2D:         return "conv2d";
  case OpKind::MATMUL:         return "matmul";
  case OpKind::ADD:            return "add";
  case OpKind::RELU:           return "relu";
  case OpKind::BATCH_NORM:     return "batch_norm";
  case OpKind::QUANTIZE:       return "quantize";
  case OpKind::DEQUANTIZE:     return "dequantize";
  case OpKind::QLINEAR_CONV:   return "qlinear_conv";
  case OpKind::QLINEAR_MATMUL: return "qlinear_matmul";
  }
  return "?";
}

struct Op {
  int id = -1;
  OpKind kind;
  std::string name;
  std::vector<TensorType> inputs;
  std::vector<TensorType> outputs;
  QuantParam output_qparam;
  CalibData calib;
  bool is_quantized = false;
};

// ======================== Graph ========================

struct Graph {
  std::string name;
  std::vector<std::unique_ptr<Op>> ops;

  Op *add_op(OpKind kind, const std::string &name) {
    auto op = std::make_unique<Op>();
    op->id = static_cast<int>(ops.size());
    op->kind = kind;
    op->name = name;
    auto *p = op.get();
    ops.push_back(std::move(op));
    return p;
  }

  void print(std::ostream &os) const {
    os << "Graph: " << name << " (" << ops.size() << " ops)\n";
    for (auto &op : ops) {
      os << "  [" << op->id << "] " << opkind_str(op->kind);
      if (op->is_quantized) os << " [QUANTIZED]";
      os << " \"" << op->name << "\"\n";
      for (auto &in : op->inputs)
        os << "       in:  " << in.str() << " (" << in.name << ")\n";
      for (auto &out : op->outputs)
        os << "       out: " << out.str() << " (" << out.name << ")\n";
    }
  }
};

}  // namespace quant_ir
