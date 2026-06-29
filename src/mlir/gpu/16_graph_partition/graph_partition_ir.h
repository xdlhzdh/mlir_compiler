// graph_partition_ir.h — Teaching IR for LLM graph partitioning (compiler-time).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gp_ir {

enum class OpKind {
  MatMul,
  Softmax,
  Add,
  LayerNorm,
  Gelu,
  Residual,
};

struct TensorRef {
  std::string name;
  int64_t bytes = 0;
};

struct GraphOp {
  std::string name;
  OpKind kind = OpKind::MatMul;
  std::string partition_hint; // "attention" | "ffn" | "residual"
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

struct Partition {
  std::string name;
  std::vector<std::string> ops;
  std::vector<std::string> boundary_tensors;
  int64_t estimated_bytes_moved = 0;
};

struct PartitionPlan {
  std::vector<Partition> partitions;
  int64_t total_comm_bytes = 0;
};

inline const char *op_kind_name(OpKind k) {
  switch (k) {
  case OpKind::MatMul:
    return "MatMul";
  case OpKind::Softmax:
    return "Softmax";
  case OpKind::Add:
    return "Add";
  case OpKind::LayerNorm:
    return "LayerNorm";
  case OpKind::Gelu:
    return "Gelu";
  case OpKind::Residual:
    return "Residual";
  }
  return "Unknown";
}

} // namespace gp_ir
