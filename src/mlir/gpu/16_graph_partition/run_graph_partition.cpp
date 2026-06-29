// run_graph_partition.cpp — Graph Partitioning teaching pipeline (P13).
//
// Simulates compiler-time partitioning of a single Transformer decoder layer into
// Attention and FFN subgraphs for multi-device execution planning.
//
//   P13 Step 0: Build layer graph (Attention + Residual + LN + FFN + Residual)
//   P13 Step 1: Annotate ops with partition hints
//   P13 Step 2: Cut at residual/LN boundaries → two partitions
//   P13 Step 3: Compute boundary tensors + estimated comm volume
//   P13 Step 4: Summary (partition sizes, comm bytes, interview talking points)
//
// Pure C++17, header-only IR, no external dependencies.

#include "graph_partition_ir.h"

#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace gp_ir;

static void sep(const char *title) {
  std::cout << "\n  ──── " << title << " ────\n\n";
}

static int64_t tensor_bytes(const std::string &name,
                              const std::map<std::string, int64_t> &sizes) {
  auto it = sizes.find(name);
  return it == sizes.end() ? 0 : it->second;
}

static std::vector<GraphOp> build_transformer_layer() {
  std::vector<GraphOp> ops;

  auto add = [&](const char *name, OpKind kind, const char *hint,
                 std::initializer_list<const char *> ins,
                 std::initializer_list<const char *> outs) {
    GraphOp op;
    op.name = name;
    op.kind = kind;
    op.partition_hint = hint;
    for (auto *in : ins) op.inputs.push_back(in);
    for (auto *out : outs) op.outputs.push_back(out);
    ops.push_back(std::move(op));
  };

  // Attention block
  add("attn_qk", OpKind::MatMul, "attention", {"Q", "Kt"}, {"scores"});
  add("attn_softmax", OpKind::Softmax, "attention", {"scores"}, {"probs"});
  add("attn_av", OpKind::MatMul, "attention", {"probs", "V"}, {"attn_out"});
  add("residual1", OpKind::Residual, "attention", {"x", "attn_out"},
      {"hidden1"});

  // Pre-FFN norm (stays with FFN partition in tensor-parallel style cuts)
  add("ln1", OpKind::LayerNorm, "ffn", {"hidden1"}, {"normed"});

  // FFN block
  add("ffn_up", OpKind::MatMul, "ffn", {"normed", "W1"}, {"up"});
  add("ffn_gelu", OpKind::Gelu, "ffn", {"up"}, {"act"});
  add("ffn_down", OpKind::MatMul, "ffn", {"act", "W2"}, {"ffn_out"});
  add("residual2", OpKind::Residual, "ffn", {"hidden1", "ffn_out"},
      {"layer_out"});

  return ops;
}

static PartitionPlan partition_by_hint(const std::vector<GraphOp> &ops,
                                       const std::map<std::string, int64_t> &sizes) {
  Partition attn;
  attn.name = "partition_attention";
  Partition ffn;
  ffn.name = "partition_ffn";

  std::set<std::string> attn_tensors;
  std::set<std::string> ffn_tensors;

  for (const auto &op : ops) {
    if (op.partition_hint == "attention") {
      attn.ops.push_back(op.name);
      for (const auto &t : op.inputs) attn_tensors.insert(t);
      for (const auto &t : op.outputs) attn_tensors.insert(t);
    } else {
      ffn.ops.push_back(op.name);
      for (const auto &t : op.inputs) ffn_tensors.insert(t);
      for (const auto &t : op.outputs) ffn_tensors.insert(t);
    }
  }

  // Boundary = tensors used by both partitions (e.g. hidden1 after residual1)
  for (const auto &t : attn_tensors) {
    if (ffn_tensors.count(t)) {
      attn.boundary_tensors.push_back(t);
      ffn.boundary_tensors.push_back(t);
      int64_t bytes = tensor_bytes(t, sizes);
      attn.estimated_bytes_moved += bytes;
      ffn.estimated_bytes_moved += bytes;
    }
  }

  PartitionPlan plan;
  plan.partitions = {attn, ffn};
  for (const auto &p : plan.partitions)
    plan.total_comm_bytes += p.estimated_bytes_moved;
  return plan;
}

int main() {
  std::cout << "====================================================\n"
            << " P13: Graph Partitioning (Transformer Layer Cut)\n"
            << "====================================================\n";

  sep("Stage 0: Build Transformer Layer Graph");
  auto ops = build_transformer_layer();
  std::map<std::string, int64_t> tensor_sizes = {
      {"x", 1 * 8 * 4096 * 4},
      {"Q", 1 * 8 * 4096 * 4},
      {"Kt", 1 * 8 * 4096 * 4},
      {"V", 1 * 8 * 4096 * 4},
      {"scores", 1 * 8 * 8 * 4},
      {"probs", 1 * 8 * 8 * 4},
      {"attn_out", 1 * 8 * 4096 * 4},
      {"hidden1", 1 * 8 * 4096 * 4},
      {"normed", 1 * 8 * 4096 * 4},
      {"W1", 4096 * 11008 * 4},
      {"up", 1 * 8 * 11008 * 4},
      {"act", 1 * 8 * 11008 * 4},
      {"W2", 11008 * 4096 * 4},
      {"ffn_out", 1 * 8 * 4096 * 4},
      {"layer_out", 1 * 8 * 4096 * 4},
  };

  for (const auto &op : ops) {
    std::cout << "  [" << op.partition_hint << "] " << op.name << " ("
              << op_kind_name(op.kind) << ")\n";
  }

  sep("Stage 1: Partition by Attention / FFN Hints");
  auto plan = partition_by_hint(ops, tensor_sizes);
  for (const auto &part : plan.partitions) {
    std::cout << "  " << part.name << " (" << part.ops.size() << " ops)\n";
    for (const auto &op : part.ops)
      std::cout << "    - " << op << "\n";
    std::cout << "  boundary tensors:";
    for (const auto &t : part.boundary_tensors)
      std::cout << " " << t;
    std::cout << "\n  est. comm bytes: " << part.estimated_bytes_moved << "\n";
  }

  sep("Stage 2: Summary (Interview Notes)");
  std::cout << "  Total cross-partition comm bytes: " << plan.total_comm_bytes
            << "\n";
  std::cout << "  Cut rationale: residual output `hidden1` is the hand-off tensor\n";
  std::cout << "    between Attention (device 0) and FFN (device 1) in pipeline parallel.\n";
  std::cout << "  Teaching scope: compiler-time partition planning only;\n";
  std::cout << "    no runtime collective / NCCL in this demo.\n";

  std::cout << "\nGraph partitioning teaching pipeline completed.\n";
  return 0;
}
