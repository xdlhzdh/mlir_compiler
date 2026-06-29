// run_level2.cpp — P4 tier 2: enhanced ONNX → StableHLO lowering
//
// 合格工程师水平：在 P4 tier 1 基础上能处理
//   (a) broadcast — numpy-style 广播 → 显式 broadcast_in_dim
//   (b) dynamic shape — 处理带 '?' 维度的 tensor
//   (c) attribute mapping — 完整的 ONNX Conv 属性映射 (auto_pad, pads, strides…)
//   (d) error handling — 输入校验 + 诊断信息
//
// 运行: ./run_lowering_l2 <model.onnx>

#include "onnx_to_shlo_utils.h"
#include <cassert>

using namespace onnx2shlo;

// ====================================================================
// Broadcast Utilities
// ====================================================================

// Compute numpy-style broadcast result shape for two operands.
// Returns empty on incompatible shapes.
static shlo::TensorType broadcast_result_type(const shlo::TensorType &a,
                                              const shlo::TensorType &b) {
  int ra = a.rank(), rb = b.rank();
  int rmax = std::max(ra, rb);
  shlo::TensorType res;
  res.elem = a.elem;
  res.dims.resize(rmax);

  for (int i = 0; i < rmax; ++i) {
    int64_t da = (i < rmax - ra) ? 1 : a.dims[i - (rmax - ra)];
    int64_t db = (i < rmax - rb) ? 1 : b.dims[i - (rmax - rb)];
    if (da == db) {
      res.dims[i] = da;
    } else if (da == 1) {
      res.dims[i] = db;
    } else if (db == 1) {
      res.dims[i] = da;
    } else if (da == -1 || db == -1) {
      res.dims[i] = -1; // dynamic → dynamic
    } else {
      return {}; // incompatible
    }
  }
  return res;
}

// Insert broadcast_in_dim when operand shape differs from result shape.
// broadcast_dimensions maps operand dim i → result dim (rmax - rank + i).
static shlo::Value maybe_broadcast(const shlo::Value &val,
                                   const shlo::TensorType &target,
                                   Context &ctx) {
  if (val.type == target) return val;

  std::vector<int64_t> dims;
  int offset = target.rank() - val.type.rank();
  for (int i = 0; i < val.type.rank(); ++i)
    dims.push_back(offset + i);

  return ctx.builder.emit_broadcast_in_dim(val, dims, target);
}

// Broadcast + binary op helper
static shlo::Value emit_broadcast_binary(const char *mnemonic,
                                         const shlo::Value &lhs,
                                         const shlo::Value &rhs,
                                         Context &ctx) {
  auto res_type = broadcast_result_type(lhs.type, rhs.type);
  if (res_type.dims.empty()) {
    ctx.error(std::string("Broadcast incompatible: ") + lhs.type.str() +
              " vs " + rhs.type.str());
    return lhs;
  }
  auto bl = maybe_broadcast(lhs, res_type, ctx);
  auto br = maybe_broadcast(rhs, res_type, ctx);
  return ctx.builder.emit_binary(mnemonic, bl, br, res_type);
}

// ====================================================================
// Auto-padding computation (ONNX auto_pad attribute)
// ====================================================================

static std::vector<std::pair<int64_t, int64_t>>
compute_auto_pad(const std::string &auto_pad,
                 const std::vector<int64_t> &input_spatial,
                 const std::vector<int64_t> &kernel_shape,
                 const std::vector<int64_t> &strides,
                 const std::vector<int64_t> &dilations) {
  int sp = static_cast<int>(strides.size());
  std::vector<std::pair<int64_t, int64_t>> pads(sp, {0, 0});

  if (auto_pad == "VALID" || auto_pad.empty() || auto_pad == "NOTSET")
    return pads;

  for (int i = 0; i < sp; ++i) {
    int64_t in_dim = input_spatial[i];
    int64_t k_eff = (kernel_shape[i] - 1) * dilations[i] + 1;
    int64_t out_dim = (in_dim + strides[i] - 1) / strides[i]; // ceil
    int64_t total = std::max<int64_t>(0, (out_dim - 1) * strides[i] + k_eff - in_dim);
    if (auto_pad == "SAME_UPPER") {
      pads[i] = {total / 2, total - total / 2};
    } else { // SAME_LOWER
      pads[i] = {total - total / 2, total / 2};
    }
  }
  return pads;
}

// ====================================================================
// P4 tier 2 converters
// ====================================================================

static void convert_add_l2(const onnx::NodeProto &node, Context &ctx) {
  auto lhs = ctx.lookup(node.input(0));
  auto rhs = ctx.lookup(node.input(1));
  if (!lhs.valid() || !rhs.valid()) {
    ctx.error("Add: missing input(s)");
    return;
  }
  auto result = emit_broadcast_binary("stablehlo.add", lhs, rhs, ctx);
  ctx.value_map[node.output(0)] = result;
}

static void convert_mul_l2(const onnx::NodeProto &node, Context &ctx) {
  auto lhs = ctx.lookup(node.input(0));
  auto rhs = ctx.lookup(node.input(1));
  if (!lhs.valid() || !rhs.valid()) {
    ctx.error("Mul: missing input(s)");
    return;
  }
  auto result = emit_broadcast_binary("stablehlo.multiply", lhs, rhs, ctx);
  ctx.value_map[node.output(0)] = result;
}

static void convert_sub_l2(const onnx::NodeProto &node, Context &ctx) {
  auto lhs = ctx.lookup(node.input(0));
  auto rhs = ctx.lookup(node.input(1));
  if (!lhs.valid() || !rhs.valid()) {
    ctx.error("Sub: missing input(s)");
    return;
  }
  auto result = emit_broadcast_binary("stablehlo.subtract", lhs, rhs, ctx);
  ctx.value_map[node.output(0)] = result;
}

static void convert_matmul_l2(const onnx::NodeProto &node, Context &ctx) {
  auto lhs = ctx.lookup(node.input(0));
  auto rhs = ctx.lookup(node.input(1));
  if (!lhs.valid() || !rhs.valid()) {
    ctx.error("MatMul: missing input(s)");
    return;
  }
  if (lhs.type.rank() < 1 || rhs.type.rank() < 1) {
    ctx.error("MatMul: operands must be at least rank-1");
    return;
  }

  int64_t lhs_contract = lhs.type.rank() - 1;
  int64_t rhs_contract = std::max<int64_t>(0, rhs.type.rank() - 2);

  // Batched matmul: leading dims that are broadcast-compatible
  std::vector<int64_t> lhs_batch, rhs_batch;
  int lhs_batch_rank = lhs.type.rank() - 2;
  int rhs_batch_rank = rhs.type.rank() - 2;
  if (lhs_batch_rank > 0 && rhs_batch_rank > 0) {
    int n = std::min(lhs_batch_rank, rhs_batch_rank);
    for (int i = 0; i < n; ++i) {
      lhs_batch.push_back(i);
      rhs_batch.push_back(i);
    }
  }

  // Result type
  shlo::TensorType res = ctx.get_type(node.output(0));
  if (res.dims.empty()) {
    res.elem = lhs.type.elem;
    for (auto b : lhs_batch) res.dims.push_back(lhs.type.dims[b]);
    if (lhs.type.rank() >= 2)
      res.dims.push_back(lhs.type.dims[lhs.type.rank() - 2]);
    if (rhs.type.rank() >= 2)
      res.dims.push_back(rhs.type.dims[rhs.type.rank() - 1]);
    else if (rhs.type.rank() == 1) {
      // matrix-vector: no trailing dim
    }
  }

  auto result = ctx.builder.emit_dot_general(lhs, rhs, lhs_batch, rhs_batch,
                                             {lhs_contract}, {rhs_contract}, res);
  ctx.value_map[node.output(0)] = result;
}

static void convert_conv_l2(const onnx::NodeProto &node, Context &ctx) {
  auto input = ctx.lookup(node.input(0));
  auto kernel = ctx.lookup(node.input(1));
  if (!input.valid() || !kernel.valid()) {
    ctx.error("Conv: missing input or kernel");
    return;
  }

  int spatial = static_cast<int>(input.type.rank()) - 2;
  if (spatial < 1) {
    ctx.error("Conv: input rank must be >= 3");
    return;
  }

  auto strides = get_ints_attr(node, "strides");
  auto pads_raw = get_ints_attr(node, "pads");
  auto dilations = get_ints_attr(node, "dilations");
  auto kernel_shape = get_ints_attr(node, "kernel_shape");
  auto auto_pad = get_str_attr(node, "auto_pad");
  int64_t group = get_int_attr(node, "group", 1);

  if (strides.empty()) strides.assign(spatial, 1);
  if (dilations.empty()) dilations.assign(spatial, 1);
  if (kernel_shape.empty())
    for (int i = 0; i < spatial; ++i)
      kernel_shape.push_back(kernel.type.dims[2 + i]);

  // Compute padding
  std::vector<std::pair<int64_t, int64_t>> padding(spatial, {0, 0});
  if (!auto_pad.empty() && auto_pad != "NOTSET") {
    std::vector<int64_t> in_sp;
    for (int i = 0; i < spatial; ++i) in_sp.push_back(input.type.dims[2 + i]);
    padding = compute_auto_pad(auto_pad, in_sp, kernel_shape, strides, dilations);
    ctx.info("auto_pad=" + auto_pad + " resolved to explicit padding");
  } else if (!pads_raw.empty()) {
    for (int i = 0; i < spatial; ++i)
      padding[i] = {pads_raw[i], pads_raw[i + spatial]};
  }

  std::vector<int64_t> lhs_dilation(spatial, 1);

  // Output shape
  auto res = ctx.get_type(node.output(0));
  if (res.dims.empty()) {
    res.elem = input.type.elem;
    int64_t N = input.type.dims[0];
    int64_t C_out = kernel.type.dims[0];
    if (group > 1) C_out = kernel.type.dims[0] * group;
    res.dims.push_back(N);
    res.dims.push_back(C_out);
    for (int i = 0; i < spatial; ++i) {
      int64_t in_d = input.type.dims[2 + i];
      int64_t k_eff = (kernel_shape[i] - 1) * dilations[i] + 1;
      if (in_d < 0) {
        res.dims.push_back(-1); // dynamic
      } else {
        int64_t out = (in_d + padding[i].first + padding[i].second - k_eff) /
                          strides[i] + 1;
        res.dims.push_back(out);
      }
    }
  }

  auto conv_out = ctx.builder.emit_convolution(
      input, kernel, strides, padding, lhs_dilation, dilations, group, 1, res);

  // Bias handling with proper broadcast
  if (node.input_size() > 2 && !node.input(2).empty()) {
    auto bias = ctx.lookup(node.input(2));
    if (bias.valid()) {
      auto bc = ctx.builder.emit_broadcast_in_dim(bias, {1}, res);
      conv_out = ctx.builder.emit_add(conv_out, bc, res);
    }
  }

  ctx.value_map[node.output(0)] = conv_out;
}

static void convert_reshape_l2(const onnx::NodeProto &node, Context &ctx) {
  auto operand = ctx.lookup(node.input(0));
  if (!operand.valid()) {
    ctx.error("Reshape: missing operand");
    return;
  }

  shlo::TensorType res = ctx.get_type(node.output(0));
  bool has_dynamic = false;

  if (res.dims.empty()) {
    auto *shape_init = ctx.get_initializer(node.input(1));
    if (shape_init) {
      res.dims = extract_int64s(*shape_init);
      res.elem = operand.type.elem;
      // Resolve -1
      int64_t known = 1, neg_idx = -1;
      for (size_t i = 0; i < res.dims.size(); ++i) {
        if (res.dims[i] == -1) neg_idx = (int64_t)i;
        else if (res.dims[i] == 0) {
          // ONNX: 0 means copy from input
          if ((int)i < operand.type.rank())
            res.dims[i] = operand.type.dims[i];
        }
        if (res.dims[i] > 0) known *= res.dims[i];
      }
      if (neg_idx >= 0 && known > 0 && operand.type.num_elements() > 0)
        res.dims[neg_idx] = operand.type.num_elements() / known;
    } else {
      has_dynamic = true;
    }
  }

  for (auto d : res.dims)
    if (d < 0) has_dynamic = true;

  if (has_dynamic && !operand.type.is_dynamic()) {
    // Shape is runtime-computed → use dynamic_reshape
    auto shape_val = ctx.lookup(node.input(1));
    if (shape_val.valid()) {
      auto result = ctx.builder.emit_dynamic_reshape(operand, shape_val, res);
      ctx.value_map[node.output(0)] = result;
      ctx.info("Reshape: emitted dynamic_reshape (runtime shape)");
      return;
    }
  }

  auto result = ctx.builder.emit_reshape(operand, res);
  ctx.value_map[node.output(0)] = result;
}

static void convert_transpose_l2(const onnx::NodeProto &node, Context &ctx) {
  auto operand = ctx.lookup(node.input(0));
  if (!operand.valid()) {
    ctx.error("Transpose: missing operand");
    return;
  }

  auto perm = get_ints_attr(node, "perm");
  if (perm.empty())
    for (int64_t i = operand.type.rank() - 1; i >= 0; --i)
      perm.push_back(i);

  if ((int64_t)perm.size() != operand.type.rank()) {
    ctx.error("Transpose: perm length (" + std::to_string(perm.size()) +
              ") != rank (" + std::to_string(operand.type.rank()) + ")");
    return;
  }

  shlo::TensorType res;
  res.elem = operand.type.elem;
  for (auto p : perm) {
    if (p < 0 || p >= operand.type.rank()) {
      ctx.error("Transpose: perm index out of range");
      return;
    }
    res.dims.push_back(operand.type.dims[p]);
  }

  auto result = ctx.builder.emit_transpose(operand, perm, res);
  ctx.value_map[node.output(0)] = result;
}

// ====================================================================
// Dispatcher
// ====================================================================

static bool dispatch_l2(const onnx::NodeProto &node, Context &ctx) {
  const auto &op = node.op_type();
  if (op == "Add")            convert_add_l2(node, ctx);
  else if (op == "Mul")       convert_mul_l2(node, ctx);
  else if (op == "Sub")       convert_sub_l2(node, ctx);
  else if (op == "MatMul")    convert_matmul_l2(node, ctx);
  else if (op == "Conv")      convert_conv_l2(node, ctx);
  else if (op == "Reshape")   convert_reshape_l2(node, ctx);
  else if (op == "Transpose") convert_transpose_l2(node, ctx);
  else {
    ctx.warn("Unsupported op: " + op);
    ++ctx.skipped;
    return false;
  }
  ++ctx.converted;
  return true;
}

// ====================================================================
// main
// ====================================================================

int main(int argc, char **argv) {
  bool mlirOnly = false;
  std::string modelPath;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--mlir-only")
      mlirOnly = true;
    else if (modelPath.empty())
      modelPath = arg;
  }
  if (modelPath.empty()) {
    std::cerr << "Usage: " << argv[0] << " [--mlir-only] <model.onnx>\n";
    return 1;
  }

  onnx::ModelProto model;
  if (!load_model(modelPath, model)) return 1;

  if (!mlirOnly) {
    std::cout
        << "====================================================\n"
        << " Level 2: ONNX -> StableHLO (Broadcast / Dynamic /\n"
        << "          Attribute Mapping / Error Handling)\n"
        << "====================================================\n\n";
    std::cout << "Model : " << modelPath << "\n";
    std::cout << "Graph : " << model.graph().name() << "\n";
    std::cout << "Nodes : " << model.graph().node_size() << "\n\n";
  }

  Context ctx(model.graph());
  ctx.init();
  ctx.create_func_args();

  if (!mlirOnly) {
    for (auto &arg : ctx.func.args)
      if (arg.type.is_dynamic())
        ctx.info("Dynamic input: " + arg.name + " : " + arg.type.str());
  }

  ctx.emit_initializers();

  if (!mlirOnly) std::cout << "\n--- Converting nodes ---\n";
  for (auto &node : model.graph().node()) {
    if (!mlirOnly) {
      std::cout << "  " << node.op_type();
      if (!node.name().empty()) std::cout << " (" << node.name() << ")";
      std::cout << " -> ";
    }
    if (dispatch_l2(node, ctx)) {
      if (!mlirOnly) {
        auto v = ctx.lookup(node.output(0));
        std::cout << "OK  " << v.name << " : " << v.type.str() << "\n";
      }
    } else if (!mlirOnly) {
      std::cout << "SKIPPED\n";
    }
  }

  ctx.finalize();

  shlo::ModuleOp module;
  module.funcs.push_back(ctx.func);

  if (mlirOnly) {
    module.print(std::cout);
    return ctx.errors > 0 ? 1 : 0;
  }

  std::cout << "\n--- Generated StableHLO MLIR ---\n\n";
  module.print(std::cout);

  ctx.print_stats();
  return ctx.errors > 0 ? 1 : 0;
}
