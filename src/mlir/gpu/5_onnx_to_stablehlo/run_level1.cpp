// run_level1.cpp — P4 tier 1: hand-written ONNX → StableHLO lowering
//
// 面试必须达到的基本能力：能手写以下 5 种 op 的 lowering 规则
//   ONNX Add       → stablehlo.add
//   ONNX MatMul    → stablehlo.dot_general
//   ONNX Conv      → stablehlo.convolution
//   ONNX Reshape   → stablehlo.reshape
//   ONNX Transpose → stablehlo.transpose
//
// 运行: ./run_lowering_l1 <model.onnx>

#include "onnx_to_shlo_utils.h"

using namespace onnx2shlo;

// ====================================================================
// P4 tier 1 converters — one function per ONNX op
// ====================================================================

// Add(A, B) → stablehlo.add A, B
// P4 tier 1 assumption: shapes of A and B are identical (no broadcast).
static void convert_add(const onnx::NodeProto &node, Context &ctx) {
  auto lhs = ctx.lookup(node.input(0));
  auto rhs = ctx.lookup(node.input(1));
  auto result = ctx.builder.emit_add(lhs, rhs);
  ctx.value_map[node.output(0)] = result;
}

// MatMul(A[M,K], B[K,N]) → stablehlo.dot_general
// The key semantic: contracting last dim of LHS with second-to-last of RHS.
static void convert_matmul(const onnx::NodeProto &node, Context &ctx) {
  auto lhs = ctx.lookup(node.input(0));
  auto rhs = ctx.lookup(node.input(1));

  int64_t lhs_contract = lhs.type.rank() - 1;
  int64_t rhs_contract = std::max<int64_t>(0, rhs.type.rank() - 2);

  shlo::TensorType res;
  res.elem = lhs.type.elem;
  if (lhs.type.rank() == 2 && rhs.type.rank() == 2) {
    res.dims = {lhs.type.dims[0], rhs.type.dims[1]};
  } else if (lhs.type.rank() == 2 && rhs.type.rank() == 1) {
    res.dims = {lhs.type.dims[0]};
  } else {
    // Batched matmul: batch dims are all leading dims except last 2
    auto out_type = ctx.get_type(node.output(0));
    if (!out_type.dims.empty()) {
      res = out_type;
    } else {
      res.dims = {lhs.type.dims[0], rhs.type.dims.back()};
    }
  }

  auto result = ctx.builder.emit_dot_general(lhs, rhs, {}, {},
                                             {lhs_contract}, {rhs_contract},
                                             res);
  ctx.value_map[node.output(0)] = result;
}

// Conv(input, kernel [, bias]) → stablehlo.convolution [+ broadcast + add]
// ONNX default layout: NCHW input, OIHW kernel.
// Attributes: strides, pads, dilations, group.
static void convert_conv(const onnx::NodeProto &node, Context &ctx) {
  auto input = ctx.lookup(node.input(0));
  auto kernel = ctx.lookup(node.input(1));

  int spatial = static_cast<int>(input.type.rank()) - 2;
  auto strides = get_ints_attr(node, "strides");
  auto pads = get_ints_attr(node, "pads");
  auto dilations = get_ints_attr(node, "dilations");
  int64_t group = get_int_attr(node, "group", 1);

  if (strides.empty()) strides.assign(spatial, 1);
  if (pads.empty()) pads.assign(spatial * 2, 0);
  if (dilations.empty()) dilations.assign(spatial, 1);

  // ONNX pads: [begin_h, begin_w, end_h, end_w]
  // StableHLO: [(begin_h, end_h), (begin_w, end_w)]
  std::vector<std::pair<int64_t, int64_t>> padding(spatial);
  for (int i = 0; i < spatial; ++i)
    padding[i] = {pads[i], pads[i + spatial]};

  std::vector<int64_t> lhs_dilation(spatial, 1);

  // Output shape from type_map or manual computation
  auto res = ctx.get_type(node.output(0));
  if (res.dims.empty()) {
    res.elem = input.type.elem;
    res.dims.push_back(input.type.dims[0]);
    res.dims.push_back(kernel.type.dims[0]);
    for (int i = 0; i < spatial; ++i) {
      int64_t k_eff = (kernel.type.dims[2 + i] - 1) * dilations[i] + 1;
      int64_t out =
          (input.type.dims[2 + i] + padding[i].first + padding[i].second -
           k_eff) /
              strides[i] +
          1;
      res.dims.push_back(out);
    }
  }

  auto conv_out = ctx.builder.emit_convolution(
      input, kernel, strides, padding, lhs_dilation, dilations, group, 1, res);

  // Bias handling: bias[C] → broadcast to [N,C,H,W] → add
  if (node.input_size() > 2 && !node.input(2).empty()) {
    auto bias = ctx.lookup(node.input(2));
    if (bias.valid()) {
      auto bc = ctx.builder.emit_broadcast_in_dim(bias, {1}, res);
      conv_out = ctx.builder.emit_add(conv_out, bc, res);
    }
  }

  ctx.value_map[node.output(0)] = conv_out;
}

// Reshape(data, shape_tensor) → stablehlo.reshape
// P4 tier 1: shape must be a known initializer.
static void convert_reshape(const onnx::NodeProto &node, Context &ctx) {
  auto operand = ctx.lookup(node.input(0));

  shlo::TensorType res = ctx.get_type(node.output(0));
  if (res.dims.empty()) {
    auto *shape_init = ctx.get_initializer(node.input(1));
    if (shape_init) {
      auto dims = extract_int64s(*shape_init);
      res.elem = operand.type.elem;
      res.dims = dims;
      // Resolve -1 (infer one dimension from total elements)
      int64_t known = 1, neg_idx = -1;
      for (size_t i = 0; i < res.dims.size(); ++i) {
        if (res.dims[i] == -1)
          neg_idx = (int64_t)i;
        else
          known *= res.dims[i];
      }
      if (neg_idx >= 0 && known > 0 && operand.type.num_elements() > 0)
        res.dims[neg_idx] = operand.type.num_elements() / known;
    }
  }

  auto result = ctx.builder.emit_reshape(operand, res);
  ctx.value_map[node.output(0)] = result;
}

// Transpose(data, perm) → stablehlo.transpose
static void convert_transpose(const onnx::NodeProto &node, Context &ctx) {
  auto operand = ctx.lookup(node.input(0));

  auto perm = get_ints_attr(node, "perm");
  if (perm.empty())
    for (int64_t i = operand.type.rank() - 1; i >= 0; --i)
      perm.push_back(i);

  shlo::TensorType res;
  res.elem = operand.type.elem;
  for (auto p : perm) res.dims.push_back(operand.type.dims[p]);

  auto result = ctx.builder.emit_transpose(operand, perm, res);
  ctx.value_map[node.output(0)] = result;
}

// Mul(A, B) → stablehlo.multiply (same-shape assumption)
static void convert_mul(const onnx::NodeProto &node, Context &ctx) {
  auto lhs = ctx.lookup(node.input(0));
  auto rhs = ctx.lookup(node.input(1));
  auto result = ctx.builder.emit_multiply(lhs, rhs);
  ctx.value_map[node.output(0)] = result;
}

// ====================================================================
// Dispatcher
// ====================================================================

static bool dispatch(const onnx::NodeProto &node, Context &ctx) {
  const auto &op = node.op_type();
  if (op == "Add")            convert_add(node, ctx);
  else if (op == "MatMul")    convert_matmul(node, ctx);
  else if (op == "Conv")      convert_conv(node, ctx);
  else if (op == "Reshape")   convert_reshape(node, ctx);
  else if (op == "Transpose") convert_transpose(node, ctx);
  else if (op == "Mul")       convert_mul(node, ctx);
  else {
    ctx.warn("Unsupported op: " + op + " (skipped)");
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
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
    return 1;
  }

  onnx::ModelProto model;
  if (!load_model(argv[1], model)) return 1;

  std::cout
      << "====================================================\n"
      << " Level 1: ONNX -> StableHLO (Basic Hand-Written)\n"
      << "====================================================\n\n";
  std::cout << "Model : " << argv[1] << "\n";
  std::cout << "Graph : " << model.graph().name() << "\n";
  std::cout << "Nodes : " << model.graph().node_size() << "\n\n";

  Context ctx(model.graph());
  ctx.init();
  ctx.create_func_args();
  ctx.emit_initializers();

  std::cout << "--- Converting nodes ---\n";
  for (auto &node : model.graph().node()) {
    std::cout << "  " << node.op_type();
    if (!node.name().empty()) std::cout << " (" << node.name() << ")";
    std::cout << " -> ";
    if (dispatch(node, ctx)) {
      auto v = ctx.lookup(node.output(0));
      std::cout << "OK  " << v.name << " : " << v.type.str() << "\n";
    } else {
      std::cout << "SKIPPED\n";
    }
  }

  ctx.finalize();

  shlo::ModuleOp module;
  module.funcs.push_back(ctx.func);

  std::cout << "\n--- Generated StableHLO MLIR ---\n\n";
  module.print(std::cout);

  ctx.print_stats();
  return ctx.errors > 0 ? 1 : 0;
}
