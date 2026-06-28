// run_level3.cpp — P4 tier 3: pattern-based ONNX → StableHLO conversion framework
//
// 进阶工程师水平：能设计通用的 conversion framework
//   (a) ConversionPattern — 每个 op 一个 pattern class (matchAndRewrite)
//   (b) ConversionTarget — 标记 legal / illegal dialect
//   (c) RewritePatternSet — pattern 集合 + 优先级排序
//   (d) applyFullConversion — 驱动转换 + 合法性验证
//   (e) Legalization pipeline — 完整的 ONNX → StableHLO 验证流水线
//
// 这是 MLIR DialectConversion 的教学级实现，覆盖面试核心考点。
//
// 运行: ./run_lowering_l3 <model.onnx>

#include "conversion_framework.h"
#include <algorithm>
#include <cassert>

using namespace onnx2shlo;
using namespace framework;

// ====================================================================
// Broadcast helper (reused from P4 tier 2)
// ====================================================================

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
    if (da == db)
      res.dims[i] = da;
    else if (da == 1)
      res.dims[i] = db;
    else if (db == 1)
      res.dims[i] = da;
    else if (da == -1 || db == -1)
      res.dims[i] = -1;
    else
      return {};
  }
  return res;
}

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

// ====================================================================
// Concrete ConversionPatterns — one class per ONNX op
// ====================================================================

class AddOpConversion : public ConversionPattern {
public:
  std::string target_op_type() const override { return "Add"; }

  LogicalResult matchAndRewrite(const onnx::NodeProto &node,
                                Context &ctx) const override {
    auto lhs = ctx.lookup(node.input(0));
    auto rhs = ctx.lookup(node.input(1));
    if (!lhs.valid() || !rhs.valid()) return LogicalResult::failure();

    auto rt = broadcast_result_type(lhs.type, rhs.type);
    if (rt.dims.empty()) return LogicalResult::failure();

    auto bl = maybe_broadcast(lhs, rt, ctx);
    auto br = maybe_broadcast(rhs, rt, ctx);
    ctx.value_map[node.output(0)] = ctx.builder.emit_add(bl, br, rt);
    return LogicalResult::success();
  }
};

class MulOpConversion : public ConversionPattern {
public:
  std::string target_op_type() const override { return "Mul"; }

  LogicalResult matchAndRewrite(const onnx::NodeProto &node,
                                Context &ctx) const override {
    auto lhs = ctx.lookup(node.input(0));
    auto rhs = ctx.lookup(node.input(1));
    if (!lhs.valid() || !rhs.valid()) return LogicalResult::failure();

    auto rt = broadcast_result_type(lhs.type, rhs.type);
    if (rt.dims.empty()) return LogicalResult::failure();

    auto bl = maybe_broadcast(lhs, rt, ctx);
    auto br = maybe_broadcast(rhs, rt, ctx);
    ctx.value_map[node.output(0)] = ctx.builder.emit_multiply(bl, br, rt);
    return LogicalResult::success();
  }
};

class SubOpConversion : public ConversionPattern {
public:
  std::string target_op_type() const override { return "Sub"; }

  LogicalResult matchAndRewrite(const onnx::NodeProto &node,
                                Context &ctx) const override {
    auto lhs = ctx.lookup(node.input(0));
    auto rhs = ctx.lookup(node.input(1));
    if (!lhs.valid() || !rhs.valid()) return LogicalResult::failure();

    auto rt = broadcast_result_type(lhs.type, rhs.type);
    if (rt.dims.empty()) return LogicalResult::failure();

    auto bl = maybe_broadcast(lhs, rt, ctx);
    auto br = maybe_broadcast(rhs, rt, ctx);
    ctx.value_map[node.output(0)] = ctx.builder.emit_subtract(bl, br, rt);
    return LogicalResult::success();
  }
};

class DivOpConversion : public ConversionPattern {
public:
  std::string target_op_type() const override { return "Div"; }

  LogicalResult matchAndRewrite(const onnx::NodeProto &node,
                                Context &ctx) const override {
    auto lhs = ctx.lookup(node.input(0));
    auto rhs = ctx.lookup(node.input(1));
    if (!lhs.valid() || !rhs.valid()) return LogicalResult::failure();

    auto rt = broadcast_result_type(lhs.type, rhs.type);
    if (rt.dims.empty() && lhs.type.rank() != 0 && rhs.type.rank() != 0)
      return LogicalResult::failure();

    auto bl = maybe_broadcast(lhs, rt, ctx);
    auto br = maybe_broadcast(rhs, rt, ctx);
    ctx.value_map[node.output(0)] = ctx.builder.emit_divide(bl, br, rt);
    return LogicalResult::success();
  }
};

class PowOpConversion : public ConversionPattern {
public:
  std::string target_op_type() const override { return "Pow"; }

  LogicalResult matchAndRewrite(const onnx::NodeProto &node,
                                Context &ctx) const override {
    auto base = ctx.lookup(node.input(0));
    auto exponent = ctx.lookup(node.input(1));
    if (!base.valid() || !exponent.valid()) return LogicalResult::failure();

    auto *exp_init = ctx.get_initializer(node.input(1));
    if (!exp_init) return LogicalResult::failure();
    auto vals = extract_floats(*exp_init);
    if (vals.size() != 1 || vals[0] != 2.0f) return LogicalResult::failure();

    // RMSNorm only needs square; use multiply so the generated StableHLO stays
    // broadly supported by the later teaching pipeline.
    ctx.value_map[node.output(0)] = ctx.builder.emit_multiply(base, base,
                                                              base.type);
    return LogicalResult::success();
  }
};

class SqrtOpConversion : public ConversionPattern {
public:
  std::string target_op_type() const override { return "Sqrt"; }

  LogicalResult matchAndRewrite(const onnx::NodeProto &node,
                                Context &ctx) const override {
    auto operand = ctx.lookup(node.input(0));
    if (!operand.valid()) return LogicalResult::failure();
    ctx.value_map[node.output(0)] = ctx.builder.emit_sqrt(operand);
    return LogicalResult::success();
  }
};

class ReduceMeanOpConversion : public ConversionPattern {
public:
  std::string target_op_type() const override { return "ReduceMean"; }

  LogicalResult matchAndRewrite(const onnx::NodeProto &node,
                                Context &ctx) const override {
    auto input = ctx.lookup(node.input(0));
    if (!input.valid() || input.type.elem != shlo::ElemType::F32)
      return LogicalResult::failure();

    auto axes = get_ints_attr(node, "axes");
    if (axes.empty())
      for (int64_t i = 0; i < input.type.rank(); ++i) axes.push_back(i);
    for (auto &axis : axes) {
      if (axis < 0) axis += input.type.rank();
      if (axis < 0 || axis >= input.type.rank())
        return LogicalResult::failure();
    }

    int64_t reduced_count = 1;
    for (auto axis : axes) {
      int64_t dim = input.type.dims[axis];
      if (dim < 0) return LogicalResult::failure();
      reduced_count *= dim;
    }

    shlo::TensorType result = ctx.get_type(node.output(0));
    if (result.dims.empty() && input.type.rank() != static_cast<int64_t>(axes.size())) {
      result.elem = input.type.elem;
      bool keepdims = get_int_attr(node, "keepdims", 1) != 0;
      for (int64_t i = 0; i < input.type.rank(); ++i) {
        bool reduced = std::find(axes.begin(), axes.end(), i) != axes.end();
        if (reduced) {
          if (keepdims) result.dims.push_back(1);
        } else {
          result.dims.push_back(input.type.dims[i]);
        }
      }
    }

    shlo::TensorType scalar{{}, input.type.elem};
    auto zero = ctx.builder.emit_constant(scalar, "0.0");
    auto sum = ctx.builder.emit_reduce(input, zero, "stablehlo.add", axes,
                                       result);
    auto denom = ctx.builder.emit_constant(scalar,
                                           std::to_string(reduced_count) + ".0");
    auto denom_b = maybe_broadcast(denom, result, ctx);
    ctx.value_map[node.output(0)] = ctx.builder.emit_divide(sum, denom_b,
                                                            result);
    return LogicalResult::success();
  }
};

class MatMulOpConversion : public ConversionPattern {
public:
  std::string target_op_type() const override { return "MatMul"; }

  LogicalResult matchAndRewrite(const onnx::NodeProto &node,
                                Context &ctx) const override {
    auto lhs = ctx.lookup(node.input(0));
    auto rhs = ctx.lookup(node.input(1));
    if (!lhs.valid() || !rhs.valid()) return LogicalResult::failure();

    int64_t lc = lhs.type.rank() - 1;
    int64_t rc = std::max<int64_t>(0, rhs.type.rank() - 2);

    std::vector<int64_t> lb, rb;
    int lb_r = lhs.type.rank() - 2, rb_r = rhs.type.rank() - 2;
    if (lb_r > 0 && rb_r > 0) {
      int n = std::min(lb_r, rb_r);
      for (int i = 0; i < n; ++i) { lb.push_back(i); rb.push_back(i); }
    }

    shlo::TensorType res = ctx.get_type(node.output(0));
    if (res.dims.empty()) {
      res.elem = lhs.type.elem;
      for (auto b : lb) res.dims.push_back(lhs.type.dims[b]);
      if (lhs.type.rank() >= 2)
        res.dims.push_back(lhs.type.dims[lhs.type.rank() - 2]);
      if (rhs.type.rank() >= 2)
        res.dims.push_back(rhs.type.dims[rhs.type.rank() - 1]);
    }

    ctx.value_map[node.output(0)] =
        ctx.builder.emit_dot_general(lhs, rhs, lb, rb, {lc}, {rc}, res);
    return LogicalResult::success();
  }
};

class ConvOpConversion : public ConversionPattern {
public:
  std::string target_op_type() const override { return "Conv"; }
  int benefit() const override { return 10; } // higher priority

  LogicalResult matchAndRewrite(const onnx::NodeProto &node,
                                Context &ctx) const override {
    auto input = ctx.lookup(node.input(0));
    auto kernel = ctx.lookup(node.input(1));
    if (!input.valid() || !kernel.valid()) return LogicalResult::failure();

    int sp = static_cast<int>(input.type.rank()) - 2;
    if (sp < 1) return LogicalResult::failure();

    auto strides = get_ints_attr(node, "strides");
    auto pads_raw = get_ints_attr(node, "pads");
    auto dilations = get_ints_attr(node, "dilations");
    auto kernel_shape = get_ints_attr(node, "kernel_shape");
    auto auto_pad = get_str_attr(node, "auto_pad");
    int64_t group = get_int_attr(node, "group", 1);

    if (strides.empty()) strides.assign(sp, 1);
    if (dilations.empty()) dilations.assign(sp, 1);
    if (kernel_shape.empty())
      for (int i = 0; i < sp; ++i)
        kernel_shape.push_back(kernel.type.dims[2 + i]);

    // Padding
    std::vector<std::pair<int64_t, int64_t>> padding(sp, {0, 0});
    if (!auto_pad.empty() && auto_pad != "NOTSET" && auto_pad != "VALID") {
      for (int i = 0; i < sp; ++i) {
        int64_t in_d = input.type.dims[2 + i];
        int64_t k_eff = (kernel_shape[i] - 1) * dilations[i] + 1;
        int64_t out_d = (in_d + strides[i] - 1) / strides[i];
        int64_t total =
            std::max<int64_t>(0, (out_d - 1) * strides[i] + k_eff - in_d);
        if (auto_pad == "SAME_UPPER")
          padding[i] = {total / 2, total - total / 2};
        else
          padding[i] = {total - total / 2, total / 2};
      }
    } else if (!pads_raw.empty()) {
      for (int i = 0; i < sp; ++i)
        padding[i] = {pads_raw[i], pads_raw[i + sp]};
    }

    std::vector<int64_t> lhs_dilation(sp, 1);

    auto res = ctx.get_type(node.output(0));
    if (res.dims.empty()) {
      res.elem = input.type.elem;
      res.dims.push_back(input.type.dims[0]);
      res.dims.push_back(kernel.type.dims[0] * (group > 1 ? group : 1));
      for (int i = 0; i < sp; ++i) {
        int64_t in_d = input.type.dims[2 + i];
        int64_t k_eff = (kernel_shape[i] - 1) * dilations[i] + 1;
        if (in_d < 0)
          res.dims.push_back(-1);
        else
          res.dims.push_back(
              (in_d + padding[i].first + padding[i].second - k_eff) /
                  strides[i] +
              1);
      }
    }

    auto conv_out = ctx.builder.emit_convolution(
        input, kernel, strides, padding, lhs_dilation, dilations, group, 1, res);

    if (node.input_size() > 2 && !node.input(2).empty()) {
      auto bias = ctx.lookup(node.input(2));
      if (bias.valid()) {
        auto bc = ctx.builder.emit_broadcast_in_dim(bias, {1}, res);
        conv_out = ctx.builder.emit_add(conv_out, bc, res);
      }
    }

    ctx.value_map[node.output(0)] = conv_out;
    return LogicalResult::success();
  }
};

class SoftmaxOpConversion : public ConversionPattern {
public:
  std::string target_op_type() const override { return "Softmax"; }
  int benefit() const override { return 20; }

  LogicalResult matchAndRewrite(const onnx::NodeProto &node,
                                Context &ctx) const override {
    auto input = ctx.lookup(node.input(0));
    if (!input.valid()) return LogicalResult::failure();
    if (input.type.elem != shlo::ElemType::F32 || input.type.rank() < 1)
      return LogicalResult::failure();

    int64_t axis = get_int_attr(node, "axis", -1);
    if (axis < 0) axis += input.type.rank();
    if (axis < 0 || axis >= input.type.rank()) return LogicalResult::failure();

    shlo::TensorType reduced;
    reduced.elem = input.type.elem;
    for (int64_t i = 0; i < input.type.rank(); ++i)
      if (i != axis) reduced.dims.push_back(input.type.dims[i]);

    shlo::TensorType scalar{{}, input.type.elem};
    auto neg_inf = ctx.builder.emit_constant(scalar, "-3.402823e+38");
    auto zero = ctx.builder.emit_constant(scalar, "0.0");

    auto max = ctx.builder.emit_reduce(input, neg_inf, "stablehlo.maximum",
                                       {axis}, reduced);
    std::vector<int64_t> broadcast_dims;
    for (int64_t i = 0; i < input.type.rank(); ++i)
      if (i != axis) broadcast_dims.push_back(i);
    auto max_b = ctx.builder.emit_broadcast_in_dim(max, broadcast_dims,
                                                   input.type);

    auto shifted = ctx.builder.emit_subtract(input, max_b, input.type);
    auto exp = ctx.builder.emit_exponential(shifted);
    auto sum = ctx.builder.emit_reduce(exp, zero, "stablehlo.add", {axis},
                                       reduced);
    auto sum_b = ctx.builder.emit_broadcast_in_dim(sum, broadcast_dims,
                                                   input.type);

    ctx.value_map[node.output(0)] = ctx.builder.emit_divide(exp, sum_b,
                                                            input.type);
    return LogicalResult::success();
  }
};

class ReshapeOpConversion : public ConversionPattern {
public:
  std::string target_op_type() const override { return "Reshape"; }

  LogicalResult matchAndRewrite(const onnx::NodeProto &node,
                                Context &ctx) const override {
    auto operand = ctx.lookup(node.input(0));
    if (!operand.valid()) return LogicalResult::failure();

    shlo::TensorType res = ctx.get_type(node.output(0));
    if (res.dims.empty()) {
      auto *si = ctx.get_initializer(node.input(1));
      if (!si) return LogicalResult::failure();
      res.dims = extract_int64s(*si);
      res.elem = operand.type.elem;
      int64_t known = 1, neg = -1;
      for (size_t i = 0; i < res.dims.size(); ++i) {
        if (res.dims[i] == -1) neg = (int64_t)i;
        else if (res.dims[i] == 0 && (int)i < operand.type.rank())
          res.dims[i] = operand.type.dims[i];
        if (res.dims[i] > 0) known *= res.dims[i];
      }
      if (neg >= 0 && known > 0 && operand.type.num_elements() > 0)
        res.dims[neg] = operand.type.num_elements() / known;
    }

    ctx.value_map[node.output(0)] = ctx.builder.emit_reshape(operand, res);
    return LogicalResult::success();
  }
};

class TransposeOpConversion : public ConversionPattern {
public:
  std::string target_op_type() const override { return "Transpose"; }

  LogicalResult matchAndRewrite(const onnx::NodeProto &node,
                                Context &ctx) const override {
    auto operand = ctx.lookup(node.input(0));
    if (!operand.valid()) return LogicalResult::failure();

    auto perm = get_ints_attr(node, "perm");
    if (perm.empty())
      for (int64_t i = operand.type.rank() - 1; i >= 0; --i)
        perm.push_back(i);

    shlo::TensorType res;
    res.elem = operand.type.elem;
    for (auto p : perm) res.dims.push_back(operand.type.dims[p]);

    ctx.value_map[node.output(0)] =
        ctx.builder.emit_transpose(operand, perm, res);
    return LogicalResult::success();
  }
};

// ====================================================================
// Pipeline: register patterns → apply → verify
// ====================================================================

static void populate_onnx_to_stablehlo_patterns(RewritePatternSet &patterns) {
  patterns.add<AddOpConversion>();
  patterns.add<MulOpConversion>();
  patterns.add<SubOpConversion>();
  patterns.add<DivOpConversion>();
  patterns.add<PowOpConversion>();
  patterns.add<SqrtOpConversion>();
  patterns.add<ReduceMeanOpConversion>();
  patterns.add<MatMulOpConversion>();
  patterns.add<ConvOpConversion>();
  patterns.add<SoftmaxOpConversion>();
  patterns.add<ReshapeOpConversion>();
  patterns.add<TransposeOpConversion>();
}

static void setup_conversion_target(ConversionTarget &target) {
  // All StableHLO ops are legal
  target.addLegalDialect("stablehlo");
  target.addLegalOp("func.return");

  // All ONNX ops we know about are illegal (must be converted)
  target.addIllegalOp("Add");
  target.addIllegalOp("Mul");
  target.addIllegalOp("Sub");
  target.addIllegalOp("Div");
  target.addIllegalOp("Pow");
  target.addIllegalOp("Sqrt");
  target.addIllegalOp("ReduceMean");
  target.addIllegalOp("MatMul");
  target.addIllegalOp("Conv");
  target.addIllegalOp("Softmax");
  target.addIllegalOp("Reshape");
  target.addIllegalOp("Transpose");
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
      << " Level 3: ONNX -> StableHLO (Conversion Framework)\n"
      << "====================================================\n\n";
  std::cout << "Model : " << argv[1] << "\n";
  std::cout << "Graph : " << model.graph().name() << "\n";
  std::cout << "Nodes : " << model.graph().node_size() << "\n\n";

  // ---- Phase 1: Setup ----
  std::cout << "--- Phase 1: Setup conversion target & patterns ---\n";

  ConversionTarget target;
  setup_conversion_target(target);
  std::cout << "  Legal dialect  : stablehlo.*\n";
  std::cout << "  Illegal ops    : Add, Mul, Sub, Div, Pow, Sqrt, ReduceMean, "
               "MatMul, Conv, Softmax, Reshape, Transpose\n";

  RewritePatternSet patterns;
  populate_onnx_to_stablehlo_patterns(patterns);
  std::cout << "  Registered patterns: " << patterns.patterns().size() << "\n";

  // ---- Phase 2: Build context and emit constants ----
  std::cout << "\n--- Phase 2: Initialize context ---\n";

  Context ctx(model.graph());
  ctx.init();
  ctx.create_func_args();
  ctx.emit_initializers();
  std::cout << "  Function args   : " << ctx.func.args.size() << "\n";
  std::cout << "  Initializers    : " << ctx.initializer_names.size() << "\n";

  // ---- Phase 3: Apply full conversion ----
  std::cout << "\n--- Phase 3: Apply full conversion ---\n";

  ConversionStats stats;
  auto result = applyFullConversion(model.graph(), target, patterns, ctx, stats);

  std::cout << "  Total nodes     : " << stats.total << "\n";
  std::cout << "  Converted       : " << stats.converted << "\n";
  std::cout << "  Already legal   : " << stats.already_legal << "\n";
  std::cout << "  Failed          : " << stats.failed << "\n";
  if (!stats.failed_ops.empty()) {
    std::cout << "  Failed ops      :";
    for (auto &op : stats.failed_ops) std::cout << " " << op;
    std::cout << "\n";
  }

  ctx.finalize();

  // ---- Phase 4: Verify legality ----
  std::cout << "\n--- Phase 4: Verify legality ---\n";

  bool legal = verifyLegality(ctx.func, target);
  std::cout << "  Legality check  : " << (legal ? "PASSED" : "FAILED") << "\n";

  // ---- Phase 5: Output ----
  shlo::ModuleOp module;
  module.funcs.push_back(ctx.func);

  std::cout << "\n--- Generated StableHLO MLIR ---\n\n";
  module.print(std::cout);

  ctx.print_stats();

  std::cout << "\n--- Pipeline summary ---\n";
  std::cout << "  Conversion : " << (result.succeeded() ? "SUCCESS" : "PARTIAL")
            << "\n";
  std::cout << "  Legality   : " << (legal ? "ALL LEGAL" : "HAS ILLEGAL OPS")
            << "\n";

  return (result.succeeded() && legal) ? 0 : 1;
}
