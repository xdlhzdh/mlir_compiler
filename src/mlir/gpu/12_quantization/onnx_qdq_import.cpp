// onnx_qdq_import.cpp — ONNX QuantizeLinear/DequantizeLinear → quant_ir (P12).

#include "onnx_qdq_import.h"

#include "onnx-ml.pb.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace quant_ir {

namespace {

static bool load_model(const std::string &path, onnx::ModelProto &model) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  return model.ParseFromIstream(&in);
}

static DType onnx_elem_dtype(int t) {
  if (t == onnx::TensorProto::FLOAT) return DType::FP32;
  if (t == onnx::TensorProto::INT8) return DType::INT8;
  if (t == onnx::TensorProto::UINT8) return DType::UINT8;
  return DType::FP32;
}

static std::vector<int64_t> shape_from_tensor(const onnx::TensorProto &tp) {
  std::vector<int64_t> dims;
  for (auto d : tp.dims()) dims.push_back(d);
  return dims;
}

static std::vector<int64_t> shape_from_value(
    const onnx::TypeProto_Tensor &tt) {
  std::vector<int64_t> dims;
  if (!tt.has_shape()) return dims;
  for (auto &d : tt.shape().dim())
    dims.push_back(d.has_dim_value() ? d.dim_value() : -1);
  return dims;
}

static float scalar_float(const onnx::TensorProto *tp) {
  if (!tp) return 1.0f;
  if (!tp->raw_data().empty()) {
    float v = 0.0f;
    std::memcpy(&v, tp->raw_data().data(), sizeof(float));
    return v;
  }
  if (tp->float_data_size() > 0) return tp->float_data(0);
  return 1.0f;
}

static int scalar_int(const onnx::TensorProto *tp) {
  if (!tp) return 0;
  if (!tp->raw_data().empty()) {
    int8_t v = 0;
    std::memcpy(&v, tp->raw_data().data(), sizeof(int8_t));
    return static_cast<int>(v);
  }
  if (tp->int32_data_size() > 0) return tp->int32_data(0);
  if (tp->int64_data_size() > 0) return static_cast<int>(tp->int64_data(0));
  return 0;
}

static QuantParam qparam_from_qdq_inputs(
    const onnx::GraphProto &graph,
    const std::unordered_map<std::string, const onnx::TensorProto *> &inits,
    const std::string &scale_name, const std::string &zp_name,
    DType qdtype) {
  QuantParam qp;
  qp.scheme = QuantScheme::PER_TENSOR_AFFINE;
  qp.target_dtype = qdtype;
  auto sit = inits.find(scale_name);
  auto zit = inits.find(zp_name);
  qp.scales = {scalar_float(sit != inits.end() ? sit->second : nullptr)};
  qp.zero_points = {scalar_int(zit != inits.end() ? zit->second : nullptr)};
  return qp;
}

static TensorType tensor_for_value(
    const std::string &name, DType dtype,
    const std::unordered_map<std::string, TensorType> &types,
    const std::unordered_map<std::string, const onnx::TensorProto *> &inits) {
  auto tit = types.find(name);
  if (tit != types.end()) {
    TensorType t = tit->second;
    t.name = name;
    t.dtype = dtype;
    return t;
  }
  auto iit = inits.find(name);
  if (iit != inits.end()) {
    TensorType t;
    t.name = name;
    t.shape = shape_from_tensor(*iit->second);
    t.dtype = dtype;
    return t;
  }
  TensorType t;
  t.name = name;
  t.dtype = dtype;
  return t;
}

}  // namespace

Graph load_onnx_qdq(const std::string &path, QdqImportStats &stats,
                    std::ostream &log) {
  stats = {};
  onnx::ModelProto model;
  if (!load_model(path, model))
    throw std::runtime_error("failed to load ONNX model: " + path);

  const auto &g = model.graph();
  std::unordered_map<std::string, const onnx::TensorProto *> inits;
  std::unordered_set<std::string> init_names;
  std::unordered_map<std::string, TensorType> types;

  for (auto &init : g.initializer()) {
    inits[init.name()] = &init;
    init_names.insert(init.name());
    TensorType t;
    t.name = init.name();
    t.shape = shape_from_tensor(init);
    t.dtype = onnx_elem_dtype(init.data_type());
    types[init.name()] = t;
  }
  for (auto &vi : g.value_info()) {
    if (!vi.has_type() || !vi.type().has_tensor_type()) continue;
    TensorType t;
    t.name = vi.name();
    t.shape = shape_from_value(vi.type().tensor_type());
    t.dtype = onnx_elem_dtype(vi.type().tensor_type().elem_type());
    types[vi.name()] = t;
  }
  for (auto &inp : g.input()) {
    if (init_names.count(inp.name())) continue;
    if (!inp.has_type() || !inp.type().has_tensor_type()) continue;
    TensorType t;
    t.name = inp.name();
    t.shape = shape_from_value(inp.type().tensor_type());
    t.dtype = onnx_elem_dtype(inp.type().tensor_type().elem_type());
    types[inp.name()] = t;
  }
  for (auto &out : g.output()) {
    if (!out.has_type() || !out.type().has_tensor_type()) continue;
    TensorType t;
    t.name = out.name();
    t.shape = shape_from_value(out.type().tensor_type());
    t.dtype = onnx_elem_dtype(out.type().tensor_type().elem_type());
    types[out.name()] = t;
  }

  Graph graph;
  graph.name = g.name().empty() ? "onnx_qdq" : g.name();

  log << "  ONNX model: " << path << "\n";
  log << "  Graph nodes: " << g.node_size() << "\n\n";

  for (auto &node : g.node()) {
    const std::string &op = node.op_type();
    OpKind kind;
    DType out_dtype = DType::FP32;
    bool quantized = false;

    if (op == "QuantizeLinear") {
      kind = OpKind::QUANTIZE;
      out_dtype = DType::INT8;
      quantized = true;
      ++stats.quantize_linear;
    } else if (op == "DequantizeLinear") {
      kind = OpKind::DEQUANTIZE;
      out_dtype = DType::FP32;
      ++stats.dequantize_linear;
    } else if (op == "MatMul") {
      kind = OpKind::MATMUL;
      ++stats.matmul;
    } else {
      ++stats.other;
      log << "  [skip] unsupported op: " << op << " (" << node.name() << ")\n";
      continue;
    }

    auto *qop = graph.add_op(kind, node.name().empty() ? op : node.name());
    qop->is_quantized = quantized;

    for (int i = 0; i < node.input_size(); ++i) {
      if (node.input(i).empty()) continue;
      DType in_dtype = DType::FP32;
      if (kind == OpKind::QUANTIZE && i == 0) in_dtype = DType::FP32;
      if (kind == OpKind::DEQUANTIZE && i == 0) in_dtype = DType::INT8;
      qop->inputs.push_back(
          tensor_for_value(node.input(i), in_dtype, types, inits));
    }

    for (int i = 0; i < node.output_size(); ++i) {
      TensorType out = tensor_for_value(node.output(i), out_dtype, types, inits);
      if (kind == OpKind::QUANTIZE || kind == OpKind::DEQUANTIZE) {
        if (node.input_size() >= 2) {
          out.qparam = qparam_from_qdq_inputs(
              g, inits, node.input(1),
              node.input_size() > 2 ? node.input(2) : "", out_dtype);
          qop->output_qparam = out.qparam;
        }
      }
      qop->outputs.push_back(out);
      types[node.output(i)] = out;
    }

    log << "  mapped " << op << " → " << opkind_str(kind) << " ("
        << qop->name << ")\n";
  }

  log << "\n  Import summary:\n";
  log << "    QuantizeLinear:   " << stats.quantize_linear << "\n";
  log << "    DequantizeLinear: " << stats.dequantize_linear << "\n";
  log << "    MatMul:           " << stats.matmul << "\n";
  if (stats.other)
    log << "    skipped ops:      " << stats.other << "\n";

  return graph;
}

}  // namespace quant_ir
