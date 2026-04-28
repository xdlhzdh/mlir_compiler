// run_onnx_to_ir.cpp — P2: ONNX → mini IR lowering
//
// 面试高频考点：将 ONNX 节点逐一 lower 到内部 IR 表示
//   Add  → ir.add      (对应 stablehlo.add)
//   MatMul → ir.dot_general (对应 stablehlo.dot_general)
// 演示 lowering 框架：op dispatch + shape 传递 + 初始化器处理。
//
// 运行：./run_onnx_to_ir <add_matmul.onnx>

#include "mini_ir.h"
#include "onnx-ml.pb.h"
#include <fstream>

using namespace mini_ir;

// ============================================================
// ONNX TensorProto → mini_ir::TensorData
// ============================================================
static TensorData tensor_from_proto(const onnx::TensorProto &tp) {
  TensorData td;
  td.type.dtype = DType::F32;
  for (int i = 0; i < tp.dims_size(); ++i)
    td.type.shape.push_back(tp.dims(i));

  int64_t n = td.type.num_elements();
  td.float_data.resize(n, 0.0f);

  if (tp.raw_data().size() > 0) {
    const float *p = reinterpret_cast<const float *>(tp.raw_data().data());
    for (int64_t i = 0; i < n && i < (int64_t)(tp.raw_data().size() / 4); ++i)
      td.float_data[i] = p[i];
  } else if (tp.float_data_size() > 0) {
    for (int64_t i = 0; i < n && i < tp.float_data_size(); ++i)
      td.float_data[i] = tp.float_data(i);
  }
  return td;
}

static TensorType type_from_value_info(const onnx::ValueInfoProto &vi) {
  TensorType tt;
  if (vi.has_type() && vi.type().has_tensor_type()) {
    auto &t = vi.type().tensor_type();
    tt.dtype = (t.elem_type() == onnx::TensorProto::FLOAT) ? DType::F32
                                                            : DType::UNKNOWN;
    if (t.has_shape())
      for (auto &d : t.shape().dim())
        tt.shape.push_back(d.has_dim_value() ? d.dim_value() : -1);
  }
  return tt;
}

// ============================================================
// Shape inference helpers (for lowered ops)
// ============================================================
static TensorType infer_elementwise(const Graph &g, const Node &node) {
  if (g.value_types.count(node.inputs[0]))
    return g.value_types.at(node.inputs[0]);
  return {};
}

static TensorType infer_matmul(const Graph &g, const Node &node) {
  TensorType res;
  if (node.inputs.size() < 2) return res;
  auto it_a = g.value_types.find(node.inputs[0]);
  auto it_b = g.value_types.find(node.inputs[1]);
  if (it_a == g.value_types.end() || it_b == g.value_types.end()) return res;
  auto &a = it_a->second;
  auto &b = it_b->second;
  if (a.shape.size() < 2 || b.shape.size() < 2) return res;
  res.dtype = a.dtype;
  res.shape = a.shape;
  res.shape.back() = b.shape.back();
  return res;
}

// ============================================================
// Per-op lowering functions
// ============================================================
static std::shared_ptr<Node> lower_add(const onnx::NodeProto &on) {
  auto n = std::make_shared<Node>();
  n->name = on.name();
  n->op_type = "ir.add";
  for (int i = 0; i < on.input_size(); ++i)
    n->inputs.push_back(on.input(i));
  for (int i = 0; i < on.output_size(); ++i)
    n->outputs.push_back(on.output(i));
  return n;
}

static std::shared_ptr<Node> lower_mul(const onnx::NodeProto &on) {
  auto n = std::make_shared<Node>();
  n->name = on.name();
  n->op_type = "ir.multiply";
  for (int i = 0; i < on.input_size(); ++i)
    n->inputs.push_back(on.input(i));
  for (int i = 0; i < on.output_size(); ++i)
    n->outputs.push_back(on.output(i));
  return n;
}

static std::shared_ptr<Node> lower_matmul(const onnx::NodeProto &on) {
  auto n = std::make_shared<Node>();
  n->name = on.name();
  n->op_type = "ir.dot_general";
  for (int i = 0; i < on.input_size(); ++i)
    n->inputs.push_back(on.input(i));
  for (int i = 0; i < on.output_size(); ++i)
    n->outputs.push_back(on.output(i));
  // dot_general dimensions: lhs_contracting=[last], rhs_contracting=[0]
  n->attrs["lhs_contracting_dimensions"] = std::vector<int64_t>{-1};
  n->attrs["rhs_contracting_dimensions"] = std::vector<int64_t>{0};
  n->attrs["lhs_batching_dimensions"] = std::vector<int64_t>{};
  n->attrs["rhs_batching_dimensions"] = std::vector<int64_t>{};
  return n;
}

static std::shared_ptr<Node> lower_conv(const onnx::NodeProto &on) {
  auto n = std::make_shared<Node>();
  n->name = on.name();
  n->op_type = "ir.convolution";
  for (int i = 0; i < on.input_size(); ++i)
    n->inputs.push_back(on.input(i));
  for (int i = 0; i < on.output_size(); ++i)
    n->outputs.push_back(on.output(i));
  for (auto &attr : on.attribute()) {
    if (attr.name() == "kernel_shape")
      n->attrs["kernel_shape"] = std::vector<int64_t>(attr.ints().begin(), attr.ints().end());
    else if (attr.name() == "strides")
      n->attrs["strides"] = std::vector<int64_t>(attr.ints().begin(), attr.ints().end());
    else if (attr.name() == "pads")
      n->attrs["pads"] = std::vector<int64_t>(attr.ints().begin(), attr.ints().end());
  }
  return n;
}

static std::shared_ptr<Node> lower_bn(const onnx::NodeProto &on) {
  auto n = std::make_shared<Node>();
  n->name = on.name();
  n->op_type = "ir.batch_norm_inference";
  for (int i = 0; i < on.input_size(); ++i)
    n->inputs.push_back(on.input(i));
  for (int i = 0; i < on.output_size(); ++i)
    n->outputs.push_back(on.output(i));
  for (auto &attr : on.attribute())
    if (attr.name() == "epsilon")
      n->attrs["epsilon"] = attr.f();
  return n;
}

static std::shared_ptr<Node> lower_transpose(const onnx::NodeProto &on) {
  auto n = std::make_shared<Node>();
  n->name = on.name();
  n->op_type = "ir.transpose";
  for (int i = 0; i < on.input_size(); ++i)
    n->inputs.push_back(on.input(i));
  for (int i = 0; i < on.output_size(); ++i)
    n->outputs.push_back(on.output(i));
  for (auto &attr : on.attribute())
    if (attr.name() == "perm")
      n->attrs["permutation"] = std::vector<int64_t>(attr.ints().begin(), attr.ints().end());
  return n;
}

static std::shared_ptr<Node> lower_generic(const onnx::NodeProto &on) {
  auto n = std::make_shared<Node>();
  n->name = on.name();
  n->op_type = "ir." + on.op_type();
  for (int i = 0; i < on.input_size(); ++i)
    n->inputs.push_back(on.input(i));
  for (int i = 0; i < on.output_size(); ++i)
    n->outputs.push_back(on.output(i));
  return n;
}

// ============================================================
// Main lowering: ONNX GraphProto → mini_ir::Graph
// ============================================================
static Graph lower_onnx_to_ir(const onnx::GraphProto &og) {
  Graph g;
  g.name = og.name();

  // Initializers
  for (auto &init : og.initializer()) {
    g.initializers[init.name()] = tensor_from_proto(init);
    g.value_types[init.name()] = g.initializers[init.name()].type;
  }

  // Inputs (skip pure initializers that also appear in input)
  for (auto &inp : og.input()) {
    if (g.initializers.count(inp.name())) continue;
    g.input_names.push_back(inp.name());
    g.value_types[inp.name()] = type_from_value_info(inp);
  }

  // Value info (intermediate shapes from ONNX shape inference)
  for (auto &vi : og.value_info())
    g.value_types[vi.name()] = type_from_value_info(vi);

  // Outputs
  for (auto &out : og.output()) {
    g.output_names.push_back(out.name());
    if (out.has_type())
      g.value_types[out.name()] = type_from_value_info(out);
  }

  // Lower each node
  for (int i = 0; i < og.node_size(); ++i) {
    auto &on = og.node(i);
    std::string op = on.op_type();
    std::shared_ptr<Node> n;

    if (op == "Add")                   n = lower_add(on);
    else if (op == "Mul")              n = lower_mul(on);
    else if (op == "MatMul")           n = lower_matmul(on);
    else if (op == "Conv")             n = lower_conv(on);
    else if (op == "BatchNormalization") n = lower_bn(on);
    else if (op == "Transpose")        n = lower_transpose(on);
    else                               n = lower_generic(on);

    // Infer output type in IR
    if (op == "Add" || op == "Mul" || op == "Sub" || op == "Div")
      for (auto &o : n->outputs) g.value_types[o] = infer_elementwise(g, *n);
    else if (op == "MatMul")
      for (auto &o : n->outputs) g.value_types[o] = infer_matmul(g, *n);
    else if (op == "Conv" || op == "BatchNormalization" || op == "Transpose") {
      // Use already-populated value_types from ONNX shape inference
    }

    g.nodes.push_back(n);
  }

  return g;
}

// ============================================================
int main(int argc, char **argv) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
    return 1;
  }

  onnx::ModelProto model;
  {
    std::ifstream ifs(argv[1], std::ios::binary);
    if (!ifs || !model.ParseFromIstream(&ifs)) {
      std::cerr << "[ERROR] Cannot load " << argv[1] << "\n";
      return 1;
    }
  }

  std::cout << "=== ONNX → IR Lowering ===\n";
  Graph g = lower_onnx_to_ir(model.graph());

  std::cout << "\n--- Lowered IR ---\n";
  g.dump(std::cout);

  std::cout << "\n--- Type map ---\n";
  for (auto &[name, ty] : g.value_types)
    std::cout << "  %" << name << " : " << ty.str() << "\n";

  std::cout << "\n✓ P2 ONNX→IR Lowering complete.\n";
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
