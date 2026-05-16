// run_onnx_parse.cpp — P1 (1_onnx_parse): ONNX GraphProto 解析（L0 交换格式）
//
// 面试高频考点：用 Protobuf C++ API 读取 ONNX 模型，遍历
//   GraphProto / NodeProto / TensorProto / AttributeProto
// 并手动实现简易 shape inference 演示。
//
// 编译后运行：./run_onnx_parse <model.onnx>

#include "onnx-ml.pb.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

using ShapeMap = std::unordered_map<std::string, std::vector<int64_t>>;

// ============================================================
// 辅助：打印 shape
// ============================================================
static std::string shape_str(const std::vector<int64_t> &s) {
  std::string r = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) r += ", ";
    r += std::to_string(s[i]);
  }
  return r + "]";
}

// ============================================================
// 1. 加载 ModelProto
// ============================================================
static bool load_model(const std::string &path, onnx::ModelProto &model) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    std::cerr << "[ERROR] Cannot open " << path << "\n";
    return false;
  }
  if (!model.ParseFromIstream(&ifs)) {
    std::cerr << "[ERROR] Failed to parse ONNX model\n";
    return false;
  }
  return true;
}

// ============================================================
// 2. 遍历 GraphProto：inputs、outputs、initializers
// ============================================================
static void dump_graph_info(const onnx::GraphProto &g) {
  std::cout << "=== Graph: " << g.name() << " ===\n";

  std::cout << "\n[Inputs]\n";
  for (auto &inp : g.input()) {
    std::cout << "  " << inp.name();
    if (inp.has_type() && inp.type().has_tensor_type()) {
      auto &tt = inp.type().tensor_type();
      std::cout << "  elem_type=" << tt.elem_type();
      if (tt.has_shape()) {
        std::cout << "  shape=[";
        for (int i = 0; i < tt.shape().dim_size(); ++i) {
          if (i) std::cout << ",";
          auto &d = tt.shape().dim(i);
          if (d.has_dim_value())
            std::cout << d.dim_value();
          else
            std::cout << "?";
        }
        std::cout << "]";
      }
    }
    std::cout << "\n";
  }

  std::cout << "\n[Outputs]\n";
  for (auto &out : g.output()) {
    std::cout << "  " << out.name();
    if (out.has_type() && out.type().has_tensor_type()) {
      auto &tt = out.type().tensor_type();
      std::cout << "  elem_type=" << tt.elem_type();
      if (tt.has_shape()) {
        std::cout << "  shape=[";
        for (int i = 0; i < tt.shape().dim_size(); ++i) {
          if (i) std::cout << ",";
          auto &d = tt.shape().dim(i);
          if (d.has_dim_value())
            std::cout << d.dim_value();
          else
            std::cout << "?";
        }
        std::cout << "]";
      }
    }
    std::cout << "\n";
  }

  std::cout << "\n[Initializers] (" << g.initializer_size() << ")\n";
  for (auto &t : g.initializer()) {
    std::cout << "  " << t.name() << "  dtype=" << t.data_type() << "  shape=[";
    for (int i = 0; i < t.dims_size(); ++i) {
      if (i) std::cout << ",";
      std::cout << t.dims(i);
    }
    std::cout << "]  raw_bytes=" << t.raw_data().size() << "\n";
  }
}

// ============================================================
// 3. 遍历 NodeProto：op_type、inputs、outputs、attributes
// ============================================================
static void dump_nodes(const onnx::GraphProto &g) {
  std::cout << "\n[Nodes] (" << g.node_size() << ")\n";
  for (int ni = 0; ni < g.node_size(); ++ni) {
    auto &node = g.node(ni);
    std::cout << "  #" << ni << " op=" << node.op_type()
              << "  name=\"" << node.name() << "\"\n";
    std::cout << "      inputs : [";
    for (int i = 0; i < node.input_size(); ++i) {
      if (i) std::cout << ", ";
      std::cout << node.input(i);
    }
    std::cout << "]\n      outputs: [";
    for (int i = 0; i < node.output_size(); ++i) {
      if (i) std::cout << ", ";
      std::cout << node.output(i);
    }
    std::cout << "]\n";

    // 4. 遍历 AttributeProto
    if (node.attribute_size() > 0) {
      std::cout << "      attrs  :\n";
      for (auto &attr : node.attribute()) {
        std::cout << "        " << attr.name() << " (type="
                  << attr.type() << ") = ";
        switch (attr.type()) {
        case onnx::AttributeProto::FLOAT:
          std::cout << attr.f();
          break;
        case onnx::AttributeProto::INT:
          std::cout << attr.i();
          break;
        case onnx::AttributeProto::STRING:
          std::cout << "\"" << attr.s() << "\"";
          break;
        case onnx::AttributeProto::INTS:
          std::cout << "[";
          for (int i = 0; i < attr.ints_size(); ++i) {
            if (i) std::cout << ",";
            std::cout << attr.ints(i);
          }
          std::cout << "]";
          break;
        case onnx::AttributeProto::FLOATS:
          std::cout << "[";
          for (int i = 0; i < attr.floats_size(); ++i) {
            if (i) std::cout << ",";
            std::cout << attr.floats(i);
          }
          std::cout << "]";
          break;
        default:
          std::cout << "<type=" << attr.type() << ">";
        }
        std::cout << "\n";
      }
    }
  }
}

// ============================================================
// 5. 简易 Shape Inference（Add / MatMul / Conv / BN / Transpose）
// ============================================================
static std::vector<int64_t> get_shape_from_type(
    const onnx::TypeProto &tp) {
  std::vector<int64_t> s;
  if (tp.has_tensor_type() && tp.tensor_type().has_shape())
    for (auto &d : tp.tensor_type().shape().dim())
      s.push_back(d.has_dim_value() ? d.dim_value() : -1);
  return s;
}

static void simple_shape_inference(const onnx::GraphProto &g) {
  std::cout << "\n[Simple Shape Inference]\n";
  ShapeMap shapes;

  for (auto &inp : g.input())
    if (inp.has_type())
      shapes[inp.name()] = get_shape_from_type(inp.type());

  for (auto &init : g.initializer()) {
    std::vector<int64_t> s(init.dims().begin(), init.dims().end());
    shapes[init.name()] = s;
  }

  for (auto &vi : g.value_info())
    if (vi.has_type())
      shapes[vi.name()] = get_shape_from_type(vi.type());

  for (int ni = 0; ni < g.node_size(); ++ni) {
    auto &node = g.node(ni);
    std::string op = node.op_type();

    if (op == "Add" || op == "Mul" || op == "Sub" || op == "Div") {
      if (node.input_size() >= 2 && shapes.count(node.input(0))) {
        auto out_shape = shapes[node.input(0)];
        if (node.output_size() > 0) {
          shapes[node.output(0)] = out_shape;
          std::cout << "  " << op << " -> " << node.output(0)
                    << " shape=" << shape_str(out_shape) << "\n";
        }
      }
    } else if (op == "MatMul") {
      if (node.input_size() >= 2 && shapes.count(node.input(0)) &&
          shapes.count(node.input(1))) {
        auto &a = shapes[node.input(0)];
        auto &b = shapes[node.input(1)];
        if (a.size() >= 2 && b.size() >= 2) {
          auto out_shape = a;
          out_shape.back() = b.back();
          shapes[node.output(0)] = out_shape;
          std::cout << "  MatMul " << shape_str(a) << " x " << shape_str(b)
                    << " -> " << node.output(0) << " shape="
                    << shape_str(out_shape) << "\n";
        }
      }
    } else if (op == "Conv") {
      if (shapes.count(node.input(0)) && shapes.count(node.input(1))) {
        auto &x = shapes[node.input(0)];
        auto &w = shapes[node.input(1)];
        if (x.size() == 4 && w.size() == 4) {
          // NCHW: simple no-pad stride=1 inference
          std::vector<int64_t> ks = {w[2], w[3]};
          for (auto &attr : node.attribute())
            if (attr.name() == "kernel_shape" && attr.ints_size() >= 2)
              ks = {attr.ints(0), attr.ints(1)};
          auto out = std::vector<int64_t>{x[0], w[0], x[2] - ks[0] + 1,
                                          x[3] - ks[1] + 1};
          shapes[node.output(0)] = out;
          std::cout << "  Conv " << shape_str(x) << " * " << shape_str(w)
                    << " -> " << node.output(0) << " shape="
                    << shape_str(out) << "\n";
        }
      }
    } else if (op == "BatchNormalization") {
      if (shapes.count(node.input(0))) {
        shapes[node.output(0)] = shapes[node.input(0)];
        std::cout << "  BN -> " << node.output(0) << " shape="
                  << shape_str(shapes[node.input(0)]) << "\n";
      }
    } else if (op == "Transpose") {
      if (shapes.count(node.input(0))) {
        auto in_shape = shapes[node.input(0)];
        std::vector<int64_t> perm;
        for (auto &attr : node.attribute())
          if (attr.name() == "perm")
            perm.assign(attr.ints().begin(), attr.ints().end());
        if (perm.empty()) {
          perm.resize(in_shape.size());
          std::iota(perm.rbegin(), perm.rend(), 0);
        }
        std::vector<int64_t> out_shape(in_shape.size());
        for (size_t i = 0; i < perm.size(); ++i)
          out_shape[i] = in_shape[perm[i]];
        shapes[node.output(0)] = out_shape;
        std::cout << "  Transpose " << shape_str(in_shape) << " perm="
                  << shape_str(perm) << " -> " << node.output(0)
                  << " shape=" << shape_str(out_shape) << "\n";
      }
    } else {
      std::cout << "  (skip " << op << ")\n";
    }
  }
}

// ============================================================
int main(int argc, char **argv) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
    return 1;
  }

  onnx::ModelProto model;
  if (!load_model(argv[1], model)) return 1;

  std::cout << "ir_version=" << model.ir_version()
            << "  opset=";
  for (auto &op : model.opset_import())
    std::cout << op.domain() << ":" << op.version() << " ";
  std::cout << "\n";

  auto &g = model.graph();
  dump_graph_info(g);
  dump_nodes(g);
  simple_shape_inference(g);

  std::cout << "\n✓ P1 ONNX Parse complete.\n";
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
