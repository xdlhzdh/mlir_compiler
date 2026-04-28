// run_graph_rewrite.cpp — P3: Graph Rewrite
//
// 面试高频考点：在自定义图级 IR 上实现经典优化 pass
//   1. Conv + BN Fusion     (最经典的 graph-level fusion)
//   2. Transpose Elimination (自逆 transpose 对消除)
//   3. Constant Folding      (所有输入都是常量的 op，编译期求值)
//
// 运行：./run_graph_rewrite <conv_bn.onnx|transpose.onnx|const_fold.onnx>
//   或不带参数，使用内置测试图。

#include "mini_ir.h"
#include "onnx-ml.pb.h"
#include <algorithm>
#include <fstream>

using namespace mini_ir;

// ============================================================
// 从 ONNX 加载到 mini_ir (复用 P2 逻辑的精简版)
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

static Graph load_onnx_to_ir(const std::string &path) {
  Graph g;
  onnx::ModelProto model;
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs || !model.ParseFromIstream(&ifs)) {
    std::cerr << "[ERROR] Cannot load " << path << "\n";
    return g;
  }
  auto &og = model.graph();
  g.name = og.name();
  for (auto &init : og.initializer()) {
    g.initializers[init.name()] = tensor_from_proto(init);
    g.value_types[init.name()] = g.initializers[init.name()].type;
  }
  for (auto &inp : og.input()) {
    if (g.initializers.count(inp.name())) continue;
    g.input_names.push_back(inp.name());
    TensorType tt; tt.dtype = DType::F32;
    if (inp.has_type() && inp.type().has_tensor_type() &&
        inp.type().tensor_type().has_shape())
      for (auto &d : inp.type().tensor_type().shape().dim())
        tt.shape.push_back(d.has_dim_value() ? d.dim_value() : -1);
    g.value_types[inp.name()] = tt;
  }
  for (auto &out : og.output()) g.output_names.push_back(out.name());
  for (int i = 0; i < og.node_size(); ++i) {
    auto &on = og.node(i);
    auto n = std::make_shared<Node>();
    n->name = on.name();
    n->op_type = on.op_type();
    for (int j = 0; j < on.input_size(); ++j) n->inputs.push_back(on.input(j));
    for (int j = 0; j < on.output_size(); ++j) n->outputs.push_back(on.output(j));
    for (auto &attr : on.attribute()) {
      if (attr.type() == onnx::AttributeProto::FLOAT)
        n->attrs[attr.name()] = attr.f();
      else if (attr.type() == onnx::AttributeProto::INT)
        n->attrs[attr.name()] = attr.i();
      else if (attr.type() == onnx::AttributeProto::INTS)
        n->attrs[attr.name()] = std::vector<int64_t>(attr.ints().begin(), attr.ints().end());
    }
    g.nodes.push_back(n);
  }
  return g;
}

// ============================================================
// Pass 1: Conv + BN Fusion
// ============================================================
// BN(Conv(x, w, b), scale, bias, mean, var) →
//   Conv(x, w_new, b_new)
//   where w_new = w * (gamma / sqrt(var+eps))
//         b_new = gamma*(b - mean)/sqrt(var+eps) + beta
static int pass_conv_bn_fusion(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t ni = 0; ni < g.nodes.size(); ++ni) {
      auto &bn = g.nodes[ni];
      if (bn->op_type != "BatchNormalization") continue;
      if (bn->inputs.empty()) continue;

      auto conv = g.find_producer(bn->inputs[0]);
      if (!conv || conv->op_type != "Conv") continue;

      // All BN params must be initializers
      if (bn->inputs.size() < 5) continue;
      bool all_init = true;
      for (int k = 1; k <= 4; ++k)
        if (!g.is_initializer(bn->inputs[k])) { all_init = false; break; }
      if (!all_init) continue;

      // Conv weight must be initializer
      if (conv->inputs.size() < 2 || !g.is_initializer(conv->inputs[1])) continue;

      auto &scale_data = g.initializers[bn->inputs[1]];
      auto &bias_data  = g.initializers[bn->inputs[2]];
      auto &mean_data  = g.initializers[bn->inputs[3]];
      auto &var_data   = g.initializers[bn->inputs[4]];
      float eps = 1e-5f;
      if (bn->attrs.count("epsilon"))
        eps = std::get<float>(bn->attrs["epsilon"]);

      int64_t C = static_cast<int64_t>(scale_data.float_data.size());

      // Compute fused scale/bias per channel
      std::vector<float> A(C), B(C);
      for (int64_t c = 0; c < C; ++c) {
        float a = scale_data.float_data[c] /
                  std::sqrt(var_data.float_data[c] + eps);
        A[c] = a;
        B[c] = bias_data.float_data[c] - mean_data.float_data[c] * a;
      }

      // Fuse into conv weight (OC is dim 0 in ONNX)
      auto &w = g.initializers[conv->inputs[1]];
      int64_t total = static_cast<int64_t>(w.float_data.size());
      int64_t per_oc = total / C;
      for (int64_t c = 0; c < C; ++c)
        for (int64_t j = 0; j < per_oc; ++j)
          w.float_data[c * per_oc + j] *= A[c];

      // Fuse into conv bias (create if not present)
      std::string conv_bias_name;
      if (conv->inputs.size() >= 3 && !conv->inputs[2].empty() &&
          g.is_initializer(conv->inputs[2])) {
        conv_bias_name = conv->inputs[2];
        auto &cb = g.initializers[conv_bias_name];
        for (int64_t c = 0; c < C; ++c)
          cb.float_data[c] = cb.float_data[c] * A[c] + B[c];
      } else {
        conv_bias_name = conv->outputs[0] + "_fused_bias";
        TensorData cb;
        cb.type.shape = {C};
        cb.type.dtype = DType::F32;
        cb.float_data = B;
        g.initializers[conv_bias_name] = cb;
        g.value_types[conv_bias_name] = cb.type;
        if (conv->inputs.size() < 3)
          conv->inputs.push_back(conv_bias_name);
        else
          conv->inputs[2] = conv_bias_name;
      }

      // Rewire: conv outputs what BN used to output
      conv->outputs[0] = bn->outputs[0];
      g.erase_node(bn);

      std::cout << "  [Conv+BN Fusion] fused " << conv->name << " + " << bn->name << "\n";
      ++count;
      changed = true;
      break;
    }
  }
  return count;
}

// ============================================================
// Pass 2: Transpose Elimination
// ============================================================
// T(perm) ∘ T(perm) with same perm that is self-inverse → identity
// General: T(p2) ∘ T(p1) → T(compose(p2,p1)); if identity, remove both
static int pass_transpose_elimination(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t ni = 0; ni < g.nodes.size(); ++ni) {
      auto &t2 = g.nodes[ni];
      if (t2->op_type != "Transpose") continue;

      auto t1 = g.find_producer(t2->inputs[0]);
      if (!t1 || t1->op_type != "Transpose") continue;

      // Get perms
      auto get_perm = [](const Node &n) -> std::vector<int64_t> {
        auto it = n.attrs.find("perm");
        if (it == n.attrs.end()) it = n.attrs.find("permutation");
        if (it != n.attrs.end())
          return std::get<std::vector<int64_t>>(it->second);
        return {};
      };
      auto p1 = get_perm(*t1);
      auto p2 = get_perm(*t2);
      if (p1.empty() || p2.empty() || p1.size() != p2.size()) continue;

      // Compose: composed[i] = p1[p2[i]]
      size_t rank = p1.size();
      std::vector<int64_t> composed(rank);
      for (size_t i = 0; i < rank; ++i) composed[i] = p1[p2[i]];

      bool is_identity = true;
      for (size_t i = 0; i < rank; ++i)
        if (composed[i] != (int64_t)i) { is_identity = false; break; }

      if (is_identity) {
        // Rewire: all consumers of t2's output → use t1's input
        std::string old_out = t2->outputs[0];
        std::string new_in = t1->inputs[0];
        for (auto &node : g.nodes) {
          for (auto &inp : node->inputs)
            if (inp == old_out) inp = new_in;
        }
        for (auto &oname : g.output_names)
          if (oname == old_out) oname = new_in;

        g.erase_node(t2);
        g.erase_node(t1);
        std::cout << "  [Transpose Elim] removed " << t1->name << " + " << t2->name
                  << " (identity)\n";
        count += 2;
        changed = true;
        break;
      } else {
        // Merge into single transpose
        t1->attrs["perm"] = composed;
        t1->outputs[0] = t2->outputs[0];
        g.erase_node(t2);
        std::cout << "  [Transpose Merge] merged " << t1->name << " + " << t2->name << "\n";
        ++count;
        changed = true;
        break;
      }
    }
  }
  return count;
}

// ============================================================
// Pass 3: Constant Folding
// ============================================================
// If all inputs of a node are initializers, evaluate at compile time.
static int pass_constant_folding(Graph &g) {
  int count = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t ni = 0; ni < g.nodes.size(); ++ni) {
      auto &node = g.nodes[ni];
      bool all_const = true;
      for (auto &inp : node->inputs)
        if (!g.is_initializer(inp)) { all_const = false; break; }
      if (!all_const || node->inputs.empty()) continue;

      // Only fold element-wise ops we know
      if (node->op_type != "Add" && node->op_type != "Mul" &&
          node->op_type != "Sub" && node->op_type != "Div")
        continue;

      auto &a = g.initializers[node->inputs[0]];
      auto &b = g.initializers[node->inputs[1]];
      if (a.float_data.size() != b.float_data.size()) continue;

      TensorData result;
      result.type = a.type;
      result.float_data.resize(a.float_data.size());

      for (size_t i = 0; i < a.float_data.size(); ++i) {
        if (node->op_type == "Add")
          result.float_data[i] = a.float_data[i] + b.float_data[i];
        else if (node->op_type == "Mul")
          result.float_data[i] = a.float_data[i] * b.float_data[i];
        else if (node->op_type == "Sub")
          result.float_data[i] = a.float_data[i] - b.float_data[i];
        else if (node->op_type == "Div")
          result.float_data[i] = a.float_data[i] / b.float_data[i];
      }

      std::string out = node->outputs[0];
      g.initializers[out] = result;
      g.value_types[out] = result.type;
      g.erase_node(node);

      std::cout << "  [Const Fold] folded " << node->name << " → " << out
                << " = " << result.summary() << "\n";
      ++count;
      changed = true;
      break;
    }
  }
  return count;
}

// ============================================================
// Built-in test graphs (when no ONNX file given)
// ============================================================
static Graph make_builtin_conv_bn() {
  Graph g;
  g.name = "builtin_conv_bn";
  g.input_names = {"X"};
  g.output_names = {"Out"};

  g.value_types["X"] = TensorType{{1, 3, 8, 8}, DType::F32};

  auto make_init = [&](const std::string &name, std::vector<int64_t> shape,
                       float val) {
    TensorData td;
    td.type = {shape, DType::F32};
    int64_t n = 1;
    for (auto d : shape) n *= d;
    td.float_data.assign(n, val);
    g.initializers[name] = td;
    g.value_types[name] = td.type;
  };

  make_init("conv_w", {16, 3, 3, 3}, 0.1f);
  make_init("conv_b", {16}, 0.0f);
  make_init("bn_scale", {16}, 1.0f);
  make_init("bn_bias", {16}, 0.0f);
  make_init("bn_mean", {16}, 0.5f);
  make_init("bn_var", {16}, 1.0f);

  auto conv = std::make_shared<Node>();
  conv->name = "conv_0"; conv->op_type = "Conv";
  conv->inputs = {"X", "conv_w", "conv_b"};
  conv->outputs = {"conv_out"};
  conv->attrs["kernel_shape"] = std::vector<int64_t>{3, 3};
  g.nodes.push_back(conv);

  auto bn = std::make_shared<Node>();
  bn->name = "bn_0"; bn->op_type = "BatchNormalization";
  bn->inputs = {"conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"};
  bn->outputs = {"Out"};
  bn->attrs["epsilon"] = 1e-5f;
  g.nodes.push_back(bn);

  return g;
}

static Graph make_builtin_transpose() {
  Graph g;
  g.name = "builtin_transpose";
  g.input_names = {"X"};
  g.output_names = {"Out"};
  g.value_types["X"] = TensorType{{2, 3, 4}, DType::F32};

  auto t1 = std::make_shared<Node>();
  t1->name = "t1"; t1->op_type = "Transpose";
  t1->inputs = {"X"}; t1->outputs = {"t1_out"};
  t1->attrs["perm"] = std::vector<int64_t>{1, 0, 2};
  g.nodes.push_back(t1);

  auto t2 = std::make_shared<Node>();
  t2->name = "t2"; t2->op_type = "Transpose";
  t2->inputs = {"t1_out"}; t2->outputs = {"Out"};
  t2->attrs["perm"] = std::vector<int64_t>{1, 0, 2};
  g.nodes.push_back(t2);
  return g;
}

static Graph make_builtin_const_fold() {
  Graph g;
  g.name = "builtin_const_fold";
  g.input_names = {"X"};
  g.output_names = {"Out"};
  g.value_types["X"] = TensorType{{2, 3}, DType::F32};

  auto make_init = [&](const std::string &name, float val) {
    TensorData td;
    td.type = {{2, 3}, DType::F32};
    td.float_data.assign(6, val);
    g.initializers[name] = td;
    g.value_types[name] = td.type;
  };
  make_init("A", 2.0f);
  make_init("B", 3.0f);

  auto add = std::make_shared<Node>();
  add->name = "const_add"; add->op_type = "Add";
  add->inputs = {"A", "B"}; add->outputs = {"add_out"};
  g.nodes.push_back(add);

  auto mul = std::make_shared<Node>();
  mul->name = "mul_0"; mul->op_type = "Mul";
  mul->inputs = {"X", "add_out"}; mul->outputs = {"Out"};
  g.nodes.push_back(mul);
  return g;
}

// ============================================================
int main(int argc, char **argv) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  auto run_all_passes = [](Graph &g, const std::string &label) {
    std::cout << "\n=== " << label << " (before) ===\n";
    g.dump(std::cout);

    int c1 = pass_conv_bn_fusion(g);
    int c2 = pass_transpose_elimination(g);
    int c3 = pass_constant_folding(g);

    std::cout << "\n=== " << label << " (after) ===\n";
    g.dump(std::cout);
    std::cout << "  rewrites: conv_bn_fusion=" << c1
              << " transpose_elim=" << c2
              << " const_fold=" << c3 << "\n";
  };

  if (argc >= 2) {
    Graph g = load_onnx_to_ir(argv[1]);
    if (g.nodes.empty()) return 1;
    run_all_passes(g, std::string("ONNX: ") + argv[1]);
  } else {
    std::cout << "No ONNX file given; running built-in test graphs.\n";
    {
      Graph g = make_builtin_conv_bn();
      run_all_passes(g, "Built-in Conv+BN");
    }
    {
      Graph g = make_builtin_transpose();
      run_all_passes(g, "Built-in Transpose");
    }
    {
      Graph g = make_builtin_const_fold();
      run_all_passes(g, "Built-in Const Fold");
    }
  }

  std::cout << "\n✓ P3 Graph Rewrite complete.\n";
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
