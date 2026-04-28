// run_quantization.cpp — Quantization & Mixed-Precision 7-Stage Pipeline
//
// Transforms a floating-point graph to lower-precision inference:
//
//   Stage 0: Graph Setup         — build a small Conv+BN+ReLU+MatMul inference graph
//   Stage 1: Calibration         — collect min/max/histogram per tensor (simulated)
//   Stage 2: Scale Computation   — compute per-tensor/per-channel scale + zero_point
//   Stage 3: Operator Fusion     — fuse patterns for quantized execution (Conv+BN+ReLU → QConv)
//   Stage 4: Graph Rewrite       — insert quantize/dequantize ops, lower to int8 compute
//   Stage 5: Mixed Precision     — decide FP16 vs INT8 per operator (sensitivity analysis)
//   Stage 6: Summary & Speedup   — compare model size, estimated throughput
//
// Pure C++17, header-only IR, no external dependencies.

#include "quant_ir.h"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace quant_ir;

static void sep(const char *title) {
  std::cout << "\n  ──── " << title << " ────\n\n";
}

// =====================================================================
// Stage 0: Graph Setup
// =====================================================================
// Build: Input → Conv2d → BatchNorm → ReLU → Conv2d → MatMul → Output

static Graph stage0_setup() {
  sep("Stage 0: Graph Setup — Conv+BN+ReLU+Conv+MatMul");

  Graph g;
  g.name = "resnet_block";

  auto *conv1 = g.add_op(OpKind::CONV2D, "conv1");
  conv1->inputs.push_back({"input", {1, 3, 224, 224}, DType::FP32});
  conv1->inputs.push_back({"conv1.weight", {64, 3, 7, 7}, DType::FP32});
  conv1->inputs.push_back({"conv1.bias", {64}, DType::FP32});
  conv1->outputs.push_back({"conv1_out", {1, 64, 112, 112}, DType::FP32});

  auto *bn = g.add_op(OpKind::BATCH_NORM, "bn1");
  bn->inputs.push_back(conv1->outputs[0]);
  bn->inputs.push_back({"bn1.weight", {64}, DType::FP32});
  bn->inputs.push_back({"bn1.bias", {64}, DType::FP32});
  bn->inputs.push_back({"bn1.mean", {64}, DType::FP32});
  bn->inputs.push_back({"bn1.var", {64}, DType::FP32});
  bn->outputs.push_back({"bn1_out", {1, 64, 112, 112}, DType::FP32});

  auto *relu = g.add_op(OpKind::RELU, "relu1");
  relu->inputs.push_back(bn->outputs[0]);
  relu->outputs.push_back({"relu1_out", {1, 64, 112, 112}, DType::FP32});

  auto *conv2 = g.add_op(OpKind::CONV2D, "conv2");
  conv2->inputs.push_back(relu->outputs[0]);
  conv2->inputs.push_back({"conv2.weight", {128, 64, 3, 3}, DType::FP32});
  conv2->inputs.push_back({"conv2.bias", {128}, DType::FP32});
  conv2->outputs.push_back({"conv2_out", {1, 128, 56, 56}, DType::FP32});

  auto *mm = g.add_op(OpKind::MATMUL, "fc");
  mm->inputs.push_back({"fc_in", {1, 401408}, DType::FP32});
  mm->inputs.push_back({"fc.weight", {401408, 1000}, DType::FP32});
  mm->outputs.push_back({"fc_out", {1, 1000}, DType::FP32});

  g.print(std::cout);
  return g;
}

// =====================================================================
// Stage 1: Calibration
// =====================================================================
// Simulate collecting activation statistics by running representative data.

static void stage1_calibration(Graph &g) {
  sep("Stage 1: Calibration — Collect Activation Statistics");

  std::mt19937 rng(42);

  std::cout << "  Running 100 calibration batches (simulated)...\n\n";

  for (auto &op : g.ops) {
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float scale_factor = 1.0f;
    if (op->kind == OpKind::CONV2D) scale_factor = 2.5f;
    if (op->kind == OpKind::RELU) scale_factor = 1.5f;
    if (op->kind == OpKind::MATMUL) scale_factor = 5.0f;

    float mn = 0, mx = 0;
    for (int batch = 0; batch < 100; ++batch) {
      float val = dist(rng) * scale_factor;
      mn = std::min(mn, val);
      mx = std::max(mx, val);
    }
    if (op->kind == OpKind::RELU) mn = 0.0f;

    op->calib.observed_min = mn;
    op->calib.observed_max = mx;

    if (op->kind == OpKind::CONV2D && !op->outputs.empty()) {
      int channels = static_cast<int>(op->outputs[0].shape[1]);
      op->calib.per_channel_min.resize(channels);
      op->calib.per_channel_max.resize(channels);
      for (int c = 0; c < channels; ++c) {
        float ch_scale = 0.5f + (rng() % 100) / 50.0f;
        op->calib.per_channel_min[c] = mn * ch_scale;
        op->calib.per_channel_max[c] = mx * ch_scale;
      }
    }

    std::cout << "  " << std::left << std::setw(12) << op->name
              << " activation range: [" << std::fixed << std::setprecision(3)
              << op->calib.observed_min << ", " << op->calib.observed_max << "]\n";
  }

  std::cout << "\n  Calibration methods compared:\n";
  std::cout << "    MinMax:    use observed min/max directly (simple, may be sensitive to outliers)\n";
  std::cout << "    Percentile: clip to 99.99th percentile (robust to outliers)\n";
  std::cout << "    Entropy:   minimize KL-divergence between FP32 and quantized distribution\n";
  std::cout << "    MSE:       minimize mean squared error of quantization\n";
}

// =====================================================================
// Stage 2: Scale Computation
// =====================================================================
// Compute quantization parameters from calibration data.

static void stage2_scale_computation(Graph &g) {
  sep("Stage 2: Scale Computation — Derive scale & zero_point");

  std::cout << "  Quantization formula:\n";
  std::cout << "    Symmetric:  scale = max(|min|, |max|) / 127\n";
  std::cout << "                zp = 0, q(x) = clamp(round(x / scale), -128, 127)\n\n";
  std::cout << "    Affine:     scale = (max - min) / 255\n";
  std::cout << "                zp = round(-min / scale)\n";
  std::cout << "                q(x) = clamp(round(x / scale) + zp, 0, 255)\n\n";

  for (auto &op : g.ops) {
    float mn = op->calib.observed_min;
    float mx = op->calib.observed_max;

    QuantParam qp;
    qp.target_dtype = DType::INT8;

    if (op->kind == OpKind::RELU) {
      qp.scheme = QuantScheme::PER_TENSOR_AFFINE;
      qp.target_dtype = DType::UINT8;
      float scale = mx / 255.0f;
      qp.scales = {scale};
      qp.zero_points = {0};
    } else if (op->kind == OpKind::CONV2D &&
               !op->calib.per_channel_min.empty()) {
      qp.scheme = QuantScheme::PER_CHANNEL_SYMMETRIC;
      qp.channel_axis = 0;
      int ch = static_cast<int>(op->calib.per_channel_min.size());
      qp.scales.resize(ch);
      qp.zero_points.resize(ch, 0);
      for (int c = 0; c < ch; ++c) {
        float absmax = std::max(std::abs(op->calib.per_channel_min[c]),
                                std::abs(op->calib.per_channel_max[c]));
        qp.scales[c] = absmax / 127.0f;
        if (qp.scales[c] == 0.0f) qp.scales[c] = 1e-7f;
      }
    } else {
      qp.scheme = QuantScheme::PER_TENSOR_SYMMETRIC;
      float absmax = std::max(std::abs(mn), std::abs(mx));
      float scale = absmax / 127.0f;
      if (scale == 0.0f) scale = 1e-7f;
      qp.scales = {scale};
      qp.zero_points = {0};
    }

    op->output_qparam = qp;

    std::cout << "  " << op->name << ":\n";
    qp.print(std::cout);
  }
}

// =====================================================================
// Stage 3: Operator Fusion for Quantization
// =====================================================================

static void stage3_fusion(Graph &g) {
  sep("Stage 3: Quantization-Aware Fusion");

  std::cout << "  Common fusion patterns for INT8 inference:\n\n";

  std::cout << "  Pattern 1: Conv + BN + ReLU → QLinearConv (with fused BN + ReLU)\n";
  std::cout << "    Before: conv1 → bn1 → relu1\n";
  std::cout << "    After:  qlinear_conv1 (BN folded into weight/bias, ReLU clipped by uint8)\n\n";

  auto *conv1 = g.ops[0].get();
  auto *relu1 = g.ops[2].get();

  std::cout << "    BN folding math:\n";
  std::cout << "      w_folded = w * gamma / sqrt(var + eps)\n";
  std::cout << "      b_folded = gamma * (b - mean) / sqrt(var + eps) + beta\n\n";

  std::cout << "    ReLU fusion: post-conv activation range [0, max]\n";
  std::cout << "      → Use UINT8 (0-255) instead of INT8 for output\n";
  std::cout << "      → Zero point = 0, scale = max / 255\n\n";

  conv1->kind = OpKind::QLINEAR_CONV;
  conv1->name = "qconv1_bn_relu";
  conv1->is_quantized = true;
  conv1->outputs[0].dtype = DType::UINT8;
  conv1->output_qparam.target_dtype = DType::UINT8;
  conv1->output_qparam.scheme = QuantScheme::PER_TENSOR_AFFINE;

  std::cout << "  Pattern 2: MatMul → QLinearMatMul\n";
  std::cout << "    Before: fc (fp32 matmul)\n";
  std::cout << "    After:  qlinear_fc (int8 × int8 → int32 → rescale → int8)\n\n";

  auto *mm = g.ops[4].get();
  mm->kind = OpKind::QLINEAR_MATMUL;
  mm->name = "qlinear_fc";
  mm->is_quantized = true;

  std::cout << "    INT8 GEMM accumulation:\n";
  std::cout << "      C_i32[m,n] = sum_k( A_i8[m,k] * B_i8[k,n] )  // int32 accumulation\n";
  std::cout << "      C_i8[m,n] = clamp(round(C_i32[m,n] * s_a * s_b / s_c) + zp_c)\n";
  std::cout << "      where s_a, s_b, s_c are scale factors\n\n";

  std::cout << "  Post-fusion graph:\n";
  g.print(std::cout);
}

// =====================================================================
// Stage 4: Graph Rewrite — Insert Q/DQ ops
// =====================================================================

static void stage4_graph_rewrite(Graph &g) {
  sep("Stage 4: Graph Rewrite — Insert Quantize/Dequantize");

  std::cout << "  Two quantization styles:\n\n";

  std::cout << "  Style A: Q/DQ (Quantize-Dequantize) insertion — ONNX/TensorRT style\n";
  std::cout << "    fp32_input → [Quantize] → int8 → [DQ] → fp32 → [Conv] → fp32 → [Q] → int8\n";
  std::cout << "    Advantage: graph stays in FP32 semantics; backends fuse Q/DQ into operators\n\n";

  std::cout << "  Style B: Native INT8 graph — compiler-level lowering\n";
  std::cout << "    int8_input → [QLinearConv] → int8 → [QLinearMatMul] → int8\n";
  std::cout << "    Advantage: explicit int8 compute; no ambiguity\n\n";

  std::cout << "  We use Style B (QLinear operators with explicit scale/zp):\n\n";

  std::cout << "  Rewritten IR:\n";
  std::cout << "    // Input quantization\n";
  std::cout << "    %input_q = quantize(%input_f32, scale=0.0196, zp=0) : tensor<1x3x224x224xui8>\n\n";

  std::cout << "    // QLinearConv: int8 × int8 → int32 → rescale → uint8\n";
  std::cout << "    %conv1_out = qlinear_conv(\n";
  std::cout << "        %input_q,        x_scale=0.0196, x_zp=0,\n";
  std::cout << "        %weight_q,       w_scale=[per_channel], w_zp=0,\n";
  std::cout << "                         y_scale=0.0118, y_zp=0\n";
  std::cout << "    ) {fused_relu=true} : tensor<1x64x112x112xui8>\n\n";

  std::cout << "    // QLinearMatMul: int8 × int8 → int32 → rescale → int8\n";
  std::cout << "    %fc_out = qlinear_matmul(\n";
  std::cout << "        %fc_in_q,        a_scale=0.0098, a_zp=0,\n";
  std::cout << "        %weight_q,       b_scale=0.0392, b_zp=0,\n";
  std::cout << "                         y_scale=0.0196, y_zp=0\n";
  std::cout << "    ) : tensor<1x1000xi8>\n\n";

  std::cout << "    // Output dequantization\n";
  std::cout << "    %output = dequantize(%fc_out, scale=0.0196, zp=0) : tensor<1x1000xf32>\n";
}

// =====================================================================
// Stage 5: Mixed Precision — sensitivity analysis
// =====================================================================

static void stage5_mixed_precision(Graph &g) {
  sep("Stage 5: Mixed Precision — Sensitivity Analysis");

  std::cout << "  Approach: quantize one layer at a time, measure accuracy drop\n\n";

  struct Sensitivity {
    std::string name;
    float fp32_acc;
    float int8_acc;
    float fp16_acc;
    float drop_int8;
    float drop_fp16;
    std::string decision;
  };

  std::vector<Sensitivity> table = {
    {"conv1 (7×7)",      76.13f, 76.02f, 76.12f, 0.11f, 0.01f, "INT8"},
    {"bn1+relu1",        76.13f, 76.10f, 76.13f, 0.03f, 0.00f, "INT8 (fused)"},
    {"conv2 (3×3)",      76.13f, 75.89f, 76.11f, 0.24f, 0.02f, "INT8"},
    {"fc (classifier)",  76.13f, 75.21f, 76.09f, 0.92f, 0.04f, "FP16"},
  };

  std::cout << "  " << std::left
            << std::setw(20) << "Layer"
            << std::setw(10) << "FP32"
            << std::setw(10) << "INT8"
            << std::setw(10) << "FP16"
            << std::setw(12) << "Drop(INT8)"
            << std::setw(12) << "Drop(FP16)"
            << "Decision\n";
  std::cout << "  " << std::string(86, '-') << "\n";

  for (auto &s : table) {
    std::cout << "  " << std::left
              << std::setw(20) << s.name
              << std::setw(10) << std::fixed << std::setprecision(2) << s.fp32_acc
              << std::setw(10) << s.int8_acc
              << std::setw(10) << s.fp16_acc
              << std::setw(12) << s.drop_int8 << "%"
              << std::setw(11) << s.drop_fp16 << "%"
              << s.decision << "\n";
  }

  std::cout << "\n  Decision rule:\n";
  std::cout << "    if drop_int8 < 0.5%  → use INT8 (best speedup)\n";
  std::cout << "    elif drop_fp16 < 0.1% → use FP16 (good balance)\n";
  std::cout << "    else                  → keep FP32 (sensitive layer)\n\n";

  std::cout << "  Final mixed-precision plan:\n";
  std::cout << "    conv1:      INT8  (insensitive, large speedup)\n";
  std::cout << "    bn1+relu1:  INT8  (fused into conv1)\n";
  std::cout << "    conv2:      INT8  (moderate sensitivity, acceptable drop)\n";
  std::cout << "    fc:         FP16  (high sensitivity to int8, fp16 is safe)\n";
}

// =====================================================================
// Stage 6: Summary & Speedup Estimate
// =====================================================================

static void stage6_summary(const Graph &g) {
  sep("Stage 6: Summary — Model Size & Throughput Estimate");

  struct LayerStats {
    std::string name;
    int64_t param_count;
    int64_t fp32_bytes;
    DType quant_type;
    int64_t quant_bytes;
    float compute_gflops;
  };

  std::vector<LayerStats> stats = {
    {"conv1 (7×7, 3→64)",  64*3*7*7+64,          (64*3*7*7+64)*4L,
     DType::INT8,  64*3*7*7+64,               2*1*64*112*112*3*7*7 / 1e9f},
    {"conv2 (3×3, 64→128)", 128*64*3*3+128,       (128*64*3*3+128)*4L,
     DType::INT8,  128*64*3*3+128,             2*1*128*56*56*64*3*3 / 1e9f},
    {"fc (401408→1000)",   401408*1000+1000,       (401408*1000L+1000)*4L,
     DType::FP16,  (401408*1000L+1000)*2,      2*401408*1000 / 1e9f},
  };

  int64_t total_fp32 = 0, total_quant = 0;
  float total_gflops = 0;

  std::cout << "  " << std::left
            << std::setw(24) << "Layer"
            << std::setw(14) << "Params"
            << std::setw(14) << "FP32 size"
            << std::setw(8) << "QType"
            << std::setw(14) << "Quant size"
            << std::setw(10) << "Ratio"
            << "GFLOPs\n";
  std::cout << "  " << std::string(96, '-') << "\n";

  for (auto &s : stats) {
    total_fp32 += s.fp32_bytes;
    total_quant += s.quant_bytes;
    total_gflops += s.compute_gflops;
    float ratio = static_cast<float>(s.quant_bytes) / s.fp32_bytes;
    std::cout << "  " << std::left
              << std::setw(24) << s.name
              << std::setw(14) << s.param_count
              << std::setw(14) << (std::to_string(s.fp32_bytes / (1024*1024)) + " MB")
              << std::setw(8) << dtype_str(s.quant_type)
              << std::setw(14) << (std::to_string(s.quant_bytes / (1024*1024)) + " MB")
              << std::setw(10) << (std::to_string(static_cast<int>(ratio * 100)) + "%")
              << std::fixed << std::setprecision(2) << s.compute_gflops << "\n";
  }

  float overall_ratio = static_cast<float>(total_quant) / total_fp32;
  std::cout << "\n  Total model size: "
            << total_fp32 / (1024*1024) << " MB (FP32) → "
            << total_quant / (1024*1024) << " MB (mixed) = "
            << static_cast<int>(overall_ratio * 100) << "% of original\n";
  std::cout << "  Total compute: " << std::fixed << std::setprecision(2)
            << total_gflops << " GFLOPs\n\n";

  std::cout << "  Estimated throughput improvement (GPU, batch=1):\n";
  std::cout << "    INT8 Conv:    ~2-4× speedup (INT8 Tensor Cores)\n";
  std::cout << "    FP16 MatMul:  ~2× speedup (FP16 Tensor Cores)\n";
  std::cout << "    Memory bound: ~2-4× less bandwidth → faster for small batches\n\n";

  std::cout << "  Accuracy impact:\n";
  std::cout << "    FP32 baseline: 76.13% top-1\n";
  std::cout << "    Mixed INT8/FP16: ~75.9% top-1 (estimated, <0.3% drop)\n";
}

// =====================================================================
// main
// =====================================================================

int main() {
  std::cout << "========================================================\n";
  std::cout << " Stage 14: Quantization & Mixed-Precision Pipeline\n";
  std::cout << "========================================================\n";

  auto g = stage0_setup();
  stage1_calibration(g);
  stage2_scale_computation(g);
  stage3_fusion(g);
  stage4_graph_rewrite(g);
  stage5_mixed_precision(g);
  stage6_summary(g);

  std::cout << "\n========================================================\n";
  std::cout << " Quantization Pipeline Complete!\n";
  std::cout << "========================================================\n";
  return 0;
}
