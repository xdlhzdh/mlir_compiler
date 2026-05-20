# MLIR Compiler

本仓库包含两条主线：

1. **传统编译器实现**：基于 LLVM 的多版解释器（V1–V4）、ANTLR 解析、NFA/DFA、LLVM IR 生成与 Pass 优化
2. **AI 编译器全流程**：基于 MLIR 的 15 阶段 pipeline，从 ONNX 模型解析到 GPU 代码生成，覆盖工业级 AI 编译器核心技术栈

```
src/
├── mlir/
│   ├── cpu/    # CPU/RISC-V 手工 lowering 流水线（独立构建，见 cpu/README.md）
│   └── gpu/    # AI 编译器 15 阶段全流程（本仓库 CMake 构建）
├── pass/       # LLVM Pass 插件（SimplePass 等）
└── ast         # 解释器 V1-V4、ANTLR、NFA/DFA
```

---

## 构建

### 基础构建（纯 C++ 阶段，无外部依赖）

```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

这会构建 **P5（`7_stablehlo_opt`）及 P6–P12**（纯 C++17，无外部依赖）。未安装的可选依赖会在 cmake 阶段以 STATUS 提示自动跳过（如 `ANTLR4 not found`、`GTest not found`），不影响基础构建。

### 完整构建（含 ONNX 解析与 MLIR Pass）

| 可选依赖 | 启用的阶段 | 安装 / 配置方式 |
|---------|-----------|----------------|
| **ANTLR4 C++ runtime** | `run_interpreter_antlr`（ANTLR 解析器） | Ubuntu: `sudo apt install libantlr4-runtime-dev`；Fedora: `sudo dnf install antlr4-cpp-runtime-devel` |
| **GTest** | `tests/` 单元测试（V1–V4、Pass UT） | Ubuntu: `sudo apt install libgtest-dev`；Fedora: `sudo dnf install gtest-devel` |
| **Protobuf** | P1–P3（ONNX 解析/前端图优化）、P4（ONNX→StableHLO） | Ubuntu: `sudo apt install libprotobuf-dev protobuf-compiler`；Fedora: `sudo dnf install protobuf-devel` |
| **Python numpy + onnx** | P1–P3、P4 的测试模型生成脚本（`gen_test_models.py` 等） | `pip install --break-system-packages numpy onnx`（须系统级安装，不可仅装在 venv 中） |
| **MLIR + StableHLO** | P5（`6_stablehlo_passes`，真实 MLIR Pass 插件 + mlir-opt） | 见下方说明；环境安装步骤见 [`src/mlir/cpu/README.md`](src/mlir/cpu/README.md) §1.2–1.5 |
| **LLVM（`llvm-config`）** | `src/pass/` LLVM IR Pass 插件（SimplePass 等） | 安装 LLVM 后 `export PATH="$LLVM_INSTALL_PREFIX/bin:$PATH"`，使 `llvm-config`、`opt` 可用 |
| **Python + PyTorch + torch-mlir** | `conv_bn_optimized`（PyTorch → StableHLO 导出） | `pip install --break-system-packages torch`；torch-mlir 需源码编译，见 [`src/mlir/cpu/README.md`](src/mlir/cpu/README.md) §1.6 |

#### P5（`6_stablehlo_passes`）的 MLIR 配置

见 [`src/mlir/cpu/README.md`](src/mlir/cpu/README.md) §1.4（StableHLO 安装）、§1.5（让 `mlir_compiler` 找到 MLIR）。找不到 MLIR 时 P5 静默跳过。

---

## AI 编译器 Pipeline 总览（`src/mlir/gpu/`）

完整的 15 阶段 pipeline，从模型输入到机器码/GPU 代码输出：

分层命名以 [`src/mlir/README.md`](src/mlir/README.md) 为准：

| 层级 | 名称 | 代表 IR / 对应 Px |
|------|------|-------------------|
| **入口** | 原始 / 交换图 | ONNX `GraphProto`、原始 TorchScript；P1 读取 |
| **L1** | Frontend Graph Layer（前端图层） | ONNX / Torch 前端图、`mini_ir`；P1–P3 |
| **L2** | Tensor Operator Layer / High-Level Math Layer（张量算子层 / 高级数学层） | StableHLO；辅助 P5 的 `4_`、P4、P5 |
| **L3** | Structured Op & Memory Layer（结构化算子与内存层） | Linalg on tensor → OSB → Linalg on memref；P6–P7 |
| **L4** | Kernel Loop & Vector Layer（内核循环与矢量层） | SCF / Affine / Vector → LLVM / GPU codegen；P8–P12 |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  P1–P3: 入口 + L1 前端图 (需 Protobuf)                                        │
│  ONNX 解析 → 自定义 mini IR Lowering → 图级优化 (Conv+BN Fusion, 常量折叠)      │
├─────────────────────────────────────────────────────────────────────────────┤
│  辅助P5(`4_`) / P4 / P5: L2 StableHLO 高级张量算子                             │
│  PyTorch→StableHLO (Python) → ONNX→StableHLO (tier 1/2/3) → MLIR Pass 插件 │
│  → StableHLO 6-Step 优化                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  P6–P7: L3 Linalg 结构化算子与内存                                             │
│  Linalg Tiling & Fusion → One-Shot Bufferization                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  P8–P9: L4 Kernel Loop / Vector / LLVM 后端 (纯 C++)                          │
│  SCF/Affine 循环优化 → Vector 向量化 → LLVM Backend (ISel/RegAlloc/Sched)     │
├─────────────────────────────────────────────────────────────────────────────┤
│  P10–P12: L4 GPU 映射 & 部署优化 (纯 C++)                                      │
│  GPU Codegen (NVVM/PTX) → 量化 & 混合精度 → 内存规划 & Buffer 复用             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 阶段目录与依赖

| 目录 | Px | 说明 | 依赖 | make target |
|------|-----|------|------|-------------|
| `common/` | — | ONNX Protobuf 绑定、mini IR 定义、测试模型生成 | Protobuf | — |
| `1_onnx_parse/` | **P1** | ONNX 模型解析：遍历 GraphProto/NodeProto/TensorProto + Shape 推断 | Protobuf | `run_graph` |
| `2_onnx_to_ir/` | **P2** | ONNX → 自定义 IR Lowering：Add→ir.add, MatMul→ir.dot_general, Conv→ir.convolution | Protobuf | `run_graph` |
| `3_graph_optimize/` | **P3** | 图级优化：Conv+BN Fusion、Transpose 消除、常量折叠 | Protobuf | `run_graph` |
| `4_torch_to_stablehlo/` | — | PyTorch → StableHLO 导出（conv_bn_model.py，为 `6_` 提供输入 MLIR） | torch + torch-mlir | `conv_bn_optimized` |
| `5_onnx_to_stablehlo/` | **P4** | ONNX → StableHLO 三级 Lowering（tier 1 基础 / tier 2 broadcast+dynamic / tier 3 conversion framework；整体目标是 **L2 StableHLO**） | Protobuf | `run_lowering` |
| `6_stablehlo_passes/` | **P5** | 真实 MLIR Pass 插件：StableHLO Conv+BN Fusion（DialectPass .so + mlir-opt 加载） | MLIR | `conv_bn_optimized` |
| `7_stablehlo_opt/` | **P5** | StableHLO 6-Step 优化：Canon→Shape→Graph(CSE/DCE/Fusion)→Layout→Clean→Legal(→Linalg) | **无** | `run_shlo_opt` |
| `8_linalg_opt/` | **P6** | Linalg 7-Step Fusion：依赖分析→候选构建→代价模型→Tile-aware→融合重写→清理 | **无** | `run_linalg` |
| `9_bufferize/` | **P7** | One-Shot Bufferization 8-Step：别名分析→In-place→WAR/RAW 冲突→Copy 插入→Dealloc | **无** | `run_buf` |
| `10_scf_affine/` | **P8** | SCF/Affine Loop 9-Step：Linalg→Loop→Tiling→Interchange→Fusion→Parallel→向量化 | **无** | `run_scf` |
| `11_vector/` | **P8** | Vector Dialect 6-Step：前置检查→Core(transfer_read/write/fma)→Register Blocking→LLVM | **无** | `run_vec` |
| `12_llvm_lowering/` | **P9** | LLVM Backend 7-Step：Vector Lowering→LLVM IR→ISel→RegAlloc→Scheduling→Machine Code | **无** | `run_llvm_lower` |
| `13_gpu_codegen/` | **P10** | GPU Codegen 7-Step：并行检测→Thread Mapping→GPU Dialect→Shared Mem→NVVM→PTX→Occupancy | **无** | `run_gpu` |
| `14_quantization/` | **P11** | 量化 7-Step：校准统计→Scale 计算→量化融合→QLinear 重写→混合精度分析→Speedup 估算 | **无** | `run_quant` |
| `15_memory_planning/` | **P12** | 内存规划 7-Step：Liveness→干涉图→Offset 分配→Buffer 复用→In-place 优化→峰值分析 | **无** | `run_memplan` |

---

## 运行命令参考

所有命令在 **build 目录**下执行（或从项目根目录使用 `cmake --build build --target <target>`）。

### AI 编译器各阶段运行

#### Px 与文件夹对照

| Px | 文件夹 | make target |
|----|--------|-------------|
| P1 | `1_onnx_parse/` | `run_graph` |
| P2 | `2_onnx_to_ir/` | `run_graph` |
| P3 | `3_graph_optimize/` | `run_graph` |
| — | `4_torch_to_stablehlo/` | `conv_bn_optimized`（Python 导出） |
| P4 | `5_onnx_to_stablehlo/` | `run_lowering` |
| P5 | `6_stablehlo_passes/` | `conv_bn_optimized`（MLIR Pass） |
| P5 | `7_stablehlo_opt/` | `run_shlo_opt` |
| P6 | `8_linalg_opt/` | `run_linalg` |
| P7 | `9_bufferize/` | `run_buf` |
| P8 | `10_scf_affine/` | `run_scf` |
| P8 | `11_vector/` | `run_vec` |
| P9 | `12_llvm_lowering/` | `run_llvm_lower` |
| P10 | `13_gpu_codegen/` | `run_gpu` |
| P11 | `14_quantization/` | `run_quant` |
| P12 | `15_memory_planning/` | `run_memplan` |

> `conv_bn_optimized` 跨两个目录：`4_` 的 Python 脚本生成 MLIR，`6_` 的 C++ Pass 插件做 Conv+BN Fusion。

#### 快速入门：逐阶段运行

```bash
cd build

# ── 前端（需 Protobuf + numpy + onnx）──
make run_graph              # P1+P2+P3: ONNX 解析 → IR Lowering → 图优化
make run_lowering           # P4: ONNX → StableHLO 三级 Lowering (tier 1+2+3，目标 L2)
make conv_bn_optimized      # P5: PyTorch 导出 → mlir-opt Conv+BN Fusion（需 torch + torch-mlir）

# ── L2 StableHLO 优化（纯 C++，无需任何依赖）──
make run_shlo_opt           # P5: StableHLO 6-Step 优化 pipeline

# ── L3 Linalg 结构化算子（纯 C++）──
make run_linalg             # P6: L3 Linalg Tensor 级 Tiling & Fusion pipeline
make run_buf                # P7: One-Shot Bufferization pipeline

# ── L4 后端（纯 C++）──
make run_scf                # P8: SCF/Affine Loop 级优化 pipeline
make run_vec                # P8: Vector Dialect 向量化 pipeline
make run_llvm_lower         # P9: Vector → LLVM Backend (ISel/RegAlloc/Sched/Emit)

# ── L4 GPU 映射 & 部署优化（纯 C++）──
make run_gpu                # P10: GPU Code Generation (→ NVVM → PTX → Occupancy)
make run_quant              # P11: Quantization & 混合精度 pipeline
make run_memplan            # P12: Memory Planning & Buffer 复用 pipeline
```

#### 通过 DOMAIN/PASS 参数运行

`make run DOMAIN=mlir` 系统支持通过 `PASS` 参数选择特定阶段：

```bash
# 运行全部 MLIR 阶段（P1–P12 所有已构建的 target）
make run_mlir
make run DOMAIN=mlir

# 通过 PASS 参数指定单个阶段
make run DOMAIN=mlir PASS=graph           # P1+P2+P3
make run DOMAIN=mlir PASS=lowering        # P4
make run DOMAIN=mlir PASS=conv_bn_fusion  # P5
make run DOMAIN=mlir PASS=shlo_opt        # P5
make run DOMAIN=mlir PASS=linalg          # P6
make run DOMAIN=mlir PASS=bufferize       # P7
make run DOMAIN=mlir PASS=scf_affine      # P8
make run DOMAIN=mlir PASS=vector          # P8
make run DOMAIN=mlir PASS=llvm_lower      # P9
make run DOMAIN=mlir PASS=gpu_codegen     # P10
make run DOMAIN=mlir PASS=quantization    # P11
make run DOMAIN=mlir PASS=memplan         # P12
```

#### 完整 make target 与 PASS 参数对照表

| make target | 等价 DOMAIN/PASS | Px | 说明 |
|-------------|-----------------|-----|------|
| `make run_graph` | `DOMAIN=mlir PASS=graph` | P1–P3 | 生成 ONNX 测试模型 → 解析 → IR Lowering → 图优化（需 Protobuf + numpy + onnx） |
| `make run_lowering` | `DOMAIN=mlir PASS=lowering` | P4 | ONNX→StableHLO tier 1(基础) + tier 2(broadcast/dynamic) + tier 3(framework)，产物属 L2 StableHLO |
| `make conv_bn_optimized` | `DOMAIN=mlir PASS=conv_bn_fusion` | P5 | `4_` Python 导出 MLIR + `6_` mlir-opt Conv+BN Fusion（需 torch + torch-mlir） |
| `make run_shlo_opt` | `DOMAIN=mlir PASS=shlo_opt` | P5 | StableHLO 图: 24 ops → Canon/Shape/CSE/DCE/Fusion/Layout/Legal → 7 ops |
| `make run_linalg` | `DOMAIN=mlir PASS=linalg` | P6 | GEMM+Bias+ReLU: 11 ops → 依赖分析/代价模型/Tile-and-Fuse → 5 ops |
| `make run_buf` | `DOMAIN=mlir PASS=bufferize` | P7 | tensor→memref: 7×8KB=57KB → 别名/In-place → 2×8KB=16KB(71%↓) |
| `make run_scf` | `DOMAIN=mlir PASS=scf_affine` | P8 | GEMM+Bias+ReLU: Linalg→scf.for→Tiling→Interchange→Fusion→Parallel→Vec |
| `make run_vec` | `DOMAIN=mlir PASS=vector` | P8 | 标量循环→vector.transfer_read/write/fma, 4-row register blocking, tail masking |
| `make run_llvm_lower` | `DOMAIN=mlir PASS=llvm_lower` | P9 | Vector→LLVM IR→ISel(vfmadd231ps)→RegAlloc(12/16 YMM, 0 spill)→Sched→MachineCode |
| `make run_gpu` | `DOMAIN=mlir PASS=gpu_codegen` | P10 | GEMM: 并行检测→GPU grid/block 映射→Shared mem tiling→NVVM→PTX→Occupancy 分析 |
| `make run_quant` | `DOMAIN=mlir PASS=quantization` | P11 | ResNet block: 校准→Scale→Conv+BN+ReLU→QLinearConv→INT8/FP16 混合精度 |
| `make run_memplan` | `DOMAIN=mlir PASS=memplan` | P12 | ResNet block: Liveness→干涉图→Offset 分配→Buffer 复用→In-place→峰值分析 |
| `make run_mlir` | `DOMAIN=mlir` | P1–P12 | 依次运行所有 MLIR target；缺少可选依赖的 target 自动跳过 |

#### 从项目根目录运行

```bash
# 使用 cmake --build
cmake --build build --target run_shlo_opt       # 运行 P5
cmake --build build --target run_gpu             # 运行 P10

# 传递 DOMAIN/PASS（注意 -- 后传参数）
cmake --build build --target run -- DOMAIN=mlir PASS=linalg
cmake --build build --target run -- DOMAIN=mlir PASS=gpu_codegen
```

---

## 各阶段详细说明

### P1–P3: ONNX 前端 (`1_onnx_parse/` → `3_graph_optimize/`)

**运行：** `make run_graph`

使用 Protobuf C++ API 解析 ONNX 模型，构建自定义 mini IR，执行图级优化。
测试模型由 `common/gen_test_models.py` 生成（add_matmul, conv_bn, transpose, const_fold）。

Pipeline: `gen_test_models.py → run_onnx_parse → run_onnx_to_ir → run_graph_rewrite`

### P4: ONNX → StableHLO 三级 Lowering (`5_onnx_to_stablehlo/`)

**运行：** `make run_lowering`

这里的 `level1/2/3` 是 **P4 内部难度分级（tier 1/2/3）**，不是本文统一分层里的 L1/L2/L3；三者的共同目标都是生成 **L2 StableHLO**。

| Tier | 可执行文件 | 覆盖内容 |
|------|-----------|---------|
| **tier 1** | `run_lowering_l1` | Add/MatMul/Conv/Reshape/Transpose → stablehlo 对应 op |
| **tier 2** | `run_lowering_l2` | broadcast → `broadcast_in_dim`；dynamic shape；完整 Conv 属性映射；错误处理 |
| **tier 3** | `run_lowering_l3` | `ConversionPattern` + `matchAndRewrite`；`ConversionTarget`；`applyFullConversion` |

### P5（`6_stablehlo_passes/`）: StableHLO MLIR Pass 插件

**运行：** `make conv_bn_optimized`

唯一使用真实 MLIR C++ API 的阶段。编译为 `DialectPass.so`，通过 `mlir-opt` 的 pass/dialect plugin 机制加载：

```bash
mlir-opt --load-pass-plugin=./DialectPass.so \
         --load-dialect-plugin=./DialectPass.so \
         --pass-pipeline='builtin.module(func.func(conv-bn-fusion))' \
         conv_bn_model.mlir -o conv_bn_fusion.mlir
```

### P5（`7_stablehlo_opt/`）: StableHLO 6-Step 优化

**运行：** `make run_shlo_opt`

| Step | Pass | 关键操作 |
|------|------|---------|
| 1 | Canonicalization | `x+0→x`, `x*1→x`, identity reshape/transpose 消除 |
| 2 | Shape Optimization | shape inference, `get_dimension_size→const`, dynamic→static |
| 3 | Graph Optimization | CSE, DCE, 常量折叠, **Conv+BN Fusion**, elementwise chain 融合检测 |
| 4 | Layout/Transpose | `T(p2)∘T(p1)→T(compose)`, transpose push-through |
| 5 | Cleanup | 再跑 canonicalize + DCE |
| 6 | Legalization | `stablehlo.*` → `linalg.*` / `arith.*` / `tensor.*` |

综合测试：24 ops → 7 ops（所有 pass 同时作用）

### P6: L3 Linalg Tensor 级 Fusion (`8_linalg_opt/`)

**运行：** `make run_linalg`

| Step | Pass | 关键概念 |
|------|------|---------|
| 0 | Pre-clean | canonicalize + CSE + DCE |
| 1 | Dependence Analysis | SSA use-def 图 + alias check |
| 2 | Fusion Candidate Build | producer-consumer 图 + elementwise chain |
| 3 | Cost Model Filtering | 中间 tensor 消除收益 vs 多用户约束 |
| 4 | Tile-aware Fusion | iterator 兼容性：elem+elem=直接, reduction+elem=tile-and-fuse |
| 5 | Fusion Rewrite | merge indexing maps + merge iterators + tiling |
| 6 | Post-clean | canonicalize + CSE + DCE |

测试：`matmul → bias_add → relu → square → residual → scaling`，11 ops → 5 ops

### P7: One-Shot Bufferization (`9_bufferize/`)

**运行：** `make run_buf`

| Step | Pass | 关键概念 |
|------|------|---------|
| 0 | Pre-clean | CSE + DCE |
| 1 | Alias Analysis | SSA use-def 链 + 潜在 alias group |
| 2 | In-place Analysis | WAR(Write-After-Read) 冲突检测 |
| 3 | Bufferization Decision | INPLACE / OUT_OF_PLACE 标记 |
| 4 | Rewrite | tensor → memref, buffer assignment |
| 5 | Copy Insertion | 冲突驱动的 memref.copy |
| 6 | Buffer Deallocation | 所有权模型、lifetime 分析 |
| 7 | Post-clean | 验证 allocs == deallocs（无泄漏） |

核心演示：`%biased` 有两个 reader（relu 和 neg），relu in-place 覆写 → neg 需要 copy。7×8KB=57KB → 2×8KB=16KB（**71% 内存减少**）

### P8（`10_scf_affine/`）: SCF/Affine Loop 级优化

**运行：** `make run_scf`

| Step | Pass | 关键操作 |
|------|------|---------|
| 0 | Pre-clean | 死代码消除 |
| 1 | Linalg → Loop | matmul → 3 层 scf.for + memref.load/store |
| 2 | Loop Canonicalization | 死循环消除 |
| 3 | Tiling | 多级 cache-aware: [TM=32, TN=8, TK=32], 工作集 ≤ L1 32KB |
| 4 | Loop Transform | interchange (stride-1 访问) + loop fusion (bias+relu) |
| 5 | Parallelization | scf.parallel + thread mapping |
| 6 | Memory Optimization | B tile stack promotion + C register promotion |
| 7 | Vectorization Prep | `vector<8xf32>` (broadcast + fma) |
| 8 | Cleanup | 统计: loops/ops/parallel/vector |

测试：GEMM+Bias+ReLU `A[64×128]×B[128×32]`，8 loops/13 ops → 7 loops/14 ops（2 parallel, 8 vector, 8× SIMD 吞吐）

### P8（`11_vector/`）: Vector Dialect Pipeline

**运行：** `make run_vec`

| Step | Pass | 关键概念 |
|------|------|---------|
| 0 | Precondition Check | canonical loop, stride 分析 |
| 1 | Vectorization Prep | alignment 32-byte, trip count = SIMD 倍数 |
| 2 | Vectorization Core | scf.for → vector.transfer_read/write/fma/broadcast |
| 3 | Vector Shaping | register blocking (unroll ii×4) + tail masking |
| 4 | Vector Optimization | load coalescing, small-op fusion, dead elim |
| 5 | Lowering to LLVM | vector → llvm.intr.fmuladd (AVX2: vfmadd231ps) |

核心：4 row × `vector<8xf32>` register blocking → 峰值 16 FLOP/cycle

### P9: LLVM Backend Pipeline (`12_llvm_lowering/`)

**运行：** `make run_llvm_lower`

| Step | Pass | 关键操作 |
|------|------|---------|
| 0 | Vector IR Input | 接收 P8（`11_vector/`）输出（4-row register-blocked FMA chain） |
| 1 | Vector Lowering | memref → GEP, transfer_read → load, fma → @llvm.fma.v8f32 |
| 2 | LLVM IR Generation | PHI nodes, basic block CFG, SSA |
| 3 | Instruction Selection | DAG pattern match: fma→vfmadd231ps, GEP+load 融合 |
| 4 | Register Allocation | 12/16 YMM used, **0 spill**, linear scan |
| 5 | Instruction Scheduling | port model, load‖broadcast 并行, 13cy→11cy |
| 6 | Machine Code Emission | VEX 3-byte 编码, ~145 bytes, AArch64 NEON/SVE 对比 |

性能：64 FLOP/iter ÷ 11 cy = 5.8 FLOP/cycle

### P10: GPU Code Generation (`13_gpu_codegen/`)

**运行：** `make run_gpu`

| Step | Pass | 关键操作 |
|------|------|---------|
| 0 | Parallel Detection | linalg.generic iterator_types 分析: parallel → GPU, reduction → 循环 |
| 1 | Thread Mapping | tile [64,64,32], block (16,16), thread tile 4×4 |
| 2 | GPU Dialect Emission | gpu.launch_func, gpu.block_id/thread_id, gpu.barrier |
| 3 | Shared Memory Tiling | bank conflict 分析, coalescing, double buffering |
| 4 | NVVM Lowering | gpu.block_id → nvvm.read.ptx.sreg.ctaid.x, gpu.barrier → nvvm.barrier0 |
| 5 | PTX Emission | .entry gemm_kernel, ld.global.v4, fma.rn, bar.sync |
| 6 | Occupancy Analysis | threads/registers/shared memory 三重资源约束, wave 分析 |

测试：GEMM C[1024,1024] = A[1024,512]·B[512,1024]，shared memory 24KB, occupancy 分析

### P11: Quantization & Mixed Precision (`14_quantization/`)

**运行：** `make run_quant`

| Step | Pass | 关键操作 |
|------|------|---------|
| 0 | Graph Setup | Conv+BN+ReLU+Conv+MatMul 推理图 |
| 1 | Calibration | 收集 min/max/histogram（MinMax/Percentile/Entropy/MSE 对比） |
| 2 | Scale Computation | symmetric/affine, per-tensor/per-channel scale+zp |
| 3 | Fusion | Conv+BN+ReLU → QLinearConv（BN folding + ReLU clip uint8） |
| 4 | Graph Rewrite | 插入 Q/DQ 或 QLinear 算子，INT8 GEMM 累加流程 |
| 5 | Mixed Precision | 逐层敏感度分析: drop<0.5%→INT8, drop<0.1%→FP16, else→FP32 |
| 6 | Summary | 模型大小压缩比, 吞吐提升估算 |

### P12: Memory Planning (`15_memory_planning/`)

**运行：** `make run_memplan`

| Step | Pass | 关键操作 |
|------|------|---------|
| 0 | Graph Setup | ResNet residual block (Conv→BN→ReLU→Conv→Add→ReLU) |
| 1 | Liveness Analysis | 每个 tensor 的 [first_use, last_use] + 时间线可视化 |
| 2 | Interference Graph | 活跃区间重叠检测 → 冲突矩阵 |
| 3 | Offset Planning | Best-Fit Decreasing 策略, 非冲突 buffer 共享偏移 |
| 4 | Buffer Reuse | 枚举可复用的 buffer 对, 潜在节省量 |
| 5 | In-place Optimization | elementwise op 的输出 alias 输入（WAR 安全性检查） |
| 6 | Summary | 峰值内存, 压缩比, 碎片率, 高级技术参考 |

---

## 传统编译器运行

本仓库 CMake 构建的传统编译器部分（解释器、ANTLR、NFA/DFA、LLVM IR Pass），与上方 AI 编译器 pipeline 及下方 `src/mlir/cpu/` 手工流水线相互独立。所有命令在 **build 目录**下执行。

### AST / 解释器

```bash
make run_ast                # 运行全部解释器：V1-V4 + ANTLR + NFA/DFA
make run DOMAIN=ast         # 等价
```

### LLVM Pass 插件

依赖 **「完整构建」表中的 LLVM（`llvm-config`）一行**：`llvm-config` 与 `opt` 在 `PATH` 中即可，**不必**传 `-DMLIR_DIR`/`-DLLVM_DIR`。

```bash
# 运行全部 pass
make run_pass               # 等价于 make run DOMAIN=pass
make run DOMAIN=pass

# 运行指定 pass
make run_simple_pass                       # SimplePass: my-peephole + my-cfg
make run_remove_trivial_block              # RemoveTrivialBlockPass
make run_remove_trivial_loop               # RemoveTrivialLoopPass
make run_remove_zero_trip_loop             # RemoveZeroTripLoopPass

make run DOMAIN=pass PASS=simple_pass      # 等价写法
make run DOMAIN=pass PASS=remove_trivial_block
```

等价的手动命令：

```bash
opt -load-pass-plugin=build/src/pass/SimplePass.so \
    -passes="my-peephole,my-cfg" \
    src/pass/simple_pass.ll -S -o build/src/pass/simple_pass_opt.ll
```

### 综合运行

```bash
make run                    # 运行全部（ast + pass，不含 mlir）
make run DOMAIN=ast         # 只运行 ast 域
make run DOMAIN=pass        # 只运行 pass 域
```

### 测试

```bash
make test                   # 运行全部测试
make test_ast               # 只运行 ast 域测试
make test_pass              # 只运行 pass 域测试
make run_tests DOMAIN=ast   # 等价于 make test_ast
make run_tests DOMAIN=pass  # 等价于 make test_pass
```

Pass 域测试包含：
- **Pass_SimplePass_Src**：C++ 单元测试（直接调用 `SimplePeepholePass` / `RemoveEmptyBlockPass`，解析 IR 后断言优化结果）
- **Pass_SimplePass**：端到端脚本测试（opt + SimplePass 插件，校验输出 IR）

---

## `src/mlir/cpu/` — CPU/RISC-V 手工流水线

独立于本仓库 CMake 构建。在 `src/mlir/cpu/` 下按 [`src/mlir/cpu/README.md`](src/mlir/cpu/README.md) 的说明进行环境准备和运行。

---

## 附加文档

| 文件 | 说明 |
|------|------|
| [`src/mlir/gpu/AI_COMPILER_INTERVIEW.md`](src/mlir/gpu/AI_COMPILER_INTERVIEW.md) | AI 编译器面试高频问题与答案（30+ 题，覆盖全部 15 个阶段） |
| [`src/mlir/gpu/RESUME_PROJECT.md`](src/mlir/gpu/RESUME_PROJECT.md) | 简历项目描述（详细版/精简版/英文版） |
| [`src/mlir/README.md`](src/mlir/README.md) | cpu/ 与 gpu/ 目录关系说明 |
| [`src/mlir/cpu/README.md`](src/mlir/cpu/README.md) | CPU/RISC-V 流水线环境与运行说明 |
