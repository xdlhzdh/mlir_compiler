# AI 编译器面试高频问题与答案

> 基于本项目 **P1–P12** 编译器 pipeline（见 `gpu/CMakeLists.txt` 顶部 Px 表；`4_torch_to_stablehlo` 为辅助脚本非标准 Px），覆盖从模型解析到机器码生成的全流程。各目录内子 Pass 记为 **Px Step N**。

---

## 目录

1. [编译器整体架构](#1-编译器整体架构)
2. [前端与图表示](#2-前端与图表示)
3. [图级优化](#3-图级优化)
4. [Dialect Lowering 与 StableHLO](#4-dialect-lowering-与-stablehlo)
5. [MLIR Pass 基础设施](#5-mlir-pass-基础设施)
6. [Linalg 与 Tensor 级优化](#6-linalg-与-tensor-级优化)
7. [Bufferization](#7-bufferization)
8. [循环优化 (SCF / Affine)](#8-循环优化-scf--affine)
9. [向量化 (Vector Dialect)](#9-向量化-vector-dialect)
10. [后端代码生成 (LLVM)](#10-后端代码生成-llvm)
11. [GPU 代码生成](#11-gpu-代码生成)
12. [量化与混合精度](#12-量化与混合精度)
13. [内存优化](#13-内存优化)
14. [算子融合专题](#14-算子融合专题)
15. [性能分析与调优](#15-性能分析与调优)

---

## 1. 编译器整体架构

### Q1.1: AI 编译器的完整 pipeline 包含哪些阶段？每个阶段的输入输出是什么？

**答：** 一个典型的 AI 编译器 pipeline：

| 阶段 | 输入 | 输出 | 关键技术 |
|------|------|------|----------|
| 1. 模型解析 | ONNX/TorchScript/TF | 图 IR (DAG) | Protobuf 解析, Shape 推断 |
| 2. 图 IR 构建 | 模型格式 | High-level IR | Op 映射, 类型推导 |
| 3. 图级优化 | Graph IR | Optimized Graph | 算子融合, 常量折叠, 死代码消除 |
| 4. HLO/StableHLO | Graph IR | StableHLO ops | Dialect Conversion |
| 5. Linalg 化 | StableHLO | Linalg on tensors | Legalization |
| 6. Tensor 级优化 | Linalg tensors | Tiled Linalg | Tiling, Fusion |
| 7. Bufferization | Tensors | MemRef | Alias 分析, In-place |
| 8. 循环优化 | SCF/Affine | Optimized loops | Tiling, Interchange, Parallel |
| 9. 向量化 | Scalar loops | Vector ops | transfer_read/write, contract |
| 10. LLVM 后端 | Vector/LLVM dialect | 机器码 | ISel, RegAlloc, Scheduling |
| 11. GPU Codegen | Parallel ops | PTX/SASS | Thread mapping, Shared mem |
| 12. 量化 | FP32 graph | INT8/FP16 graph | Calibration, Scale 计算 |
| 13. 内存规划 | Op graph | Memory plan | Liveness, Buffer reuse |

### Q1.2: MLIR 相比传统编译器（如 LLVM）的核心优势是什么？

**答：**

1. **多层次 IR（Multi-level IR）**：MLIR 允许在同一框架中定义多个抽象层级的 dialect（如 StableHLO → Linalg → SCF → LLVM），每层 IR 语义精确，避免了传统编译器"悬崖式降级"（一次从高级语义降到低级 IR，丢失优化机会）。

2. **可扩展的 Dialect 机制**：用户可以自定义 dialect + operations + types + attributes，天然适合领域特定优化（DSL）。而 LLVM IR 是固定的。

3. **Pass 基础设施复用**：CSE、DCE、Canonicalization、Inlining 等通用 pass 可跨 dialect 复用。

4. **显式的 Region/Block 结构**：支持嵌套控制流（如 `scf.for` 内含 `linalg.generic`），适合表示 AI 算子的复杂语义。

5. **生态整合**：StableHLO（Google）、Torch-MLIR（PyTorch）、ONNX-MLIR 等项目统一在 MLIR 框架下，避免了生态碎片化。

### Q1.3: 解释 MLIR 中 Dialect, Operation, Type, Attribute 的关系

**答：**

- **Dialect**：命名空间 + 语义集合。例如 `stablehlo` dialect 包含 AI 高级算子，`linalg` dialect 包含张量代数算子。每个 dialect 注册自己的 operations, types, attributes。

- **Operation**：IR 中的基本计算单元。格式为 `%result = "dialect.op_name"(%operand) {attrs} : (input_types) -> output_types`。Operation 可以包含 Region（嵌套代码块）。

- **Type**：值的类型系统。如 `tensor<4x8xf32>`、`memref<4x8xf32>`、`index`。不同 dialect 可以扩展类型系统。

- **Attribute**：编译期常量元数据。如 `dense<1.0>`（常量张量）、`affine_map<(d0,d1)->(d0,d1)>`（索引映射）。

层次关系：Dialect 包含多个 Operation 定义；每个 Operation 使用 Types 描述操作数/结果的类型，使用 Attributes 携带编译期参数。

---

## 2. 前端与图表示

### Q2.1: ONNX 模型的核心数据结构是什么？解析时需要处理哪些关键信息？

**答：**

ONNX 模型由 Protobuf 定义，核心结构：

- **ModelProto**：顶层容器，包含 `opset_import`（算子版本）和 `GraphProto`
- **GraphProto**：计算图，包含：
  - `node[]`：NodeProto 列表（算子节点），每个节点有 `op_type`、`input[]`、`output[]`、`attribute[]`
  - `initializer[]`：TensorProto（模型权重）
  - `input[]` / `output[]`：ValueInfoProto（图的输入输出）

解析时关键处理：
1. **Shape 推断**：根据算子语义传播形状信息
2. **Initializer 处理**：区分可训练参数和图输入
3. **Attribute 解析**：不同算子有不同属性（kernel_size, padding, strides 等）
4. **Graph 拓扑排序**：保证依赖顺序

### Q2.2: 为什么 AI 编译器需要自己的 IR 而不是直接使用 ONNX？

**答：**

1. **ONNX 是交换格式，不是编译器 IR**：ONNX 设计目标是框架间模型互换，不适合编译优化。缺少 SSA 形式、没有控制流表示、无法表示中间优化状态。

2. **算子粒度不适合优化**：ONNX Conv 算子是高度封装的，编译器需要将其分解为更细粒度的操作（乘法、加法、归约）才能做融合和 tiling。

3. **缺少硬件语义**：ONNX 不区分 tensor/memref/register，不表达内存布局、并行维度等硬件相关信息。

4. **版本碎片**：ONNX opset 版本繁多，同一算子在不同版本语义不同，编译器 IR 需要规范化。

---

## 3. 图级优化

### Q3.1: Conv + BatchNorm 融合的数学原理是什么？为什么这是最经典的图优化？

**答：**

**数学推导：**

BatchNorm（推理模式）：
```
y = gamma * (x - mean) / sqrt(var + eps) + beta
  = gamma / sqrt(var + eps) * x + (beta - gamma * mean / sqrt(var + eps))
  = alpha * x + bias_new
```

其中 `alpha = gamma / sqrt(var + eps)`，`bias_new = beta - gamma * mean / sqrt(var + eps)`

Conv 后接 BN：
```
conv_out = W * input + b
bn_out   = alpha * conv_out + bias_new
         = alpha * (W * input + b) + bias_new
         = (alpha * W) * input + (alpha * b + bias_new)
```

**融合后：**
```
W_fused = alpha * W     (逐输出通道乘 alpha)
b_fused = alpha * b + bias_new
```

**为什么重要：**
1. 消除了 BN 的 5 个参数的运行时计算
2. 减少了一次完整的 tensor 遍历（memory-bound 优化）
3. 在量化场景下 BN 必须融合（否则精度损失大）
4. 几乎所有 CNN 都有 Conv+BN，覆盖面极广

### Q3.2: 常量折叠（Constant Folding）在 AI 编译器中有哪些特殊考量？

**答：**

1. **模型大小权衡**：折叠大型常量表达式（如 `Reshape(constant_weights)`）会将计算结果物化为新常量，增加模型二进制大小。需要设阈值。

2. **Shape-dependent 折叠**：Shape 算子（`Shape`、`Gather`、`Squeeze`）在输入 shape 已知时可折叠为常量，这对消除动态 shape 开销至关重要。

3. **Symbolic vs Concrete**：动态 shape 场景下只能做 partial constant folding（符号常量传播）。

4. **与融合的交互**：某些常量折叠必须在融合之前做（如 BN 参数的预计算），某些在之后做（如融合后的 scale 合并）。

### Q3.3: 请解释 CSE（公共子表达式消除）和 DCE（死代码消除）在图级的工作原理

**答：**

**CSE**：
- 遍历图中所有操作，如果两个操作有相同的 `op_type`、相同的输入、相同的属性，则它们的输出是等价的
- 将后者的所有使用者重定向到前者的输出，删除后者
- 在 SSA 形式下天然高效（值定义唯一）
- AI 编译器中常见场景：多个分支共享相同的 Reshape/Transpose

**DCE**：
- 从图的输出反向标记所有可达的操作
- 删除所有未被标记的操作（它们的计算结果不被使用）
- AI 编译器中常见场景：融合后被替代的旧算子、shape 推断产生的临时计算

两者通常在 canonicalize pass 前后反复执行，形成 `canonicalize → CSE → DCE` 的标准清理流水线。

### Q3.4: Layout 真折叠（NCHW↔NHWC）在 mlir_pass 中如何实现？

**答：** `layout-fold` 对 P4 风格 conv（`[b,f,0,1]` + 常量 kernel perm `{2,3,1,0}`）改写 `dimension_numbers` 与 result layout，消除冗余 transpose。`layout_bridge_legalize.mlir` 仅演示「shape 不变时消 transpose」；完整 NHWC 语义以 `lowering_layout_conv.onnx` + `test_layout_e2e` 为准。

**本项目对应：** [`LayoutFold.cpp`](../../../../../../mlir_pass/lib/Transforms/LayoutFold.cpp)；[`run_layout_e2e.sh`](../../../../../../mlir_pass/scripts/run_layout_e2e.sh)。

**收窄说明：** 非 cudnn layout kernel；失败 shape 时回退为仅删 transpose。

---

## 4. Dialect Lowering 与 StableHLO

### Q4.1: 什么是 StableHLO？它与 HLO/MHLO 的关系是什么？

**答：**

- **HLO（High Level Optimizer）**：XLA 编译器的内部 IR，定义了 ~100 个高级张量操作（dot_general, convolution, reduce 等）。不是公开标准。

- **MHLO（Meta HLO）**：HLO 的 MLIR dialect 化版本，曾用于 TF/JAX → MLIR 的桥梁。但跟随 XLA 内部变化频繁。

- **StableHLO**：OpenXLA 社区制定的 **稳定版本的 HLO 规范**，承诺向后兼容性。是当前推荐的高级 AI dialect。与 MHLO 语义基本一致但有版本保证。

**关键操作：**
- `stablehlo.dot_general`：广义矩阵乘（支持 batched + contracting dimensions）
- `stablehlo.convolution`：参数化卷积（支持任意维度、dilation、padding）
- `stablehlo.reduce`：任意归约操作
- `stablehlo.broadcast_in_dim`：带维度映射的广播

### Q4.2: 解释 Dialect Conversion 的三种模式：Full, Partial, Dynamic

**答：**

MLIR 的 `DialectConversion` 框架支持三种模式：

1. **Full Conversion**：
   - 所有操作必须被转换，未被转换的操作视为错误
   - 用 `ConversionTarget::addIllegalDialect<SourceDialect>()` 标记源 dialect 为非法
   - 适用于阶段间的完整降级（如 StableHLO → Linalg）

2. **Partial Conversion**：
   - 只转换能匹配 pattern 的操作，其余保留
   - 用 `addDynamicallyLegalOp` 标记部分操作为合法
   - 适用于增量 lowering（如只 lower 特定算子）

3. **Dynamic Legality**：
   - 操作的合法性在运行时按条件判断
   - 用 lambda 函数判断每个具体 operation 实例
   - 适用于有条件的 lowering（如只 lower 特定 shape 的 conv）

**Pattern 优先级**：`benefit` 值越高越优先匹配。当多个 pattern 都能匹配时，框架选择 benefit 最高的。

### Q4.3: ONNX → StableHLO lowering 中，MatMul 如何映射到 dot_general？

**答：**

ONNX MatMul `C = A @ B`（A: `[M,K]`, B: `[K,N]`）映射为：

```mlir
%C = stablehlo.dot_general %A, %B,
    batching_dims = [] x [],
    contracting_dims = [1] x [0]
    : (tensor<MxKxf32>, tensor<KxNxf32>) -> tensor<MxNxf32>
```

关键映射：
- `contracting_dims`：A 的最后一维（K）与 B 的第一维（K）收缩
- `batching_dims`：无 batch（如果有 batch 维则填入）
- Batched MatMul `A: [B,M,K], B: [B,K,N]` → `batching_dims = [0] x [0], contracting_dims = [2] x [1]`

`dot_general` 的通用性使其能表示所有矩阵乘变体（matmul, bmm, outer product, dot），是 AI 编译器中最核心的算子之一。

---

## 5. MLIR Pass 基础设施

### Q5.1: 解释 PatternRewriter 的工作流程，以及 Greedy Pattern Rewrite 的执行策略

**答：**

**PatternRewriter 工作流程：**
1. 定义 `RewritePattern`，实现 `matchAndRewrite(op, rewriter)`
2. 在 `match` 阶段检查操作是否满足模式条件
3. 在 `rewrite` 阶段使用 `rewriter` API 修改 IR：
   - `rewriter.replaceOp(op, newValues)` — 替换操作
   - `rewriter.eraseOp(op)` — 删除操作
   - `rewriter.create<NewOp>(...)` — 创建新操作
4. 返回 `success()` 或 `failure()`

**Greedy Pattern Rewrite（贪心重写）：**
```
worklist = all ops in region
while worklist not empty:
    op = worklist.pop()
    for pattern in patterns (sorted by benefit):
        if pattern.matchAndRewrite(op, rewriter) succeeds:
            add modified/new ops to worklist
            break
```

- **贪心策略**：每次选 benefit 最高的匹配 pattern
- **不保证全局最优**：但实践中足够好
- **固定点迭代**：重复直到无 pattern 匹配成功
- **防无限循环**：有最大迭代次数限制

### Q5.2: MLIR 中 OperationPass 和 InterfacePass 的区别？Pass 的调度机制是什么？

**答：**

**OperationPass\<OpT>**：
- 只运行在特定类型的操作上（如 `OperationPass<func::FuncOp>` 只处理函数）
- 最常用的 pass 类型
- 保证并行安全：同一模块内的不同函数可以并行运行 pass

**InterfacePass**：
- 运行在实现了特定 interface 的任何操作上
- 更通用，但使用较少

**Pass 调度（PassManager）：**
```cpp
pm.addPass(createCanonicalizerPass());
pm.addNestedPass<func::FuncOp>(createCSEPass());
pm.addPass(createInlinerPass());
```

- **OpPassManager**：`pm.nest<func::FuncOp>()` 创建嵌套 pass manager
- **并行调度**：`func::FuncOp` 级别的 pass 可以跨函数并行执行
- **Pass Pipeline**：通过字符串描述（如 `builtin.module(func.func(canonicalize,cse))`）

---

## 6. Linalg 与 Tensor 级优化

### Q6.1: Linalg dialect 的核心设计思想是什么？indexing_maps 和 iterator_types 如何工作？

**答：**

**核心思想：** Linalg 用 `linalg.generic` 统一表示所有张量操作，通过两个关键属性参数化：

**indexing_maps（索引映射）：**
```mlir
// GEMM: C[m,n] += A[m,k] * B[k,n]
indexing_maps = [
  affine_map<(m, n, k) -> (m, k)>,   // A 的访问模式
  affine_map<(m, n, k) -> (k, n)>,   // B 的访问模式
  affine_map<(m, n, k) -> (m, n)>    // C 的访问模式
]
```
每个 affine_map 描述一个操作数的索引表达式，参数是迭代变量。

**iterator_types（迭代器类型）：**
```
iterator_types = ["parallel", "parallel", "reduction"]
```
- `parallel`：可并行的维度（输出中存在）
- `reduction`：归约维度（输出中不存在，被累加）

**为什么这个抽象好？**
1. **统一性**：matmul、conv、elementwise 都能用 generic 表示
2. **Tiling 自然**：parallel dims 独立 tile，reduction dims 顺序 tile
3. **Fusion 判断**：通过 indexing_maps 的兼容性判断两个 op 能否融合
4. **向量化**：parallel dims 可直接映射到 vector 维度

### Q6.2: 解释 Tile-and-Fuse 策略以及它在 AI 编译器中的重要性

**答：**

**Tile-and-Fuse 是性能优化的核心策略：**

1. **Tiling（分块）**：将大的迭代空间划分为小块
   ```
   // 原始：C[1024, 1024] += A[1024, 512] * B[512, 1024]
   // Tiled：
   for m_tile in range(0, 1024, 64):
     for n_tile in range(0, 1024, 64):
       for k_tile in range(0, 512, 32):
         C[m_tile:m_tile+64, n_tile:n_tile+64] +=
           A[m_tile:m_tile+64, k_tile:k_tile+32] *
           B[k_tile:k_tile+32, n_tile:n_tile+64]
   ```

2. **Fusion（融合）**：将 tiled 的 producer 和 consumer 合并到同一个 tile 循环内
   ```
   // Unfused: conv → relu (两次内存遍历)
   // Fused:
   for tile in tiles:
     tmp = conv(input[tile])
     output[tile] = relu(tmp)  // tmp 留在 cache/register
   ```

**重要性：**
- **数据局部性**：tiled 数据适合 cache（L1: ~32KB, L2: ~256KB）
- **减少内存带宽**：fusion 消除中间 tensor 的内存 round-trip
- **GPU 适用**：tile 大小映射到 thread block 的共享内存
- **几乎所有高性能 AI kernel 都基于这个策略**

**本项目对应（教学边界）：** `custom-linalg-tile` 对 `linalg.matmul` 做 2×2 strip-mine 并打 `aicom.linalg_tiled`；`linalg_tile_fuse_smoke.mlir` 演示 matmul+bias 经 Linalg 后再 tile（elementwise 暂为独立 `linalg.generic`，非工业 fuse kernel）。`buildLinalgOptimizationStage` 中 **CSE 在 tile 之前**，避免 fold 掉 tile nest；tile 之后不再跑第二次 `linalg-fuse-elementwise-ops`（会撤销分块）。

### Q6.3: 如何判断两个 linalg.generic 操作能否融合？

**答：**

**融合条件检查：**

1. **Producer-Consumer 关系**：producer 的输出是 consumer 的输入
2. **Iterator 兼容性**：
   - Elementwise + Elementwise：总是可融合
   - Matmul + Elementwise：可融合（consumer 只访问 parallel dims）
   - Matmul + Matmul：通常不融合（reduction dim 冲突）
3. **Indexing Map 兼容性**：consumer 的 input map（对应 producer output）必须是 producer 的 output map 的排列或投影
4. **无环检查**：融合后不能产生循环依赖
5. **Tile 兼容性**：两者的 tiling 方案必须兼容

**代价模型考虑：**
- 融合增加寄存器压力（两个 op 的活跃值同时存在）
- 融合可能阻止更好的 tiling 选择
- 大型 reduction（如大 K 的 matmul）不适合与后续 op 融合

---

## 7. Bufferization

### Q7.1: 什么是 One-Shot Bufferization？它解决了什么问题？

**答：**

**问题背景：**
Tensor 语义是值语义（immutable），但硬件执行需要 buffer（mutable memory）。Bufferization 的任务是将 tensor 操作转换为 memref 操作，同时尽量减少内存分配和拷贝。

**传统方法的缺陷：**
- 逐 dialect 做 bufferization（先 linalg, 再 scf, 再 arith），需要混合 tensor/memref 的中间状态，容易出错

**One-Shot Bufferization（OSB）核心步骤：**

1. **Alias Analysis**：分析 tensor 值之间的别名关系（谁可能指向同一块内存）
2. **In-Place Analysis**：对每个写操作，判断能否原地修改输入 buffer
   - 检查 Write-After-Read (WAR) 冲突
   - 检查 Read-After-Write (RAW) 冲突
3. **Bufferization Decision**：标记 INPLACE 或 OUT-OF-PLACE
4. **Rewrite**：将 tensor 操作替换为 memref 操作
   - `tensor.extract_slice` → `memref.subview`
   - `linalg.generic` (tensor) → `linalg.generic` (memref)
5. **Copy Insertion**：对 OUT-OF-PLACE 操作插入 `memref.copy`
6. **Buffer Deallocation**：基于 ownership 分析插入 `memref.dealloc`

### Q7.2: In-place bufferization 的冲突检测是如何工作的？

**答：**

```
例子：
  %0 = linalg.generic ins(%input) outs(%output) { ... }
  // %input 和 %output 是否能指向同一块 buffer？
```

**WAR (Write-After-Read) 冲突检测：**
- 如果操作 op 写入 buffer B，但 B 的值在 op 之后还有其他读取者 → 冲突
- 解决：op 必须使用 copy 后的 buffer

**RAW (Read-After-Write) 冲突检测：**
- 如果操作 op 读取 buffer B，但 B 在 op 之前被另一个操作修改了 → 冲突

**判定算法：**
```
for each op with write:
  B = write target buffer
  for each other_use of B:
    if other_use reads B and happens after op:
      mark as WAR conflict → OUT-OF-PLACE
    if other_use writes B and happens before op reads:
      mark as RAW conflict → OUT-OF-PLACE
  if no conflicts:
    mark as INPLACE
```

**Destination-Passing Style (DPS)：**
Linalg 的操作采用 DPS 风格（`outs(...)` 参数），天然支持 in-place：输出写到 `outs` 参数指定的 buffer。这大大简化了 bufferization。

---

## 8. 循环优化 (SCF / Affine)

### Q8.1: Affine dialect 和 SCF dialect 的区别？各自适用场景？

**答：**

| 特性 | Affine Dialect | SCF Dialect |
|------|---------------|-------------|
| 循环边界 | 必须是仿射表达式 | 任意值 |
| 分析能力 | 完全的依赖分析、合法性判断 | 有限分析 |
| 适用场景 | 规则的稠密计算（矩阵乘、卷积） | 通用控制流、动态边界 |
| 优化能力 | 自动 tiling, interchange, fusion | 手动/pattern-based |
| 表达力 | 受限（仿射约束） | 完全通用 |

**典型 lowering 路径：**
```
linalg.generic → scf.for (or affine.for)
scf.for → llvm 循环 (br + phi)
affine.for → scf.for → llvm 循环
```

**实际选择：** AI 编译器中常使用 SCF（更灵活），对性能关键的 dense kernel 使用 Affine 做分析后降级到 SCF。

### Q8.2: Loop Tiling 的 tile size 如何选择？有哪些考量因素？

**答：**

**关键考量因素：**

1. **Cache 大小**：
   - L1 cache (~32KB): 最内层 tile 的工作集应适合 L1
   - L2 cache (~256KB): 中间层 tile 适合 L2
   - 例如 GEMM: `tile_M × tile_K + tile_K × tile_N ≤ L1_size`

2. **向量宽度对齐**：
   - AVX2: 256-bit = 8 × float32 → tile 的最内层维度应是 8 的倍数
   - AVX-512: 512-bit = 16 × float32

3. **寄存器数量**：
   - 内循环的累加器数量不超过可用寄存器
   - x86-64: 16 个 YMM (AVX2) → 最多 ~14 个累加器
   - 例如 4×8 register blocking: 4 个 `vector<8xf32>` 累加器

4. **GPU 特有**：
   - Shared memory 大小（通常 48KB-164KB）
   - Thread block 大小（通常 128-256 threads）
   - Warp 大小（32 threads）对齐

5. **自动调优**：
   - 搜索空间爆炸（多个 tile 维度的组合）
   - 方法：AutoTVM、Ansor（基于代价模型 + 搜索）、poly optimization

### Q8.3: Loop Interchange 的合法性条件是什么？

**答：**

**合法性条件：** 交换两个循环的顺序后，数据依赖关系不被违反。

**形式化定义（使用依赖向量）：**
- 计算所有循环间的依赖向量 `d = (d1, d2, ..., dn)`
- 交换后依赖向量仍然是字典序非负的（lexicographically non-negative）

**例子：**
```c
// 可以交换 i,j（独立循环）
for i: for j: C[i][j] = A[i][j] + B[i][j]

// 不能交换 i,j（依赖方向会反转）
for i: for j: A[i][j] = A[i-1][j-1] + 1
// 依赖向量 (1,1)，交换后变 (1,1) → 仍合法
// 但如果依赖向量是 (1,-1)，交换后变 (-1,1) → 非法
```

**AI 编译器中的典型应用：**
- GEMM 的 `m-n-k` 交换为 `m-k-n` 以改善 B 矩阵的局部性
- 卷积的通道维度和空间维度交换以适配不同布局

---

## 9. 向量化 (Vector Dialect)

### Q9.1: MLIR Vector dialect 的核心操作有哪些？transfer_read/write 的设计意图是什么？

**答：**

**核心操作：**

| 操作 | 功能 | 示例 |
|------|------|------|
| `vector.transfer_read` | 从 memref 读取到 vector | `%v = vector.transfer_read %m[%i, %j] : memref → vector<8xf32>` |
| `vector.transfer_write` | 从 vector 写入 memref | `vector.transfer_write %v, %m[%i, %j]` |
| `vector.contract` | 向量化的 contraction（矩阵乘核心） | `%c = vector.contract %a, %b, %c {indexing_maps, ...}` |
| `vector.broadcast` | 标量/低维 → 高维广播 | `%v = vector.broadcast %s : f32 to vector<8xf32>` |
| `vector.fma` | Fused Multiply-Add | `%r = vector.fma %a, %b, %c` |
| `vector.shape_cast` | 改变 vector 形状 | `vector<4x8xf32> → vector<32xf32>` |
| `vector.maskedload/store` | 带掩码的访存 | 处理边界条件 |

**transfer_read/write 的设计意图：**
1. **抽象层级正确**：它是 memref 和 vector 世界的桥梁
2. **处理边界条件**：支持 padding value（越界时返回默认值）
3. **支持 permutation_map**：读取时可以做维度转换
4. **延迟具体化**：在后续 lowering 中才决定具体的 load/store 指令（aligned vs unaligned, masked vs unmasked）

### Q9.2: 向量化一个 GEMM 微内核的思路？

**答：**

**目标：** 将标量 GEMM 循环转换为 vector 操作

**策略（Register Blocking）：**
```
// 标量版本
for m in [0, M): for n in [0, N): for k in [0, K):
    C[m,n] += A[m,k] * B[k,n]

// 向量化版本（4×8 register blocking, AVX2）
for mo in [0, M, 4):        // 外层 m，步长 4
  for no in [0, N, 8):      // 外层 n，步长 8（vector width）
    acc[0..3] = transfer_read C[mo+0..3, no] : 4 × vector<8xf32>
    for k in [0, K):
      vb = transfer_read B[k, no] : vector<8xf32>
      va0 = broadcast A[mo+0, k] : vector<8xf32>
      acc[0] = fma(va0, vb, acc[0])
      va1 = broadcast A[mo+1, k] : vector<8xf32>
      acc[1] = fma(va1, vb, acc[1])
      // ... acc[2], acc[3]
    transfer_write acc[0..3] → C[mo+0..3, no]
```

**关键决策：**
- N 维度映射到 vector 宽度（8 for AVX2 float32）
- M 维度展开为多个累加器（4 行 = 4 个 vector 寄存器）
- K 维度保持循环（reduction）
- 共需要 4 个累加器 + 1 个 B vector + 1 个 A broadcast = 6 个 YMM 寄存器

---

## 10. 后端代码生成 (LLVM)

### Q10.1: 从 Vector dialect 到可执行代码的完整路径是什么？

**答：**

```
Vector dialect
    ↓ vector → LLVM dialect lowering
LLVM dialect (GEP, load, store, fma intrinsics, shufflevector)
    ↓ MLIR → LLVM IR translation
LLVM IR (textual .ll)
    ↓ LLVM NVPTX/X86 backend
    ├── Instruction Selection (DAG pattern matching)
    ├── Register Allocation (graph coloring / linear scan)
    ├── Instruction Scheduling (list scheduling)
    └── Machine Code Emission
Assembly (.s) / Object (.o)
    ↓ linker
Executable / Shared library
```

**关键 lowering 规则：**
- `vector.transfer_read` → `llvm.load` 或 `llvm.masked.load`
- `vector.fma` → `llvm.intr.fma`
- `vector.broadcast` → `llvm.shufflevector` (splat)
- `scf.for` → `llvm.br` + `llvm.cond_br` + `llvm.phi`
- `memref` → `llvm.struct` (base ptr, offset, sizes, strides)

### Q10.2: 寄存器分配算法有哪些？AI 编译器为什么要关心寄存器分配？

**答：**

**主要算法：**

1. **Linear Scan**：按线性顺序扫描活跃区间，O(n log n)。LLVM 的 fast regalloc。
2. **Graph Coloring**：构建干涉图，用图着色分配寄存器。经典但开销大。
3. **PBQP**：基于优化的方法，LLVM 的 greedy regalloc 是其变体。

**AI 编译器为什么关心：**
- **高寄存器压力**：GEMM 微内核需要大量累加器寄存器（4×8 blocking = 4 个 YMM），如果 spill 到内存则性能急剧下降
- **GPU 更敏感**：GPU 的寄存器数量直接影响 occupancy（寄存器用多了 → 并行 block 数减少）
- **Tile size 选择**：tile size 最终决定了寄存器需求，编译器的 tiling 策略必须考虑目标架构的寄存器数量

---

## 11. GPU 代码生成

### Q11.1: AI 编译器如何将并行计算映射到 GPU 的 grid/block/thread 层次？

**答：**

**GPU 层次结构：**
```
Grid (整个 kernel)
  └── Block (thread block, 共享 shared memory)
       └── Warp (32 threads, SIMT 执行)
            └── Thread (最小执行单元)
```

**映射策略（以 GEMM 为例）：**
```
C[M, N] = A[M, K] × B[K, N]

1. Parallel dims (M, N) → GPU grid + block
   gridDim.y = M / TILE_M     (每个 block 处理 TILE_M 行)
   gridDim.x = N / TILE_N     (每个 block 处理 TILE_N 列)
   blockDim.y = TILE_M / THREAD_M  (每个 thread 处理 THREAD_M 行)
   blockDim.x = TILE_N / THREAD_N  (每个 thread 处理 THREAD_N 列)

2. Reduction dim (K) → sequential loop inside kernel
   for ko = 0 to K step TILE_K:
     load A_tile, B_tile to shared memory
     __syncthreads()
     for ki = 0 to TILE_K:
       accumulate FMA
     __syncthreads()
```

**关键设计决策：**
- Block size 通常 128-256 threads（平衡 occupancy 和资源）
- Tile size 决定 shared memory 用量
- Thread tile 决定寄存器用量
- K-tile 决定 shared memory + compute 的比率

### Q11.2: GPU Shared Memory 的作用和优化策略？

**答：**

**作用：** Shared memory 是片上 SRAM（~48-164KB/SM），所有同一 block 内的 threads 共享，延迟 ~20 cycles（vs global memory ~400 cycles）。

**在 GEMM 中的作用：**
```
不用 shared memory:
  每个 thread 独立从 global memory 读 A 和 B → K × (M + N) 次 global loads

使用 shared memory:
  threads 协作加载 tile 到 shared memory → K/TILE_K × (TILE_M × TILE_K + TILE_K × TILE_N) global loads
  然后从 shared memory 读 → fast
  数据复用率 = TILE_N (A) 或 TILE_M (B) 倍
```

**优化策略：**

1. **Bank Conflict 避免**：
   - Shared memory 分为 32 banks（4 字节粒度）
   - 同一 warp 的不同 threads 访问相同 bank → conflict → 串行化
   - 解决：padding（加 1 列避免 stride 是 32 的倍数）

2. **Double Buffering**：
   - 两份 shared memory buffer，交替使用
   - 加载 tile[ko+1] 的同时计算 tile[ko]
   - 隐藏 global memory 延迟

3. **异步拷贝**：
   - `cp.async`（SM ≥ 8.0）：global → shared 不经过寄存器
   - 进一步隐藏延迟

### Q11.3: 什么是 GPU Occupancy？如何优化？

**答：**

**定义：** Occupancy = 每个 SM 上活跃 warp 数 / SM 的最大 warp 数

**影响 occupancy 的三个资源：**
1. **Registers/thread**：每个 SM 的寄存器总数固定（如 65536），用得多 → 能并行的 block 少
2. **Shared memory/block**：每个 SM 的 shared memory 固定（如 48KB），用得多 → block 少
3. **Threads/block**：每个 SM 最大线程数固定（如 2048）

**计算：**
```
blocks_per_sm = min(
    max_threads / threads_per_block,
    max_registers / (threads_per_block × regs_per_thread),
    max_shared_mem / shared_mem_per_block,
    max_blocks_per_sm
)
occupancy = blocks_per_sm × threads_per_block / max_threads
```

**优化策略：**
- 不盲目追求 100% occupancy（有时低 occupancy + 高 ILP 更快）
- 减少寄存器：`--maxrregcount` 或减小 thread tile
- 减少 shared memory：减小 tile_k 或使用 multi-stage pipeline
- CUDA Occupancy Calculator 工具辅助分析

---

## 12. 量化与混合精度

### Q12.1: Post-Training Quantization (PTQ) 的完整流程？

**答：**

```
1. 准备校准数据集（~100-1000 个代表性样本）
        ↓
2. 运行 FP32 模型，收集每层激活的统计信息
   - Min/Max（简单但对异常值敏感）
   - 直方图 + Entropy 校准（KL 散度最小化，TensorRT 默认）
   - Percentile（截断异常值）
        ↓
3. 计算量化参数
   - 对称量化：scale = max(|min|, |max|) / 127, zp = 0
   - 非对称量化：scale = (max - min) / 255, zp = round(-min / scale)
   - 权重通常用 per-channel 对称量化（精度更高）
   - 激活通常用 per-tensor 量化（速度更快）
        ↓
4. 算子融合（Conv+BN+ReLU → QLinearConv）
        ↓
5. 插入 Q/DQ 节点或转换为 QLinear 算子
        ↓
6. 混合精度决策（敏感层保持 FP16/FP32）
        ↓
7. 验证精度
```

**本项目串联（A6）：** `run_calib_demo.py` 生成 min/max → scale/zp JSON；`run_calib_to_quant.sh` 调用 `run_quantization` Stage 2 打印 scale 对照（教学，非 INT8 kernel）。`mlir_pass` 侧 `test_quant_e2e` 验证 Q/DQ 图标注。

### Q12.2: Per-tensor 和 Per-channel 量化的区别？为什么权重通常用 Per-channel？

**答：**

**Per-tensor：** 整个 tensor 共用一个 scale 和 zero_point
```
q(x) = clamp(round(x / scale) + zp, qmin, qmax)
```

**Per-channel：** 每个输出通道有独立的 scale 和 zero_point
```
q(x[c]) = clamp(round(x[c] / scale[c]) + zp[c], qmin, qmax)
```

**权重用 Per-channel 的原因：**
- 卷积权重不同通道的值域可能差异很大（例如某些通道 [-0.1, 0.1]，某些 [-5, 5]）
- Per-tensor 量化时，小值通道的精度会被大值通道"牺牲"
- Per-channel 可以为每个通道独立选择最优 scale，精度损失小很多
- 性能代价很小（scale 向量只需在输出时做一次逐通道乘法）

**激活用 Per-tensor 的原因：**
- 激活的 scale 需要在运行时动态计算，per-channel 会增加开销
- 激活值通常跨通道分布更均匀

### Q12.3: INT8 矩阵乘的计算过程和误差来源

**答：**

**计算过程：**
```
// 输入：A (int8, scale_a, zp_a), B (int8, scale_b, zp_b)
// 输出：C (int8, scale_c, zp_c)

// Step 1: INT8 × INT8 → INT32 累加
C_i32[m,n] = Σ_k (A_i8[m,k] - zp_a) × (B_i8[k,n] - zp_b)

// Step 2: 反量化 + 重量化（rescaling）
C_f32[m,n] = C_i32[m,n] × scale_a × scale_b
C_i8[m,n] = clamp(round(C_f32[m,n] / scale_c) + zp_c, -128, 127)
```

**实际优化（对称量化 zp=0 时）：**
```
C_i32 = A_i8 × B_i8              // 硬件 INT8 GEMM（Tensor Core IMMA）
C_i8 = clamp(round(C_i32 × M))   // M = scale_a × scale_b / scale_c（常量）
```

**误差来源：**
1. **量化误差**：将 FP32 映射到 INT8 时的舍入误差
2. **累积误差**：K 次乘法的误差累积（K 越大误差越大）
3. **Rescale 误差**：INT32 → INT8 的二次量化
4. **Outlier 效应**：少量大值导致 scale 偏大，多数小值精度损失

---

## 13. 内存优化

### Q13.1: AI 推理引擎中的内存规划是怎么做的？

**答：**

**步骤：**

1. **Liveness Analysis（活跃分析）：**
   - 对每个中间 tensor，计算其 [first_use, last_use] 区间
   - first_use = 产生该 tensor 的算子
   - last_use = 最后一个读取该 tensor 的算子

2. **Interference Graph（干涉图）：**
   - 两个 tensor 的活跃区间重叠 → 不能共享内存
   - 构建干涉图（类似寄存器分配的干涉图）

3. **Offset Assignment（偏移分配）：**
   - 所有 activation tensor 共享一个大的 memory pool
   - 使用 Best-Fit Decreasing 策略：
     - 按大小降序排列 tensor
     - 每个 tensor 找最小的能放下的 gap
   - 非重叠的 tensor 可以使用同一段内存

4. **In-place Optimization：**
   - elementwise 操作的输出直接写到输入 buffer（如 ReLU）
   - 条件：单一消费者、相同 shape、无 WAR 冲突

**效果：** 对于 ResNet-50，内存规划可将 activation 内存从 ~200MB 降到 ~50MB。

### Q13.2: 训练场景的内存优化有哪些特殊技术？

**答：**

1. **Gradient Checkpointing（梯度检查点）：**
   - 前向传播时不保存所有中间激活
   - 反向传播时按需重新计算
   - 时间换空间：约 30% 额外计算，但内存降至 O(√n)

2. **混合精度训练（AMP）：**
   - 前向/反向用 FP16，权重更新保持 FP32 master copy
   - 内存约减半（激活 + 梯度都是 FP16）
   - 需要 loss scaling 防止梯度下溢

3. **Tensor 换出（Offloading）：**
   - 将暂时不用的 tensor 从 GPU 内存换出到 CPU 内存
   - 需要重叠数据传输和计算
   - DeepSpeed ZeRO-Offload 的核心技术

4. **内存池（Memory Pool）：**
   - PyTorch 的 CUDACachingAllocator
   - 预分配大块内存，按需分配/回收
   - 避免频繁的 cudaMalloc/cudaFree

---

## 14. 算子融合专题

### Q14.1: 总结 AI 编译器中常见的融合模式

**答：**

| 融合模式 | 示例 | 收益 | 难度 |
|---------|------|------|------|
| **Elementwise 链融合** | relu(bias_add(x)) | 减少 N 次内存遍历为 1 次 | 低 |
| **Conv + BN + Activation** | conv → bn → relu | BN 折入权重，一次 kernel | 中 |
| **Producer-Consumer 融合** | matmul → softmax | 减少中间 tensor 写回 | 中 |
| **Horizontal 融合** | 多个独立小 GEMM | 合并为一个大 GEMM，提升 GPU 利用率 | 中 |
| **Reduction + Elementwise** | softmax (reduce + exp + div) | 减少 pass 数 | 高 |
| **Attention 融合** | Q·K^T → scale → mask → softmax → ·V | FlashAttention 核心思想 | 高 |
| **Tiled Fusion** | 在 tile 内融合 producer/consumer | L1 cache 内复用 | 高 |

### Q14.2: FlashAttention 的核心思想是什么？为什么它不需要物化 attention matrix？

**答：**

**标准 Attention 的内存瓶颈：**
```
S = Q @ K^T      → [N, N] 矩阵，N=seq_len
P = softmax(S)   → [N, N]
O = P @ V        → [N, d]
```
当 N=4096 时，S 和 P 各占 4096² × 4B = 64MB，且需要全部 softmax 后才能计算 O。

**FlashAttention 核心思想——分块在线 Softmax：**

1. **将 Q, K, V 沿 seq_len 分成 tiles**
2. **对每个 Q tile，依次与所有 K tiles 计算**
3. **在线更新 softmax**（不需要看到所有 S 值）：
   ```
   // 在线 softmax 的数学基础：
   m_new = max(m_old, max(S_tile))
   l_new = l_old * exp(m_old - m_new) + sum(exp(S_tile - m_new))
   O_new = O_old * (l_old * exp(m_old - m_new) / l_new)
         + exp(S_tile - m_new) @ V_tile / l_new
   ```
4. **S 矩阵从不物化**：每个 tile 的 S 在 SRAM 中计算并立即用于 softmax + V 乘法

**结果：**
- 内存复杂度从 O(N²) 降到 O(N)
- IO 复杂度从 O(N² d) 降到 O(N² d² / M)（M = SRAM 大小）
- 这是 **算子融合 + 内存优化** 的完美结合

**本项目对应：** `attention-legalize` 标注 `aicom.scaled_dot_product_attention`；`flash-attention-tile` 在 SDP 子图的 `dot_general` 上追加 `aicom.flash_tile` + tile size 属性（**编译期标注**，非 GPU kernel）。与 Q14.10 producer-consumer 同属「子图边界识别」，不物化 N×N 矩阵。

### Q14.3: GELU 在编译器里通常如何 lower？与 ReLU 相比难点在哪？

**答：**

**数学定义（常用近似）：**
```
GELU(x) ≈ 0.5 * x * (1 + erf(x / sqrt(2)))
```

**Lowering 路径：**
1. **分解为 primitive ops**：`Div`（除以 √2）→ `Erf` → `Add`（+1）→ `Mul`（×0.5×x）
2. **Erf 方言**：StableHLO 无原生 `erf`，常落到 `chlo.erf` 或自定义 lowering
3. **Fusion 机会**：整链可融合为单 kernel，避免中间 tensor

**与 ReLU 对比：** GELU 含超越函数 `erf`，融合难度高于纯 elementwise 的 ReLU。

**本项目对应：** `mlir_compiler` P4 `lowering_gelu.onnx` + `chlo.erf`；`mlir_pass` `gelu-legalize` 标注 `aicom.gelu_canonicalized`。

### Q14.4: SwiGLU 是什么？编译器如何识别并优化？

**答：**

**定义（LLM FFN 常用）：** `SwiGLU(gate, up) = silu(gate) * up`，其中 `silu(x) = x / (1 + exp(-x))`。

**图结构特征：** `gate` 经 `Neg → Exp → Add(1) → Div` 得 sigmoid，再与 `gate` 相乘得 silu，最终与 `up` 逐元素乘。

**编译优化：** 子图匹配标注 SwiGLU 边界；silu + 逐元素乘可 kernel 融合。

**本项目对应：** `lowering_swiglu.onnx`；`swiglu-legalize` 标注 `aicom.swiglu_canonicalized`。

### Q14.5: MatMul + Bias 融合为什么常见？如何实现？

**答：**

线性层 `Y = X @ W + b` 极常见；分离 GEMM 与 bias add 多一次内存往返。融合后 bias 在 GEMM epilogue 累加。

**图模式：** `add(dot_general(x,w), broadcast(bias))`。

**本项目对应：** `matmul-bias-fusion` 标注 `aicom.matmul_bias_fused`；P4 `lowering_matmul_bias.onnx`。

### Q14.6: 大模型推理中的 Graph Partitioning 解决什么问题？

**答：**

单卡放不下权重/激活时需切分图并插入通信。常见策略：Tensor Parallel（权重切分 + AllReduce）、Pipeline Parallel（按层 stage + P2P）、Expert Parallel（MoE AllToAll）。

编译器职责：划分 subgraph、在边界插入通信节点、估算通信量与计算量平衡。

**本项目对应：** P13 `16_graph_partition/` 教学模拟（无真实多卡 runtime）。

### Q14.7: ONNX Q/DQ 量化图在编译器前端如何处理？

**答：**

`DQ: (q - zp) * scale` 将 int8 恢复为 float；`Q` 反向量化。编译器识别 `DQ → Compute → Q` 子图，对双端 DQ 的 MatMul 可标注 QDQ 边界或改写为 QLinear 算子。

**本项目对应：** P11 `quant_qdq_matmul.onnx`；P4 `lowering_qdq_matmul.onnx`；`mlir_pass` `qdq-legalize` + `test_quant_e2e`。

### Q14.8: Horizontal GEMM 融合是什么？编译器如何做？

**答：**

**场景：** 多个小 GEMM 共享同一输入激活 `X`，输出在特征维拼接，常见于 MoE 门控或多头投影拆分：
```
Y1 = X @ W1
Y2 = X @ W2
Y = concat(Y1, Y2, axis=-1)
```

**优化思路：** 将 `W1`、`W2` 在输出维拼接为 `W = [W1|W2]`，一次大 GEMM `Y = X @ W` 替代两次小 GEMM + concat，提升 GPU 利用率、减少 kernel launch。

**编译器实现：**
1. **模式匹配**：`Concat` 的操作数均为 `DotGeneral`，且 `lhs` 相同
2. **合法性**：contracting 维一致；concat 维为输出特征维
3. **改写**：合并权重常量，或标注融合边界供后续 codegen 选用 batched/strided GEMM

**本项目对应：** `horizontal-gemm-fusion` 标注 `aicom.horizontal_gemm_fused`；P4 `lowering_horizontal_gemm.onnx`。

### Q14.9: Elementwise 链融合在 mlir_pass 中如何体现？

**答：** 识别 `maximum(add(x,c), 0)` 改写为 `clamp(0, add, +inf)` 并 **erase** `maximum`；或已有 `clamp` 时标注 `aicom.elementwise_chain_fused`（StableHLO 图级折叠，非 fused CUDA kernel）。

**本项目对应：** `elementwise-chain-legalize`；LIT `elementwise_chain_legalize.mlir`（`CHECK-NOT: stablehlo.maximum`）。

### Q14.10: Producer-Consumer 融合（MatMul → Softmax）如何标注？

**答：** `dot → subtract(max) → exp` 教学子图中，将 `exp(scores-max)` 改写为 `exp(scores)*exp(-max)` 并删除 `subtract`；在 `multiply` 上标注 `aicom.producer_consumer_fused`。

**本项目对应：** `producer-consumer-legalize`；LIT `CHECK-NOT: stablehlo.subtract`；P4 `lowering_matmul_softmax.onnx`。

### Q14.11: 动态 Shape 跨仓库 e2e 如何验证？

**答：** P4 tier 2 `lowering_dynamic.onnx`（动态 batch + bias 输入）经 `run_level2 --mlir-only` 导出 StableHLO，在 `mlir_pass` 跑 fusion/linalg 并 grep `tensor<?x` 与 `stablehlo.dot_general`；`run_golden` 对 batch=2/4 两档做 ORT vs NumPy。

**本项目对应：** `scripts/run_dynamic_e2e.sh`；`test_dynamic_e2e`；`run_lowering_golden.check_dynamic`。

### Q14.12: CHLO 与 StableHLO 分层、GELU 如何跑通 Linalg？

**答：** StableHLO 核心无 `erf`；P4 GELU lowering 落到 `chlo.erf`。在 Linalg 前须跑 `chlo-legalize-to-stablehlo`（将 erf 等 CHLO 算子分解为 StableHLO 原语），再 `stablehlo-legalize-to-linalg`。`mlir_pass` 在 `buildStableHloToLinalgStage` 中自动插入该 pass。

**本项目对应：** `lib/Transforms/StableHLOToLinalg.cpp`；`test/lit/gelu_linalg_smoke.mlir`；shell regression「gelu_linalg_smoke → no chlo.erf + linalg」。

### Q14.13: numpy-style Broadcast 图优化如何验证？

**答：** P4 tier 2 将 ONNX Add/Mul 的隐式广播显式化为 `stablehlo.broadcast_in_dim`。`mlir_pass` 的 `broadcast-simplify` 消除恒等广播、折叠嵌套 broadcast 链；`lowering_broadcast.onnx` 有 golden + `run_broadcast_e2e` 跨仓库验证 fusion/linalg。

**本项目对应：** `BroadcastSimplify.cpp`；`test/lit/broadcast_simplify.mlir`；`scripts/run_broadcast_e2e.sh`；`run_lowering_golden.check_broadcast`。

### Q14.14: Shape 收窄与 get_dimension_size 教学路径（A1）？

**答：** 官方 `stablehlo-refine-shapes` 在静态 bias/权重参与时把 `?×3` 收窄为 `2×3`；`shape_get_dimension_size.mlir` 演示 `get_dimension_size` + `tensor.cast` 组合。无自写 shape pass。

**本项目对应：** `StableHLOToLinalg.cpp`；`shape_refine_batch.mlir` / `shape_get_dimension_size.mlir`；shell + LIT。

**收窄说明：** 无自写 shape pass；非任意 rank 符号推断。

### Q14.15: Layout 真 NHWC 折叠（A2）？

**答：** `layout-fold` 对 P4 风格 conv（`[b,f,0,1]` + weight perm `{2,3,1,0}`）改写 `dimension_numbers` 与 result layout；LIT 简单 fixture 仅消 transpose。

**本项目对应：** `LayoutFold.cpp`；`test_layout_e2e`；`lowering_layout_conv.onnx`。

**收窄说明：** LIT fixture 不代表完整 NHWC 权重重排路径。

### Q14.16: torch-mlir Conv+BN 跨仓库 e2e（A4）？

**答：** `run_torch_e2e.sh` 导出或 fixture → fusion，断言无 `batch_norm`；Conv **尚不能** CPU JIT（无 conv→LLVM 路径）。

**本项目对应：** `run_torch_e2e.sh`；`conv_bn_torch.mlir` fixture。

**收窄说明：** 无 Conv JIT 数值闭环；勿写「torch 端到端 JIT」。

### Q14.17: P4 GELU / SwiGLU 全链路 JIT（A5）？

**答：** `gelu_p4_jit.mlir` / `swiglu_p4_jit.mlir`；JIT golden `atol=1e-3`（GELU erf）；`jit_gelu.mlir` 保留 sigmoid 快速路径。

**本项目对应：** `run_jit_golden.py` **6 项**；`gelu_linalg_smoke.mlir`。

**收窄说明：** 仅 nullary fixture；含参图不能直 JIT。

### Q14.18: KV decode + P12 内存规划（B2）？

**答：** P4 `lowering_decode_step.onnx` → `kvcache-legalize`；`run_kvcache_e2e.sh` 断言 `kvcache_boundary` + P12 decode slot。

**本项目对应：** `test_kvcache_e2e`；`kvcache_legalize.mlir`。

### Q14.19: Graph Partition 通信量 golden（B3）？

**答：** `sync_partition_fixture.py` 解析 P13 stdout，硬 golden **262144** comm bytes；`run_partition_smoke.sh` 串联 P13 + `graph_partition_smoke.mlir`。

**本项目对应：** `test_partition_smoke`。

### Q14.20: 多动态维 MatMul（B5）？

**答：** `lowering_dynamic_mn.onnx`（`?×K` @ `K×?`）；`run_dynamic_e2e.sh` 双模型；`decode_loop.mlir` 演示 `scf.while`（fusion stop）。

**本项目对应：** `run_lowering_golden.check_dynamic_mn`；`test_dynamic_e2e`。

### Q14.21: FP16 MatMul lowering golden（B6）？

**答：** `lowering_matmul_f16.onnx` 发射 f16；golden cast 到 f32 比较（`rtol/atol=1e-2`）。无 FP16 LLVM 优化。

**本项目对应：** `run_lowering_golden.check_matmul_f16`；P11 混合精度 Step 5 概念对齐。

---

## 15. 性能分析与调优

### Q15.1: 如何分析一个 AI kernel 是 compute-bound 还是 memory-bound？

**答：**

**Roofline Model 分析：**

1. **计算 Arithmetic Intensity (AI)**：
   ```
   AI = FLOPs / Bytes_accessed
   ```

2. **比较与硬件的 Ridge Point**：
   ```
   Ridge Point = Peak_FLOPS / Peak_Bandwidth
   ```
   - 如果 AI > Ridge Point → **Compute-bound**
   - 如果 AI < Ridge Point → **Memory-bound**

3. **常见算子分析**：

| 算子 | FLOPs | Bytes | AI | 瓶颈 |
|------|-------|-------|-----|------|
| GEMM [M,K]×[K,N] | 2MKN | 4(MK+KN+MN) | ~K/4 | Compute (large K) |
| Elementwise | N | 8N (read+write) | 0.125 | Memory |
| BatchNorm | ~8N | ~12N | ~0.67 | Memory |
| Softmax | ~5N | ~8N | ~0.625 | Memory |
| Conv (3×3, C=256) | ~2×9×256²×HW | ... | ~100+ | Compute |

**实际工具：**
- NVIDIA Nsight Compute：详细的 kernel profiling
- `nvprof` / `ncu`：查看实际的 SM utilization 和 memory throughput
- Roofline 图：直观判断优化方向

### Q15.2: TensorRT、XLA、TVM 的核心差异是什么？

**答：**

| 特性 | TensorRT | XLA | TVM |
|------|----------|-----|-----|
| **定位** | NVIDIA 推理优化器 | Google 训练+推理编译器 | 通用深度学习编译器 |
| **前端** | ONNX/TF/PyTorch | JAX/TF/PyTorch (torch_xla) | Relay/TIR (多前端) |
| **中间表示** | 内部图 IR | HLO → StableHLO | Relay (图级) + TIR (算子级) |
| **优化方式** | 手写 kernel + 模板匹配 | Pattern-based fusion + LLVM | 搜索式自动调优 (AutoTVM/Ansor) |
| **算子融合** | 规则匹配 (强大但固定) | HLO fusion heuristics | Relay level + compute/schedule 分离 |
| **量化** | INT8/FP16 强大 | 有限 | 支持但不完善 |
| **目标硬件** | NVIDIA GPU 专用 | CPU/GPU/TPU | 几乎所有硬件 |
| **性能** | NVIDIA GPU 最优 | TPU 最优，GPU 良好 | 依赖搜索时间，峰值可比拟手写 |
| **灵活性** | 低（闭源） | 中 | 高（完全开源） |

**面试观点：** TensorRT 追求极致性能但限制平台；XLA 与 Google 生态深度绑定；TVM 通过 auto-scheduling 追求跨平台通用性。MLIR 试图成为统一的底层框架，所有编译器都可以在 MLIR 上构建。

### Q15.3: Polyhedral Model 在 AI 编译器中的作用和局限性？

**答：**

**作用：**
多面体模型将循环变换问题转化为整数线性规划（ILP）问题：
- 每个循环迭代映射为多维整数空间中的点
- 数据依赖映射为约束
- 自动找到满足依赖的最优调度（最大并行 + 最优局部性）

**在 AI 编译器中的应用：**
- Linalg 的 `affine_map` 天然可以做多面体分析
- 自动 tiling、interchange、fusion、parallelization
- MLIR Affine dialect 支持多面体分析

**局限性：**
1. **只处理仿射访问**：动态 shape、indirect access 不支持
2. **代价模型不精确**：ILP 优化的目标函数难以准确建模 cache/register 行为
3. **编译时间**：对大问题实例 ILP 求解时间可能很长
4. **AI 算子的特殊性**：很多算子（softmax, attention）的访问模式不是纯仿射的
5. **实践中**：TVM/Ansor 的搜索式方法通常比纯多面体方法效果更好

---

## 附录 B：第三档概念答法（简历勿夸大）

> 第三档 C1–C10 **仅概念**，无大工程代码。能力映射 §2.1 链接至此。

### C1: 自定义 MLIR Dialect

| 段 | 内容 |
|----|------|
| **是什么** | 用 TableGen 定义自有 op/attr/type，形成独立 dialect |
| **工业方案** | `mhlo`/`stablehlo`/`linalg`/`tosa` 等 |
| **本项目边界** | 无独立 dialect；用 StableHLO + `aicom.*` 标注属性 |
| **面试一句话** | 「理解 dialect 扩展方式；本项目用 decomposition + 标注降低工程量」 |

### C2: PyTorch Dynamo / FX / AOT

| 段 | 内容 |
|----|------|
| **是什么** | 捕获 PyTorch 计算图并导出/编译 |
| **工业方案** | Dynamo→FX→Inductor；`torch.export` |
| **本项目边界** | 无 Dynamo；`torch-mlir` Conv+BN 导出 demo |
| **面试一句话** | 「熟悉 PyTorch 编译栈分层；主路径 ONNX→StableHLO」 |

### C3: TorchScript / JAX / TensorFlow

| 段 | 内容 |
|----|------|
| **是什么** | 多前端统一到 HLO/XLA 类 IR |
| **工业方案** | JAX→XLA；TF→XLA；TS→ONNX |
| **本项目边界** | 以 ONNX + StableHLO 为共同 IR |
| **面试一句话** | 「StableHLO 是跨框架交换层；我练的是 ONNX 解析 + SHLO lowering」 |

### C4: QAT vs PTQ、FP8

| 段 | 内容 |
|----|------|
| **是什么** | 训练后量化 vs 量化感知训练；NV FP8 训练/推理 |
| **工业方案** | QAT 插入 fake quant；PTQ 校准 scale |
| **本项目边界** | PTQ `run_calib_demo` + `run_calib_to_quant`；无 QAT/FP8 kernel |
| **面试一句话** | 「能讲 PTQ 校准链；QAT/FP8 仅概念」 |

### C5: Tensor Parallel / Pipeline Parallel

| 段 | 内容 |
|----|------|
| **是什么** | 模型切分到多卡；插入通信 |
| **工业方案** | NCCL AllReduce/AllGather；Megatron-LM |
| **本项目边界** | P13 编译期切分 + comm bytes 估算；无 NCCL |
| **面试一句话** | 「P13 教编译期 partition；runtime 并行另论」 |

### C6: 全模型端到端编译

| 段 | 内容 |
|----|------|
| **是什么** | 整网单 pass 编译到硬件 |
| **工业方案** | XLA whole-graph；TensorRT builder |
| **本项目边界** | 子图 StableHLO demo；禁止假 LLaMA benchmark |
| **面试一句话** | 「子图 lowering + fusion 闭环；不说全模型性能数字」 |

### C7: 真实 GPU 下发

| 段 | 内容 |
|----|------|
| **是什么** | PTX/SASS kernel launch |
| **工业方案** | cuBLAS/cuDNN；Triton；CUTLASS |
| **本项目边界** | P10 PTX 文本模拟；`mlir_pass` CPU LLVM JIT |
| **面试一句话** | 「GPU 映射理解 + PTX 教学；数值验证在 CPU JIT」 |

### C8: Benchmark 与性能声称

| 段 | 内容 |
|----|------|
| **是什么** | 可复现的性能测试方法论 |
| **工业方案** | Roofline；Nsight；固定 batch/warmup |
| **本项目边界** | Q15.1 Roofline 概念；无虚构 speedup |
| **面试一句话** | 「用算术强度讲瓶颈，不编百分比」 |

### C9: Auto-tuning / TVM

| 段 | 内容 |
|----|------|
| **是什么** | 搜索 tile/schedule 最优点 |
| **工业方案** | AutoTVM/Ansor；Halide autoscheduler |
| **本项目边界** | 手工 `custom-linalg-tile` 2×2 教学 |
| **面试一句话** | 「理解搜索式调度；本项目固定 tile 教原理」 |

### C10: IREE / XLA 插件生态

| 段 | 内容 |
|----|------|
| **是什么** | 以 MLIR/StableHLO 为枢纽的多后端 |
| **工业方案** | IREE；OpenXLA；PJRT |
| **本项目边界** | 自建 `pipe-demo` pipeline；概念对齐 |
| **面试一句话** | 「StableHLO 是生态枢纽；我实现的是教学级 lowering 链」 |

---

## 附录 C：工业级全栈缺口速查

> 与 [能力映射 §2.2](./编译器能力映射.md#22-工业级全栈编译器缺口本项目刻意未做) 同构；面试「诚实边界」用。

| 知识域 | 简历禁止表述 |
|--------|-------------|
| Conv/GPU JIT | 「Conv 端到端 JIT / GPU 下发」 |
| 真 tile-fuse | 「完整 tile-fuse pipeline」 |
| 校准→INT8 kernel | 「校准 JSON 驱动 INT8 加速」 |
| FlashAttention kernel | 「实现了 FlashAttention kernel」 |
| QAT/FP8 | 「QAT/FP8 推理提升 XX%」 |
| TP/PP runtime | 「多卡 NCCL 分布式」 |
| Dynamo/Dialect | 「实现 Dynamo / 自定义 Dialect」 |
| 全模型 benchmark | 「LLaMA 全编译 / 吞吐 XX%」 |
| Paged KV | 「PagedAttention runtime」 |

第三档概念答法（C1–C10）见 **附录 B**；本项目主线为 ONNX→StableHLO→mlir_pass CPU JIT 子图闭环。

---

## 附录：面试快速参考

### 必须掌握的核心概念清单

- [ ] MLIR 多层次 IR 与 Dialect 体系
- [ ] SSA 形式与 use-def chain
- [ ] PatternRewriter + GreedyPatternRewrite
- [ ] Linalg indexing_maps + iterator_types
- [ ] Tile-and-Fuse 策略
- [ ] One-Shot Bufferization
- [ ] Tensor → MemRef 转换
- [ ] Loop Tiling + Interchange + Parallelization
- [ ] Vector dialect 与 SIMD 向量化
- [ ] GPU thread hierarchy 映射
- [ ] Shared memory tiling + bank conflict
- [ ] Occupancy 计算
- [ ] INT8 量化原理（symmetric/affine, per-tensor/per-channel）
- [ ] Conv+BN Fusion 数学推导
- [ ] GELU / SwiGLU lowering 与融合边界
- [ ] MatMul+Bias epilogue 融合
- [ ] Horizontal GEMM（共享 LHS + concat）
- [ ] Graph Partitioning（TP/PP 与通信插入）
- [ ] ONNX Q/DQ 图识别与 DQ 链 lowering
- [ ] Memory liveness analysis + buffer reuse
- [ ] Roofline model（compute-bound vs memory-bound）
- [ ] FlashAttention 在线 softmax 原理
- [ ] FlashAttention tile 标注（`aicom.flash_tile`，非 kernel）
- [ ] 第三档概念 C1–C10（附录 B，简历勿夸大）
- [ ] 工业级全栈缺口（附录 C / 能力映射 §2.2，简历勿夸大）
