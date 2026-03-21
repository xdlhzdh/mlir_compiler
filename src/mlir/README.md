# `src/mlir/`：`cpu/` 与 `gpu/` 的区别

本仓库将 MLIR 相关示例分为两个目录：**编译栈不同**，与机器上是否插有独立 GPU **无必然对应关系**。

- **`cpu/`**：面向 **CPU 类目标**（主机 ISA、**RISC-V 仿真或真机**等）。演示的是 **自 Linalg（tensor）起的 lowering 管线**：经 **bufferization**、**LLVM Dialect**，再到 **LLVM 后端**（`llc` / `clang` 等生成目标代码）。所谓 cpu 指 **走通用处理器 codegen 这条栈**，而非「仅在笔记本 CPU 上演示」。
- **`gpu/`**：面向 **深度学习编译里常见的「图级 → 算子级」路径**：在 **StableHLO / ONNX 等 L1 图 IR** 上做领域 Pass（如 Conv+BN 融合），与后续 **L2（Linalg / TOSA）→ L3（循环调度 / TensorIR 等）→ L4（LLVM / GPU ISA）** 的工业界分层 **概念对齐**。目录名 **gpu** 表示加速器/图编译一侧；**本仓库 CMake 仅构建 StableHLO Pass 与 `mlir-opt` 演示**，**不包含** GPU kernel 下发。

---

### 分层一览（与常见「L1–L4」说法对齐）

| 层级 | 常见称谓 | `cpu/` 示例关注点 | `gpu/`（本仓库）关注点 |
|------|----------|-------------------|------------------------|
| **L1** | 图级 IR（交换 / 图优化） | 多为 **Torch → Linalg** 入口，不强调 StableHLO | **StableHLO**（生态上可对接 **ONNX** 等） |
| **L2** | 张量级 / 结构化算子 | **`linalg` on `tensor`**：以 **tiling / fusion / genericization** 为主，优先减少 **DRAM 访问** | 下游可接 **Linalg / TOSA**（本仓示例 **止于 L1 Pass**） |
| **L3** | Kernel / 循环与调度 | **`memref` / `vector`** 上的显式循环、展开、向量化、并行化，重点压榨 **Compute/ALU** | 常叠 **TensorIR**、**GPU launch / 线程块** 等（**本仓未展开**） |
| **L4** | 后端 / 指令形式 | **LLVM** + 具体 ISA（例：**RISC-V**） | **LLVM IR / GPU ISA**（**本仓未接 codegen**） |

### L2 / 中转 / L3 的职责切分

把中端进一步拆开时，可以把 **bufferization** 单独看成 **L2 与 L3 之间的中转层**：

| 阶段 | 主要动作 | 数据类型 | 核心关注点 |
|------|----------|----------|------------|
| **L2** | `linalg` 上的 **Tiling / Fusion / Genericization** | `tensor` | 优先减少 **DRAM / 带宽压力**，让后续计算在更好的数据局部性上展开 |
| **中转** | **One-Shot Bufferize** | `tensor -> memref` | 决定 **缓冲区分配、别名与原地更新**，把值语义切到存储语义 |
| **L3** | **Linalg to Loops / Unrolling / Vectorization / Parallel** | `memref` / `vector` | 围绕循环与 SIMD/并行映射，尽量压满 **计算单元 / ALU** |

这三步不是说每条 pipeline 都严格线性且不可穿插，而是提供一个更便于理解的**优化重心划分**：**L2 更像带宽侧调度**，**中转负责内存语义落地**，**L3 更像算力侧调度**。

### StableHLO 往下是不是 Linalg？从哪一步起 `cpu/` 和 `gpu/` 算「汇合」？

1. **StableHLO → Linalg**  
   在常见 **MLIR 系** AI 编译栈里，**StableHLO（L1）往下** 往往会通过一批 **lowering / conversion pass** 落到 **`linalg` on `tensor`（L2）**——与 **Torch → Linalg** 的 **L2 形态是同一种抽象**（都是结构化张量算子，便于再接 bufferize、循环等）。  
   **注意**：也有栈会走 **StableHLO → TOSA** 或其它中间层，**Linalg 不是唯一合法下一站**，但在 **「统一到 Linalg 做中端」** 的路线里，你的理解 **成立**。

2. **从 Linalg 起是否算 `cpu/` 与 `gpu/` 汇合？**  
   **概念上：是。** 一旦都变成 **`linalg.generic` / `linalg.matmul` 等 on tensor**，后面 **bufferization、展循环、落到 LLVM Dialect** 等 **中段 Pass 族** 与 **CPU 侧教程** 讲的是 **同一套 MLIR 机制**——可以认为两条前端在 **L2 Linalg** **汇合**。  
   **实际上**：接 **GPU** 时往往在 **L3 或更早** 再 **分叉**（例如映射 **block/thread**、**shared memory**、**GPU Dialect / launch**，或走 **LLVM NVPTX/AMDGPU** 等），**不会**与 **本仓 `cpu/` 里 RISC-V `llc` 命令** 逐字相同；**汇合的是 IR 层与大量 Pass 思路，不是整条 shell 一键复用**。

3. **本仓库现状**  
   **`gpu/`** 的 CMake 流程 **停在 StableHLO + 图级 Pass**，**没有** 再接一段 **StableHLO→Linalg** 写进脚本；**`cpu/`** 从 **Torch→Linalg** 起跳。**因此仓库内两条线并未在一份命令里真正拼成一条**，但按业界分层，**若在 Linalg 处对接，后面与 `cpu/` 文档中的 lowering 故事是同一脉络**。

---

### L2→L3：Bufferize、Tiling、SCF、Affine、Vector 分别是什么？

**约定**：这里把 **L2** 看作 **`tensor` 世界里的局部性优化**，把 **bufferize** 单独看作 **L2→L3 的中转**，而 **L3（Kernel 段）** 指计算已主要落在 **`memref` / `vector` + 显式循环 nest（或等价可调度结构）** 上、即将对接目标 ABI 或并行映射的那一段 IR。表中条目 **既有「Pass 变换」也有「方言形态」**，不是并列的「七个层次」。

**执行顺序**：**存在常见先后，但无唯一标准 pipeline**。概念上，**bufferize** 与 **linalg → 循环** 在不同管线里 **可先后互换**；但按本文当前采用的口径，以及本仓库 `cpu/` 示例的主线，**更适合把 bufferize 看作进入 L3 前的中转**，再在 **`memref` / `vector`** 世界里展开显式循环、向量化与并行化。标「可选」的步骤可省略或重复。更重要的是看清**每一步优化的目标**：先改善数据搬运，再把内存形态落地，最后榨取循环与向量硬件能力。

| 次序 | 名称 | 类型 | 说明 |
|------|------|------|------|
| ① 可选 | **Tiling（分块）** | 变换 | 典型作用在 **`linalg`（常仍为 `tensor`）** 或已有循环上；产出更小 **tile** 与嵌套。它首先服务于 **L2 的局部性/减少访存**，不是与 Affine 同级的独立「一层 IR」，而是贯穿 L2/L3 的 **调度/形状策略**。 |
| ② | **Bufferize**（如 one-shot-bufferize） | 变换（**bufferization**） | **`tensor`（值语义）→ `memref`（存储语义）**。核心是确定 **分配、别名、是否可原地更新**，把张量语义安全地落到缓冲区。按本文与本仓 `cpu/` 示例的主线，可把它看成 **进入 L3 前的中转**。 |
| ③ | **Linalg → 显式循环** | 变换 + **SCF** 形态 | `linalg.*` 展开为 **`scf.for` / `scf.if`** 等。放在本文当前口径里，它更适合被理解为 **bufferize 之后进入 L3** 的一步：把计算显式落到 **`memref`** 上的循环结构。**Structured Control Flow**：结构清晰，**迭代域不必仿射**；经典 **polyhedral** 工具 **未必直接适用**。 |
| ④ 可选 | **Affine** | 方言（循环子类） | **`affine.for`** 等：**仿射** 索引与界，利于 **依赖分析、融合、自动 tiling**。可视为 **SCF 可分析子集**（**不同方言**，不是 SCF 的简单语法糖）。 |
| ⑤ 可选 | **Vector** | 方言 | **SIMD / lane** 级类型与算子；常配合 **向量化、展开、并行映射** 等 L3 优化，再经 **`convert-vector-to-llvm`** 进入 LLVM。 |
| — | **Loop nest** | 术语 | MLIR **无单独顶级 Loop 方言**；通常指 **`scf` / `affine`** 中 `for` 形成的 **嵌套**。 |

---

### 与 `cpu/` 示例、及 `mlir-opt` / `llvm-opt` 的关系

1. **语义上**：`cpu/` 中 matmul 路线可读成 **L2（tensor 局部性）→ bufferize 中转 → L3（memref/vector 算力调度）→ LLVM Dialect → LLVM IR**，与工业界 CPU lowering **同一套路**；若接 GPU L3/L4，**整体形状相似**，但必须增加 **线程层次、共享存储与 launch** 等，**不能**照搬本仓 RISC-V 命令序列。
2. **工具分工**：Kernel 段 lowering **主要在 `mlir-opt`** 中完成；**`mlir-translate --mlir-to-llvmir`** 得到 **`.ll`**。**`llvm-opt`** 仅作用于 **已是 LLVM IR** 的文件，负责 **LLVM 中间层优化**，**不能替代** MLIR 侧的 tensor/memref/scf lowering；实践中可 **串联**：`mlir-opt` → `mlir-translate` →（可选）`llvm-opt` → `llc`。
3. **是否还要「做编译器」**：**Lowering 与规范化 Pass 一般由上游实现**，无需自研每条 rewrite；工作重心通常是 **编排 pipeline**、**调 pass 选项**、编写 **图级或领域 Pass**（如本仓库 **`gpu/`** 目录中的 **Conv+BN 融合**）、或 **对接自研后端**。**「不必手写每个 convert」≠「零编译知识」**。

与 **matmul 逐步命令** 的一一对应见 **[cpu/README.md](cpu/README.md)** §1.1（数据流与步骤表）。

---

| | **`cpu/`** | **`gpu/`** |
|---|------------|------------|
| **定位** | CPU/RISC-V：**Linalg → LLVM** 手工管线与示例 IR | **L1 StableHLO** Pass + Python 导出（图侧） |
| **本仓库 CMake** | **否**（无 `cpu/CMakeLists.txt`） | **是**（`find_package(MLIR)` 成功则 `add_subdirectory(gpu)`） |
| **产物** | 已提交 `.mlir` / `.ll`；可用 `mlir-opt`、`llc`、`clang` 本地生成二进制 | `libDialectPass.so`、`conv_bn_*.mlir` 等 |
| **入口文档** | [cpu/README.md](cpu/README.md) | 根目录 README「MLIR GPU」 |
| **依赖** | `mlir-opt`、可选 `torch-mlir`（跑 `matmul.py`） | MLIR + StableHLO、Python、与插件同树的 `mlir-opt` |

## 根目录 CMake 如何挂接

- 仅在 **`src/CMakeLists.txt`**：`find_package(MLIR QUIET CONFIG)` **成功**时 `add_subdirectory(mlir/gpu)`。
- **`mlir/cpu`** 不 `add_subdirectory`，故 **无 cpu target**；**有意**弱化对本机 MLIR 安装路径的绑定。

## 运行 `gpu/`（本仓库已接好）

在 **build 目录**：

```bash
cmake --build . --target conv_bn_optimized
# 或
make run DOMAIN=mlir PASS=conv_bn_fusion
```

输出一般在 `build/src/mlir/gpu/` 下的 `conv_bn_model.mlir`、`conv_bn_fusion.mlir`。详见项目根目录 **README.md** 的「MLIR GPU」小节。

## 运行 `cpu/`（手工流水线）

1. 阅读 [cpu/README.md](cpu/README.md)：完成 **一、环境准备**，再在 **二、代码运行与流水线** 中于 `src/mlir/cpu` 下执行。
2. 若执行到 **2.6** 的 QEMU 步骤，可得到运行输出 `riscv_run.txt` 作为结果记录。
3. 若只读示例 IR，可直接打开 `matmul*.mlir`、`matmul.ll` 等文件，无需构建 **gpu** CMake 目标。

## 命名小结

- **`cpu/`**：**通用处理器 lowering 教程栈**（Linalg 起至 LLVM）；含 **RISC-V**，非仅限 x86。
- **`gpu/`**：**图编译 / 加速器管线命名**；本仓库 **仅实现 L1 StableHLO Pass**，L2–L4 **未**在本目录全覆盖。
