# `src/mlir/`：`cpu/` 与 `gpu/` 的区别

本仓库将 MLIR 相关示例分为两个目录。**二者不是「CPU 机器 vs GPU 机器」**，而是 **编译器工作侧重点不同**：

| | **`gpu/`** | **`cpu/`** |
|---|------------|------------|
| **在编译器里的角色** | **MLIR 路线的前端 + 全链路分段教学**（原始模型 → L1 前端图 → L2 StableHLO → L3 Linalg/OSB → L4 Loop/Vector/LLVM） | **从 L3 Linalg 起的后端 lowering 实战**（从 **`linalg` on `tensor`** 起，一路到 **LLVM / RISC-V 可执行**） |
| **你主要练什么** | 模型怎么进 IR、图怎么规范化到 StableHLO、怎么降到 Linalg/Loop/LLVM，各层 Pass **叫什么、顺序如何** | 怎么用 **`mlir-opt` / `mlir-translate` / `llc`** 把 **已有 `.mlir`** 降到 **真实 `.ll` 与二进制** |
| **IR 形态与进入方式** | **多轨并存**：① 原始输入 — **P1** `.onnx` 解析；② **L1** — **P2/P3** `mini_ir` 图 Pass；③ **L2** — **P4/P5** ONNX/Torch→StableHLO、真实 `stablehlo`（`6`+`mlir-opt`）与 **`shlo_graph`**（`7`）；④ **L3–L4** — **P6–P9** 等 `*_ir` + `run_*`（按 **Px Step** 打印 Pass 链，**不是** `mlir-opt` 跑全程） | **Torch → L3 Linalg**（`matmul.py`）或 **`matmul_l3_*.mlir`** + 手工 **`mlir-opt` 命令链** |
| **是否进 CMake** | **是**（`add_subdirectory(mlir/gpu)`） | **否**（无 `cpu/CMakeLists.txt`，避免绑死本机 MLIR 路径） |
| **MLIR 与上游工具** | **P5（`6`）** 真实 StableHLO + `mlir-opt`；**P6–P12** 多为 `*_ir` 教学二进制；**P1–P4** Protobuf / mini_ir / 手写 StableHLO | **全程真实 `.mlir` + `mlir-opt` / `mlir-translate` / `llc`**，可到 **QEMU/RISC-V** |
| **典型产物** | `run_*` 输出；**P5** 可生成 **`conv_bn_*.mlir`** | **`matmul_l3_linalg_tensor.mlir` → … → `matmul.ll`**（仓库内） |
| **与硬件的关系** | 目录名 **gpu** 来自业界「AI/加速器编译器」习惯叫法；**不向 GPU 设备下发 kernel**（**P10** 只讲 PTX/线程映射 **概念**） | **cpu** 指 **通用处理器 codegen 栈**（含 **RISC-V**），不是「只能在笔记本 CPU 上演示」 |

**一句话：** `gpu/` = **把 MLIR 系 AI 编译器从前到后拆开学**；`cpu/` = **假定你已站在 L3 Linalg，专注把 lowering 跑到机器码**。

---

### 分层一览（L1–L4）

`ONNX GraphProto` / 原始 TorchScript 属于**模型交换格式或框架导出物**，可作为入口材料；下面的 **L1–L4** 才是本文统一使用的编译器抽象层级。

| 层级 | 称谓 | 典型 IR / 形态 | 核心职责 | `gpu/` 本仓库 | `cpu/` 本仓库 |
|------|------|----------------|----------|---------------|---------------|
| **入口** | 原始 / 交换图（尚未算作编译器分层） | **ONNX `GraphProto`**（Protobuf）、原始 TorchScript 等 | 读取模型结构、权重、框架导出信息 | `1_onnx_parse/`：只 **读模型结构**，不做图优化 | **不涉及**（无 ONNX 解析示例） |
| **L1** | **Frontend Graph Layer**（前端图层） | **ONNX Dialect / `mini_ir`**、Torch Dialect 等 | 承接框架图、处理拓扑与框架元数据，并向通用 IR 收拢 | `2`–`3`：Protobuf → **mini_ir** → 图 Pass | 多为 **Torch → Linalg** 一跳到 L3，不展开 L1 文档与脚本 |
| **L2** | **Tensor Operator Layer / High-Level Math Layer**（张量算子层 / 高级数学层） | **StableHLO Dialect**（以及同级高层张量数学 IR） | 硬件无关的数学执行语义：形状推导、布局传播、常量折叠、图级融合等 | `4`/`5`/`6`/`7`：Torch/ONNX→StableHLO、StableHLO Pass / 优化练习 | 不直接产出 StableHLO；概念上可与 `gpu/` 在 L2→L3 处对接 |
| **L3** | **Structured Op & Memory Layer**（结构化算子与内存层） | **`linalg` on `tensor` → OSB → `linalg` on `memref`** | 结构化算子、tiling/fusion、bufferization、别名与 in-place 决策 | `8`：Linalg tiling/fusion；`9`：**OSB**（产物仍属 L3） | `matmul_l3_linalg_tensor.mlir` / `matmul_l3_linalg_generic.mlir` / `matmul_l3_linalg_memref.mlir` |
| **L4** | **Kernel Loop & Vector Layer**（内核循环与向量层） | **`scf` / `affine` + `memref.load/store`**、`vector`，再接 LLVM Dialect / LLVM IR / 机器码 | 显式循环控制流、向量化、并行映射、后端出码 | `10`–`12`：循环、向量、LLVM；`13`：GPU 映射（概念） | `matmul_llvm.mlir`、`matmul.ll`、`llc`/`clang`、可选 QEMU |

**进入 L4 的判据：** 算子已通过 **`linalg-to-loops`（或同类 lowering Pass）** 展开为 **显式控制流 + 逐元素/逐块访存**，而不是「类型里出现了 `memref`」。

---

### StableHLO 为什么放在 L2

虽然 StableHLO 在 MLIR 内部也以 SSA / 数据流图组织，但它和 ONNX 这种“前端图”有本质区别：

| **特性** | **L1：前端图层（如 ONNX / Torch / mini_ir）** | **L2：高级张量算子层（StableHLO）** |
|----------|-----------------------------------------------|-------------------------------------|
| **核心关注点** | **模型结构与框架元数据**（topology & metadata） | **纯粹的数学执行语义**（mathematical semantics） |
| **算子纯度** | 可能混杂训练/推理开关、框架特有 layout 标记、导出遗留节点等 | 更接近**纯函数式**张量计算：输入 tensor、输出 tensor，强调数学语义 |
| **类型系统** | 允许较多前端动态性或框架侧约定，shape 信息可能仍在补全 | **强类型**；rank、element type、shape 约束必须能被 verifier / shape 推导严谨处理 |
| **状态表达** | 可能保留权重、initializer、框架对象等模型状态 | 权重通常已**常数化**或作为函数参数进入，只看张量数据流 |

一句话：**ONNX 叫“图”，是因为它承载神经网络模型拓扑；StableHLO 叫“高级算子 IR”，是因为它已经剥离框架外衣，成为一组针对多维张量的通用数学函数。**

业界常把 StableHLO 这一层称为：

- **High-Level Operator IR**：处于算子生命周期的高层，尚未被 lowering 成循环。
- **Tensor-Level IR**：输入输出仍是抽象 `tensor`，不是物理内存 `memref` 或标量循环。
- **Hardware-Agnostic Mathematical IR**：只描述数学逻辑，不绑定 CPU/GPU、线程、cache、寄存器等硬件细节。

---

### L3 的两种形态：`tensor` 与 `memref`（都在结构化算子层）

| | **`linalg` on `tensor`** | **`linalg` on `memref`（OSB 之后）** |
|---|--------------------------|--------------------------------------|
| **层级** | **L3** | **仍是 L3**（结构化算子级） |
| **语义** | 值语义、无固定缓冲区 | 存储已落地，但 **仍是 `linalg.matmul` / `linalg.generic` 等算子** |
| **典型优化** | Tiling、Fusion、Genericization | 可在 memref 上继续做 **结构化算子级** 融合/切片；**别名与 in-place** 在 OSB 阶段决定 |
| **何时离开 L3** | 经 **`linalg-to-loops` 等 lowering** → **L4**（`scf.for` + `load`/`store`） | 同上；**不是 bufferize 本身** |

**OSB（One-Shot Bufferize）** 在分层上是 **L3 内的类型落地 Pass**（`tensor`→`memref`），**不是单独一层「L3.5」**；本仓 `9_bufferize/` 单独成阶段，是为了 **把缓冲区规划讲清楚**，不代表结构化算子层结束。

**与 L4 里 `memref` 的区别：**

| | **L3：`linalg` on `memref`** | **L4：循环 + `memref` 访存** |
|---|------------------------------|-------------------------------|
| **结构** | 算子节点（如 `linalg.generic`） | **`scf.for` / `affine.for` 嵌套** + `memref.load`/`store` 或 vector transfer |
| **优化目标** | 算子融合、tile 形状、缓冲区复用 | 循环交换、展开、并行化、向量化、线程映射 |
| **本仓目录** | `8`（tensor）、`9`（OSB，仍属 L3 语境） | `10`、`11`、`12` |

---

### 三个易混编号（与 L1–L4 无关）

| 说法 | 含义 |
|------|------|
| **Px（`gpu/` 标准阶段）** | **P1** 解析 → **P2** mini_ir → **P3** 图优化 → **P4** ONNX→StableHLO（tier 1/2/3）→ **P5** StableHLO（`6`+`7`）→ **P6–P12** 见下表 |
| **`5_onnx_to_stablehlo/` 的 tier 1/2/3** | **P4 内部难度**，不是编译器 L1/L2/L3 分层 |
| **`4_torch_to_stablehlo/`** | **非标准 Px**：torch-mlir 导出 StableHLO，供 **P5** 使用 |
| **业界「L1–L4」** | 本文表格中的 **编译器抽象层级** |

**ONNX 在本仓：**

1. **入口 — `GraphProto`（Protobuf）**：`1_onnx_parse/`，**原生交换图**，尚未进入本文 L1–L4。
2. **L1 — `mini_ir`**：`2_onnx_to_ir/`、`3_graph_optimize/`，对标 **ONNX MLIR Dialect / Torch Dialect** 的前端图语义（本仓用自研 IR 模拟，未链完整 `onnx-mlir`）。
3. **L2 — StableHLO**：**P4**/`5`、`6`+`7`（**P5**）；降到 **`linalg` on `tensor` 才是 L3**。

---

### `gpu/` Px 速查（P1–P12）

完整命令见 [根 README](../../README.md)。各目录内 **子 Pass 链** 在源码注释中记为 **「Px Step N」**（勿与目录 Px 混淆）。

| Px | 目录 | 分层 | 典型 target |
|----|------|------|-------------|
| **P1** | `1_onnx_parse/` | 入口（原始交换图） | `run_graph` |
| **P2** | `2_onnx_to_ir/` | **L1**（mini_ir） | `run_graph` |
| **P3** | `3_graph_optimize/` | **L1** | `run_graph` |
| — | `4_torch_to_stablehlo/` | 辅助 **P5 / L2** | — |
| **P4** | `5_onnx_to_stablehlo/` | 入口/L1→**L2** | `run_lowering` |
| **P5** | `6_stablehlo_passes/` + `7_stablehlo_opt/` | **L2** | `conv_bn_optimized` / `run_shlo_opt` |
| **P6** | `8_linalg_opt/` | **L3**（`linalg` on `tensor`） | `run_linalg` |
| **P7** | `9_bufferize/` | **L3**（OSB） | `run_buf` |
| **P8** | `10_scf_affine/` + `11_vector/` | **L4** | `run_scf` / `run_vec` |
| **P9** | `12_llvm_lowering/` | **L4** 后端出码 | `run_llvm_lower` |
| **P10** | `13_gpu_codegen/` | **L4** GPU 映射概念 | `run_gpu` |
| **P11** | `14_quantization/` | 横切 | `run_quant` |
| **P12** | `15_memory_planning/` | 横切 | `run_memplan` |

**P6–P12** 多为 **C++ 教学模拟**；**P5（`6_stablehlo_passes`）** 为默认真 **`mlir-opt`** 路径。面试细节见 [`gpu/docs/AI_COMPILER_INTERVIEW.md`](gpu/docs/AI_COMPILER_INTERVIEW.md)；学习主文档见 [`gpu/docs/两仓库学习路径与代码导读.md`](gpu/docs/两仓库学习路径与代码导读.md)。

---

### 两条线何时「汇合」？

1. **L1 → L2**：ONNX / Torch / mini_ir 等前端图规范化为 **StableHLO** 这类高层张量数学 IR。
2. **L2 → L3 汇合点**：StableHLO 经 legalization / conversion 到 **`linalg` on `tensor`**；`cpu/` 的 Torch 入口则直接导出 L3 Linalg。
3. **L3 内部**：Linalg 做 tiling/fusion/genericization；OSB 把 `tensor` 落到 `memref`，但仍是 **L3 结构化算子层**。
4. **L3 → L4**：须经 **`linalg-to-loops`**（及后续循环/向量 Pass）；`cpu/` matmul 文档逐步对应，`gpu/` 在 **P8**（`10`/`11`）展开。
5. **L4 分叉**：接 GPU 时增加 **block/thread、shared mem、launch**（`13`）；`cpu/` 走 **RISC-V `llc`**，**命令不通用，Pass 思想通用**。
6. **仓库现状**：**未** 提供一条 shell 把 `gpu/` 全阶段与 `cpu/` matmul **串成一键**；需在 **L3** 处 **概念对接** 后分别练习。

**关于 TensorIR：** 属 **TVM** 生态（TIR），**不是** MLIR 栈必选层。本文的 L4 对应 **`scf`/`affine`/`vector` + memref 访存**，后半段继续接 **LLVM Dialect / LLVM IR / 机器码**。

---

### L3→L4 Pass 速查（与 `cpu/` matmul 对齐）

| 次序 | 名称 | 所在层 | 说明 |
|------|------|--------|------|
| ① 可选 | **Tiling / Fusion** | **L3**（多为 `tensor`，也可在 memref 算子上） | 结构化算子级局部性 |
| ② | **OSB（Bufferize）** | **L3 内 Pass** | `tensor`→`memref`，**仍保留 `linalg` 算子** |
| ③ | **`linalg-to-loops` 等** | **L3→L4 边界** | 生成 **`scf` + 显式访存** |
| ④ 可选 | **Affine / 循环变换** | **L4** | 交换、融合、并行化 |
| ⑤ 可选 | **Vector** | **L4** | SIMD；再 **`convert-vector-to-llvm`** 进入 LLVM 后端 |

`cpu/` 主线：**L3 tiling/genericize（tensor）→ OSB（仍 L3）→ linalg-to-loops（进 L4）→ vector/并行 → LLVM Dialect**。详见 [cpu/README.md](cpu/README.md) §1.1。

---

### 工具分工（主要服务于 `cpu/` 实战）

- **`mlir-opt`**：L3–L4 的 MLIR Pass 编排（含 bufferize、展循环、降到 LLVM Dialect）。
- **`mlir-translate --mlir-to-llvmir`**：MLIR → **`.ll`**。
- **`llvm-opt`**：只优化 **已是 LLVM IR** 的文件，**不能替代** MLIR lowering。

**P6–P12** **不依赖** 本机 `mlir-opt` 即可运行；要与 `cpu/` **对照**，再在本地对 `cpu/` 的 `.mlir` 跑同一类 Pass 名。

---

## 根目录 CMake

- `find_package(MLIR)` **成功** → `add_subdirectory(mlir/gpu)`。
- **`mlir/cpu`** 不加入构建。
- **`gpu/`**：**P1–P4** 需 **Protobuf**；**P5（`6`）** 需 **MLIR + StableHLO**；**P5（`7`）–P12** 无外部 MLIR 依赖。

## 运行 `gpu/`

```bash
cmake --build . --target run_graph      # P1–P3
cmake --build . --target run_lowering   # P4
cmake --build . --target run_shlo_opt   # P5 (7_stablehlo_opt)
cmake --build . --target run_linalg     # P6
cmake --build . --target run_buf        # P7
cmake --build . --target run_scf        # P8 (10_scf_affine)
cmake --build . --target run_vec        # P8 (11_vector)
cmake --build . --target run_llvm_lower # P9
cmake --build . --target run_gpu        # P10
cmake --build . --target conv_bn_optimized   # P5 (6_stablehlo_passes, 需 MLIR)
```

更多：`make run DOMAIN=mlir PASS=...`（根 README）。

## 运行 `cpu/`

1. [cpu/README.md](cpu/README.md)：**环境准备** → **二、流水线**（`src/mlir/cpu` 下执行）。
2. 可选 **QEMU** 得 `riscv_run.txt`。
3. 只读 IR：直接打开 `matmul_l3_*.mlir`、`matmul_llvm.mlir`、`matmul.ll`，**无需** 构建 `gpu`。

## 命名小结

- **`gpu/`**：**MLIR 路线 AI 编译器前端 + P1–P12 分段教程**（原始输入 → L4）；**不是**「必须在 GPU 上跑」。
- **`cpu/`**：**L3 起的后端 lowering 手工实验**（真实 IR + RISC-V）；**不是**「只能 x86 笔记本」。
