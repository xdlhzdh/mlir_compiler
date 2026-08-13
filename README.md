# MLIR Compiler

本仓库两条主线：

1. **传统编译器**：解释器 V1–V4、ANTLR、NFA/DFA、LLVM IR Pass
2. **AI 编译器教学链路**（`src/mlir/gpu/` P1–P14）：每个阶段一个可执行文件，在自定义 C++ IR 上演示对应概念，输出以 stdout 为主

本仓库不链接 MLIR。与兄弟仓的交接见文末 [与 mlir_pass 的接口](#mlir-pass-interface)。

```
src/
├── mlir/
│   ├── cpu/    # CPU/RISC-V 手工 lowering 流水线（独立构建，见 cpu/README.md）
│   └── gpu/    # AI 编译器 P1–P14 教学全链路（本仓库 CMake 构建）
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

这会构建无需外部依赖的阶段（P5–P14，纯 C++17）。P1–P4 需要 Protobuf，见下一节。未安装的可选依赖会在 cmake 阶段以 STATUS 提示自动跳过（如 `ANTLR4 not found`、`GTest not found`），不影响基础构建。

### 完整构建（可选依赖）


| 可选依赖                        | 启用的阶段                                                                            | 安装 / 配置方式                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **ANTLR4 C++ runtime**      | `run_interpreter_antlr`（ANTLR 解析器）                                               | Ubuntu: `sudo apt install libantlr4-runtime-dev`；Fedora: `sudo dnf install antlr4-cpp-runtime-devel`   |
| **GTest**                   | `tests/` 单元测试（V1–V4、Pass UT）                                                     | Ubuntu: `sudo apt install libgtest-dev`；Fedora: `sudo dnf install gtest-devel`                         |
| **Protobuf**                | P1–P3（ONNX 解析/前端图优化）、P4（ONNX→StableHLO）                                          | Ubuntu: `sudo apt install libprotobuf-dev protobuf-compiler`；Fedora: `sudo dnf install protobuf-devel` |
| **Python numpy + onnx**     | 生成测试 `.onnx` 的脚本（`gen_test_models.py`、`gen_lowering_models.py` 等）                | `pip install --break-system-packages numpy onnx`（须系统级安装，不可仅装在 venv 中）                                  |
| **Python onnxruntime**      | 仅 `run_onnx_golden`：CMake 调用 `python3 run_onnx_golden.py`。本仓 C++ **不链接、不调用** ORT | 使 `python3 -c "import onnxruntime"` 成功即可。Ubuntu: `sudo apt install python3-onnxruntime`                |
| **LLVM**（`llvm-config`） | `src/pass/` LLVM IR Pass 插件（SimplePass 等）                                        | 安装 LLVM 后 `export PATH="$LLVM_INSTALL_PREFIX/bin:$PATH"`，使 `llvm-config`、`opt` 可用                      |


> 真实 MLIR Pass / JIT 在 [mlir_pass](../mlir_pass/)。本仓不链接 MLIR 运行库。P4 `--mlir-only` 是唯一写给兄弟仓的产物（标准 StableHLO 文本）。

---



## AI 编译器 Pipeline 总览（`src/mlir/gpu/`）

P1–P14 各自独立：读入本阶段的图或 `.onnx`，在自定义 C++ IR 上跑完演示，把结果打到 stdout。阶段之间**不通过文件互相消费**（唯一例外：P4 可选导出 `.mlir` 给 `mlir_pass`）。分层命名以 [src/mlir/README.md](src/mlir/README.md) 为准。


| 层级       | 名称           | 对应 Px   | 本阶段产物                                                         |
| -------- | ------------ | ------- | ------------------------------------------------------------- |
| **入口**   | ONNX 交换图     | P1      | 解析后的 GraphProto 打印                                            |
| **L1**   | 前端图          | P2–P3   | 自定义 `mini_ir`                                                 |
| **L2**   | 张量算子         | P4、P5   | P4：StableHLO 文本；P5：自定义 `shlo_graph`                           |
| **L3**   | 结构化算子 / 内存   | P6–P7   | 自定义 Linalg / buffer IR                                        |
| **L4**   | 循环 / 向量 / 后端 | P8–P11  | 自定义 loop / vector / LLVM / GPU 概念 IR                          |
| **独立专题** | 量化、内存规划、图分区  | P12–P14 | 各自的 `quant_ir` / `memory_ir` / `graph_partition_ir`，不接到 L4 之后 |


```
P1–P3   ONNX → mini_ir → 图优化                         产物：stdout
P4      ONNX → StableHLO 文本                            产物：stdout；--mlir-only 可重定向成 .mlir
P5      自定义 shlo_graph 上的图优化                      产物：stdout
P6–P11  Linalg → Bufferize → SCF/Affine → Vector → LLVM → GPU  产物：stdout
P12–P14 量化 / 内存规划 / 图分区                          产物：stdout
```



### 阶段目录与依赖


| 目录                     | Px      | 说明                                                                                                                      | 依赖       | make target                |
| ---------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------- | -------- | -------------------------- |
| `common/`              | —       | ONNX Protobuf 绑定、mini IR 定义、测试模型生成                                                                                      | Protobuf | —                          |
| `1_onnx_parse/`        | **P1**  | ONNX 模型解析：遍历 GraphProto/NodeProto/TensorProto + Shape 推断                                                                | Protobuf | `run_graph`                |
| `2_onnx_to_ir/`        | **P2**  | ONNX → 自定义 IR Lowering：Add→ir.add, MatMul→ir.dot_general, Conv→ir.convolution                                           | Protobuf | `run_graph`                |
| `3_graph_optimize/`    | **P3**  | 图级优化：Conv+BN Fusion、Transpose 消除、常量折叠                                                                                   | Protobuf | `run_graph`                |
| `4_onnx_to_stablehlo/` | **P4**  | ONNX → StableHLO 三级 Lowering（tier 1 基础 / tier 2 broadcast+dynamic / tier 3 conversion framework；整体目标是 **L2 StableHLO**） | Protobuf | `run_lowering`             |
| `5_stablehlo_opt/`     | **P5**  | StableHLO 6-Step 优化：Canon→Shape→Graph(CSE/DCE/Fusion)→Layout→Clean→Legal(→Linalg)                                       | **无**    | `run_shlo_opt`             |
| `6_linalg_opt/`        | **P6**  | Linalg 7-Step Fusion：依赖分析→候选构建→代价模型→Tile-aware→融合重写→清理                                                                  | **无**    | `run_linalg`               |
| `7_bufferize/`         | **P7**  | One-Shot Bufferization 8-Step：别名分析→In-place→WAR/RAW 冲突→Copy 插入→Dealloc                                                  | **无**    | `run_buf`                  |
| `8_scf_affine/`        | **P8**  | SCF/Affine Loop 9-Step：Linalg→Loop→Tiling→Interchange→Fusion→Parallel→向量化                                               | **无**    | `run_scf`                  |
| `9_vector/`            | **P9**  | Vector Dialect 6-Step：前置检查→Core(transfer_read/write/fma)→Register Blocking→LLVM                                         | **无**    | `run_vec`                  |
| `10_llvm_lowering/`    | **P10** | LLVM Backend 7-Step：Vector Lowering→LLVM IR→ISel→RegAlloc→Scheduling→Machine Code                                       | **无**    | `run_llvm_lower`           |
| `11_gpu_codegen/`      | **P11** | GPU Codegen 7-Step：并行检测→Thread Mapping→GPU Dialect→Shared Mem→NVVM→PTX→Occupancy                                        | **无**    | `run_gpu`                  |
| `12_quantization/`     | **P12** | 量化：校准 → scale → qlinear 改写 → 混合精度（自定义 `quant_ir`）                                                                       | **无**    | `run_quant`                |
| `13_memory_planning/`  | **P13** | 内存规划 + KVCache decode 峰值估算（自定义 `memory_ir`）                                                                             | **无**    | `run_memplan`              |
| `14_graph_partition/`  | **P14** | Transformer Attention/FFN 切分与通信量估算（自定义 `graph_partition_ir`）                                                            | **无**    | `run_graph_partition_demo` |


---



## 运行命令参考

所有命令在 **build 目录**下执行（或从项目根目录使用 `cmake --build build --target <target>`）。

### AI 编译器各阶段运行



#### 快速入门：逐阶段运行

```bash
cd build

# ── 只生成 .onnx（可选；run_graph / run_lowering 会自动依赖）──
make gen_test_models        # P1–P3 → build/src/mlir/gpu/models/
make gen_lowering_models    # P4    → build/src/mlir/gpu/lowering_models/

# ── 前端（需 Protobuf + numpy + onnx）──
make run_graph              # P1+P2+P3: 生成模型 → 解析 → IR Lowering → 图优化
make run_lowering           # P4 教学演示（stdout 带 banner，不是 e2e 输入）
make run_onnx_golden        # P4 fixture：ORT vs NumPy（不跑 C++）

# ── L2 StableHLO 优化（纯 C++，无需任何依赖）──
make run_shlo_opt           # P5: StableHLO 6-Step 优化 pipeline

# ── L3 Linalg 结构化算子（纯 C++）──
make run_linalg             # P6: L3 Linalg Tensor 级 Tiling & Fusion pipeline
make run_buf                # P7: One-Shot Bufferization pipeline

# ── L4 后端（纯 C++）──
make run_scf                # P8: SCF/Affine Loop 级优化 pipeline
make run_vec                # P9: Vector Dialect 向量化 pipeline
make run_llvm_lower         # P10: Vector → LLVM Backend (ISel/RegAlloc/Sched/Emit)

# ── L4 GPU 映射（纯 C++）──
make run_gpu                # P11: GPU Code Generation (→ NVVM → PTX → Occupancy)

# ── 独立专题（横切，不是 P11 之后的下一层）──
make run_quant              # P12: 图级量化前端（校准 / QDQ → qlinear），不读 P11 产物
make run_memplan            # P13: 编译期内存规划（对标 bufferize 附近，不读 P11 产物）
make run_graph_partition_demo  # P14: 图级切分（Attention/FFN + 通信量），不读 P11 产物
```



#### 通过 DOMAIN/PASS 参数运行

可通过 `PASS` 参数选择特定阶段：

```bash
# 运行主要 AI 编译器阶段（P1–P13 所有已构建的 target；P14 单独见 run_graph_partition_demo）
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
make run DOMAIN=mlir PASS=vector          # P9
make run DOMAIN=mlir PASS=llvm_lower      # P10
make run DOMAIN=mlir PASS=gpu_codegen     # P11
make run DOMAIN=mlir PASS=quantization    # P12
make run DOMAIN=mlir PASS=memplan         # P13
```



#### 完整 make target 与 PASS 参数对照表


| make target                     | 等价 DOMAIN/PASS                  | Px     | 说明                                                                                                |
| ------------------------------- | ------------------------------- | ------ | ------------------------------------------------------------------------------------------------- |
| `make gen_test_models`          | —                               | P1–P3  | **只生成** `models/*.onnx`，不跑解析。`run_graph` 会自动依赖它                                                   |
| `make gen_lowering_models`      | —                               | P4     | **只生成** `lowering_models/*.onnx`。`run_lowering` / `run_onnx_golden` 会自动依赖它                        |
| `make gen_quant_models`         | —                               | P12    | **只生成** `quant_qdq_matmul.onnx`。P12 与 mlir_pass 都会用到，见 [两处用法](#quant-qdq-onnx)                   |
| `make run_graph`                | `DOMAIN=mlir PASS=graph`        | P1–P3  | 先 `gen_test_models`，再依次跑 `run_onnx_parse` → `run_onnx_to_ir` → `run_graph_rewrite`                |
| `make run_lowering`             | `DOMAIN=mlir PASS=lowering`     | P4     | 先生成 `.onnx`，再依次跑 `l1`/`l2`/`l3` **教学演示**（stdout 含 banner）。**不**带 `--mlir-only`，产物**不**给 e2e        |
| `make run_onnx_golden`          | —                               | P4     | Python：ORT vs NumPy 校验 `.onnx` fixture。不编译、不跑本仓 C++。**不**给 e2e                                    |
| `make run_calib_to_quant`       | —                               | P12    | 校准 JSON → `run_quantization` Stage 2 教学串联                                                         |
| `make run_shlo_opt`             | `DOMAIN=mlir PASS=shlo_opt`     | P5     | StableHLO 图: 24 ops → Canon/Shape/CSE/DCE/Fusion/Layout/Legal → 7 ops                             |
| `make run_linalg`               | `DOMAIN=mlir PASS=linalg`       | P6     | GEMM+Bias+ReLU: 11 ops → 依赖分析/代价模型/Tile-and-Fuse → 5 ops                                          |
| `make run_buf`                  | `DOMAIN=mlir PASS=bufferize`    | P7     | tensor→memref: 7×8KB=57KB → 别名/In-place → 2×8KB=16KB(71%↓)                                        |
| `make run_scf`                  | `DOMAIN=mlir PASS=scf_affine`   | P8     | GEMM+Bias+ReLU: Linalg→scf.for→Tiling→Interchange→Fusion→Parallel→Vec                             |
| `make run_vec`                  | `DOMAIN=mlir PASS=vector`       | P9     | 标量循环→vector.transfer_read/write/fma, 4-row register blocking, tail masking                        |
| `make run_llvm_lower`           | `DOMAIN=mlir PASS=llvm_lower`   | P10    | Vector→LLVM IR→ISel(vfmadd231ps)→RegAlloc(12/16 YMM, 0 spill)→Sched→MachineCode                   |
| `make run_gpu`                  | `DOMAIN=mlir PASS=gpu_codegen`  | P11    | GEMM: 并行检测→GPU grid/block 映射→Shared mem tiling→NVVM→PTX→Occupancy 分析                              |
| `make run_quant`                | `DOMAIN=mlir PASS=quantization` | P12    | ResNet block: 校准→Scale→Fusion→**真实 Q/DQ 图改写**（`quantize`/`dequantize`/`qlinear_`*）→INT8/FP16 混合精度 |
| `make run_quant_qdq`            | —                               | P12    | ONNX QDQ 导入（`quant_qdq_matmul.onnx`）→ `qlinear_matmul` 改写                                         |
| `make run_memplan`              | `DOMAIN=mlir PASS=memplan`      | P13    | ResNet block + KVCache decode: Liveness→干涉图→Offset 分配→Buffer 复用→In-place→峰值分析                     |
| `make run_graph_partition_demo` | —                               | P14    | 单层 Transformer：Attention/FFN 切分，stdout 打印 boundary 与通信量                                          |
| `make run_mlir`                 | `DOMAIN=mlir`                   | P1–P13 | 依次运行已构建的 P1–P13 target；缺少可选依赖的 target 自动跳过                                                     |




<a id="onnx-fixtures"></a>

#### 测试 ONNX 怎么单独生成



P1–P4 读的 `.onnx` **不进 git**，由 Python 脚本在 `build/` 下写出。三个 CMake custom target 只负责生成文件，不跑 C++：


| CMake target          | 脚本                                                        | 写出目录（相对仓库根）                                                   | 谁消费                       |
| --------------------- | --------------------------------------------------------- | ------------------------------------------------------------- | ------------------------- |
| `gen_test_models`     | `src/mlir/gpu/common/gen_test_models.py`                  | `build/src/mlir/gpu/models/`（4 个）                             | P1–P3                     |
| `gen_lowering_models` | `src/mlir/gpu/4_onnx_to_stablehlo/gen_lowering_models.py` | `build/src/mlir/gpu/lowering_models/`（22 个 `lowering_*.onnx`） | P4                        |
| `gen_quant_models`    | `src/mlir/gpu/12_quantization/gen_quant_models.py`        | `build/src/mlir/gpu/quant_models/`（`quant_qdq_matmul.onnx`）   | P12 改写量化算子；mlir_pass 经 P4 导出后识别标注 |


<a id="quant-qdq-onnx"></a>

同一份 `quant_qdq_matmul.onnx` 两处用法不同。P12 `run_quant_qdq` 直接读 ONNX，把 `DQ→MatMul→DQ` 折成 `qlinear_matmul`。mlir_pass 不读这份 ONNX：先经 P4 `--mlir-only` 变成 float StableHLO，再由 `qdq-legalize` 识别子图并打 `aicom.qdq_matmul_canonicalized`。


```bash
# 在仓库根目录；需系统 Python 能 import numpy、onnx（见上方依赖表）
cmake --build build --target gen_test_models gen_lowering_models gen_quant_models

# 等价：不经过 CMake，直接调脚本（目录需自己 mkdir）
python3 src/mlir/gpu/common/gen_test_models.py \
  build/src/mlir/gpu/models
python3 src/mlir/gpu/4_onnx_to_stablehlo/gen_lowering_models.py \
  build/src/mlir/gpu/lowering_models
python3 src/mlir/gpu/12_quantization/gen_quant_models.py \
  build/src/mlir/gpu/quant_models
```

`run_graph` / `run_lowering` / `run_onnx_golden` 的 `DEPENDS` 会**自动先跑**对应的 `gen_`*，所以只 `make run_graph` 也能得到模型。若要自己喂给二进制、或给 `mlir_pass` e2e 准备输入，就单独跑上面的 `gen_*`。

CMake 里两类名字不要混：


| 名字                                                   | 类型                  | `cmake --build build --target <名>` 实际做什么 |
| ---------------------------------------------------- | ------------------- | ---------------------------------------- |
| `gen_*`、`run_graph`、`run_lowering`、`run_onnx_golden` | `add_custom_target` | **执行**（生成文件或跑 pipeline）                  |
| `run_onnx_parse`、`run_onnx_to_ir`、`run_graph_rewrite`、`run_lowering_l1` / `_l2` / `_l3`   | `add_executable`    | **只编译**该二进制，不跑、也不传 `--mlir-only`         |




#### 从项目根目录运行

```bash
# 使用 cmake --build
cmake --build build --target run_shlo_opt       # 运行 P5
cmake --build build --target run_gpu             # 运行 P11

# 传递 DOMAIN/PASS（注意 -- 后传参数）
cmake --build build --target run -- DOMAIN=mlir PASS=linalg
cmake --build build --target run -- DOMAIN=mlir PASS=gpu_codegen
```

---



## 各阶段详细说明

P1–P4 读 `gen_*` 生成的 `.onnx`（[怎么生成](#onnx-fixtures)）。P5–P14 用源码里的图。各阶段输出都是 stdout；唯一写成文件的是 P4 `--mlir-only`（[与 mlir_pass 的接口](#mlir-pass-interface)）。

### P1–P3: ONNX 前端 (`1_onnx_parse/` → `3_graph_optimize/`)

用 Protobuf 解析 ONNX，建成自定义 `mini_ir`，再做 Conv+BN 融合、Transpose 消除、常量折叠。无 `.mlir` 产物。

#### 测试模型：`gen_test_models`

4 个文件由 `common/gen_test_models.py` 用 `onnx.helper` 构图后 `onnx.save`，写出到 `build/src/mlir/gpu/models/`（不进 git）。

```bash
# 推荐：CMake 创建目录并调用脚本
cmake --build build --target gen_test_models

# 等价手工调用
python3 src/mlir/gpu/common/gen_test_models.py build/src/mlir/gpu/models
```


| 文件                | 图内容                       | 谁读                 |
| ----------------- | ------------------------- | ------------------ |
| `add_matmul.onnx` | Add + MatMul              | P1 解析、P2 → mini_ir |
| `conv_bn.onnx`    | Conv + BatchNormalization | P1、P2、P3 fusion    |
| `transpose.onnx`  | Transpose(Transpose(x))   | P3 消除              |
| `const_fold.onnx` | 常量子图                      | P3 常量折叠            |


`make run_graph` 的 `DEPENDS` 含 `gen_test_models`，所以**只跑一键 target 也会先生成这 4 个文件**。单独 `gen_test_models` 的用途：看文件、或自己把路径传给下面的二进制。

#### 怎么跑

编排在 `common/run_graph.cmake.in`：先 P1 两个模型，再 P2 两个模型，再 P3 三个模型。

```bash
# 一键：生成 .onnx → 编译三个二进制 → 按上面顺序执行（stdout）
cmake --build build --target run_graph
# 或在 build/ 下：make run_graph
```

三个可执行文件是 `add_executable`，`cmake --build build --target run_onnx_parse` **只编译不跑**。要单步演示，先确保模型已生成，再直接调二进制：

```bash
cmake --build build --target gen_test_models run_onnx_parse run_onnx_to_ir run_graph_rewrite

./build/src/mlir/gpu/run_onnx_parse \
  build/src/mlir/gpu/models/add_matmul.onnx
./build/src/mlir/gpu/run_onnx_to_ir \
  build/src/mlir/gpu/models/conv_bn.onnx
./build/src/mlir/gpu/run_graph_rewrite \
  build/src/mlir/gpu/models/transpose.onnx
```


| 项目   | 内容                                         |
| ---- | ------------------------------------------ |
| 一键命令 | `make run_graph`                           |
| 输入   | `build/src/mlir/gpu/models/*.onnx`（上表 4 个） |
| 输出   | stdout（GraphProto 遍历、mini_ir、优化后的图）        |
| 目的   | 走完「交换格式 → 编译器内部图 → 图级优化」                   |




### P4: ONNX → StableHLO 三级 Lowering (`4_onnx_to_stablehlo/`)

把 `.onnx` 降成标准 StableHLO 文本（`module { func.func @main ... }`）。实现是本目录**自写的 C++ lowering**（`stablehlo_ir.h` 拼 dialect 语法），只链 Protobuf，**不是**官方 MLIR C++ API / Pass。tier 3 的 `ConversionPattern` / `applyFullConversion` 是教学仿写，不是 `mlir::DialectConversion`。

这里的 **tier 1/2/3** 是 P4 内部难度分级，不是总览里的 L1/L2/L3。三者目标都是同一份 StableHLO 文本，能力递进，不是互相替代。

#### 三条互不消费的路径

同一批 `.onnx` 会走三条独立路径，**彼此不读对方的 stdout**：

```text
lowering_*.onnx / quant_qdq_matmul.onnx
        │
        ├─① make run_lowering        C++ 教学：依次跑 l1 / l2 / l3，stdout 带 banner
        │                            不传 --mlir-only
        │
        ├─② run_onnx_golden          Python：ORT vs NumPy，校验 fixture 与参考实现
        │                            不编译、不调用 run_lowering_l*
        │
        └─③ run_lowering_l3 --mlir-only → 纯 StableHLO .mlir
                                     交给 mlir_pass（对方 ninja test_e2e 消费）
```

跨仓接口只走路径 ③：本仓出 `.mlir`，`mlir_pass` 拿去跑 fusion。路径 ①、② 都不进对方。`make run_lowering` 虽然也跑 L3，但 stdout 带 banner，不能当接口文件。

#### 测试模型：`gen_lowering_models`（+ 一个 P12 文件）

```bash
cmake --build build --target gen_lowering_models gen_quant_models
```


| CMake target          | 写出目录                                  | 内容                                               |
| --------------------- | ------------------------------------- | ------------------------------------------------ |
| `gen_lowering_models` | `build/src/mlir/gpu/lowering_models/` | 22 个 `lowering_*.onnx`（tier 1/2/3 共用）            |
| `gen_quant_models`    | `build/src/mlir/gpu/quant_models/`    | `quant_qdq_matmul.onnx`（工业 QDQ：Q + DQ + MatMul） |


`run_lowering` / `run_onnx_golden` 已 `DEPENDS` 这两个 target，一键跑时会自动生成。给 e2e 准备输入时单独跑 `gen_*` 即可（路径 ③ 的脚本自己读这些目录）。

等价手工调用：

```bash
python3 src/mlir/gpu/4_onnx_to_stablehlo/gen_lowering_models.py \
  build/src/mlir/gpu/lowering_models
python3 src/mlir/gpu/12_quantization/gen_quant_models.py \
  build/src/mlir/gpu/quant_models
```



#### 路径 ①：教学演示 `run_lowering`（CMake 能一键跑）

`run_lowering.cmake.in` 按 tier 调用二进制，**不加** `--mlir-only`，stdout 含阶段标题 + StableHLO 文本。这是本仓的 P4 演示，不是 e2e 的输入。

```bash
cmake --build build --target run_lowering
# 或：make run_lowering / make run DOMAIN=mlir PASS=lowering
```

依赖链：编译 `run_lowering_l1/l2/l3` → `gen_lowering_models` + `gen_quant_models` → 按表执行。


| 二进制                               | cmake `--target`    | 实际做什么                         |
| --------------------------------- | ------------------- | ----------------------------- |
| `run_lowering_l1` / `_l2` / `_l3` | `add_executable`    | **只编译**                       |
| `run_lowering`                    | `add_custom_target` | 生成模型并跑完三个二进制（无 `--mlir-only`） |


单步（先 `gen_lowering_models`，再直接跑二进制）：

```bash
./build/src/mlir/gpu/run_lowering_l1 \
  build/src/mlir/gpu/lowering_models/lowering_basic.onnx
./build/src/mlir/gpu/run_lowering_l3 \
  build/src/mlir/gpu/lowering_models/lowering_softmax.onnx
```


| Tier       | 可执行文件             | `run_lowering` 喂哪些模型                                                     | 覆盖内容                                      |
| ---------- | ----------------- | ------------------------------------------------------------------------ | ----------------------------------------- |
| **tier 1** | `run_lowering_l1` | `lowering_basic` / `_conv` / `_reshape_transpose`                        | Add/MatMul/Conv/Reshape/Transpose         |
| **tier 2** | `run_lowering_l2` | `_broadcast` / `_conv_full` / `_dynamic` / `_dynamic_mn` / `_matmul_f16` | broadcast、dynamic shape、完整 Conv 属性、错误处理   |
| **tier 3** | `run_lowering_l3` | 复跑部分 tier 1/2 图，再跑 Softmax/Attention/RMSNorm/RoPE/GELU/… 和 `quant_qdq_matmul.onnx` | 教学版 conversion framework；接口导出用 `--mlir-only` |




#### 路径 ②：fixture 校验 `run_onnx_golden`（CMake 能一键跑，不碰 C++）

确认这些 `.onnx` 能被 ONNX Runtime 加载，且脚本里的 NumPy 参考实现与 ORT 数值一致。**不验证 lowering 代码**，也不产出 `.mlir`。

```bash
cmake --build build --target run_onnx_golden

# 等价（需已有 lowering_models / quant_models）
python3 src/mlir/gpu/4_onnx_to_stablehlo/run_onnx_golden.py \
  build/src/mlir/gpu/lowering_models \
  --quant-dir build/src/mlir/gpu/quant_models
```

依赖的是**跑脚本的那个 Python**：`numpy`、`onnx`、`onnxruntime`。和本仓 C++ 二进制无关。


| 项目  | 内容                                                 |
| --- | -------------------------------------------------- |
| 命令  | `cmake --build build --target run_onnx_golden`     |
| 输入  | 17 个 `lowering_*.onnx` + `quant_qdq_matmul.onnx`   |
| 做什么 | 每个图：ORT 跑一遍，NumPy 按同一输入再算一遍，`np.allclose`          |
| 输出  | 每项 `PASS <name>`，最后 `All 18 golden checks passed.` |


18 项 = 脚本 `CHECKS`（17）+ `check_quant_qdq_matmul`（1）：


| #   | 函数                        | 输入                                     |
| --- | ------------------------- | -------------------------------------- |
| 1   | `check_basic`             | `lowering_basic.onnx`                  |
| 2   | `check_softmax`           | `lowering_softmax.onnx`                |
| 3   | `check_attention`         | `lowering_attention.onnx`              |
| 4   | `check_rmsnorm`           | `lowering_rmsnorm.onnx`                |
| 5   | `check_layernorm`         | `lowering_layernorm.onnx`              |
| 6   | `check_rope`              | `lowering_rope.onnx`                   |
| 7   | `check_gelu`              | `lowering_gelu.onnx`                   |
| 8   | `check_swiglu`            | `lowering_swiglu.onnx`                 |
| 9   | `check_matmul_bias`       | `lowering_matmul_bias.onnx`            |
| 10  | `check_qdq_matmul`        | `lowering_qdq_matmul.onnx`             |
| 11  | `check_horizontal_gemm`   | `lowering_horizontal_gemm.onnx`        |
| 12  | `check_broadcast`         | `lowering_broadcast.onnx`              |
| 13  | `check_transformer_block` | `lowering_transformer_block.onnx`      |
| 14  | `check_dynamic`           | `lowering_dynamic.onnx`                |
| 15  | `check_decode_step`       | `lowering_decode_step.onnx`            |
| 16  | `check_dynamic_mn`        | `lowering_dynamic_mn.onnx`             |
| 17  | `check_matmul_f16`        | `lowering_matmul_f16.onnx`             |
| 18  | `check_quant_qdq_matmul`  | `quant_qdq_matmul.onnx`（`--quant-dir`） |




<a id="p4-mlir-only"></a>

#### 路径 ③：给 mlir_pass 的接口就是一份 StableHLO 文件

接口命令：

```text
run_lowering_l3 --mlir-only <file.onnx>  >  file.mlir
```

```bash
# 本仓准备二进制和 .onnx（一次即可）
cmake --build build --target run_lowering_l3 gen_lowering_models gen_quant_models

# 对方消费：脚本内部调 --mlir-only，再跑自己的回归
ninja -C ../mlir_pass/build test_e2e
```

只想看某一份 IR：

```bash
./build/src/mlir/gpu/run_lowering_l3 --mlir-only \
  build/src/mlir/gpu/lowering_models/lowering_attention.onnx \
  > attention_p4.mlir
```

`test_e2e` 验什么见 [mlir_pass README · 跨仓 e2e](../mlir_pass/README.md#cross-repo-e2e)。




### P5（`5_stablehlo_opt/`）: StableHLO 6-Step 优化

**运行：** `make run_shlo_opt`


| Step | Pass               | 关键操作                                                                           |
| ---- | ------------------ | ------------------------------------------------------------------------------ |
| 1    | Canonicalization   | `x+0→x`, `x*1→x`, identity reshape/transpose 消除                                |
| 2    | Shape Optimization | shape inference, `get_dimension_size→const`, dynamic→static                    |
| 3    | Graph Optimization | CSE, DCE, 常量折叠, **Conv+BN Fusion**, elementwise chain 融合检测                     |
| 4    | Layout/Transpose   | `T(p2)∘T(p1)→T(compose)`, transpose push-through, **NCHW↔NHWC conv layout 折叠** |
| 5    | Cleanup            | 再跑 canonicalize + DCE                                                          |
| 6    | Legalization       | `stablehlo.*` → `linalg.*` / `arith.*` / `tensor.*`                            |


综合测试：24 ops → 7 ops（所有 pass 同时作用）

### P6: L3 Linalg Tensor 级 Fusion (`6_linalg_opt/`)

**运行：** `make run_linalg`


| Step | Pass                   | 关键概念                                                    |
| ---- | ---------------------- | ------------------------------------------------------- |
| 0    | Pre-clean              | canonicalize + CSE + DCE                                |
| 1    | Dependence Analysis    | SSA use-def 图 + alias check                             |
| 2    | Fusion Candidate Build | producer-consumer 图 + elementwise chain                 |
| 3    | Cost Model Filtering   | 中间 tensor 消除收益 vs 多用户约束                                 |
| 4    | Tile-aware Fusion      | iterator 兼容性：elem+elem=直接, reduction+elem=tile-and-fuse |
| 5    | Fusion Rewrite         | merge indexing maps + merge iterators + tiling          |
| 6    | Post-clean             | canonicalize + CSE + DCE                                |


测试：`matmul → bias_add → relu → square → residual → scaling`，11 ops → 5 ops

### P7: One-Shot Bufferization (`7_bufferize/`)

**运行：** `make run_buf`


| Step | Pass                   | 关键概念                               |
| ---- | ---------------------- | ---------------------------------- |
| 0    | Pre-clean              | CSE + DCE                          |
| 1    | Alias Analysis         | SSA use-def 链 + 潜在 alias group     |
| 2    | In-place Analysis      | WAR(Write-After-Read) 冲突检测         |
| 3    | Bufferization Decision | INPLACE / OUT_OF_PLACE 标记          |
| 4    | Rewrite                | tensor → memref, buffer assignment |
| 5    | Copy Insertion         | 冲突驱动的 memref.copy                  |
| 6    | Buffer Deallocation    | 所有权模型、lifetime 分析                  |
| 7    | Post-clean             | 验证 allocs == deallocs（无泄漏）         |


核心演示：`%biased` 有两个 reader（relu 和 neg），relu in-place 覆写 → neg 需要 copy。7×8KB=57KB → 2×8KB=16KB（**71% 内存减少**）

### P8（`8_scf_affine/`）: SCF/Affine Loop 级优化

**运行：** `make run_scf`


| Step | Pass                  | 关键操作                                                |
| ---- | --------------------- | --------------------------------------------------- |
| 0    | Pre-clean             | 死代码消除                                               |
| 1    | Linalg → Loop         | matmul → 3 层 scf.for + memref.load/store            |
| 2    | Loop Canonicalization | 死循环消除                                               |
| 3    | Tiling                | 多级 cache-aware: [TM=32, TN=8, TK=32], 工作集 ≤ L1 32KB |
| 4    | Loop Transform        | interchange (stride-1 访问) + loop fusion (bias+relu) |
| 5    | Parallelization       | scf.parallel + thread mapping                       |
| 6    | Memory Optimization   | B tile stack promotion + C register promotion       |
| 7    | Vectorization Prep    | `vector<8xf32>` (broadcast + fma)                   |
| 8    | Cleanup               | 统计: loops/ops/parallel/vector                       |


测试：GEMM+Bias+ReLU `A[64×128]×B[128×32]`，8 loops/13 ops → 7 loops/14 ops（2 parallel, 8 vector, 8× SIMD 吞吐）

### P9（`9_vector/`）: Vector Dialect Pipeline

**运行：** `make run_vec`


| Step | Pass                | 关键概念                                               |
| ---- | ------------------- | -------------------------------------------------- |
| 0    | Precondition Check  | canonical loop, stride 分析                          |
| 1    | Vectorization Prep  | alignment 32-byte, trip count = SIMD 倍数            |
| 2    | Vectorization Core  | scf.for → vector.transfer_read/write/fma/broadcast |
| 3    | Vector Shaping      | register blocking (unroll ii×4) + tail masking     |
| 4    | Vector Optimization | load coalescing, small-op fusion, dead elim        |
| 5    | Lowering to LLVM    | vector → llvm.intr.fmuladd (AVX2: vfmadd231ps)     |


核心：4 row × `vector<8xf32>` register blocking → 峰值 16 FLOP/cycle

### P10: LLVM Backend Pipeline (`10_llvm_lowering/`)

**运行：** `make run_llvm_lower`


| Step | Pass                   | 关键操作                                                      |
| ---- | ---------------------- | --------------------------------------------------------- |
| 0    | Vector IR Input        | 接收 P9（`9_vector/`）输出（4-row register-blocked FMA chain）    |
| 1    | Vector Lowering        | memref → GEP, transfer_read → load, fma → @llvm.fma.v8f32 |
| 2    | LLVM IR Generation     | PHI nodes, basic block CFG, SSA                           |
| 3    | Instruction Selection  | DAG pattern match: fma→vfmadd231ps, GEP+load 融合           |
| 4    | Register Allocation    | 12/16 YMM used, **0 spill**, linear scan                  |
| 5    | Instruction Scheduling | port model, load‖broadcast 并行, 13cy→11cy                  |
| 6    | Machine Code Emission  | VEX 3-byte 编码, ~145 bytes, AArch64 NEON/SVE 对比            |


性能：64 FLOP/iter ÷ 11 cy = 5.8 FLOP/cycle

### P11: GPU Code Generation (`11_gpu_codegen/`)

**运行：** `make run_gpu`


| Step | Pass                 | 关键操作                                                                   |
| ---- | -------------------- | ---------------------------------------------------------------------- |
| 0    | Parallel Detection   | linalg.generic iterator_types 分析: parallel → GPU, reduction → 循环       |
| 1    | Thread Mapping       | tile [64,64,32], block (16,16), thread tile 4×4                        |
| 2    | GPU Dialect Emission | gpu.launch_func, gpu.block_id/thread_id, gpu.barrier                   |
| 3    | Shared Memory Tiling | bank conflict 分析, coalescing, double buffering                         |
| 4    | NVVM Lowering        | gpu.block_id → nvvm.read.ptx.sreg.ctaid.x, gpu.barrier → nvvm.barrier0 |
| 5    | PTX Emission         | .entry gemm_kernel, ld.global.v4, fma.rn, bar.sync                     |
| 6    | Occupancy Analysis   | threads/registers/shared memory 三重资源约束, wave 分析                        |


测试：GEMM C[1024,1024] = A[1024,512]·B[512,1024]，shared memory 24KB, occupancy 分析

### P12: Quantization (`12_quantization/`)

目录排在 P11 后面只是教学模块序号，**不是**「GPU codegen 之后再量化」。本阶段在自定义 `quant_ir` 上走量化**前端**（图级）：校准、算 scale、把浮点算子改成 `qlinear_*`、按敏感度选 INT8 / FP16 / FP32。不读 P6–P11 的 IR，也不往 L3/L4 降。工业编译器里这一步通常在 L1/L2 张量图上做；int8 kernel / MMA 才属于后面的 linalg 或 loop 阶段，本仓没有接到那一层。


| 项目  | 内容                                                                                                                                 |
| --- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 命令  | `make run_quant`：内置 Conv+BN+ReLU+MatMul 图走 7 步；`make run_quant_qdq`：读 `quant_qdq_matmul.onnx`，把 `DQ→MatMul→DQ` 折成 `qlinear_matmul` |
| 输入  | 默认：源码硬编码的教学图。`run_quant_qdq`：`quant_models/quant_qdq_matmul.onnx`（`gen_quant_models.py` 生成）                                        |
| 输出  | stdout（节点、scale、混合精度决策）。内存里是 `quant_ir::Graph`                                                                                     |
| 目的  | 演示量化流水线；本仓唯一把 QDQ 改写成量化算子的代码                                                                                                       |


`run_quant` 的 `quant_ir` 不进 `mlir_pass`。


| Step | 做什么                                              |
| ---- | ------------------------------------------------ |
| 0    | 搭建 Conv+BN+ReLU+MatMul 图                         |
| 1    | 校准：min/max / histogram                           |
| 2    | 计算 per-tensor / per-channel scale 与 zero_point   |
| 3    | Conv+BN+ReLU → QLinearConv                       |
| 4    | 插入 quantize/dequantize，Conv/MatMul → `qlinear_*` |
| 5    | 按误差阈值分层：INT8 / FP16 / FP32                       |
| 6    | 打印模型体积与吞吐估算                                      |




### P13: Memory Planning (`13_memory_planning/`)

在自定义 `memory_ir` 上做编译期内存规划：liveness、干涉图、offset、buffer 复用、in-place，并估算 Transformer decode 的 KVCache 峰值字节。


| 项目  | 内容                                                     |
| --- | ------------------------------------------------------ |
| 命令  | `make run_memplan`                                     |
| 输入  | 源码硬编码的 ResNet-like block；Stage 7 用固定 shape 模拟多层 decode |
| 输出  | stdout（live interval、峰值内存、KV bytes）                    |
| 目的  | 在编译期给 tensor 分配 buffer offset，并估算 decode 阶段的 KV 占用     |


`memory_ir` 不进 `mlir_pass`。


| Step | 做什么                                 |
| ---- | ----------------------------------- |
| 0    | 搭建 Conv→BN→ReLU→Conv→Add→ReLU       |
| 1    | 每个 tensor 的 `[first_use, last_use]` |
| 2    | 活跃区间重叠 → 冲突矩阵                       |
| 3    | Best-Fit Decreasing 分配 offset       |
| 4    | 非重叠 tensor 共享 buffer                |
| 5    | elementwise 输出 alias 输入（WAR 检查）     |
| 6    | 峰值、压缩比、碎片率                          |
| 7    | 多层 decode 的 K/V live interval 与峰值字节 |




### P14: Graph Partition (`14_graph_partition/`)

把一层 Transformer 按 hint 切成 Attention / FFN 两个 partition，识别边界张量并估算跨分区通信量。


| 项目  | 内容                                                                                    |
| --- | ------------------------------------------------------------------------------------- |
| 命令  | `cmake --build build --target run_graph_partition_demo`（可执行文件名 `run_graph_partition`） |
| 输入  | 源码硬编码的单层 Transformer 图（固定 shape）                                                      |
| 输出  | stdout，含 `boundary tensors` 与 `Total cross-partition comm bytes: 262144`              |
| 目的  | 演示按 hint 切图，并估算边界张量的通信字节数                                                             |


`graph_partition_ir` 不进 `mlir_pass`。


| Step | 做什么                                |
| ---- | ---------------------------------- |
| 0    | 搭建 Attention + Residual + LN + FFN |
| 1    | `partition_hint`：attention / ffn   |
| 2    | 按 hint 分成两个 partition              |
| 3    | 两边共用的 tensor 记为 boundary，累加字节数     |
| 4    | 打印切分结果与通信量                         |


---



<a id="mlir-pass-interface"></a>

## 与 mlir_pass 的接口



两仓库独立构建，不链接对方的库。P4 不用官方 MLIR C++，但打印的文本按 StableHLO 格式来，所以对方 `parseSourceFile` 能收下。脚本把 `@main` 改成 `@inference`。

```text
.onnx  →  run_lowering_l3 --mlir-only  →  *.mlir  →  mlir_pass
```

```bash
# mlir_compiler
cmake --build build --target run_lowering_l3 gen_lowering_models gen_quant_models

# mlir_pass
ninja -C build test_e2e
```

`test_e2e` 验什么见 [mlir_pass README · 跨仓 e2e](../mlir_pass/README.md#cross-repo-e2e)。`make run_lowering` 的 stdout 和 `run_onnx_golden` 都不进对方。

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

独立于本仓库 CMake 构建。在 `src/mlir/cpu/` 下按 [src/mlir/cpu/README.md](src/mlir/cpu/README.md) 的说明进行环境准备和运行。

---



## 附加文档


| 文件 | 说明 |
| --- | --- |
| [两仓库学习路径与代码导读](src/mlir/gpu/docs/两仓库学习路径与代码导读.md) | 两仓库 AI 编译器学习主文档（路径导读 + 面试边界） |
| [编译器能力映射](src/mlir/gpu/docs/编译器能力映射.md) | 能力边界与工业级缺口对照 |
| [AI 编译器面试题](src/mlir/gpu/docs/AI_COMPILER_INTERVIEW.md) | AI 编译器面试高频问题与答案（30+ 题，覆盖全部 15 个阶段） |
| [src/mlir/README.md](src/mlir/README.md) | cpu/ 与 gpu/ 目录关系说明 |
| [src/mlir/cpu/README.md](src/mlir/cpu/README.md) | CPU/RISC-V 流水线环境与运行说明 |


