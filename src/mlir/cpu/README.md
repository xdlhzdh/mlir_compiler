# `src/mlir/cpu` — PyTorch → MLIR → RISC-V MatMul 端到端

本目录**不参与**仓库根 CMake（无 `CMakeLists.txt`），用于 **自 L3（`linalg` on `tensor`）起的后端 lowering 实战**：真实 `.mlir`、**mlir-opt**、**LLVM / RISC-V** 直至可执行。与 `**src/mlir/gpu/`**（MLIR 路线 **前端 + L1–L4 分段教学**，由 CMake 构建）的分工见 **[../README.md](../README.md)**。

本文档分为两个部分：


| 部分           | 内容                                                                                |
| ------------ | --------------------------------------------------------------------------------- |
| **环境准备**     | 一次性的工具链：LLVM/MLIR、StableHLO（供 `gpu/` L2 / P5）、torch-mlir、可选 CUDA / RISC-V sysroot |
| **代码运行与流水线** | 在 `src/mlir/cpu` 下跑 matmul：导出 IR → `mlir-opt` → LLVM → 仿真                         |


---

## 一、环境准备

> 升级工具链或换机器时必须执行。**跑第二节流水线前，请先保证** `mlir-opt` **等工具就绪。**

### 1.1 总体数据流（L1–L4 与工具链）

与 **[../README.md](../README.md)** 中 **「分层一览」「L3→L4」** 使用同一套术语；本节的每一步对应如下 **lowering 管线**（自 PyTorch 导出起）：

```
PyTorch
  → Torch-MLIR (Torch Dialect)            …… L1 前端图语义（见下「Torch Dialect、StableHLO 与 L1 / L2 / L3」）
  → StableHLO（可选路径）                  …… L2 高级张量算子 / 硬件无关数学 IR
  → linalg on tensor                      …… L3 结构化算子（如 `linalg.matmul`；适合做 tiling / fusion）
  → 规范化（如 generalize / canonicalize） …… L3：仍在结构化算子语义上整理 IR（如 `linalg.generic`）
  → one-shot-bufferize + 缓冲释放相关      …… L3 内 Pass（OSB）：tensor → memref，仍保留 linalg 算子
  → convert-linalg-to-loops + scf/vector (+ 可选 affine) on memref
                                          …… L4：显式循环、向量与并行映射
  → convert-*-to-llvm / vector-to-llvm    …… L4 后半段：LLVM Dialect
  → mlir-translate → LLVM IR (.ll)        …… L4 后端出码：LLVM IR
  → llc / clang → 目标文件 / 链接          …… L4 后端出码：机器码（例：RISC-V + RVV）
  → QEMU / Spike                          …… 仿真或真机
```

**把 CPU 路线拆成 L3 / L4 来看（OSB 仍在 L3，见 [../README.md](../README.md)「L3 的两种形态」）：**


| 阶段                     | 主要动作                                                       | 数据类型                       | 核心关注点                               |
| ---------------------- | ---------------------------------------------------------- | -------------------------- | ----------------------------------- |
| **L3**（`tensor`）       | **Tiling / Fusion / Genericization**（`linalg` on `tensor`） | `tensor`                   | 以**数据局部性**为主，尽量减少不必要的访存             |
| **L3**（OSB 后 `memref`） | **One-Shot Bufferize**                                     | `tensor` → `memref`        | **仍是结构化算子层**：分配、别名、原地更新；**不是**进入 L4 |
| **L4**                 | **Linalg to Loops / Unrolling / Vectorization / Parallel** | `memref` / `vector` + 显式循环 | 面向循环和 SIMD/并行，尽量压满 **Compute/ALU**  |


这不是说每条 pipeline 都必须按同一个命令顺序展开，而是用来理解 **CPU lowering 每一段主要在优化什么**：**L3 侧重结构化算子局部性与缓冲区规划（含 OSB）**，**L4 侧重循环、向量与并行执行效率**。

**Torch Dialect、StableHLO 与 L1 / L2 / L3 的分界（以 [../README.md](../README.md) 为准）**

- **L1 — Frontend Graph Layer**：ONNX Dialect / `mini_ir`、**Torch Dialect** 等，负责承接框架图、拓扑与元数据。
- **L2 — Tensor Operator / High-Level Math Layer**：**StableHLO**，更接近硬件无关的纯数学张量算子 IR，适合形状推导、布局传播、常量折叠、图级融合等。
- **L3 — Structured Op & Memory Layer**：`**linalg` on `tensor` → OSB → `linalg` on `memref`**。**降到 `linalg` on `tensor` 才进入 L3**（**P6** / `8_linalg_opt`）。勿与 **P4 内部的 tier 1/2/3**（难度分级）混淆。
- **Torch Dialect**：**torch-mlir** 表示 **PyTorch 侧算子** 的 MLIR 方言，属于 L1 前端图语义，不是 StableHLO（不同 op 集合与规范：OpenXLA vs PyTorch 映射）。
- **本示例 `matmul.py`**：`output_type="linalg-on-tensors"` 会沿 torch-mlir **尽快落到 L3**，中间可能短暂经过 Torch 相关 IR，**通常不生成** 供对照阅读的 **StableHLO 文本**。要看 **StableHLO 导出 + L2 高级张量算子 Pass**，用 `**src/mlir/gpu/`**（如 `conv_bn_model.py`）。
- **L2 → L3 汇合**：**StableHLO → `linalg` on `tensor`**（也可能先经 **TOSA** 等，视编译器而定），与本文 **Torch → Linalg** **在 L3 汇合**；之后 **OSB（仍 L3）→ `linalg-to-loops`（进 L4）→ LLVM Dialect** 等与 **[../README.md](../README.md)** 中 **「L3→L4 Pass 速查」** 一致。接 GPU 时 **L4 后半段** 仍会再分叉，命令流与 `cpu/` 不完全相同。

**本仓库步骤 ↔ 分层（速查）**


| 步骤（与 §2 命令一致）                                                                                           | 分层                 | 说明                                                                     |
| ------------------------------------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------------- |
| `linalg.matmul` on `tensor`                                                                             | **L3**             | 张量值语义，无显式缓冲区；这一层更适合做 **tiling / fusion / genericization** 一类结构化算子变换    |
| `--linalg-generalize-named-ops`、`--canonicalize`                                                        | **L3**             | 先把 named op 规整为更通用的 `linalg.generic` 形态，便于继续做 L3 侧变换                   |
| `--one-shot-bufferize`、`-buffer-deallocation-pipeline`                                                  | **L3 内 Pass（OSB）** | `tensor` → `memref`；关键在 **分配、别名和原地更新**；**仍保留 `linalg` 算子**             |
| `-convert-linalg-to-loops` 及后续 `convert-*-to-llvm`、`finalize-memref-to-llvm`、`convert-vector-to-llvm` 等 | **L4**             | 在 **memref / vector** 世界展开循环、继续 lowering，并把循环/向量形态对接到 **LLVM Dialect** |
| `mlir-translate --mlir-to-llvmir`                                                                       | **L4 后端出码**        | 输出 `.ll`                                                               |


**Affine / Tiling**：本示例 **未**走完整 **affine 循环**（`affine.for`）管线；若提升为仿射 nest，更利于 **polyhedral** 类 tiling/融合。放到上面的 **L3 / L4** 视角里，**tiling 更偏 L3 的数据局部性优化**，而 **vectorize / unroll / parallel 更偏 L4 的算力调度**。GPU 路径在 L4 **形状类似**（buffer + 循环/调度），但必须再映射 **block/thread、shared memory** 等，**不等价**于本条 RISC-V 命令流。

**工具**：Kernel 与 LLVM Dialect 之前 **主要用 `mlir-opt`**；出 `.ll` 用 `mlir-translate`。`llvm-opt` 只处理 **已是 LLVM IR** 的 `.ll`，与 MLIR 侧 lowering **分工不同**、**不可替代**，可串在 `mlir-translate` 之后。详见 **[../README.md](../README.md)「工具分工」**。

### 1.2 获取源码

直接使用 **torch-mlir 内置的 llvm-project 与 stablehlo 子模块**，版本由 torch-mlir 锁定。

```bash
export WORK_HOME="${WORK_HOME:-/opt}"
export TORCH_MLIR_HOME="${TORCH_MLIR_HOME:-$WORK_HOME/torch-mlir}"

mkdir -p "$WORK_HOME"
cd "$WORK_HOME"

git clone https://github.com/llvm/torch-mlir.git "$TORCH_MLIR_HOME"
git -C "$TORCH_MLIR_HOME" submodule update --init externals/llvm-project externals/stablehlo
```

LLVM 源码与（建议的）构建目录：`$TORCH_MLIR_HOME/externals/llvm-project`、`$TORCH_MLIR_HOME/externals/llvm-project/build`。

### 1.3 编译并安装 LLVM + MLIR（推荐统一前缀）

在 **torch-mlir 的 `externals/llvm-project`** 下构建。

```bash
export LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-$TORCH_MLIR_HOME/externals/llvm-project}"
export LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$LLVM_SOURCE_DIR/build}"

cd "$LLVM_SOURCE_DIR"
mkdir -p build && cd build

sudo apt install -y build-essential cmake ninja-build gcc g++ python3-dev python3-pip python3-venv
command -v cmake > /dev/null && command -v ninja >/dev/null && dpkg -l | grep python3-dev >/dev/null && command -v pip > /dev/null && command -v python3 -m venv >/dev/null && echo "toolchain install succeed!" || echo "toolchain install failed!"

python3 -m venv /opt/venv
source /opt/venv/bin/activate
pip install nanobind
export nanobind_DIR=$(python -m nanobind --cmake-dir)

export CC=gcc
export CXX=g++

export LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-/usr/local}"

cmake -G Ninja ../llvm \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="RISCV;X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX"

ninja
ninja mlir-tblgen mlir-python-sources (可选)
ninja install (安装工具链到PATH以及LLVMConfig.cmake/MLIRConfig.cmake到CMAKE_INSTALL_PREFIX)
```

安装后把工具链放进 `PATH`：

```bash
# export并且写入系统环境变量 /etc/profile.d/llvm-mlir.sh（root 写一次）
export PATH="$LLVM_INSTALL_PREFIX/bin:$PATH"
```

### 1.4 在同一前缀下构建并安装 StableHLO（仅当需要编 P5 插件）

与 **§1.3 相同安装前缀**，形成「带 StableHLO 的一套 LLVM/MLIR」。**StableHLO 用于 `gpu/` 的 L2（P5）等目标**；本目录 **matmul 流水线不经过 StableHLO**，但共用同一套 MLIR 安装前缀（`mlir_compiler` 的 **gpu** 目标亦依赖此布局）。

```bash
export STABLEHLO_INSTALL_PREFIX="${STABLEHLO_INSTALL_PREFIX:-$LLVM_INSTALL_PREFIX}"

cd "$TORCH_MLIR_HOME/externals/stablehlo"
rm -rf build && mkdir build && cd build

cmake -G Ninja -S .. -B . \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DCMAKE_INSTALL_PREFIX="$STABLEHLO_INSTALL_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF

ninja
ninja install
```

**头文件**：上游 StableHLO 常未把 `.h` 写进 `install`，需在 build 目录下**手动拷贝**：

```bash
STABLEHLO_SRC="$TORCH_MLIR_HOME/externals/stablehlo"
STABLEHLO_INSTALL_PREFIX="${STABLEHLO_INSTALL_PREFIX:-/usr/local}"
mkdir -p "$STABLEHLO_INSTALL_PREFIX/include"
cp -r "$STABLEHLO_SRC/stablehlo" "$STABLEHLO_INSTALL_PREFIX/include/"
cp -r stablehlo/* "$STABLEHLO_INSTALL_PREFIX/include/stablehlo/" 2>/dev/null || true
# 自定义前缀时需额外指定：`STABLEHLO_INCLUDE_DIR`、`STABLEHLO_LIB_DIR`。
```

### 1.5 配置 mlir_compiler 找到 MLIR（仅当需要编 P5 插件）

§1.3 `ninja install` 后，root 把路径写入系统环境（普通用户读不到 `$LLVM_BUILD_DIR`，**不要依赖 build 目录**）：

```bash
# export并且写入系统环境变量 /etc/profile.d/llvm-mlir.sh（root 写一次）
# 二选一，让 find_package(MLIR) / find_package(LLVM) 找到 Config.cmake：
export CMAKE_PREFIX_PATH="$LLVM_INSTALL_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
# 或者不用 CMAKE_PREFIX_PATH，改用下面两行：
# export MLIR_DIR="$LLVM_INSTALL_PREFIX/lib/cmake/mlir"
# export LLVM_DIR="$LLVM_INSTALL_PREFIX/lib/cmake/llvm"
# 环境变量设置等同于将-DMLIR_DIR/-DLLVM_DIR传入CMake
# 通常只写 -DMLIR_DIR 即可：MLIRConfig.cmake 会把 LLVM_DIR 推导为同级的 ../llvm
# 若 configure 报找不到 LLVM，再补 -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm"
```


| 方式     | 变量                      | 说明                                                                 |
| ------ | ----------------------- | ------------------------------------------------------------------ |
| **推荐** | `CMAKE_PREFIX_PATH`     | 一个前缀即可，CMake 自动找 `lib/cmake/mlir` 与 `lib/cmake/llvm`               |
| **备选** | `MLIR_DIR` + `LLVM_DIR` | 分别给 `find_package(MLIR)`、`find_package(LLVM)` 用；先找 MLIR，找到后才找 LLVM |


与 `llvm-config`（`src/pass/`）无关，只需 `PATH` 里有 `llvm-config`。找不到 MLIR 时 P5 静默跳过。

### 1.6 编译 Torch-MLIR 与 Python 依赖

**重要**：配置 torch-mlir 时 **必须**使用 LLVM 的 **构建目录** `LLVM_BUILD_DIR`，不要用仅安装后的 `/usr/local`（否则 `check-torch-mlir` 会缺 `FileCheck` 等 target）。

```bash
export LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$TORCH_MLIR_HOME/externals/llvm-project/build}"
export TORCH_MLIR_BUILD_DIR="$TORCH_MLIR_HOME/build"
export PYTORCH_CHANNEL="${PYTORCH_CHANNEL:-cpu}"

command -v clang >/dev/null && command -v clang++ >/dev/null
export CC=clang
export CXX=clang++

test -f "$LLVM_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake"
ninja -C "$LLVM_BUILD_DIR"
ninja -C "$LLVM_BUILD_DIR" mlir-tblgen mlir-python-sources

rm -rf "$TORCH_MLIR_BUILD_DIR"
cmake -G Ninja -S "$TORCH_MLIR_HOME" -B "$TORCH_MLIR_BUILD_DIR" \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=torch_mlir \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
  -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON \
  ${EXTRA_TORCH_MLIR_CMAKE_ARGS:-}
ninja -C "$TORCH_MLIR_BUILD_DIR"
ninja -C "$TORCH_MLIR_BUILD_DIR" TorchMLIRPythonModules

python -m pip install -U pip
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url "https://download.pytorch.org/whl/${PYTORCH_CHANNEL}" torch torchvision torchaudio
python -m pip install numpy
```

`**PYTHONPATH**`：

```bash
set_torch_mlir_pythonpath() {
  local root="$1"
  local paths=(
    "$root/python"
    "$root/projects/pt1/python"
    "$root/build/python_packages/torch_mlir"
    "$root/build/tools/torch-mlir/python_packages/torch_mlir"
  )
  for p in "${paths[@]}"; do
    [ -d "$p" ] && export PYTHONPATH="$p:${PYTHONPATH:-}"
  done
}
set_torch_mlir_pythonpath "$TORCH_MLIR_HOME"
python -c "import torch_mlir._mlir_libs._jit_ir_importer as m; print('jit_ir_importer OK')"

# 可选: 持久化 PYTHONPATH（重开终端后仍生效）
grep -q 'torch-mlir/build/python_packages/torch_mlir' ~/.bashrc || \
  echo "export PYTHONPATH=\"$TORCH_MLIR_HOME/python:$TORCH_MLIR_HOME/projects/pt1/python:$TORCH_MLIR_HOME/build/python_packages/torch_mlir:$TORCH_MLIR_HOME/build/tools/torch-mlir/python_packages/torch_mlir:\$PYTHONPATH\"" >> /etc/profile.d/llvm-mlir.sh
```

**注意**：若报错信息类似
`The dependency target "FileCheck" of target "check-torch-mlir" does not exist`，多半是 torch-mlir 的 `MLIR_DIR` / `LLVM_DIR` 指到了 LLVM **安装目录**（如 `/usr/local`），而不是 **构建目录**。应改为使用 `**$TORCH_MLIR_HOME/externals/llvm-project/build`**，删掉 torch-mlir 的 build 目录后重新 `cmake` + `ninja`：

```bash
export LLVM_BUILD_DIR="$TORCH_MLIR_HOME/externals/llvm-project/build"
rm -rf "$TORCH_MLIR_HOME/build"
# 再执行 §1.6 中自 cmake 起的命令
```

### 1.7 CUDA 版 PyTorch（可选）

默认 `PYTORCH_CHANNEL=cpu`。需要 GPU 版时：

```bash
export PYTORCH_CHANNEL=cu128   # 按驱动改为 cu121/cu124 等
export CUDA_HOME=/usr/local/cuda
export CUDAToolkit_ROOT=$CUDA_HOME
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export EXTRA_TORCH_MLIR_CMAKE_ARGS="-DCUDAToolkit_ROOT=$CUDAToolkit_ROOT"
# 然后重做 §1.6 中自 cmake 起的步骤
```

### 1.8 RISC-V sysroot（可选：要链接出可 QEMU 的 ELF 时）

若仅生成 `matmul_riscv.so` 做反汇编查看，可跳过；要 `**qemu-riscv64 ./matmul_riscv**`，需要完整 sysroot。示例（Ubuntu + riscv-gnu-toolchain）：

```bash
sudo apt update
sudo apt install -y build-essential git \
  libgmp-dev libmpfr-dev libmpc-dev zlib1g-dev libexpat1-dev \
  bison flex gawk texinfo patchutils python3

git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --with-arch=rv64gc --with-abi=lp64d
make linux -j"$(nproc)"

test -f /opt/riscv/sysroot/usr/lib/libc.so
```

链接与运行时示例见 **§2.6**。

---

## 二、代码运行与流水线

> **前置**：`mlir-opt`、`mlir-translate`、`llc`、`clang` 在 `PATH`（来自 §1.3）。若需用 `**matmul.py`** 生成入口 IR，还需完成 **§1.6** 的 Python/torch-mlir。

### 2.1 工作目录

```bash
cd /path/to/mlir_compiler/src/mlir/cpu
```

以下命令默认在此目录执行；若从**仓库根**执行，请把脚本路径改成 `src/mlir/cpu/...`。

### 2.2 PyTorch 前端：导出 matmul（`matmul.py`）

`[matmul.py](matmul.py)` 核心：用 `torchscript.compile(..., output_type="linalg-on-tensors")` 得到带 `linalg.matmul` 的 MLIR。

```bash
python matmul.py > matmul.mlir
```

期望在 `matmul.mlir` 中看到类似：

```mlir
linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
              outs(%init : tensor<4x4xf32>)
```

### 2.3 L3：Linalg 规范化（tensor）

```bash
mlir-opt matmul.mlir \
  --linalg-generalize-named-ops \
  --canonicalize \
  > matmul_l2.mlir
```

这里**没有显式做 tiling**。当前示例选择的是一条**最小可读**流水线：先把 `linalg.matmul` 规整到更通用的 L3 形态，再经 bufferize 和循环 lowering 走到 LLVM。

若你想把这里的 **L3 局部性优化** 落到具体命令上，需要额外插入 **tiling / fusion** 相关 pass。`mlir-opt` **并不是不支持 tiling**，而是**没有一个跨版本都稳定、且适用于所有 Linalg 场景的统一写法**：有的版本/管线会用专门的 tiling pass，有的更偏向 **Transform Dialect**，也有一些只在特定 IR 形态上可用。因此本文先说明概念与主干 pipeline，不把某个易变的 tiling 命令固定成教程中的唯一写法。

### 2.4 One-Shot Bufferize

**语义**：**bufferize（缓冲化）** 的核心工作是把 IR 里的 `tensor`（值语义）转换为 `memref`（显式内存块），并插入/整理分配与释放（`-buffer-deallocation-pipeline` 负责生命周期管理）。在分层上它是 L3 内的类型落地 Pass（OSB，见 [../README.md](../README.md)）：重点在于决定缓冲区如何落地、哪些结果可原地更新、哪些必须新分配；产物仍是 L3 结构化算子（`linalg` on `memref`），还不是进入 L4 的标志——须经 `-convert-linalg-to-loops` 才生成显式循环。**注意**：bufferize **不是**「把整个 Linalg 方言替换成 memref 方言」——`memref` 只是**类型/内存抽象**，算子仍可能是 `linalg` on `memref`，或在后续 pass 中进一步拆成 `memref.load`/`store` + 循环。

```bash
mlir-opt matmul_l2.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  -buffer-deallocation-pipeline \
  > matmul_buffer.mlir
```

### 2.5 L4：展开循环并 Lowering 到 LLVM Dialect

```bash
mlir-opt matmul_buffer.mlir \
  -convert-bufferization-to-memref \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -convert-vector-to-llvm \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -convert-arith-to-llvm \
  -convert-index-to-llvm \
  -convert-math-to-llvm \
  -finalize-memref-to-llvm \
  -reconcile-unrealized-casts \
  > matmul_llvm.mlir
```

### 2.6 转 LLVM IR、编译 RISC-V 目标文件

```bash
mlir-translate --mlir-to-llvmir matmul_llvm.mlir > matmul.ll

llc matmul.ll \
  -mtriple=riscv64 \
  -mattr=+v \
  -O3 \
  -filetype=obj \
  -o matmul.o
```

**链接**（二选一）：

**A — sysroot 不完整时**：生成共享库，用 `llvm-objdump` 等静态查看：

```bash
clang --target=riscv64-unknown-linux-gnu -fuse-ld=lld -nostdlib -shared matmul.o -o matmul_riscv.so
llvm-objdump -d matmul_riscv.so | sed -n '1,20p'
llvm-readelf -h matmul_riscv.so
```

**B — 有完整 RISC-V sysroot（见 §1.8）**：可执行文件 + QEMU：

```bash
clang --target=riscv64-unknown-linux-gnu \
  --gcc-toolchain=/opt/riscv \
  --sysroot=/opt/riscv/sysroot \
  matmul.o driver_main.c -o matmul_riscv
file matmul_riscv

qemu-riscv64 -L /opt/riscv/sysroot ./matmul_riscv | tee riscv_run.txt
```

**Spike**（同样需要可运行 ELF）：

```bash
spike pk matmul_riscv
```

说明：`matmul_riscv.so` **不能**直接交给 `qemu-riscv64` 当程序运行。

### 2.7 一键流水线（等价于分步 §2.3–§2.6 前半）

可在本目录保存为 `pipeline.sh` 并执行：

```bash
mlir-opt matmul.mlir \
  --linalg-generalize-named-ops \
  --canonicalize \
  --one-shot-bufferize="bufferize-function-boundaries" \
  -buffer-deallocation-pipeline \
  -convert-bufferization-to-memref \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -convert-vector-to-llvm \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -convert-arith-to-llvm \
  -convert-index-to-llvm \
  -convert-math-to-llvm \
  -finalize-memref-to-llvm \
  -reconcile-unrealized-casts \
  | mlir-translate --mlir-to-llvmir \
  > matmul.ll
```

然后继续 **§2.6** 的 `llc` / `clang`。

---

## 三、本目录文件（速查）


| 文件                           | 作用                               |
| ---------------------------- | -------------------------------- |
| `matmul.py`                  | PyTorch → Linalg-on-tensors MLIR |
| `driver_main.c`              | 与 `matmul.o` 链接、调用 kernel        |
| `matmul*.mlir` / `matmul.ll` | 流水线中间产物（部分已提交示例）                 |


---

## 四、文档维护说明

- **权威正文**：以本文档 `**src/mlir/cpu/README.md`** 为准（环境准备 / 代码运行业已分区）。
- `**docs/MLIR端到端实战指南.md`**：保留为**跳转页**，避免旧链接失效。

若你使用 CUDA 版 PyTorch 却缺少本机 CUDA，可能出现：

`Your installed Caffe2 version uses CUDA but I cannot find the CUDA libraries`

此时要么补齐 CUDA，要么改回 CPU wheel（见 **§1.7**）。