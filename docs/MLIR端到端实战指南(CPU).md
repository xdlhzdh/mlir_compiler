# 🚀 PyTorch → MLIR → 硬件仿真 MatMul 端到端实战指南 (CPU/CUDA)

---

# 0️⃣ 总体数据流

```
PyTorch
  ↓
Torch-MLIR (torch dialect)
  ↓
{linalg.matmul (tensor)}
  ↓ 
Linalg Tiling + Vectorization
  ↓ 
{scf.for on tensor(动态不限步长) or affine.for on tensor(静态限整数步长)}
  ↓
One-Shot Bufferize
  ↓
{scf.for on memref or affine.for on memref}
  ↓
Convert to LLVM dialect
  ↓
{cf.br}
  ↓
LLVM IR
  ↓
LLVM RISC-V backend (RVV)
  ↓
RISC-V binary
  ↓
QEMU / Spike
```

✔ 这是当前 MLIR for CPU / RISC-V 最标准的路径

---

# 1️⃣ 环境准备(root)

## 1.1 获取源码

直接使用 **torch-mlir 仓库内的 llvm-project 与 stablehlo 子模块**，无需单独再 clone 一份 llvm-project，版本由 torch-mlir 锁定，避免 tblgen/ODS 与后续编 torch-mlir 时不一致。

```bash
# 统一路径变量（从第一步开始）
export WORK_HOME="${WORK_HOME:-/opt}"
export TORCH_MLIR_HOME="${TORCH_MLIR_HOME:-$WORK_HOME/torch-mlir}"

mkdir -p "$WORK_HOME"
cd "$WORK_HOME"

git clone https://github.com/llvm/torch-mlir.git "$TORCH_MLIR_HOME"
# 拉取构建所需的子模块：llvm-project 与 stablehlo（均位于 torch-mlir 仓库内）
git -C "$TORCH_MLIR_HOME" submodule update --init externals/llvm-project externals/stablehlo
```

之后 LLVM 源码与构建目录为：`$TORCH_MLIR_HOME/externals/llvm-project`、`$TORCH_MLIR_HOME/externals/llvm-project/build`（见 1.2.1）。

---

## 1.2 编译启用 StableHLO 的 LLVM + MLIR（推荐：统一工具链）

本指南推荐使用 **启用 StableHLO 的 LLVM**：在同一安装前缀下安装 LLVM/MLIR 与 StableHLO，得到一套工具链，**不再使用“单独 LLVM-project + 单独安装 StableHLO”的分散方式**。这样 simple_compiler 的 MLIR GPU 目标只需指向该工具链即可。

### 1.2.1 构建 LLVM + MLIR（使用 torch-mlir 内的 llvm-project）

在 **torch-mlir 的 externals/llvm-project** 下配置并构建，无需单独 clone 的 llvm-project。

```bash
export LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-$TORCH_MLIR_HOME/externals/llvm-project}"
export LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$LLVM_SOURCE_DIR/build}"

cd "$LLVM_SOURCE_DIR"
mkdir -p build && cd build

# torch-mlir/MLIR 推荐使用 clang 工具链，避免 gcc 无法识别部分 -W 参数
command -v clang >/dev/null && command -v clang++ >/dev/null
export CC=clang
export CXX=clang++

# 统一安装前缀：与后续 StableHLO 安装到同一前缀，形成“启用 StableHLO 的 LLVM”
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

# 后续要编 torch-mlir 与 StableHLO，这里必须先完成 LLVM/MLIR 全量构建
ninja
ninja mlir-tblgen mlir-python-sources
# 安装到统一前缀（与 StableHLO 同前缀，即“启用 StableHLO 的 LLVM”工具链）
ninja install
```

设置环境变量（使用**安装后的** bin 目录）：

```bash
export PATH="$LLVM_INSTALL_PREFIX/bin:$PATH"

# 持久化到当前用户 shell（重开终端后仍生效）
grep -q "$LLVM_INSTALL_PREFIX/bin" ~/.bashrc || \
  echo "export PATH=\"$LLVM_INSTALL_PREFIX/bin:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
```

### 1.2.2 在同一前缀下构建并安装 StableHLO（形成统一工具链）

使用 **torch-mlir 已拉取的 stablehlo**（与 1.1 一致、版本匹配），在 1.2.1 完成后，将 StableHLO **安装到与 LLVM 相同的前缀**，这样得到的就是“启用 StableHLO 的 LLVM”一套工具链。

```bash
# 与 1.2.1 相同的前缀（必须一致，否则不是“一套”工具链）
# LLVM_BUILD_DIR 已在 1.2.1 中设为 torch-mlir/externals/llvm-project/build
export STABLEHLO_INSTALL_PREFIX="${STABLEHLO_INSTALL_PREFIX:-$LLVM_INSTALL_PREFIX}"

cd "$TORCH_MLIR_HOME/externals/stablehlo"
rm -rf build && mkdir build && cd build

cmake -G Ninja -S .. -B . \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DCMAKE_INSTALL_PREFIX="$STABLEHLO_INSTALL_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF

ninja
ninja install
```

**说明：为何头文件不会自动装到安装前缀**

当前 **StableHLO 上游 CMake**（torch-mlir 使用的 `externals/stablehlo`）**没有配置头文件的 install 规则**：只对库目标做了安装，头文件与构建生成的 `.inc` 未包含在 `ninja install` 中。若希望头文件也在同一前缀下，需在 `ninja install` 后**手动拷贝**：

```bash
# 在 externals/stablehlo/build 目录下执行；前缀为 /usr/local 时加 sudo
STABLEHLO_SRC="$TORCH_MLIR_HOME/externals/stablehlo"
PREFIX="${STABLEHLO_INSTALL_PREFIX:-/usr/local}"
mkdir -p "$PREFIX/include"
cp -r "$STABLEHLO_SRC/stablehlo" "$PREFIX/include/"
cp -r stablehlo/* "$PREFIX/include/stablehlo/" 2>/dev/null || true
```

安装完成后，**库**在 `$STABLEHLO_INSTALL_PREFIX/lib/`（如 `libStablehloOps.a`、`libStablehloBase.a`、`libStablehloRegister.a` 等），**头文件**在 `$PREFIX/include/stablehlo/`（若已执行上述拷贝）。此时 LLVM/MLIR 与 StableHLO 位于同一前缀，即 **启用 StableHLO 的 LLVM**。

### 1.2.3 simple_compiler 使用该工具链

配置 simple_compiler 时，指定 **安装后的** MLIR/LLVM（与 StableHLO 同前缀）：

```bash
cd simple_compiler/build
cmake .. \
  -DMLIR_DIR="$LLVM_INSTALL_PREFIX/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_INSTALL_PREFIX/lib/cmake/llvm"
```

若 StableHLO 已按 1.2.2 安装到同一前缀（如 `/usr/local`），gpu CMake 会在该前缀下自动找到头与库，**无需**再设 `STABLEHLO_INCLUDE_DIR` 或 `STABLEHLO_LIB_DIR`。若安装到自定义前缀，则加上：

```bash
-DSTABLEHLO_INCLUDE_DIR="$STABLEHLO_INSTALL_PREFIX/include" \
-DSTABLEHLO_LIB_DIR="$STABLEHLO_INSTALL_PREFIX/lib"
```

**不推荐**：仅构建 LLVM 而不安装、或 StableHLO 安装到与 LLVM 不同的前缀，会导致需要单独指定多处路径，与本指南“启用 StableHLO 的 LLVM”统一工具链目标不一致。

---

## 1.3 编译 Torch-MLIR

**重要**：配置 torch-mlir 时 **必须** 使用 LLVM 的 **构建目录**（`LLVM_BUILD_DIR`），不能使用安装目录（如 `/usr/local`）。lit 测试依赖的 `FileCheck`、`count`、`not` 等 target 仅存在于 LLVM 构建树中，使用安装目录会报错：`The dependency target "FileCheck" of target "check-torch-mlir" does not exist`。

```bash
# 必须使用 LLVM 的构建目录（1.2.1 中 ninja 所在目录），不要用安装前缀 /usr/local
export LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$TORCH_MLIR_HOME/externals/llvm-project/build}"
export TORCH_MLIR_BUILD_DIR="$TORCH_MLIR_HOME/build"

# CPU/CUDA 通用：PyTorch wheel 通道，默认 cpu；CUDA 示例见 1.4
export PYTORCH_CHANNEL="${PYTORCH_CHANNEL:-cpu}"

# 1) 子模块已在 1.1 中 init（externals/llvm-project、externals/stablehlo），若需更新可再执行：
#    git -C "$TORCH_MLIR_HOME" submodule update --init externals/llvm-project externals/stablehlo

# 2) 配置编译器
command -v clang >/dev/null && command -v clang++ >/dev/null
export CC=clang
export CXX=clang++

# 3) 确保 LLVM/MLIR 产物齐全（torch-mlir 链接依赖完整库；1.2.1 已在该目录构建并安装）
test -f "$LLVM_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake"
ninja -C "$LLVM_BUILD_DIR"
ninja -C "$LLVM_BUILD_DIR" mlir-tblgen mlir-python-sources

# 4) 重配并编译 torch-mlir（MLIR_DIR/LLVM_DIR 必须指向构建目录，不能指向 /usr/local）
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

# 5) 安装 Python 依赖（默认 CPU-only）
python -m pip install -U pip
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url "https://download.pytorch.org/whl/${PYTORCH_CHANNEL}" torch torchvision torchaudio
python -m pip install numpy

# 6) 统一设置 PYTHONPATH
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

# 持久化 PYTHONPATH（重开终端后仍生效）
grep -q 'torch-mlir/build/python_packages/torch_mlir' ~/.bashrc || \
  echo "export PYTHONPATH=\"$TORCH_MLIR_HOME/python:$TORCH_MLIR_HOME/projects/pt1/python:$TORCH_MLIR_HOME/build/python_packages/torch_mlir:$TORCH_MLIR_HOME/build/tools/torch-mlir/python_packages/torch_mlir:\$PYTHONPATH\"" >> ~/.bashrc
source ~/.bashrc
```

**若出现 `The dependency target "FileCheck"/"count"/"not" of target "check-torch-mlir" does not exist`**：说明当前用的是 LLVM **安装目录**（如 `/usr/local`）作为 `MLIR_DIR`/`LLVM_DIR`。应改用 LLVM **构建目录**（即 1.2.1 中执行 `ninja` 的目录，例如 `$TORCH_MLIR_HOME/externals/llvm-project/build`），然后删掉 torch-mlir 的 build 再重新配置并编译：

```bash
export LLVM_BUILD_DIR="$TORCH_MLIR_HOME/externals/llvm-project/build"
rm -rf "$TORCH_MLIR_HOME/build"
# 再执行上面 4) 的 cmake 与 ninja 命令
```

## 1.4 CUDA 版本支持（可选）

默认 `PYTORCH_CHANNEL=cpu`。如果你要 CUDA，切换通道并增加 CUDA 参数后，重新执行 **1.3** 即可。

```bash
# 示例：cu128，可按你机器驱动版本改成 cu121/cu124/cu126...
export PYTORCH_CHANNEL=cu128
export CUDA_HOME=/usr/local/cuda
export CUDAToolkit_ROOT=$CUDA_HOME
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# 将 CUDA 路径附加到 1.3 的 CMake 参数
export EXTRA_TORCH_MLIR_CMAKE_ARGS="-DCUDAToolkit_ROOT=$CUDAToolkit_ROOT"

# 然后重新执行 1.3 全部命令

# 自检
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available:", torch.cuda.is_available())
PY
```

若你看到：

```text
Your installed Caffe2 version uses CUDA but I cannot find the CUDA libraries
```

说明当前环境是 CUDA 版 PyTorch，但系统 CUDA runtime/toolkit 不完整。此时二选一：

1. 补齐系统 CUDA（并确认 `CUDA_HOME`/`LD_LIBRARY_PATH` 正确）
2. 切回 CPU-only PyTorch（按上面的 CPU 安装命令）


---

# 2️⃣ PyTorch 前端：导出 matmul (User)

## matmul.py

```python
import torch
import torch_mlir.torchscript as torchscript

class MatmulModule(torch.nn.Module):
    def forward(self, A, B):
        return torch.matmul(A, B)

model = MatmulModule().eval()

A = torch.randn(4, 4)
B = torch.randn(4, 4)

module = torchscript.compile(
    model,
    (A, B),
    output_type="linalg-on-tensors"
)

print(module)
```

运行：

```bash
python matmul.py > matmul.mlir

# 生成 Python baseline（golden_ref.txt）
python golden_output.py
```

你会得到：

```mlir
linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
              outs(%init : tensor<4x4xf32>)
```

---

# 3️⃣ Linalg 规范化 + Loop 化 (tensor)

## pipeline

```bash
mlir-opt matmul.mlir \
  --linalg-generalize-named-ops \
  --canonicalize \
  -convert-linalg-to-loops \
  -canonicalize \
  > matmul_tiled.mlir
```

作用：

✔ 将 named linalg op 规约为更通用形式
✔ 生成 scf.for
✔ 清理中间 IR（canonicalize）

---

# 4️⃣ One-Shot Bufferize

```bash
mlir-opt matmul_tiled.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  -buffer-deallocation-pipeline \
  > matmul_bufferized.mlir
```

tensor → memref：

```mlir
memref<4x4xf32>
```

---

# 5️⃣ Lower 到 LLVM Dialect

```bash
mlir-opt matmul_bufferized.mlir \
  -convert-bufferization-to-memref \
  -convert-linalg-to-loops \
  -canonicalize \
  -convert-vector-to-llvm \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -convert-arith-to-llvm \
  -convert-index-to-llvm \
  -convert-math-to-llvm \
  -finalize-memref-to-llvm \
  -reconcile-unrealized-casts \
  > matmul_llvm.mlir
```

---

# 6️⃣ 转 LLVM IR

```bash
mlir-translate \
  --mlir-to-llvmir \
  matmul_llvm.mlir > matmul.ll
```

---

# 7️⃣ 编译为 RISC-V RVV 二进制

```bash
llc matmul.ll \
  -mtriple=riscv64 \
  -mattr=+v \
  -O3 \
  -filetype=obj \
  -o matmul.o

# 优先尝试生成可运行 ELF；若 sysroot 不完整则自动回退到 .so（保证命令成功）
if [ -f /usr/riscv64-linux-gnu/sys-root/usr/lib/libc.so ] || \
   [ -f /usr/riscv64-linux-gnu/sys-root/lib/libc.so ]; then
  clang --target=riscv64-unknown-linux-gnu matmul.o driver_main.c -o matmul_riscv
  file matmul_riscv
else
  echo "RISC-V sysroot 不完整，回退为共享库产物 matmul_riscv.so"
  clang --target=riscv64-unknown-linux-gnu -fuse-ld=lld -nostdlib -shared matmul.o -o matmul_riscv.so
  file matmul_riscv.so
  llvm-objdump -d matmul_riscv.so | sed -n '1,20p'
  llvm-readelf -h matmul_riscv.so
fi
```

### 7.1 安装 RISC-V sysroot（可生成可运行 ELF）

如果你想从 `matmul.o` 链接出可执行 `matmul_riscv`（而不是 `.so`），需要完整 sysroot（`libc` 及 `libgcc.a` 或 `libgcc_s.so`）。

```bash
# 1) 安装构建依赖
sudo dnf -y groupinstall "Development Tools"
sudo dnf -y install \
  gmp-devel mpfr-devel libmpc-devel zlib-devel expat-devel \
  bison flex gawk texinfo patchutils python3

# 2) 构建 riscv-gnu-toolchain（会生成 sysroot）
git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --with-arch=rv64gc --with-abi=lp64d
make linux -j"$(nproc)"

# 3) 验证 sysroot（libc 必需；libgcc 可为 libgcc.a 静态版或 libgcc_s.so 动态版，任一即可）
test -f /opt/riscv/sysroot/usr/lib/libc.so
test -f /opt/riscv/lib/gcc/riscv64-unknown-linux-gnu/*/libgcc.a || \
  test -f /opt/riscv/lib/gcc/riscv64-unknown-linux-gnu/*/libgcc_s.so

# 4) 用 sysroot 链接可执行文件
cd ~/simple_compiler/src/mlir/cpu
clang --target=riscv64-unknown-linux-gnu \
  --gcc-toolchain=/opt/riscv \
  --sysroot=/opt/riscv/sysroot \
  matmul.o driver_main.c -o matmul_riscv
file matmul_riscv

# 5) QEMU 验证可执行文件（应打印 first row + PASS）
qemu-riscv64 -L /opt/riscv/sysroot ./matmul_riscv | tee riscv_run.txt

# 6) 与 Python baseline 比对（应输出 PASS）
python compare_outputs.py
```

---

# 8️⃣ 硬件仿真运行

说明：`matmul_riscv.so` 是共享库，不能直接作为程序交给 `qemu-riscv64` 运行。  
默认方案请用 `llvm-objdump` 做静态验证；仅“方案 B 可执行 ELF”可做 QEMU/Spike 仿真。

## QEMU

```bash
# 仅当你完成了上面的“方案 B（可运行 ELF）”时：
qemu-riscv64 -L /opt/riscv/sysroot ./matmul_riscv | tee riscv_run.txt
python compare_outputs.py
```

## Spike

```bash
# 仅当你完成了上面的“方案 B（可运行 ELF）”时：
spike pk matmul_riscv
```

---

# 🎯 到这里你已经完成

PyTorch → MLIR → RVV → 仿真

完整闭环 ✅

---

# 📦 一键跑通工程模板

---

## mlir/pipeline.sh

```bash
mlir-opt matmul.mlir \
  --linalg-generalize-named-ops \
  --canonicalize \
  -convert-linalg-to-loops \
  -canonicalize \
  --one-shot-bufferize="bufferize-function-boundaries" \
  -buffer-deallocation-pipeline \
  -convert-bufferization-to-memref \
  -convert-linalg-to-loops \
  -canonicalize \
  -convert-vector-to-llvm \
  -convert-scf-to-cf \
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

---

## run_pipeline.sh

```bash
#!/usr/bin/env bash

python cpu/matmul.py > matmul.mlir
python cpu/golden_output.py

bash mlir/pipeline.sh

llc matmul.ll -mtriple=riscv64 -mattr=+v -O3 -filetype=obj -o matmul.o

if [ -f /opt/riscv/sysroot/usr/lib/libc.so ]; then
  clang --target=riscv64-unknown-linux-gnu \
    --gcc-toolchain=/opt/riscv \
    --sysroot=/opt/riscv/sysroot \
    matmul.o driver_main.c -o matmul_riscv
  qemu-riscv64 -L /opt/riscv/sysroot ./matmul_riscv | tee riscv_run.txt
  python cpu/compare_outputs.py
else
  clang --target=riscv64-unknown-linux-gnu -fuse-ld=lld -nostdlib -shared matmul.o -o matmul_riscv.so
  llvm-objdump -d matmul_riscv.so | sed -n '1,20p'
fi
```
