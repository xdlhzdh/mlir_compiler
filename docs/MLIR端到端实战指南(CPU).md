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

# 1️⃣ 环境准备

## 1.1 获取源码

```bash
# 统一路径变量（从第一步开始）
export WORK_HOME="${WORK_HOME:-$HOME/mlir-matmul}"
export LLVM_HOME="${LLVM_HOME:-$WORK_HOME/llvm-project}"
export TORCH_MLIR_HOME="${TORCH_MLIR_HOME:-$WORK_HOME/torch-mlir}"

mkdir -p "$WORK_HOME"
cd "$WORK_HOME"

git clone https://github.com/llvm/torch-mlir.git "$TORCH_MLIR_HOME"
# 仅拉取构建需要的子模块（stablehlo），避免全量子模块开销
git -C "$TORCH_MLIR_HOME" submodule update --init externals/stablehlo

# 关键：用 torch-mlir 锁定的 LLVM 提交，避免 tblgen/ODS 版本不匹配
if [ -f "$TORCH_MLIR_HOME/build_tools/llvm_version.txt" ]; then
  LLVM_COMMIT=$(cat "$TORCH_MLIR_HOME/build_tools/llvm_version.txt")
else
  LLVM_COMMIT=$(git -C "$TORCH_MLIR_HOME" ls-tree HEAD externals/llvm-project | awk '{print $3}')
fi
git clone https://github.com/llvm/llvm-project.git "$LLVM_HOME"
git -C "$LLVM_HOME" checkout "$LLVM_COMMIT"
```

---

## 1.2 编译 LLVM + MLIR

```bash
cd "$LLVM_HOME"

if [ -f "$TORCH_MLIR_HOME/build_tools/llvm_version.txt" ]; then
  LLVM_COMMIT=$(cat "$TORCH_MLIR_HOME/build_tools/llvm_version.txt")
else
  LLVM_COMMIT=$(git -C "$TORCH_MLIR_HOME" ls-tree HEAD externals/llvm-project | awk '{print $3}')
fi
git fetch --all
git checkout "$LLVM_COMMIT"

mkdir -p build && cd build

# torch-mlir/MLIR 推荐使用 clang 工具链，避免 gcc 无法识别部分 -W 参数
command -v clang >/dev/null && command -v clang++ >/dev/null
export CC=clang
export CXX=clang++

cmake -G Ninja ../llvm \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="RISCV;X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON

# 注意：后续要编 torch-mlir，这里必须先完成 LLVM/MLIR 库的全量构建
ninja
ninja mlir-tblgen mlir-python-sources
```

设置环境变量：

```bash
export PATH="$LLVM_HOME/build/bin:$PATH"

# 持久化到当前用户 shell（重开终端后仍生效）
grep -q 'llvm-project/build/bin' ~/.bashrc || \
  echo "export PATH=\"$LLVM_HOME/build/bin:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
```

---

## 1.3 编译 Torch-MLIR

```bash
# 复用 1.1 中已设置的路径变量（若单独执行本节，请先设置这两个变量）
export LLVM_BUILD_DIR="$LLVM_HOME/build"
export TORCH_MLIR_BUILD_DIR="$TORCH_MLIR_HOME/build"

# CPU/CUDA 通用：PyTorch wheel 通道，默认 cpu；CUDA 示例见 1.4
export PYTORCH_CHANNEL="${PYTORCH_CHANNEL:-cpu}"

# /opt 场景可直接复用：自动以仓库 owner 身份执行 git（避免 dubious ownership）
run_git_as_owner() {
  repo="$1"; shift
  owner="$(stat -c '%U' "$repo")"
  if [ "$(id -un)" = "$owner" ]; then
    git -C "$repo" "$@"
  else
    sudo -u "$owner" git -C "$repo" "$@"
  fi
}

# 1) 对齐 torch-mlir 子模块（只拉必要 stablehlo）
run_git_as_owner "$TORCH_MLIR_HOME" submodule update --init externals/stablehlo

# 2) 校验 llvm-project 提交与 torch-mlir 期望一致
if [ -f "$TORCH_MLIR_HOME/build_tools/llvm_version.txt" ]; then
  LLVM_EXPECTED_COMMIT=$(cat "$TORCH_MLIR_HOME/build_tools/llvm_version.txt")
else
  LLVM_EXPECTED_COMMIT=$(run_git_as_owner "$TORCH_MLIR_HOME" ls-tree HEAD externals/llvm-project | awk '{print $3}')
fi
run_git_as_owner "$LLVM_HOME" fetch --all
run_git_as_owner "$LLVM_HOME" checkout "$LLVM_EXPECTED_COMMIT"
test "$(run_git_as_owner "$LLVM_HOME" rev-parse HEAD)" = "$LLVM_EXPECTED_COMMIT"

# 3) 配置编译器
command -v clang >/dev/null && command -v clang++ >/dev/null
export CC=clang
export CXX=clang++

# 4) 确保 LLVM/MLIR 产物齐全（torch-mlir 链接依赖完整库）
test -f "$LLVM_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake"
ninja -C "$LLVM_BUILD_DIR"
ninja -C "$LLVM_BUILD_DIR" mlir-tblgen mlir-python-sources

# 5) 重配并编译 torch-mlir
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

# 6) 安装 Python 依赖（默认 CPU-only）
python -m pip install -U pip
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url "https://download.pytorch.org/whl/${PYTORCH_CHANNEL}" torch torchvision torchaudio
python -m pip install numpy

# 7) 统一设置 PYTHONPATH
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

# 2️⃣ PyTorch 前端：导出 matmul

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
set_torch_mlir_pythonpath "$HOME/mlir-matmul/torch-mlir"
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
cd ~/simple_compiler/src/mlir/python
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
