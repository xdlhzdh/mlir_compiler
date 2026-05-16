# 非标准 Px：借助 PyTorch + torch-mlir 导出 StableHLO 文本，供 P5（6_stablehlo_passes / mlir-opt）使用。
from torch import Tensor
import argparse
import torch
import torch.nn as nn
from torch_mlir import fx


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层：输入通道数3，输出通道数8，卷积核大小3x3（默认stride=1, padding=0）
        self.conv = nn.Conv2d(3, 8, 3)
        # 批量归一化层：对8个输出通道分别做归一化(使得均值0, 方差1)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        return self.bn(self.conv(x))


model = Model().eval()

# 1. 做一次前向追踪（tracing）
# 用这个张量对 model 做一次前向执行，把实际经过的算子（conv、bn 等）和调用顺序记录下来，得到一张计算图。
# 没有这个输入，编译器就不知道模型里到底有哪些 op、怎么连的，也就没法生成对应的 MLIR/StableHLO。
# 2. 确定形状和类型
# example = torch.randn(1, 3, 32, 32) 给出了：
# 形状：[1, 3, 32, 32]（batch=1, channels=3, H=32, W=32）
# 类型：默认 float32
# 编译器会沿着这次前向传播推断出中间和输出张量的形状与 dtype（例如 conv 后、bn 后的 tensor 的 shape），
# 这样生成的 MLIR/StableHLO 里可以带上具体的、静态的 shape 信息，便于后续优化和 lowering。
example: Tensor = torch.randn(1, 3, 32, 32)

module = fx.export_and_import(model, example, output_type="stablehlo")

print(module)


def main():
    parser = argparse.ArgumentParser(
        description="Export Conv+BN model to StableHLO MLIR"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="conv_bn_model.mlir",
        help="Output MLIR file path (default: conv_bn_model.mlir)",
    )
    args = parser.parse_args()
    with open(args.output, "w") as f:
        f.write(str(module))
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
