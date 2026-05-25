# output_type="linalg-on-tensors"：导出直接为 L3「linalg on tensor」；
# print(module) 不会呈现以 torch 方言（如 torch.aten.*）为主的 L1 IR。
# 若要看 Torch Dialect，需改用会停在 torch 层的 output_type（依 torch-mlir 版本文档为准）。
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