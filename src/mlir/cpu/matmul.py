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