module attributes {torch.debug_module_name = "MatmulModule"} {
  func.func @forward(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %2 : tensor<4x4xf32>
  }
}

