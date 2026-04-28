"""Generate test ONNX models for L1 graph level exercises.

Models generated:
  1. add_matmul.onnx — Add + MatMul chain for P1 (parse) and P2 (lowering)
  2. conv_bn.onnx    — Conv + BN for P3 (graph rewrite: fusion)
  3. transpose.onnx  — Transpose(Transpose(x)) for P3 (transpose elimination)
  4. const_fold.onnx  — constant-only subgraph for P3 (constant folding)
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _save(model: onnx.ModelProto, name: str, out_dir: Path) -> None:
    onnx.checker.check_model(model)
    path = out_dir / name
    onnx.save(model, str(path))
    print(f"  saved {path}")


def make_add_matmul(out_dir: Path) -> None:
    # x(2,3) + bias(2,3) -> MatMul(result, W(3,4)) -> out(2,4)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Out = helper.make_tensor_value_info("Out", TensorProto.FLOAT, [2, 4])

    bias_val = numpy_helper.from_array(
        np.ones((2, 3), dtype=np.float32), name="bias"
    )
    w_val = numpy_helper.from_array(
        np.eye(3, 4, dtype=np.float32), name="W"
    )

    add_node = helper.make_node("Add", ["X", "bias"], ["add_out"], name="add_0")
    mm_node = helper.make_node("MatMul", ["add_out", "W"], ["Out"], name="mm_0")

    graph = helper.make_graph(
        [add_node, mm_node], "add_matmul_graph", [X], [Out],
        initializer=[bias_val, w_val],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model = onnx.shape_inference.infer_shapes(model)
    _save(model, "add_matmul.onnx", out_dir)


def make_conv_bn(out_dir: Path) -> None:
    # Conv(x, w, b) -> BatchNormalization -> out
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
    Out = helper.make_tensor_value_info("Out", TensorProto.FLOAT, [1, 16, 6, 6])

    conv_w = numpy_helper.from_array(
        np.random.randn(16, 3, 3, 3).astype(np.float32), name="conv_w"
    )
    conv_b = numpy_helper.from_array(np.zeros(16, dtype=np.float32), name="conv_b")
    bn_scale = numpy_helper.from_array(np.ones(16, dtype=np.float32), name="bn_scale")
    bn_bias = numpy_helper.from_array(np.zeros(16, dtype=np.float32), name="bn_bias")
    bn_mean = numpy_helper.from_array(
        np.random.randn(16).astype(np.float32), name="bn_mean"
    )
    bn_var = numpy_helper.from_array(
        np.abs(np.random.randn(16).astype(np.float32)) + 0.1, name="bn_var"
    )

    conv = helper.make_node(
        "Conv", ["X", "conv_w", "conv_b"], ["conv_out"],
        name="conv_0", kernel_shape=[3, 3],
    )
    bn = helper.make_node(
        "BatchNormalization",
        ["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["Out"], name="bn_0", epsilon=1e-5,
    )
    graph = helper.make_graph(
        [conv, bn], "conv_bn_graph", [X], [Out],
        initializer=[conv_w, conv_b, bn_scale, bn_bias, bn_mean, bn_var],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model = onnx.shape_inference.infer_shapes(model)
    _save(model, "conv_bn.onnx", out_dir)


def make_transpose(out_dir: Path) -> None:
    # Transpose(perm=[1,0,2]) -> Transpose(perm=[1,0,2]) == identity
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    Out = helper.make_tensor_value_info("Out", TensorProto.FLOAT, [2, 3, 4])
    t1 = helper.make_node(
        "Transpose", ["X"], ["t1_out"], name="t1", perm=[1, 0, 2]
    )
    t2 = helper.make_node(
        "Transpose", ["t1_out"], ["Out"], name="t2", perm=[1, 0, 2]
    )
    graph = helper.make_graph([t1, t2], "transpose_graph", [X], [Out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model = onnx.shape_inference.infer_shapes(model)
    _save(model, "transpose.onnx", out_dir)


def make_const_fold(out_dir: Path) -> None:
    # Constant A(2,3) + Constant B(2,3) -> add_out (can fold) -> Mul(x, add_out) -> Out
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Out = helper.make_tensor_value_info("Out", TensorProto.FLOAT, [2, 3])
    a_val = numpy_helper.from_array(
        np.full((2, 3), 2.0, dtype=np.float32), name="A"
    )
    b_val = numpy_helper.from_array(
        np.full((2, 3), 3.0, dtype=np.float32), name="B"
    )
    add_node = helper.make_node("Add", ["A", "B"], ["add_out"], name="const_add")
    mul_node = helper.make_node("Mul", ["X", "add_out"], ["Out"], name="mul_0")
    graph = helper.make_graph(
        [add_node, mul_node], "const_fold_graph", [X], [Out],
        initializer=[a_val, b_val],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model = onnx.shape_inference.infer_shapes(model)
    _save(model, "const_fold.onnx", out_dir)


def main() -> None:
    import sys
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Generating test ONNX models ...")
    make_add_matmul(out_dir)
    make_conv_bn(out_dir)
    make_transpose(out_dir)
    make_const_fold(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
