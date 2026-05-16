"""Generate test ONNX models for P4 (ONNX → StableHLO lowering exercises).

Models:
  P4 tier 1 (basic):
    lowering_basic.onnx             — Add + MatMul chain
    lowering_conv.onnx              — Conv (no bias, basic attrs)
    lowering_reshape_transpose.onnx — Reshape + Transpose

  P4 tier 2 (enhanced):
    lowering_broadcast.onnx         — Add/Mul with numpy-style broadcast
    lowering_conv_full.onnx         — Conv with bias, strides, pads
    lowering_dynamic.onnx           — Dynamic batch dimension
"""

from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _save(model, name, out_dir):
    onnx.checker.check_model(model)
    path = out_dir / name
    onnx.save(model, str(path))
    print(f"  saved {path}")


def make_lowering_basic(out_dir):
    """X(2,3) + bias(2,3) → MatMul(add_out, W(3,4)) → Y(2,4)"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    bias = numpy_helper.from_array(np.ones((2, 3), dtype=np.float32), "bias")
    W = numpy_helper.from_array(np.eye(3, 4, dtype=np.float32), "W")

    add_n = helper.make_node("Add", ["X", "bias"], ["add_out"], name="add_0")
    mm_n = helper.make_node("MatMul", ["add_out", "W"], ["Y"], name="mm_0")

    graph = helper.make_graph([add_n, mm_n], "basic", [X], [Y],
                              initializer=[bias, W])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_basic.onnx", out_dir)


def make_lowering_conv(out_dir):
    """X(1,1,8,8) → Conv(W(2,1,3,3), no bias, stride=1, pad=0) → Y(1,2,6,6)"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 6, 6])
    W = numpy_helper.from_array(
        np.random.randn(2, 1, 3, 3).astype(np.float32), "W")

    conv = helper.make_node("Conv", ["X", "W"], ["Y"], name="conv_0",
                            kernel_shape=[3, 3])
    graph = helper.make_graph([conv], "conv", [X], [Y], initializer=[W])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_conv.onnx", out_dir)


def make_lowering_reshape_transpose(out_dir):
    """X(2,3,4) → Reshape([6,4]) → Transpose([1,0]) → Y(4,6)"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 6])
    shape = numpy_helper.from_array(
        np.array([6, 4], dtype=np.int64), "target_shape")

    reshape = helper.make_node("Reshape", ["X", "target_shape"],
                               ["reshape_out"], name="reshape_0")
    transpose = helper.make_node("Transpose", ["reshape_out"], ["Y"],
                                 name="transpose_0", perm=[1, 0])
    graph = helper.make_graph([reshape, transpose], "reshape_transpose",
                              [X], [Y], initializer=[shape])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_reshape_transpose.onnx", out_dir)


def make_lowering_broadcast(out_dir):
    """X(2,3,4) + bias(4) → Mul(add_out, scale(1,3,1)) → Y(2,3,4)
    Tests numpy-style broadcasting for Add and Mul."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4])
    bias = numpy_helper.from_array(np.ones(4, dtype=np.float32), "bias")
    scale = numpy_helper.from_array(
        np.full((1, 3, 1), 2.0, dtype=np.float32), "scale")

    add_n = helper.make_node("Add", ["X", "bias"], ["add_out"], name="add_0")
    mul_n = helper.make_node("Mul", ["add_out", "scale"], ["Y"], name="mul_0")

    graph = helper.make_graph([add_n, mul_n], "broadcast", [X], [Y],
                              initializer=[bias, scale])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_broadcast.onnx", out_dir)


def make_lowering_conv_full(out_dir):
    """X(1,3,16,16) → Conv(W, B, stride=2, pad=1) → Y(1,8,8,8)
    Tests full Conv attribute mapping including bias broadcast."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 16, 16])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 8, 8])
    W = numpy_helper.from_array(
        np.random.randn(8, 3, 3, 3).astype(np.float32), "W")
    B = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "B")

    conv = helper.make_node("Conv", ["X", "W", "B"], ["Y"], name="conv_0",
                            kernel_shape=[3, 3], strides=[2, 2],
                            pads=[1, 1, 1, 1])
    graph = helper.make_graph([conv], "conv_full", [X], [Y],
                              initializer=[W, B])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_conv_full.onnx", out_dir)


def make_lowering_dynamic(out_dir):
    """X(?x3) + bias(3) → MatMul(add_out, W(3,4)) → Y(?x4)
    Tests dynamic (symbolic) batch dimension handling."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 4])
    bias = numpy_helper.from_array(np.ones(3, dtype=np.float32), "bias")
    W = numpy_helper.from_array(np.eye(3, 4, dtype=np.float32), "W")

    add_n = helper.make_node("Add", ["X", "bias"], ["add_out"], name="add_0")
    mm_n = helper.make_node("MatMul", ["add_out", "W"], ["Y"], name="mm_0")

    graph = helper.make_graph([add_n, mm_n], "dynamic", [X], [Y],
                              initializer=[bias, W])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_dynamic.onnx", out_dir)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Generating lowering test ONNX models ...")
    make_lowering_basic(out_dir)
    make_lowering_conv(out_dir)
    make_lowering_reshape_transpose(out_dir)
    make_lowering_broadcast(out_dir)
    make_lowering_conv_full(out_dir)
    make_lowering_dynamic(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
