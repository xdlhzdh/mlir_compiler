"""Generate ONNX QDQ fixtures for P12 quantization pipeline."""

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


def make_quant_qdq_matmul(out_dir):
    """X(2,3) → Q/DQ → MatMul(W) → Q/DQ → Y(2,4) ONNX QDQ pattern."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    W = numpy_helper.from_array(np.eye(3, 4, dtype=np.float32), "W")
    scale_x = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale_x")
    zp_x = numpy_helper.from_array(np.array(0, dtype=np.int8), "zp_x")
    scale_w = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale_w")
    zp_w = numpy_helper.from_array(np.array(0, dtype=np.int8), "zp_w")
    scale_y = numpy_helper.from_array(np.array(0.2, dtype=np.float32), "scale_y")
    zp_y = numpy_helper.from_array(np.array(0, dtype=np.int8), "zp_y")

    q_x = helper.make_node("QuantizeLinear", ["X", "scale_x", "zp_x"], ["qx"],
                           name="quantize_x")
    dq_x = helper.make_node("DequantizeLinear", ["qx", "scale_x", "zp_x"],
                            ["dq_x"], name="dequantize_x")
    q_w = helper.make_node("QuantizeLinear", ["W", "scale_w", "zp_w"], ["qw"],
                           name="quantize_w")
    dq_w = helper.make_node("DequantizeLinear", ["qw", "scale_w", "zp_w"],
                            ["dq_w"], name="dequantize_w")
    mm = helper.make_node("MatMul", ["dq_x", "dq_w"], ["mm_out"], name="matmul")
    q_y = helper.make_node("QuantizeLinear", ["mm_out", "scale_y", "zp_y"],
                           ["qy"], name="quantize_y")
    dq_y = helper.make_node("DequantizeLinear", ["qy", "scale_y", "zp_y"],
                            ["Y"], name="dequantize_y")

    inits = [W, scale_x, zp_x, scale_w, zp_w, scale_y, zp_y]
    graph = helper.make_graph(
        [q_x, dq_x, q_w, dq_w, mm, q_y, dq_y],
        "qdq_matmul", [X], [Y], initializer=inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "quant_qdq_matmul.onnx", out_dir)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Generating P12 quantization ONNX models ...")
    make_quant_qdq_matmul(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
