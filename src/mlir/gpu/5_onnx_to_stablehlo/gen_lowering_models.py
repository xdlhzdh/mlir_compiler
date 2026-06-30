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

  P4 tier 3 (framework):
    lowering_softmax.onnx           — Softmax(axis=-1) decomposition
    lowering_attention.onnx         — Scaled dot-product attention subgraph
    lowering_rmsnorm.onnx           — RMSNorm decomposition
    lowering_layernorm.onnx         — LayerNorm decomposition
    lowering_rope.onnx              — RoPE (rotate_half + sin/cos) decomposition
    lowering_gelu.onnx              — GELU via Erf decomposition
    lowering_swiglu.onnx            — SwiGLU silu(gate)*up decomposition
    lowering_matmul_bias.onnx       — MatMul + constant bias add
    lowering_qdq_matmul.onnx        — dequant chains + MatMul (P11 concept)
    lowering_horizontal_gemm.onnx   — shared-LHS MatMul pair + Concat (horizontal fusion)
    lowering_decode_step.onnx       — seq=1 tiny attention (KV decode teaching)
    lowering_dynamic_mn.onnx        — MatMul with two dynamic dims (?xK) @ (Kx?)
    lowering_matmul_f16.onnx        — single MatMul in FP16
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
    """X(?x3) + bias(?x3) → MatMul(add_out, W(3,4)) → Y(?x4)
    Tests dynamic (symbolic) batch dimension handling."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 3])
    Bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, ["batch", 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 4])
    W = numpy_helper.from_array(np.eye(3, 4, dtype=np.float32), "W")

    add_n = helper.make_node("Add", ["X", "bias"], ["add_out"], name="add_0")
    mm_n = helper.make_node("MatMul", ["add_out", "W"], ["Y"], name="mm_0")

    graph = helper.make_graph([add_n, mm_n], "dynamic", [X, Bias], [Y],
                              initializer=[W])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_dynamic.onnx", out_dir)


def make_lowering_softmax(out_dir):
    """X(2,4) → Softmax(axis=-1) → Y(2,4)
    Tests tier-3 decomposition into reduce/sub/exp/reduce/divide."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])

    softmax = helper.make_node("Softmax", ["X"], ["Y"],
                               name="softmax_0", axis=-1)
    graph = helper.make_graph([softmax], "softmax", [X], [Y])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_softmax.onnx", out_dir)


def make_lowering_attention(out_dir):
    """Q,K,V(1,2,4) → MatMul(Q,K^T) → scale → Softmax → MatMul(V) → Y.
    Tests a small scaled dot-product attention lowering path."""
    Q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, 2, 4])
    Kt = helper.make_tensor_value_info("Kt", TensorProto.FLOAT, [1, 4, 2])
    V = helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, 2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 4])
    scale = numpy_helper.from_array(
        np.array(0.5, dtype=np.float32), "scale")

    qk = helper.make_node("MatMul", ["Q", "Kt"], ["scores"],
                          name="attention_qk")
    scale_n = helper.make_node("Mul", ["scores", "scale"],
                               ["scaled_scores"], name="attention_scale")
    softmax = helper.make_node("Softmax", ["scaled_scores"], ["probs"],
                               name="attention_softmax", axis=-1)
    out = helper.make_node("MatMul", ["probs", "V"], ["Y"],
                           name="attention_pv")

    graph = helper.make_graph([qk, scale_n, softmax, out], "attention",
                              [Q, Kt, V], [Y], initializer=[scale])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_attention.onnx", out_dir)


def make_lowering_rmsnorm(out_dir):
    """X(2,4), weight(4) → RMSNorm(eps=1e-5) → Y(2,4).
    Builds RMSNorm from ONNX primitive ops so tier-3 legalization must support
    reduction, sqrt/divide and broadcasting."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    weight = numpy_helper.from_array(np.ones(4, dtype=np.float32), "weight")
    two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two")
    eps = numpy_helper.from_array(np.array(1.0e-5, dtype=np.float32), "eps")

    square = helper.make_node("Pow", ["X", "two"], ["x2"], name="rms_square")
    mean = helper.make_node("ReduceMean", ["x2"], ["mean"],
                            name="rms_mean", axes=[-1], keepdims=1)
    add_eps = helper.make_node("Add", ["mean", "eps"], ["var_eps"],
                               name="rms_add_eps")
    sqrt = helper.make_node("Sqrt", ["var_eps"], ["denom"], name="rms_sqrt")
    norm = helper.make_node("Div", ["X", "denom"], ["norm"], name="rms_div")
    mul = helper.make_node("Mul", ["norm", "weight"], ["Y"],
                           name="rms_weight")

    graph = helper.make_graph([square, mean, add_eps, sqrt, norm, mul],
                              "rmsnorm", [X], [Y],
                              initializer=[weight, two, eps])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_rmsnorm.onnx", out_dir)


def make_lowering_layernorm(out_dir):
    """X(2,4), gamma(4), beta(4) → LayerNorm(eps=1e-5) via primitive ops."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    gamma = numpy_helper.from_array(np.ones(4, dtype=np.float32), "gamma")
    beta = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "beta")
    eps = numpy_helper.from_array(np.array(1.0e-5, dtype=np.float32), "eps")

    mean = helper.make_node("ReduceMean", ["X"], ["mean"],
                            name="ln_mean", axes=[-1], keepdims=1)
    centered = helper.make_node("Sub", ["X", "mean"], ["centered"],
                                name="ln_center")
    sq = helper.make_node("Mul", ["centered", "centered"], ["sq"], name="ln_sq")
    var = helper.make_node("ReduceMean", ["sq"], ["var"],
                           name="ln_var", axes=[-1], keepdims=1)
    add_eps = helper.make_node("Add", ["var", "eps"], ["var_eps"],
                               name="ln_add_eps")
    denom = helper.make_node("Sqrt", ["var_eps"], ["denom"], name="ln_sqrt")
    norm = helper.make_node("Div", ["centered", "denom"], ["norm"],
                            name="ln_div")
    scaled = helper.make_node("Mul", ["norm", "gamma"], ["scaled"],
                              name="ln_scale")
    out = helper.make_node("Add", ["scaled", "beta"], ["Y"], name="ln_out")

    graph = helper.make_graph(
        [mean, centered, sq, var, add_eps, denom, norm, scaled, out],
        "layernorm", [X], [Y], initializer=[gamma, beta, eps])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_layernorm.onnx", out_dir)


def make_lowering_rope(out_dir):
    """X(1,2,4) + precomputed cos/sin(1,2,4) → RoPE via rotate_half decomposition.
    out = X * cos + rotate_half(X) * sin
    rotate_half: concat(-X[..., d/2:], X[..., :d/2]) on last axis."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 4])

    seq, dim = 2, 4
    pos = np.arange(seq, dtype=np.float32)
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    angles = np.outer(pos, inv_freq)
    angles_full = np.repeat(angles, 2, axis=1)
    cos = np.cos(angles_full).reshape(1, seq, dim).astype(np.float32)
    sin = np.sin(angles_full).reshape(1, seq, dim).astype(np.float32)

    cos_init = numpy_helper.from_array(cos, "cos")
    sin_init = numpy_helper.from_array(sin, "sin")
    neg_one = numpy_helper.from_array(np.array(-1.0, dtype=np.float32), "neg_one")
    starts_left = numpy_helper.from_array(
        np.array([0, 0, 0], dtype=np.int64), "starts_left")
    ends_left = numpy_helper.from_array(
        np.array([1, 2, 2], dtype=np.int64), "ends_left")
    starts_right = numpy_helper.from_array(
        np.array([0, 0, 2], dtype=np.int64), "starts_right")
    ends_right = numpy_helper.from_array(
        np.array([1, 2, 4], dtype=np.int64), "ends_right")
    axes = numpy_helper.from_array(
        np.array([0, 1, 2], dtype=np.int64), "slice_axes")

    slice_left = helper.make_node(
        "Slice", ["X", "starts_left", "ends_left", "slice_axes"],
        ["x_left"], name="rope_slice_left")
    slice_right = helper.make_node(
        "Slice", ["X", "starts_right", "ends_right", "slice_axes"],
        ["x_right"], name="rope_slice_right")
    neg_right = helper.make_node(
        "Mul", ["x_right", "neg_one"], ["neg_right"], name="rope_neg")
    x_rot = helper.make_node(
        "Concat", ["neg_right", "x_left"], ["x_rot"],
        name="rope_rotate", axis=2)
    x_cos = helper.make_node("Mul", ["X", "cos"], ["x_cos"], name="rope_x_cos")
    rot_sin = helper.make_node(
        "Mul", ["x_rot", "sin"], ["rot_sin"], name="rope_rot_sin")
    out = helper.make_node("Add", ["x_cos", "rot_sin"], ["Y"], name="rope_out")

    inits = [cos_init, sin_init, neg_one, starts_left, ends_left,
             starts_right, ends_right, axes]
    graph = helper.make_graph(
        [slice_left, slice_right, neg_right, x_rot, x_cos, rot_sin, out],
        "rope", [X], [Y], initializer=inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_rope.onnx", out_dir)


def make_lowering_gelu(out_dir):
    """X(2,4) → GELU via 0.5 * x * (1 + erf(x / sqrt(2)))."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    half = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half")
    one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one")
    two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two")

    sqrt2 = helper.make_node("Sqrt", ["two"], ["sqrt2"], name="gelu_sqrt2")
    scaled = helper.make_node("Div", ["X", "sqrt2"], ["scaled"], name="gelu_scaled")
    erf = helper.make_node("Erf", ["scaled"], ["erf_val"], name="gelu_erf")
    inner = helper.make_node("Add", ["one", "erf_val"], ["inner"], name="gelu_inner")
    half_x = helper.make_node("Mul", ["X", "half"], ["half_x"], name="gelu_half_x")
    out = helper.make_node("Mul", ["half_x", "inner"], ["Y"], name="gelu_out")

    graph = helper.make_graph(
        [sqrt2, scaled, erf, inner, half_x, out],
        "gelu", [X], [Y], initializer=[half, one, two])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_gelu.onnx", out_dir)


def make_lowering_swiglu(out_dir):
    """Gate(2,4), Up(2,4) → SwiGLU = silu(gate) * up, silu(x)=x/(1+exp(-x))."""
    Gate = helper.make_tensor_value_info("Gate", TensorProto.FLOAT, [2, 4])
    Up = helper.make_tensor_value_info("Up", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one")

    neg = helper.make_node("Neg", ["Gate"], ["neg_gate"], name="swiglu_neg")
    exp = helper.make_node("Exp", ["neg_gate"], ["exp_neg"], name="swiglu_exp")
    denom = helper.make_node("Add", ["one", "exp_neg"], ["denom"], name="swiglu_denom")
    sigmoid = helper.make_node("Div", ["one", "denom"], ["sigmoid"], name="swiglu_sigmoid")
    silu = helper.make_node("Mul", ["Gate", "sigmoid"], ["silu"], name="swiglu_silu")
    out = helper.make_node("Mul", ["silu", "Up"], ["Y"], name="swiglu_out")

    graph = helper.make_graph(
        [neg, exp, denom, sigmoid, silu, out],
        "swiglu", [Gate, Up], [Y], initializer=[one])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_swiglu.onnx", out_dir)


def make_lowering_matmul_bias(out_dir):
    """X(2,3) → MatMul(W(3,4)) → Add(bias(4)) → Y(2,4)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    W = numpy_helper.from_array(np.eye(3, 4, dtype=np.float32), "W")
    bias = numpy_helper.from_array(np.full(4, 0.25, dtype=np.float32), "bias")

    mm = helper.make_node("MatMul", ["X", "W"], ["mm_out"], name="mm_0")
    add = helper.make_node("Add", ["mm_out", "bias"], ["Y"], name="add_bias")

    graph = helper.make_graph([mm, add], "matmul_bias", [X], [Y],
                              initializer=[W, bias])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_matmul_bias.onnx", out_dir)


def make_lowering_qdq_matmul(out_dir):
    """X(2,3), W(3,4) → dq chains (x-zp)*scale → MatMul → Y(2,4)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    W = numpy_helper.from_array(np.eye(3, 4, dtype=np.float32), "W")
    scale_x = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale_x")
    zp_x = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zp_x")
    scale_w = numpy_helper.from_array(np.array(0.2, dtype=np.float32), "scale_w")
    zp_w = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zp_w")

    sub_x = helper.make_node("Sub", ["X", "zp_x"], ["sub_x"], name="sub_x")
    dq_x = helper.make_node("Mul", ["sub_x", "scale_x"], ["dq_x"], name="dq_x")
    sub_w = helper.make_node("Sub", ["W", "zp_w"], ["sub_w"], name="sub_w")
    dq_w = helper.make_node("Mul", ["sub_w", "scale_w"], ["dq_w"], name="dq_w")
    mm = helper.make_node("MatMul", ["dq_x", "dq_w"], ["Y"], name="mm_0")

    graph = helper.make_graph(
        [sub_x, dq_x, sub_w, dq_w, mm],
        "qdq_matmul", [X], [Y],
        initializer=[W, scale_x, zp_x, scale_w, zp_w])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_qdq_matmul.onnx", out_dir)


def make_lowering_transformer_block(out_dir):
    """Pre-LN teaching block: X(2,4) → RMSNorm → Attention → residual
    → RMSNorm → SwiGLU(gate=up=norm) → residual → Y(2,4)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])

    weight = numpy_helper.from_array(np.ones(4, dtype=np.float32), "weight")
    two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two")
    eps = numpy_helper.from_array(np.array(1.0e-5, dtype=np.float32), "eps")
    Kt = numpy_helper.from_array(
        np.array([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]],
                 dtype=np.float32),
        "Kt")
    V = numpy_helper.from_array(
        np.array([[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]],
                 dtype=np.float32),
        "V")
    scale = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "scale")
    shape_124 = numpy_helper.from_array(
        np.array([1, 2, 4], dtype=np.int64), "shape_124")
    shape_24 = numpy_helper.from_array(
        np.array([2, 4], dtype=np.int64), "shape_24")
    one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one")

    square1 = helper.make_node("Pow", ["X", "two"], ["x2_1"], name="rms1_square")
    mean1 = helper.make_node("ReduceMean", ["x2_1"], ["mean_1"],
                             name="rms1_mean", axes=[-1], keepdims=1)
    add_eps1 = helper.make_node("Add", ["mean_1", "eps"], ["var_eps_1"],
                                name="rms1_add_eps")
    sqrt1 = helper.make_node("Sqrt", ["var_eps_1"], ["denom_1"], name="rms1_sqrt")
    norm1 = helper.make_node("Div", ["X", "denom_1"], ["norm1"], name="rms1_div")
    norm1_w = helper.make_node("Mul", ["norm1", "weight"], ["norm1_w"],
                               name="rms1_weight")

    q = helper.make_node("Reshape", ["norm1_w", "shape_124"], ["Q"], name="reshape_q")
    scores = helper.make_node("MatMul", ["Q", "Kt"], ["scores"], name="attn_qk")
    scaled = helper.make_node("Mul", ["scores", "scale"], ["scaled_scores"],
                              name="attn_scale")
    probs = helper.make_node("Softmax", ["scaled_scores"], ["probs"],
                             name="attn_softmax", axis=-1)
    attn = helper.make_node("MatMul", ["probs", "V"], ["attn_out"], name="attn_pv")
    attn_2d = helper.make_node("Reshape", ["attn_out", "shape_24"], ["attn_2d"],
                               name="reshape_attn")

    h = helper.make_node("Add", ["X", "attn_2d"], ["h"], name="res1")

    square2 = helper.make_node("Pow", ["h", "two"], ["x2_2"], name="rms2_square")
    mean2 = helper.make_node("ReduceMean", ["x2_2"], ["mean_2"],
                             name="rms2_mean", axes=[-1], keepdims=1)
    add_eps2 = helper.make_node("Add", ["mean_2", "eps"], ["var_eps_2"],
                                name="rms2_add_eps")
    sqrt2 = helper.make_node("Sqrt", ["var_eps_2"], ["denom_2"], name="rms2_sqrt")
    norm2 = helper.make_node("Div", ["h", "denom_2"], ["norm2"], name="rms2_div")
    norm2_w = helper.make_node("Mul", ["norm2", "weight"], ["norm2_w"],
                               name="rms2_weight")

    neg = helper.make_node("Neg", ["norm2_w"], ["neg_gate"], name="swiglu_neg")
    exp = helper.make_node("Exp", ["neg_gate"], ["exp_neg"], name="swiglu_exp")
    denom = helper.make_node("Add", ["one", "exp_neg"], ["denom_swiglu"],
                             name="swiglu_denom")
    sigmoid = helper.make_node("Div", ["one", "denom_swiglu"], ["sigmoid"],
                               name="swiglu_sigmoid")
    silu = helper.make_node("Mul", ["norm2_w", "sigmoid"], ["silu"],
                            name="swiglu_silu")
    ffn = helper.make_node("Mul", ["silu", "norm2_w"], ["ffn_out"],
                           name="swiglu_out")

    out = helper.make_node("Add", ["h", "ffn_out"], ["Y"], name="res2")

    nodes = [
        square1, mean1, add_eps1, sqrt1, norm1, norm1_w,
        q, scores, scaled, probs, attn, attn_2d, h,
        square2, mean2, add_eps2, sqrt2, norm2, norm2_w,
        neg, exp, denom, sigmoid, silu, ffn, out,
    ]
    inits = [weight, two, eps, Kt, V, scale, shape_124, shape_24, one]
    graph = helper.make_graph(nodes, "transformer_block", [X], [Y], initializer=inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_transformer_block.onnx", out_dir)


def make_lowering_matmul_softmax(out_dir):
    """A(2,4) @ B(4,4) → MatMul → Softmax(axis=-1) → Y(2,4).
    Exercises producer-consumer fusion (GEMM scores → softmax exp)."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    B_init = numpy_helper.from_array(
        np.eye(4, dtype=np.float32), "B")

    mm = helper.make_node("MatMul", ["A", "B"], ["scores"], name="mm_0")
    softmax = helper.make_node("Softmax", ["scores"], ["Y"],
                               name="softmax_0", axis=-1)
    graph = helper.make_graph([mm, softmax], "matmul_softmax", [A], [Y],
                              initializer=[B_init])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_matmul_softmax.onnx", out_dir)


def make_lowering_layout_conv(out_dir):
    """X(NCHW 1x2x2x2) → Conv → Transpose[0,2,3,1] → Y(NHWC layout teaching)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 2, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 2, 2])
    W = numpy_helper.from_array(
        np.full((2, 2, 1, 1), 0.5, dtype=np.float32), "W")

    conv = helper.make_node("Conv", ["X", "W"], ["conv_out"], name="conv_0",
                            kernel_shape=[1, 1], pads=[0, 0, 0, 0],
                            strides=[1, 1])
    transpose = helper.make_node(
        "Transpose", ["conv_out"], ["Y"], name="to_nhwc", perm=[0, 2, 3, 1])
    graph = helper.make_graph([conv, transpose], "layout_conv", [X], [Y],
                              initializer=[W])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_layout_conv.onnx", out_dir)


def make_lowering_decode_step(out_dir):
    """Q(1,1,4), Kt(1,4,3), V(1,3,4) → scaled dot-product attention → Y(1,1,4).
    seq=1 decode step: one query token attends over a small KV cache."""
    Q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, 1, 4])
    Kt = helper.make_tensor_value_info("Kt", TensorProto.FLOAT, [1, 4, 3])
    V = helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, 3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4])
    scale = numpy_helper.from_array(
        np.array(0.5, dtype=np.float32), "scale")

    qk = helper.make_node("MatMul", ["Q", "Kt"], ["scores"],
                          name="decode_qk")
    scale_n = helper.make_node("Mul", ["scores", "scale"],
                               ["scaled_scores"], name="decode_scale")
    softmax = helper.make_node("Softmax", ["scaled_scores"], ["probs"],
                               name="decode_softmax", axis=-1)
    out = helper.make_node("MatMul", ["probs", "V"], ["Y"],
                           name="decode_pv")

    graph = helper.make_graph([qk, scale_n, softmax, out], "decode_step",
                              [Q, Kt, V], [Y], initializer=[scale])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_decode_step.onnx", out_dir)


def make_lowering_dynamic_mn(out_dir):
    """A(?xK) @ B(Kx?) → Y(?x?). Both MatMul operands have symbolic M/N dims."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, ["K", "N"])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["M", "N"])

    mm = helper.make_node("MatMul", ["A", "B"], ["Y"], name="mm_0")
    graph = helper.make_graph([mm], "dynamic_mn", [A, B], [Y])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_dynamic_mn.onnx", out_dir)


def make_lowering_matmul_f16(out_dir):
    """X(2,3) → MatMul(W(3,4)) → Y(2,4) in FP16."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 4])
    W = numpy_helper.from_array(
        np.eye(3, 4, dtype=np.float16), "W")

    mm = helper.make_node("MatMul", ["X", "W"], ["Y"], name="mm_0")
    graph = helper.make_graph([mm], "matmul_f16", [X], [Y], initializer=[W])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_matmul_f16.onnx", out_dir)


def make_lowering_horizontal_gemm(out_dir):
    """X(2,3) → MatMul(W1), MatMul(W2) → Concat(axis=1) → Y(2,4)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    W1 = numpy_helper.from_array(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32), "W1")
    W2 = numpy_helper.from_array(
        np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float32), "W2")

    mm1 = helper.make_node("MatMul", ["X", "W1"], ["y1"], name="mm_1")
    mm2 = helper.make_node("MatMul", ["X", "W2"], ["y2"], name="mm_2")
    concat = helper.make_node("Concat", ["y1", "y2"], ["Y"],
                              name="concat_0", axis=1)

    graph = helper.make_graph([mm1, mm2, concat], "horizontal_gemm", [X], [Y],
                              initializer=[W1, W2])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m = onnx.shape_inference.infer_shapes(m)
    _save(m, "lowering_horizontal_gemm.onnx", out_dir)


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
    make_lowering_softmax(out_dir)
    make_lowering_attention(out_dir)
    make_lowering_rmsnorm(out_dir)
    make_lowering_layernorm(out_dir)
    make_lowering_rope(out_dir)
    make_lowering_gelu(out_dir)
    make_lowering_swiglu(out_dir)
    make_lowering_matmul_bias(out_dir)
    make_lowering_qdq_matmul(out_dir)
    make_lowering_matmul_softmax(out_dir)
    make_lowering_layout_conv(out_dir)
    make_lowering_horizontal_gemm(out_dir)
    make_lowering_transformer_block(out_dir)
    make_lowering_decode_step(out_dir)
    make_lowering_dynamic_mn(out_dir)
    make_lowering_matmul_f16(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
