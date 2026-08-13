#!/usr/bin/env python3
"""ONNX Runtime vs NumPy reference checks for P4 fixture `.onnx` files.

Does not run lowering or read StableHLO.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is required for run_onnx_golden", file=sys.stderr)
    sys.exit(1)

RTOL = 1e-5
ATOL = 1e-5


def _init_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _run_ort(model_path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    return dict(zip(
        [o.name for o in sess.get_outputs()],
        sess.run(None, {k: v for k, v in feeds.items() if k in names}),
    ))


def _assert_close(name: str, got: np.ndarray, expected: np.ndarray) -> None:
    if not np.allclose(got, expected, rtol=RTOL, atol=ATOL):
        diff = np.max(np.abs(got.astype(np.float64) - expected.astype(np.float64)))
        raise AssertionError(
            f"{name}: max abs diff {diff:.3e} exceeds rtol={RTOL}, atol={ATOL}")


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    left = x[..., : d // 2]
    right = x[..., d // 2 :]
    return np.concatenate([-right, left], axis=-1)


def check_basic(model_dir: Path) -> None:
    path = model_dir / "lowering_basic.onnx"
    model = onnx.load(str(path))
    inits = _init_map(model)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    expected = (x + inits["bias"]) @ inits["W"]
    out = _run_ort(path, {"X": x})["Y"]
    _assert_close("lowering_basic", out, expected)
    print("  PASS lowering_basic")


def check_softmax(model_dir: Path) -> None:
    path = model_dir / "lowering_softmax.onnx"
    x = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]], dtype=np.float32)
    expected = _softmax(x, axis=-1)
    out = _run_ort(path, {"X": x})["Y"]
    _assert_close("lowering_softmax", out, expected)
    print("  PASS lowering_softmax")


def check_attention(model_dir: Path) -> None:
    path = model_dir / "lowering_attention.onnx"
    model = onnx.load(str(path))
    inits = _init_map(model)
    q = np.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=np.float32)
    kt = np.array([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    v = np.array([[[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]], dtype=np.float32)
    scores = q @ kt
    scaled = scores * inits["scale"]
    probs = _softmax(scaled, axis=-1)
    expected = probs @ v
    out = _run_ort(path, {"Q": q, "Kt": kt, "V": v})["Y"]
    _assert_close("lowering_attention", out, expected)
    print("  PASS lowering_attention")


def check_rmsnorm(model_dir: Path) -> None:
    path = model_dir / "lowering_rmsnorm.onnx"
    model = onnx.load(str(path))
    inits = _init_map(model)
    x = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    eps = float(inits["eps"])
    weight = inits["weight"]
    var = np.mean(x * x, axis=-1, keepdims=True)
    expected = (x / np.sqrt(var + eps)) * weight
    out = _run_ort(path, {"X": x})["Y"]
    _assert_close("lowering_rmsnorm", out, expected)
    print("  PASS lowering_rmsnorm")


def check_layernorm(model_dir: Path) -> None:
    path = model_dir / "lowering_layernorm.onnx"
    model = onnx.load(str(path))
    inits = _init_map(model)
    x = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    eps = float(inits["eps"])
    gamma = inits["gamma"]
    beta = inits["beta"]
    mean = np.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True)
    expected = centered / np.sqrt(var + eps) * gamma + beta
    out = _run_ort(path, {"X": x})["Y"]
    _assert_close("lowering_layernorm", out, expected)
    print("  PASS lowering_layernorm")


from math import erf


def check_rope(model_dir: Path) -> None:
    path = model_dir / "lowering_rope.onnx"
    model = onnx.load(str(path))
    inits = _init_map(model)
    x = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    cos, sin = inits["cos"], inits["sin"]
    expected = x * cos + _rotate_half(x) * sin
    out = _run_ort(path, {"X": x})["Y"]
    _assert_close("lowering_rope", out, expected)
    print("  PASS lowering_rope")


def check_gelu(model_dir: Path) -> None:
    path = model_dir / "lowering_gelu.onnx"
    x = np.array([[-1.0, 0.0, 1.0, 2.0], [0.5, -0.5, 1.5, -1.5]], dtype=np.float32)
    expected = 0.5 * x * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))
    out = _run_ort(path, {"X": x})["Y"]
    _assert_close("lowering_gelu", out, expected)
    print("  PASS lowering_gelu")


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def check_swiglu(model_dir: Path) -> None:
    path = model_dir / "lowering_swiglu.onnx"
    gate = np.array([[-1.0, 0.0, 1.0, 2.0], [0.5, -0.5, 1.5, -1.5]], dtype=np.float32)
    up = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    expected = _silu(gate) * up
    out = _run_ort(path, {"Gate": gate, "Up": up})["Y"]
    _assert_close("lowering_swiglu", out, expected)
    print("  PASS lowering_swiglu")


def check_matmul_bias(model_dir: Path) -> None:
    path = model_dir / "lowering_matmul_bias.onnx"
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    w = np.eye(3, 4, dtype=np.float32)
    bias = np.full(4, 0.25, dtype=np.float32)
    expected = x @ w + bias
    out = _run_ort(path, {"X": x})["Y"]
    _assert_close("lowering_matmul_bias", out, expected)
    print("  PASS lowering_matmul_bias")


def check_qdq_matmul(model_dir: Path) -> None:
    path = model_dir / "lowering_qdq_matmul.onnx"
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    w = np.eye(3, 4, dtype=np.float32)
    scale_x, zp_x = 0.1, 0.0
    scale_w, zp_w = 0.2, 0.0
    dq_x = (x - zp_x) * scale_x
    dq_w = (w - zp_w) * scale_w
    expected = dq_x @ dq_w
    out = _run_ort(path, {"X": x})["Y"]
    _assert_close("lowering_qdq_matmul", out, expected)
    print("  PASS lowering_qdq_matmul")


def check_horizontal_gemm(model_dir: Path) -> None:
    path = model_dir / "lowering_horizontal_gemm.onnx"
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    w1 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    w2 = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float32)
    expected = np.concatenate([x @ w1, x @ w2], axis=1)
    out = _run_ort(path, {"X": x})["Y"]
    _assert_close("lowering_horizontal_gemm", out, expected)
    print("  PASS lowering_horizontal_gemm")


def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * weight


def check_transformer_block(model_dir: Path) -> None:
    path = model_dir / "lowering_transformer_block.onnx"
    model = onnx.load(str(path))
    inits = _init_map(model)
    x = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    eps = float(inits["eps"])
    weight = inits["weight"]
    kt = inits["Kt"]
    v = inits["V"]
    scale = float(inits["scale"])

    norm1 = _rmsnorm(x, weight, eps)
    q = norm1.reshape(1, 2, 4)
    scores = q @ kt
    scaled = scores * scale
    probs = _softmax(scaled, axis=-1)
    attn = (probs @ v).reshape(2, 4)
    h = x + attn
    norm2 = _rmsnorm(h, weight, eps)
    ffn = _silu(norm2) * norm2
    expected = h + ffn
    out = _run_ort(path, {"X": x})["Y"]
    _assert_close("lowering_transformer_block", out, expected)
    print("  PASS lowering_transformer_block")


def check_broadcast(model_dir: Path) -> None:
    path = model_dir / "lowering_broadcast.onnx"
    model = onnx.load(str(path))
    inits = _init_map(model)
    x = np.ones((2, 3, 4), dtype=np.float32)
    bias = inits["bias"]
    scale = inits["scale"]
    expected = (x + bias) * scale
    out = _run_ort(path, {"X": x})["Y"]
    _assert_close("lowering_broadcast", out, expected)
    print("  PASS lowering_broadcast")


def check_dynamic(model_dir: Path) -> None:
    path = model_dir / "lowering_dynamic.onnx"
    w = np.eye(3, 4, dtype=np.float32)
    for batch in (2, 4):
        x = np.arange(batch * 3, dtype=np.float32).reshape(batch, 3)
        bias = np.ones((batch, 3), dtype=np.float32)
        expected = (x + bias) @ w
        out = _run_ort(path, {"X": x, "bias": bias})["Y"]
        _assert_close(f"lowering_dynamic(batch={batch})", out, expected)
    print("  PASS lowering_dynamic")


def check_decode_step(model_dir: Path) -> None:
    path = model_dir / "lowering_decode_step.onnx"
    model = onnx.load(str(path))
    inits = _init_map(model)
    q = np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    kt = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
    v = np.array([[[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0],
                   [0.0, 0.0, 4.0, 0.0]]], dtype=np.float32)
    scores = q @ kt
    scaled = scores * inits["scale"]
    probs = _softmax(scaled, axis=-1)
    expected = probs @ v
    out = _run_ort(path, {"Q": q, "Kt": kt, "V": v})["Y"]
    _assert_close("lowering_decode_step", out, expected)
    print("  PASS lowering_decode_step")


def check_dynamic_mn(model_dir: Path) -> None:
    path = model_dir / "lowering_dynamic_mn.onnx"
    cases = (
        (2, 3, 4),
        (3, 4, 2),
    )
    for m, k, n in cases:
        a = np.arange(m * k, dtype=np.float32).reshape(m, k)
        b = np.arange(k * n, dtype=np.float32).reshape(k, n) * 0.1
        expected = a @ b
        out = _run_ort(path, {"A": a, "B": b})["Y"]
        _assert_close(f"lowering_dynamic_mn(M={m},K={k},N={n})", out, expected)
    print("  PASS lowering_dynamic_mn")


def check_matmul_f16(model_dir: Path) -> None:
    path = model_dir / "lowering_matmul_f16.onnx"
    model = onnx.load(str(path))
    inits = _init_map(model)
    x = np.arange(6, dtype=np.float16).reshape(2, 3)
    expected = (x.astype(np.float32) @ inits["W"].astype(np.float32)).astype(np.float16)
    out = _run_ort(path, {"X": x})["Y"]
    # FP16 matmul: relax tolerance vs FP32 golden checks
    if not np.allclose(out.astype(np.float32), expected.astype(np.float32),
                       rtol=1e-2, atol=1e-2):
        diff = np.max(np.abs(out.astype(np.float32) - expected.astype(np.float32)))
        raise AssertionError(
            f"lowering_matmul_f16: max abs diff {diff:.3e} exceeds rtol=1e-2, atol=1e-2")
    print("  PASS lowering_matmul_f16")


def check_quant_qdq_matmul(quant_dir: Path) -> None:
    path = quant_dir / "quant_qdq_matmul.onnx"
    if not path.is_file():
        raise FileNotFoundError(path)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = _run_ort(path, {"X": x})["Y"]
    inits = _init_map(onnx.load(str(path)))
    scale_x = float(inits["scale_x"])
    zp_x = float(inits["zp_x"])
    scale_w = float(inits["scale_w"])
    zp_w = float(inits["zp_w"])
    scale_y = float(inits["scale_y"])
    zp_y = float(inits["zp_y"])
    w = inits["W"]

    def dequant(q, scale, zp):
        return (q.astype(np.float32) - zp) * scale

    def quant(t, scale, zp):
        return np.clip(np.round(t / scale + zp), -128, 127).astype(np.int8)

    qx = quant(x, scale_x, zp_x)
    qw = quant(w, scale_w, zp_w)
    mm = dequant(qx, scale_x, zp_x) @ dequant(qw, scale_w, zp_w)
    qy = quant(mm, scale_y, zp_y)
    expected = dequant(qy, scale_y, zp_y)
    _assert_close("quant_qdq_matmul", out, expected)
    print("  PASS quant_qdq_matmul")


CHECKS = [
    check_basic,
    check_softmax,
    check_attention,
    check_rmsnorm,
    check_layernorm,
    check_rope,
    check_gelu,
    check_swiglu,
    check_matmul_bias,
    check_qdq_matmul,
    check_horizontal_gemm,
    check_broadcast,
    check_transformer_block,
    check_dynamic,
    check_decode_step,
    check_dynamic_mn,
    check_matmul_f16,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=".",
        help="Directory containing lowering_*.onnx fixtures",
    )
    parser.add_argument(
        "--quant-dir",
        default="",
        help="Directory containing quant_qdq_matmul.onnx (P11)",
    )
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    print("ONNX Runtime vs NumPy (run_onnx_golden; does not run lowering)")
    failed = 0
    for check in CHECKS:
        try:
            check(model_dir)
        except Exception as exc:  # noqa: BLE001 — test harness reports all failures
            print(f"  FAIL {check.__name__}: {exc}", file=sys.stderr)
            failed += 1

    extra = 0
    if args.quant_dir:
        try:
            check_quant_qdq_matmul(Path(args.quant_dir))
            extra = 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL check_quant_qdq_matmul: {exc}", file=sys.stderr)
            failed += 1

    total = len(CHECKS) + extra
    if failed:
        print(f"\n{failed} golden check(s) failed.", file=sys.stderr)
        return 1
    print(f"\nAll {total} golden checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
