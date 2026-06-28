#!/usr/bin/env python3
"""P4 golden numerical checks: ONNX Runtime vs NumPy reference for tier-3 fixtures."""

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
    print("ERROR: onnxruntime is required for run_lowering_golden", file=sys.stderr)
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


CHECKS = [
    check_basic,
    check_softmax,
    check_attention,
    check_rmsnorm,
    check_rope,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=".",
        help="Directory containing lowering_*.onnx fixtures",
    )
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    print("P4 golden numerical checks (ONNX Runtime vs NumPy)")
    failed = 0
    for check in CHECKS:
        try:
            check(model_dir)
        except Exception as exc:  # noqa: BLE001 — test harness reports all failures
            print(f"  FAIL {check.__name__}: {exc}", file=sys.stderr)
            failed += 1

    if failed:
        print(f"\n{failed} golden check(s) failed.", file=sys.stderr)
        return 1
    print(f"\nAll {len(CHECKS)} golden checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
