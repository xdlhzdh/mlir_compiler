#!/usr/bin/env python3
"""PTQ calibration demo: numpy min/max on a calibration set → scales (A6)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Simulated calibration batches for a single activation tensor (teaching).
NUM_BATCHES = 100
RNG = np.random.default_rng(42)


def calibrate_tensor(name: str, scale_factor: float = 1.0) -> dict:
    samples = RNG.normal(0.0, 1.0, size=NUM_BATCHES).astype(np.float32) * scale_factor
    observed_min = float(samples.min())
    observed_max = float(samples.max())
    if observed_min < 0:
        # Symmetric INT8 scale (teaching simplification).
        abs_max = max(abs(observed_min), abs(observed_max))
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        zp = 0
    else:
        scale = (observed_max - observed_min) / 255.0 if observed_max > observed_min else 1.0
        zp = int(round(-observed_min / scale)) if scale > 0 else 0
    return {
        "tensor": name,
        "observed_min": observed_min,
        "observed_max": observed_max,
        "scale": scale,
        "zero_point": zp,
    }


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    records = [
        calibrate_tensor("conv1_out", scale_factor=2.5),
        calibrate_tensor("relu1_out", scale_factor=1.5),
        calibrate_tensor("fc_out", scale_factor=5.0),
    ]
    print("P11 calibration demo (numpy min/max on simulated batches)")
    for rec in records:
        print(
            f"  {rec['tensor']}: min={rec['observed_min']:.4f} max={rec['observed_max']:.4f} "
            f"scale={rec['scale']:.6f} zp={rec['zero_point']}"
        )
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")
    print("PASS calibration demo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
