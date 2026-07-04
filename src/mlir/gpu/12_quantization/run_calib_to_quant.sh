#!/usr/bin/env bash
# P12: calibration demo JSON -> run_quantization (scale echo).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${MLIR_COMPILER_BUILD:-$ROOT/../../../../build}"
QUANT_DIR="$ROOT"
JSON="${TMPDIR:-/tmp}/aicom_calib_scales.json"
RUN_QUANT="$BUILD/src/mlir/gpu/run_quantization"

python3 "$QUANT_DIR/run_calib_demo.py" "$JSON"

if [[ ! -x "$RUN_QUANT" ]]; then
  echo "skip: missing $RUN_QUANT" >&2
  exit 0
fi

echo "== P12 run_quantization (post-calibration teaching) =="
"$RUN_QUANT" 2>&1 | grep -q 'Stage 2'

echo "Calibration -> quant pipeline passed."
echo "Scales JSON: $JSON"
