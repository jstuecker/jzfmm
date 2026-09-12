#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR=/tmp/jzfmm-s4g-matplotlib
PYTHON=${PYTHON:-/home/bender/.venvs/jzfmm/bin/python}
# Serialize GPU workloads so cost measurements do not contend with simulations.
"$PYTHON" -u checks/symplectic_o4/gpu_cost.py --n 1000000 --output checks/symplectic_o4/results/gpu_n1000000_float32.json
"$PYTHON" -u checks/symplectic_o4/gpu_cost.py --n 100000 --output checks/symplectic_o4/results/gpu_n100000_float32.json
"$PYTHON" checks/symplectic_o4/decision.py --results checks/symplectic_o4/results --cost checks/symplectic_o4/results/gpu_n100000_float32.json
if "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open("checks/symplectic_o4/results/decision.json"))["promising"] else 1)'; then
  "$PYTHON" -u checks/symplectic_o4/nbody.py --output checks/symplectic_o4/results/nbody
else
  echo 'Phase C does not exceed the 2x median threshold; skipping Phase D.'
fi
