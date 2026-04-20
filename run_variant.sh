#!/usr/bin/env bash
# Run a variant across multiple seeds, then summarize.
# Usage: ./run_variant.sh <variant> [--seeds "0 1 2"] [--steps N] [--outdir outputs]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <variant> [--seeds \"0 1 2\"] [--steps N] [--outdir DIR]" >&2
  exit 2
fi

VARIANT="$1"; shift || true
SEEDS="0 1 2"
STEPS=""
OUTDIR="outputs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

for seed in $SEEDS; do
  echo "=== $VARIANT seed=$seed ==="
  if [[ -n "$STEPS" ]]; then
    python -m src.train --variant "$VARIANT" --seed "$seed" --outdir "$OUTDIR" --steps "$STEPS"
  else
    python -m src.train --variant "$VARIANT" --seed "$seed" --outdir "$OUTDIR"
  fi
done

# Summarize over whatever seeds are present on disk
python - <<PY
from src.train import summarize
import json
print("SUMMARY", json.dumps(summarize("$VARIANT", "$OUTDIR")))
PY
