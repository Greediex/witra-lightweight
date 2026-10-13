#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if (( $# < 1 || $# > 2 )); then
  echo 'Usage: bash run_uts.sh train|test|sweep [seed]' >&2
  exit 2
fi
action="$1"
seed="${2:-42}"
python_bin="${WITRA_PYTHON:-python}"
output_root="${WITRA_RUNS:-../WiTra_Depth2_D160_FFN1_FloorProtect_BF16_Runs}"
run="$output_root/uts/MAIN/seed_$seed"
device="${WITRA_DEVICE:-cuda}"
case "$action" in
  train)
    exec "$python_bin" -B run.py train \
      --prepared "${WITRA_PREPARED_ROOT:-../WiTra_Direct_Prepared}/uts_mean_center" \
      --output-root "$output_root" --variant MAIN --seed "$seed" \
      --device "$device" --precision bf16 --attention-backend sdpa \
      --conditioning-update floor-protect --depth 2 --mlp-ratio 1 \
      --embed-dim 160 --num-heads 8 --num-bins 100 \
      --lr 0.0001 --batch-size 32 --epochs 300 \
      --num-workers "${WITRA_WORKERS:-0}"
    ;;
  test)
    exec "$python_bin" -B run.py evaluate --run-dir "$run" --device "$device"
    ;;
  sweep)
    exec "$python_bin" -B run.py sweep --run-dir "$run" \
      --output "$run/test_sweep_last100_exploratory.json" --device "$device"
    ;;
  *) echo "Unknown action: $action (use train, test, or sweep)" >&2; exit 2 ;;
esac
