#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="${1:?Usage: prepare_data.sh uji|uts|tampere [mean_center|power] [EXTERNAL_DATA_ROOT]}"
norm="${2:-mean_center}"
data_root="${3:-${WITRA_DATA_ROOT:-../WiTra_Revision_Full_20261005/WiTra_Server_New/data/indoorloc}}"
if [[ ! -d "$data_root" ]]; then
  echo 'Raw data is external. Set WITRA_DATA_ROOT or pass EXTERNAL_DATA_ROOT as argument 3.' >&2
  exit 2
fi
exec "${WITRA_PYTHON:-.venv/bin/python}" -B prepare_data.py --dataset "$dataset" --norm-method "$norm" --data-root "$data_root" --output-dir "${WITRA_PREPARED_ROOT:-../WiTra_Direct_Prepared}/${dataset}_${norm}"
