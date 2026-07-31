#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
DATASET="${DATASET:-data/processed/log_states_v3.jsonl}"
ENCODER="${ENCODER:-models/bge-small-en-v1.5}"

if [[ ! -x "$PY" ]]; then
  echo "Python environment missing: $PY" >&2
  echo "Run scripts/upgrade_micro_log_state_v3.sh . first." >&2
  exit 2
fi

if [[ ! -f "$DATASET" ]]; then
  echo "Dataset missing: $DATASET" >&2
  echo "Run the local data factory first." >&2
  exit 2
fi

"$PY" -m src.train \
  --data "$DATASET" \
  --encoder-path "$ENCODER" \
  --epochs "${EPOCHS:-60}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-256}" \
  --train-batch-size "${TRAIN_BATCH_SIZE:-512}" \
  --seed "${SEED:-1337}"

"$PY" -m src.predict --json "high load"
"$PY" -m src.predict --json "filesystem not full writable"
"$PY" -m src.predict --json "nginx.service entered failed state"
