#!/usr/bin/env bash
#
# Full background pipeline for the local TENT reimplementation.
# It runs these stages in order inside the `tent` conda environment:
# 1. Train or resume CIFAR-10 ResNet-26 training.
# 2. Evaluate the saved checkpoint with source adaptation.
# 3. Evaluate the saved checkpoint with norm adaptation.
# 4. Evaluate the saved checkpoint with tent adaptation.
# 5. Generate plots and results.csv.
#
# Common overrides:
#   CKPT_PATH=./ckpt/cifar10/resnet26_best.pth
#   EPOCHS=200
#   BATCH_SIZE=128
#   EVAL_BATCH_SIZE=256
#   TRAIN_WORKERS=4
#   EVAL_WORKERS=2
#   RESUME=1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CKPT_PATH="${CKPT_PATH:-./ckpt/cifar10/resnet26_best.pth}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
RESUME="${RESUME:-1}"

mkdir -p output ckpt/cifar10

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

CONDA_RUN=(conda run --no-capture-output -n tent)
TRAIN_CMD=(
  "${CONDA_RUN[@]}"
  python
  train_cifar10.py
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --num-workers "$TRAIN_WORKERS"
  --ckpt-path "$CKPT_PATH"
)

if [[ "$RESUME" == "1" ]]; then
  TRAIN_CMD+=(--resume)
fi

echo "[$(timestamp)] Starting CIFAR-10 training"
"${TRAIN_CMD[@]}"

echo "[$(timestamp)] Training complete, starting source evaluation"
"${CONDA_RUN[@]}" python cifar10c.py \
  --cfg cfgs/source.yaml \
  MODEL.CKPT_PATH "$CKPT_PATH" \
  TEST.NUM_WORKERS "$EVAL_WORKERS"

echo "[$(timestamp)] Source complete, starting norm evaluation"
"${CONDA_RUN[@]}" python cifar10c.py \
  --cfg cfgs/norm.yaml \
  MODEL.CKPT_PATH "$CKPT_PATH" \
  TEST.NUM_WORKERS "$EVAL_WORKERS"

echo "[$(timestamp)] Norm complete, starting tent evaluation"
"${CONDA_RUN[@]}" python cifar10c.py \
  --cfg cfgs/tent.yaml \
  MODEL.CKPT_PATH "$CKPT_PATH" \
  TEST.NUM_WORKERS "$EVAL_WORKERS"

echo "[$(timestamp)] Evaluations complete, generating plots"
"${CONDA_RUN[@]}" python plots_tent.py

echo "[$(timestamp)] Full pipeline complete"
