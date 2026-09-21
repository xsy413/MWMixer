#!/usr/bin/env bash
set -euo pipefail

pred_len="${1:-96}"

case "${pred_len}" in
  96|192|336|720) ;;
  *) echo "Unsupported prediction length: ${pred_len}" >&2; exit 2 ;;
esac

python run.py \
  --is_training 1 \
  --model MWMixer \
  --model_id "ETTh1_96_${pred_len}" \
  --des paper_reproduction \
  --data ETTh1 \
  --root_path "./dataset/ETT-small/" \
  --data_path ETTh1.csv \
  --features M \
  --freq h \
  --seq_len 96 \
  --pred_len "${pred_len}" \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --train_epochs 30 \
  --patience 10 \
  --itr 3
