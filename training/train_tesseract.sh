#!/usr/bin/env bash
# Fine-tune Tesseract's best English LSTM on our synthetic line data (CPU only, free).
#
#   python scripts/fetch_resources.py
#   python -m training.make_tess_lines --out data/train_lines --n 40000
#   bash training/train_tesseract.sh [max_iterations]
#
# Output: data/tessdata/eng_ft.traineddata
set -euo pipefail
cd "$(dirname "$0")/.."

ITER=${1:-30000}
WORK=data/train_work
BASE=data/tessdata/eng_best.traineddata
mkdir -p "$WORK"

combine_tessdata -e "$BASE" "$WORK/eng.lstm" >/dev/null

lstmtraining \
  --model_output "$WORK/eng_ft" \
  --continue_from "$WORK/eng.lstm" \
  --traineddata "$BASE" \
  --train_listfile data/train_lines/list.train \
  --eval_listfile data/train_lines/list.eval \
  --learning_rate 0.0001 \
  --max_iterations "$ITER" \
  --target_error_rate 0.005 \
  --debug_interval 0 2>&1 | tee "$WORK/train.log" | grep -E "At iteration|best|Finished" || true

# Export the best checkpoint (lowest error, encoded in the file name) as a normal .traineddata
BEST=$(ls "$WORK"/eng_ft_[0-9]*.checkpoint 2>/dev/null | awk -F'eng_ft_' '{print $2" "$0}' | sort -g | head -1 | cut -d' ' -f2)
BEST=${BEST:-$WORK/eng_ft_checkpoint}
echo "exporting $BEST"
lstmtraining --stop_training \
  --continue_from "$BEST" \
  --traineddata "$BASE" \
  --model_output data/tessdata/eng_ft.traineddata
echo "wrote data/tessdata/eng_ft.traineddata"
