#!/usr/bin/env bash
# Stage 2: continue from our stage-1 model on a mix of general + receipt-style lines.
#
#   python -m training.make_tess_lines --out data/train_lines_receipt --n 15000 --style receipt --seed 11
#   bash training/train_stage2.sh [max_iterations]
#
# Output: data/tessdata/eng_ft2.traineddata
set -euo pipefail
cd "$(dirname "$0")/.."

ITER=${1:-12000}
WORK=data/train_work2
BASE=data/tessdata/eng_best.traineddata
mkdir -p "$WORK"

combine_tessdata -e data/tessdata/eng_ft.traineddata "$WORK/eng_ft.lstm" >/dev/null
cat data/train_lines/list.train data/train_lines_receipt/list.train | shuf --random-source=<(yes) > "$WORK/list.train"
cat data/train_lines/list.eval data/train_lines_receipt/list.eval > "$WORK/list.eval"

lstmtraining \
  --model_output "$WORK/eng_ft2" \
  --continue_from "$WORK/eng_ft.lstm" \
  --traineddata "$BASE" \
  --train_listfile "$WORK/list.train" \
  --eval_listfile "$WORK/list.eval" \
  --learning_rate 0.00005 \
  --max_iterations "$ITER" \
  --target_error_rate 0.005 \
  --debug_interval 0 2>&1 | tee "$WORK/train.log" | grep -E "At iteration|best|Finished" || true

BEST=$(ls "$WORK"/eng_ft2_[0-9]*.checkpoint 2>/dev/null | awk -F'eng_ft2_' '{print $2" "$0}' | sort -g | head -1 | cut -d' ' -f2)
BEST=${BEST:-$WORK/eng_ft2_checkpoint}
echo "exporting $BEST"
lstmtraining --stop_training --continue_from "$BEST" --traineddata "$BASE" \
  --model_output data/tessdata/eng_ft2.traineddata
echo "wrote data/tessdata/eng_ft2.traineddata"
