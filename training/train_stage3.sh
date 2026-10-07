#!/usr/bin/env bash
# Stage 3: continue from our stage-2 model on REAL receipt lines (CORD train, CC BY 4.0, oversampled
# x3) mixed with a share of the synthetic general and receipt-style lines.
#
#   python -m training.make_cord_lines --out data/train_cord_lines
#   bash training/train_stage3.sh [max_iterations]
#
# Output: data/tessdata/eng_ft3.traineddata
set -euo pipefail
cd "$(dirname "$0")/.."

ITER=${1:-12000}
WORK=data/train_work3
BASE=data/tessdata/eng_best.traineddata
mkdir -p "$WORK"

combine_tessdata -e models/eng_ft2.traineddata "$WORK/eng_ft2.lstm" >/dev/null
{
  for _ in 1 2 3; do cat data/train_cord_lines/list.train; done
  shuf -n 15000 --random-source=<(yes) data/train_lines/list.train
  shuf -n 6000 --random-source=<(yes) data/train_lines_receipt/list.train
} | shuf --random-source=<(yes) > "$WORK/list.train"
cat data/train_lines/list.eval data/train_lines_receipt/list.eval > "$WORK/list.eval"

lstmtraining \
  --model_output "$WORK/eng_ft3" \
  --continue_from "$WORK/eng_ft2.lstm" \
  --traineddata "$BASE" \
  --train_listfile "$WORK/list.train" \
  --eval_listfile "$WORK/list.eval" \
  --learning_rate 0.00005 \
  --max_iterations "$ITER" \
  --target_error_rate 0.005 \
  --debug_interval 0 2>&1 | tee "$WORK/train.log" | grep -E "At iteration|best|Finished" || true

BEST=$(ls "$WORK"/eng_ft3_[0-9]*.checkpoint 2>/dev/null | awk -F'eng_ft3_' '{print $2" "$0}' | sort -g | head -1 | cut -d' ' -f2)
BEST=${BEST:-$WORK/eng_ft3_checkpoint}
echo "exporting $BEST"
lstmtraining --stop_training --continue_from "$BEST" --traineddata "$BASE" \
  --model_output data/tessdata/eng_ft3.traineddata
lstmtraining --stop_training --continue_from "$WORK/eng_ft3_checkpoint" --traineddata "$BASE" \
  --model_output data/tessdata/eng_ft3last.traineddata
echo "wrote data/tessdata/eng_ft3.traineddata and eng_ft3last.traineddata"
