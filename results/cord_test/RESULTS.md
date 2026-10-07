# real_cord_test

| Engine | real acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|
| tesseract/raw | 63.11% | 63.11% | 71.58% | 50.73% | 0.00 |
| ppocr6m/raw | 89.13% | 89.13% | 20.04% | 88.83% | 0.00 |
| previous pipeline | 76.01% | 76.01% | 49.98% | 68.27% | 0.00 |
| pipeline (voting only) | 92.57% | 92.57% | 17.16% | 87.42% | 0.00 |
| PIPELINE (voting + word prediction) | 92.58% | 92.58% | 17.03% | 87.55% | 0.00 |
