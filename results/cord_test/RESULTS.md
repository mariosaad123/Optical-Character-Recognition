# real_cord_test

| Engine | real acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|
| tesseract/raw | 63.11% | 63.11% | 71.58% | 50.73% | 0.00 |
| ppocr6m/raw | 89.92% | 89.92% | 18.48% | 90.45% | 0.00 |
| previous pipeline | 76.01% | 76.01% | 49.98% | 68.27% | 0.00 |
| pipeline (voting only) | 93.22% | 93.22% | 14.15% | 90.60% | 0.00 |
| PIPELINE (voting + word prediction) | 93.17% | 93.17% | 14.39% | 90.36% | 0.00 |
