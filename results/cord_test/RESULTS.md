# real_cord_test

| Engine | real acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|
| tesseract/raw | 63.11% | 63.11% | 71.58% | 50.73% | 0.00 |
| rapidocr/raw | 87.62% | 87.62% | 38.22% | 70.18% | 0.00 |
| ppocr6m/raw | 89.13% | 89.13% | 20.04% | 88.83% | 0.00 |
| ppocr6m/enhanced | 92.72% | 92.72% | 15.82% | 88.79% | 0.00 |
| previous pipeline | 76.01% | 76.01% | 49.96% | 68.29% | 0.00 |
| pipeline (voting only) | 92.68% | 92.68% | 16.11% | 88.38% | 0.00 |
| PIPELINE (voting + word prediction) | 92.52% | 92.52% | 17.35% | 87.21% | 0.00 |
