# real_sroie_test

| Engine | real acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|
| tesseract/raw | 87.10% | 87.10% | 37.85% | 73.04% | 0.00 |
| rapidocr/raw | 90.87% | 90.87% | 46.83% | 62.79% | 0.00 |
| ppocr6m/raw | 95.51% | 95.51% | 16.27% | 88.60% | 0.00 |
| ppocr6m/enhanced | 95.42% | 95.42% | 16.52% | 88.45% | 0.00 |
| previous pipeline | 91.91% | 91.91% | 25.06% | 81.57% | 0.00 |
| pipeline (voting only) | 95.14% | 95.14% | 15.90% | 88.73% | 0.00 |
| PIPELINE (voting + word prediction) | 95.15% | 95.15% | 16.01% | 88.61% | 0.00 |
