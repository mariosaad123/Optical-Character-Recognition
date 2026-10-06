# real_sroie_test

| Engine | real acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|
| tesseract/raw | 87.10% | 87.10% | 37.85% | 73.04% | 0.00 |
| tesseract-eng_best/raw | 88.13% | 88.13% | 36.18% | 74.16% | 0.00 |
| rapidocr/raw | 90.87% | 90.87% | 46.83% | 62.79% | 0.00 |
| tesseract-eng_best/enhanced | 88.23% | 88.23% | 35.72% | 74.70% | 0.00 |
| tesseract-eng_ft2/enhanced | 88.41% | 88.41% | 37.10% | 71.89% | 0.00 |
| tesseract-eng_ft2last/enhanced | 89.19% | 89.19% | 34.22% | 75.25% | 0.00 |
| pipeline (voting only) | 91.90% | 91.90% | 25.15% | 81.49% | 0.00 |
| PIPELINE (voting + word prediction) | 91.91% | 91.91% | 25.06% | 81.57% | 0.00 |
