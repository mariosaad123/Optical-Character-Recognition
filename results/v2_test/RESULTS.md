# bench_en_v2

| Engine | clean acc (1-CER) | medium acc (1-CER) | hard acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|---|---|
| tesseract/raw | 99.32% | 86.04% | 22.74% | 69.37% | 56.27% | 71.49% | 0.00 |
| tesseract-eng_best/raw | 99.38% | 91.45% | 34.41% | 75.08% | 46.60% | 71.95% | 0.00 |
| rapidocr/raw | 95.51% | 89.84% | 37.87% | 74.41% | 50.17% | 52.94% | 0.00 |
| tesseract-eng_best/enhanced | 99.33% | 91.95% | 66.75% | 86.01% | 27.78% | 78.14% | 0.00 |
| tesseract-eng_ft2/enhanced | 99.36% | 95.41% | 74.55% | 89.77% | 22.41% | 81.83% | 0.00 |
| tesseract-eng_ft2last/enhanced | 99.18% | 95.28% | 72.37% | 88.95% | 24.02% | 79.94% | 0.00 |
| pipeline (voting only) | 99.59% | 95.80% | 71.08% | 88.82% | 18.79% | 83.64% | 0.00 |
| PIPELINE (voting + word prediction) | 99.59% | 95.83% | 71.24% | 88.89% | 18.21% | 84.25% | 0.00 |
