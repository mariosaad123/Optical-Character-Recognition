# bench_en_v2

| Engine | clean acc (1-CER) | medium acc (1-CER) | hard acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|---|---|
| tesseract/raw | 99.32% | 86.04% | 22.74% | 69.37% | 56.27% | 71.49% | 0.00 |
| rapidocr/raw | 95.51% | 89.84% | 37.87% | 74.41% | 50.17% | 52.94% | 0.00 |
| ppocr6m/raw | 95.92% | 92.96% | 57.72% | 82.20% | 23.81% | 78.71% | 0.00 |
| ppocr6m/enhanced | 94.53% | 93.08% | 57.65% | 81.76% | 26.35% | 76.41% | 0.00 |
| tesseract-eng_ft2/enhanced | 99.36% | 95.41% | 74.55% | 89.77% | 22.41% | 81.83% | 0.00 |
| previous pipeline | 99.59% | 95.83% | 71.24% | 88.89% | 18.21% | 84.25% | 0.00 |
| pipeline (voting only) | 99.24% | 96.02% | 68.76% | 88.01% | 19.80% | 83.48% | 0.00 |
| PIPELINE (voting + word prediction) | 99.21% | 95.99% | 68.61% | 87.93% | 19.14% | 84.17% | 0.00 |
