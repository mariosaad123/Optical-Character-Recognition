# bench_en_v2

| Engine | clean acc (1-CER) | medium acc (1-CER) | hard acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|---|---|
| tesseract/raw | 99.32% | 86.04% | 22.74% | 69.37% | 56.27% | 71.49% | 0.00 |
| ppocr6m/raw | 95.92% | 92.96% | 57.72% | 82.20% | 23.81% | 78.71% | 0.00 |
| previous pipeline | 99.59% | 95.83% | 71.24% | 88.89% | 18.21% | 84.25% | 0.00 |
| pipeline (voting only) | 99.54% | 95.96% | 72.39% | 89.30% | 18.81% | 84.43% | 0.00 |
| PIPELINE (voting + word prediction) | 99.54% | 95.99% | 72.61% | 89.38% | 18.02% | 85.25% | 0.00 |
