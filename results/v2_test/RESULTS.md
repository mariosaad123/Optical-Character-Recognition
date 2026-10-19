# bench_en_v2

| Engine | clean acc (1-CER) | medium acc (1-CER) | hard acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|---|---|
| tesseract/raw | 99.32% | 86.04% | 22.74% | 69.37% | 56.27% | 71.49% | 0.00 |
| ppocr6m/raw | 99.63% | 95.01% | 59.97% | 84.87% | 20.98% | 80.86% | 0.00 |
| previous pipeline | 99.59% | 95.83% | 71.24% | 88.89% | 18.21% | 84.25% | 0.00 |
| pipeline (voting only) | 99.71% | 96.72% | 73.19% | 89.87% | 17.63% | 85.01% | 0.00 |
| PIPELINE (voting + word prediction) | 99.72% | 96.76% | 73.35% | 89.94% | 16.99% | 85.70% | 0.00 |
