# bench_en_v2

| Engine | clean acc (1-CER) | medium acc (1-CER) | hard acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|---|---|
| ppocr6m/raw | 95.92% | 92.96% | 57.72% | 82.20% | 23.81% | 78.71% | 0.00 |
| tesseract-eng_ft3last/raw | 98.65% | 93.87% | 49.42% | 80.65% | 36.93% | 73.15% | 0.00 |
| previous pipeline | 99.21% | 95.99% | 68.61% | 87.93% | 19.14% | 84.17% | 0.00 |
| pipeline (voting only) | 97.66% | 96.31% | 72.51% | 88.83% | 18.60% | 84.31% | 0.00 |
| PIPELINE (voting + word prediction) | 97.66% | 96.35% | 72.70% | 88.90% | 17.93% | 85.02% | 0.00 |
