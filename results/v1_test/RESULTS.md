# bench_en

| Engine | clean acc (1-CER) | medium acc (1-CER) | hard acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|---|---|
| tesseract/raw | 99.87% | 97.71% | 42.81% | 80.13% | 42.61% | 77.62% | 0.00 |
| rapidocr/raw | 97.61% | 94.25% | 57.64% | 83.17% | 45.72% | 57.97% | 0.00 |
| tesseract-eng_ft2/enhanced | 99.86% | 99.92% | 93.15% | 97.64% | 10.75% | 91.12% | 0.00 |
| pipeline (voting only) | 99.86% | 99.94% | 91.95% | 97.25% | 8.64% | 93.12% | 0.00 |
| PIPELINE (voting + word prediction) | 99.86% | 99.93% | 92.01% | 97.27% | 8.49% | 93.27% | 0.00 |
