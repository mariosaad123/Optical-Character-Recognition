# real_sroie_test

| Engine | real acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|
| ppocr6m/raw | 95.51% | 95.51% | 16.27% | 88.60% | 0.00 |
| tesseract-eng_ft3last/raw | 89.67% | 89.67% | 34.38% | 75.20% | 0.00 |
| previous pipeline | 95.15% | 95.15% | 16.01% | 88.61% | 0.00 |
| pipeline (voting only) | 94.97% | 94.97% | 17.05% | 87.95% | 0.00 |
| PIPELINE (voting + word prediction) | 94.97% | 94.97% | 16.99% | 88.01% | 0.00 |
