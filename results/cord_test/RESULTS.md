# real_cord_test

| Engine | real acc (1-CER) | all acc (1-CER) | all WER | all word F1 | sec/img |
|---|---|---|---|---|---|
| ppocr6m/raw | 89.13% | 89.13% | 20.04% | 88.83% | 0.00 |
| tesseract-eng_ft3last/raw | 72.90% | 72.90% | 60.10% | 59.58% | 0.00 |
| previous pipeline | 92.58% | 92.58% | 17.03% | 87.55% | 0.00 |
| pipeline (voting only) | 92.76% | 92.76% | 15.28% | 89.30% | 0.00 |
| PIPELINE (voting + word prediction) | 92.72% | 92.72% | 15.44% | 89.13% | 0.00 |
