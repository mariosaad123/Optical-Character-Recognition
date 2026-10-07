# Our trained models

Tesseract LSTM models fine-tuned from `tessdata_best/eng` (Apache 2.0) on synthetic data only
(Google Fonts, public-domain Gutenberg books, receipt-style lines). See `training/`.

| File | Training | Notes |
|---|---|---|
| `eng_ft2.traineddata` | stage 1 (40k general lines) + stage 2 (general + 15k receipt lines), best checkpoint | best single model on synthetic test |
| `eng_ft2last.traineddata` | same, last checkpoint of stage 2 | |
| `eng_ft3.traineddata` | stage 3: + 6,626 real CORD receipt lines (CC BY 4.0, x3) + synthetic mix, best checkpoint | best Tesseract model on synthetic dev (94.04%) |
| `eng_ft3last.traineddata` | stage 3, last checkpoint | best Tesseract model on receipt photos (CORD dev raw 74.7% -> 80.6%) |

Use directly with Tesseract: `tesseract page.png out --tessdata-dir models -l eng_ft2`
