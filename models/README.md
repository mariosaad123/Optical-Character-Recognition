# Our trained models

Tesseract LSTM models fine-tuned from `tessdata_best/eng` (Apache 2.0) on synthetic data only
(Google Fonts, public-domain Gutenberg books, receipt-style lines). See `training/`.

| File | Training | Notes |
|---|---|---|
| `eng_ft2.traineddata` | stage 1 (40k general lines) + stage 2 (general + 15k receipt lines), best checkpoint | best single model on synthetic test |
| `eng_ft2last.traineddata` | same, last checkpoint of stage 2 | best single model on real receipts |

Use directly with Tesseract: `tesseract page.png out --tessdata-dir models -l eng_ft2`
