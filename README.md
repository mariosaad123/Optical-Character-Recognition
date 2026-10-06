# Optical Character Recognition

A zero-cost English OCR system built from free/open-source parts plus our own fine-tuned model,
measured on reproducible benchmarks, including real scanned receipts.
See [ROADMAP.md](ROADMAP.md) for the plan.

## How it works

```
image ─┬─ enhanced (strong adaptive denoise + deskew + background normalization) ─┬─ our model (eng_ft2last)
       │                                                                          └─ Tesseract eng_best
       ├─ soft (lighter denoise)  ─────────────────────────────────────────────────── our model (eng_ft2)
       └─ raw ─────────────────────────────────────────────────────────────────────── RapidOCR (PaddleOCR)
                                │
                     word-level voting (ROVER confusion network, tuned weights)
                                │
                     word prediction (lexicon + bigram LM + OCR-confusion edit cost)
                                │
                               text
```

| Part | File | What it does |
|---|---|---|
| Preprocessing | `ocr_engine/preprocess.py` | Noise-adaptive denoising, projection-profile deskew, shadow removal. Every step was ablated on the dev split; upscaling was measured harmful and removed |
| Our model | `training/` | Tesseract `eng_best` LSTM fine-tuned on 55k synthetic lines (83 font families, public-domain books, receipt-style lines, realistic degradations) |
| Voting | `ocr_engine/ensemble.py` | Aligns readings word-by-word into a confusion network and votes with engine weights and confidences |
| Word prediction | `ocr_engine/lm.py` | Fixes misread or missing letters: `rnodern wornan` -> `modern woman`, `qu1ck brovvn` -> `quick brown`, `very tred of` -> `very tired of`. Numbers, dates, money, emails, codes, names and hyphenated words are protected |
| Pipeline | `ocr_engine/pipeline.py` | Combines everything; weights/parameters in `ocr_engine/pipeline_config.json` are tuned on dev data only |

## Results (held-out test sets, CPU only)

Accuracy = 1 - CER (character error rate). WER = word error rate (lower is better).
Word F1 = order-independent word accuracy.

**Synthetic v2 test** (450 images; fonts and books never used for training or tuning):

| System | clean | medium | hard | overall | WER | word F1 |
|---|---|---|---|---|---|---|
| Tesseract (default) | 99.32% | 86.04% | 22.74% | 69.37% | 56.27% | 71.49% |
| Tesseract eng_best | 99.38% | 91.45% | 34.41% | 75.08% | 46.60% | 71.95% |
| RapidOCR | 95.51% | 89.84% | 37.87% | 74.41% | 50.17% | 52.94% |
| Our model + enhancement | 99.36% | 95.41% | **74.55%** | **89.77%** | 22.41% | 81.83% |
| **Full pipeline** | **99.59%** | **95.83%** | 71.24% | 88.89% | **18.21%** | **84.25%** |

**Real scanned receipts** (ICDAR-2019 SROIE, 100 receipts; evaluation only, case-insensitive
because the official transcripts are upper-cased):

| System | accuracy | WER | word F1 |
|---|---|---|---|
| Tesseract (default) | 87.10% | 37.85% | 73.04% |
| Tesseract eng_best | 88.13% | 36.18% | 74.16% |
| RapidOCR | 90.87% | 46.83% | 62.79% |
| Our model + enhancement | 89.19% | 34.22% | 75.25% |
| **Full pipeline** | **91.91%** | **25.06%** | **81.57%** |

Full tables: `results/v2_test/RESULTS.md`, `results/real_test/RESULTS.md`.

## Setup

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
pip install -r requirements.txt
python scripts/fetch_resources.py        # free fonts, public-domain books, tessdata_best
```

## Reproduce

```bash
# benchmarks
python -m bench.generate --version v2 --split test --out data/bench_en_v2 --per-level 150 --seed 2024
python -m bench.generate --version v2 --split dev  --out data/dev_en_v2  --per-level 60  --seed 77
python -m bench.import_sroie --n 150 --dev 50          # real receipts (evaluation only, not committed)

# train our model (CPU, a few hours)
python -m training.make_tess_lines --out data/train_lines --n 40000
bash training/train_tesseract.sh 30000
python -m training.make_tess_lines --out data/train_lines_receipt --n 15000 --style receipt --seed 11
bash training/train_stage2.sh 10000

# read every image with every source once, tune on dev, evaluate on test
python -m bench.cache_readings --data data/dev_en_v2 --sources tesseract-eng_ft2/enhanced ...
python -m bench.tune_pipeline --data data/dev_en_v2 data/real_sroie_dev
python -m bench.evaluate_cached --data data/bench_en_v2 --out results/v2_test

python -m pytest tests
```

Use it from Python:

```python
from PIL import Image
from ocr_engine import get_engine

print(get_engine("pipeline").recognize(Image.open("page.png")))
```

The trained models live in `data/tessdata/` (not committed; regenerate with the training commands).
