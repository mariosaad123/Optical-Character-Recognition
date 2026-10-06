# Optical Character Recognition

A zero-cost OCR system built from free/open-source components, measured on a reproducible benchmark.
English first. See [ROADMAP.md](ROADMAP.md) for the plan.

## Engines

| Engine | What it is |
|---|---|
| `tesseract` | Tesseract 5 LSTM. Fast, excellent on clean scans |
| `rapidocr` | PaddleOCR (DBNet detection + SVTR recognition) via ONNX Runtime. More robust to noise and blur |
| `router` | Ours: Tesseract first; if its confidence is low, falls back to RapidOCR |

## Setup

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
pip install -r requirements.txt
```

## Benchmark

```bash
python -m bench.generate --out data/bench_en --per-level 100 --seed 1234   # test split
python -m bench.generate --out data/dev_en   --per-level 50  --seed 999    # dev split (tuning only)
python -m bench.tune_router --data data/dev_en
python -m bench.evaluate --data data/bench_en
python -m pytest tests
```

Results are written to `results/RESULTS.md` and `results/summary.json`.

## Current results (test split, 300 images, CPU only)

| Engine | clean | medium | hard | overall accuracy (1-CER) | overall WER | sec/img |
|---|---|---|---|---|---|---|
| tesseract | 99.87% | 97.71% | 42.81% | 80.13% | 42.61% | 0.32 |
| rapidocr | 97.61% | 94.25% | 57.64% | 83.17% | 45.72% | 1.97 |
| **router (ours)** | **99.87%** | **98.11%** | **63.18%** | **87.05%** | **24.26%** | 0.62 |

The router threshold (0.56) was tuned on the dev split only. The `hard` level (heavy blur, noise, skew,
low resolution, JPEG artifacts) is where the remaining work is.
