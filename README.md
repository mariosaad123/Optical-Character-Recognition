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
