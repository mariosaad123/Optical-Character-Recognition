# Roadmap

Goal: build the strongest OCR system we can, **at zero cost**, starting with **English**,
and prove every claim with numbers on a reproducible benchmark.

## Principles

1. **Zero cost**: only free/open-source models, free compute (local CPU, Kaggle / Colab free GPUs), and free data
   (synthetic rendering, born-digital PDFs, permissively licensed datasets).
2. **Measure everything**: no change is kept unless it improves CER/WER on the benchmark.
   Tune on the dev split, report on the test split.
3. **Combine, then specialize**: start by combining the best free engines, then train our own model.

## Phase 0: foundation (done)

- [x] Engine interface + two free engines: Tesseract 5 (LSTM), RapidOCR (PaddleOCR DBNet + SVTR, ONNX)
- [x] Synthetic English benchmark: 40 fonts, 3 degradation levels (clean / medium / hard), exact ground truth
- [x] Metrics: CER, WER
- [x] First own system: confidence-based **router** (Tesseract fast path → RapidOCR when confidence is low)

## Phase 1: stronger free baselines (CPU / free GPU)

- [ ] Add open VLM-based OCR models (e.g. small Qwen-VL / GOT-OCR / Florence-2 class) and measure them
- [ ] Image pre-processing: deskew, denoise, binarization, upscaling for low-DPI input
- [ ] Word-level ensemble: align outputs of several engines and vote per word (ROVER-style)
- [ ] Add real-world test sets (scanned pages, receipts, photos) with permissive licenses
- [ ] Layout: reading order, multi-column pages, tables → Markdown

## Phase 2: our own model (free GPU hours)

- [ ] Large-scale synthetic data engine (millions of lines: fonts, layouts, degradations)
- [ ] Born-digital PDF pipeline: render pages → image + exact text labels
- [ ] Fine-tune a small recognizer / VLM on Kaggle free GPUs (LoRA, mixed precision)
- [ ] Per-word confidence + bounding boxes, anti-hallucination checks
- [ ] Distill to a small fast model that runs on CPU and mobile

## Phase 3: product

- [ ] CLI + Python API + REST API
- [ ] Input: images and PDFs. Output: text, searchable PDF, Markdown, JSON with boxes
- [ ] Web UI for review and correction; corrections feed back into training (active learning)
- [ ] Then: Arabic and other languages
