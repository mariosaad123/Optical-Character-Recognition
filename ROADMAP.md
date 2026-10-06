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

- [x] Engine interface + free engines: Tesseract 5 (LSTM), RapidOCR (PaddleOCR DBNet + SVTR, ONNX)
- [x] Synthetic benchmarks: v1 (system fonts) and v2 (83 Google Font families, public-domain books,
      held-out test fonts/books, perspective / ink / texture / motion blur)
- [x] Real-world benchmark: 150 ICDAR-2019 SROIE scanned receipts (evaluation only)
- [x] Metrics: CER, WER, order-independent word F1

## Phase 1: stronger system on CPU (done)

- [x] Preprocessing with measured ablation: adaptive denoise, deskew, background normalization
- [x] Word-level ensemble: ROVER confusion network over engines x preprocessing variants
- [x] Word prediction: lexicon + bigram LM + OCR-confusion edit distance, with protections
- [x] Our own model: Tesseract eng_best fine-tuned on 55k synthetic lines (general + receipt style)
- [x] Everything tuned on dev splits, reported on held-out test splits

Result: real receipts 87.1% -> 91.9% accuracy, WER 37.9% -> 25.1%;
synthetic v2 69.4% -> 88.9% (hard images 22.7% -> 71.2%).

## Phase 2: next steps

- [ ] Image-quality router: very noisy pages favour our model alone (synthetic hard: 74.6% single vs 71.2% voting)
- [ ] Larger language model for word prediction (current bigram LM uses 15 public-domain books;
      a neural LM needs a bigger free corpus or HuggingFace access, which this environment blocks)
- [ ] Train a recognizer from scratch (CRNN / PARSeq / TrOCR) on free GPUs (Kaggle / Colab)
- [ ] Real data with permissive licenses for training (receipts, forms, photos)
- [ ] Layout: reading order for multi-column pages, tables -> Markdown

## Phase 3: product

- [ ] CLI + REST API
- [ ] Input: images and PDFs. Output: text, searchable PDF, Markdown, JSON with boxes
- [ ] Web UI for review and correction; corrections feed back into training (active learning)
- [ ] Then: Arabic and other languages
