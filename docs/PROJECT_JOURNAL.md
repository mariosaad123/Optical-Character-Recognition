# Project Journal: Zero-Cost OCR System

A running record of every phase: goals, decisions, methods, experiments (with numbers), mistakes
found and fixed, and lessons learned. Written to be the source for a project report.

- **Owner:** Mario Saad
- **Repository:** `mariosaad123/Optical-Character-Recognition`
- **Constraint:** zero cost (free/open-source models, free data, free compute)
- **Language focus:** English first (Arabic later)
- **Rule of the project:** nothing is kept unless it improves a measured number; tune on dev
  splits, report on held-out test splits.

---

## Phase map

| Phase | Scope | Status |
|---|---|---|
| 0 | Printed-text OCR system: engines, benchmarks, preprocessing, our own model, voting, word prediction | done (v1), being pushed further (v2) |
| 1 | Large handwriting benchmark (IAM + real phone photos) | planned |
| 2 | Neural handwriting models (TrOCR, GLM-OCR), speed-ups | planned (feasibility done) |
| 3 | Phone-photo enhancement: page detection, perspective, dewarping, shadows, super-resolution | planned |
| 4 | Training on free GPUs (Kaggle): handwriting recognizers | planned |
| 5 | Smart system: printed/handwriting router, ensembles, neural LM correction, confidences | planned |
| 6 | Web app: upload page with motion and transitions, free hosting on HuggingFace Spaces | planned |

---

## Phase 0: printed-text OCR

### 0.1 Starting point and environment

- Empty repository; cloud container with 4 CPU cores, 15 GB RAM, **no GPU**.
- Network at the start: PyPI and GitHub raw files reachable; HuggingFace and PyTorch blocked.
  Later the owner opened HuggingFace / PyTorch / Kaggle (used from phase 1 on).
- Cost of commercial alternatives was checked and rejected (Google Document AI / Vision: first 1,000
  pages per month free, then $1.50 per 1,000 pages; Azure Read: $1.50; Mistral OCR 4: $4 or $2 batch).

### 0.2 Baseline engines and the first benchmark (v1)

- Engines behind one interface (`ocr_engine/engines`): Tesseract 5 LSTM, RapidOCR (PaddleOCR DBNet
  detection + SVTR recognition via ONNX Runtime).
- Benchmark v1 (`bench/generate.py`): 300 synthetic images, 40 system fonts, three degradation levels
  (clean / medium / hard: rotation, resolution loss, blur, lighting gradient, noise, salt & pepper, JPEG).
  Text: document-style sequences (money, dates, emails, invoice numbers, phone numbers, names).
- Metrics (`bench/metrics.py`): CER, WER; later order-independent word F1.

Bugs found while measuring RapidOCR (both fixed):
1. Wide single-line images skipped detection entirely (library default `width_height_ratio=8`)
   and came back empty. Fix: always run detection.
2. Very wide line crops returned empty text from the recognizer. Fix: split long lines at blank
   columns between words (`_split_at_gaps`).
   RapidOCR clean accuracy: 69.49% -> 74.51% -> 97.61%.

First own system: a **confidence router** (Tesseract first; low confidence -> RapidOCR), threshold
0.56 tuned on a separate dev split.

| v1 test (300 images) | clean | medium | hard | overall | WER |
|---|---|---|---|---|---|
| Tesseract | 99.87% | 97.71% | 42.81% | 80.13% | 42.61% |
| RapidOCR | 97.61% | 94.25% | 57.64% | 83.17% | 45.72% |
| Router | 99.87% | 98.11% | 63.18% | 87.05% | 24.26% |

### 0.3 Free resources

`scripts/fetch_resources.py`
- 327 font files from 83 Google Fonts families (OFL / Apache / UFL licenses).
- 18 public-domain Project Gutenberg books (via the NLTK data mirror).
- `tessdata_best` English float LSTM model (Apache 2.0), which can be fine-tuned.
- `wordfreq` English word frequencies (200k words).

### 0.4 Honest benchmarks

- **v2 synthetic** (`--version v2`): real prose from books mixed with document lines; 10 font families
  and 3 books are **held out for test only**, 4 families and 2 books for dev; richer degradations
  (perspective, ink bleed/thinning, paper texture, motion blur). Test 450 images, dev 180.
- **Real receipts**: 150 ICDAR-2019 SROIE scanned receipts (50 dev / 100 test), evaluation only
  (research license, never committed). The official transcripts are upper-cased regardless of the
  printed case, so scoring is case-insensitive for this set.
- Order-independent word F1 added because receipts have columns whose reading order can differ.

### 0.5 Preprocessing: measured, not assumed

Ablation on the dev split (Tesseract eng_best, accuracy = 1 - CER):

| Step | medium | hard |
|---|---|---|
| none | 96.98% | 42.84% |
| denoise (3x noise sigma) | 96.98% | 59.62% |
| **upscale** | 95.55% | **-218.16%** (noise magnified into fake glyphs) |
| deskew | 97.25% | 44.52% |
| background normalization | 99.24% | 51.00% |
| denoise + normalize | 99.24% | 65.28% |
| **denoise 6x + normalize** | 99.16% | **72.25%** |
| denoise 10x + normalize | 98.79% | 66.04% |

Final `enhance()`: noise-adaptive non-local-means denoising (strength = 6 x measured noise sigma),
projection-profile deskew (only when clearly better than no rotation), background normalization.
A lighter `soft` variant (3x) gives the voting step a second view.

Bugs found and fixed in the first version of enhancement:
- Text-height estimate counted noise specks as glyphs, so it upscaled noisy images 4x.
- Rotation used edge replication, which drew fake vertical strokes at the borders.
- Skew search clipped wide, short images; fixed by padding before rotating.

### 0.6 Our own model (fine-tuned Tesseract LSTM)

- Data (`training/make_tess_lines.py`): 40,000 general lines (books + document text, 70+ train font
  families + system fonts, variable-font weights, clean/medium/hard mix) and 15,000 receipt-style
  lines (prices with decimals, codes, dates, upper case, faded thermal-printer look).
- **Critical bug found:** the first training lines had wide margins. In raw-line mode Tesseract
  scales the whole image to the network input height, so the text became tiny: the base model's
  median error on the training lines was 76%. Tight cropping brought it to 2%. Training was stopped
  and restarted on the fixed data.
- Stage 1: continue from `eng_best`, learning rate 1e-4. Training error 18% -> ~9%.
- Stage 2: general + receipt lines, learning rate 5e-5, 10,000 iterations.
- Models are committed in `models/` (`eng_ft2`, `eng_ft2last`, 15 MB each).

| dev v2, enhanced input | clean | medium | hard | overall |
|---|---|---|---|---|
| Tesseract eng_best | 99.71% | 99.32% | 73.65% | 90.89% |
| ours, mid stage 1 | 99.87% | 99.61% | 81.01% | 93.50% |
| ours, stage 2 best (`eng_ft2`) | 99.85% | 99.59% | 81.23% | 93.56% |

### 0.7 Word-level voting (ROVER)

`ocr_engine/ensemble.py`: readings from several (engine, preprocessing) sources are aligned
word-by-word with a character-aware edit distance into a confusion network; each slot votes with
source weight x (0.3 + 0.7 x confidence); inserted words get implicit "nothing here" votes.
Source subsets and weights are searched automatically on dev (`bench/tune_pipeline.py`), with
synthetic and real dev data weighted equally (macro average). The first tuning (micro average) was
dominated by the larger synthetic set and ignored the real receipts.

Final sources: our model on the enhanced image (last checkpoint), Tesseract eng_best on the
enhanced image, our model on the soft image, RapidOCR on the raw image.

### 0.8 Word prediction

`ocr_engine/lm.py`: for suspicious words, candidates = dictionary words within edit distance 2
(SymSpell-style delete index over 80k words) + readings proposed by other engines. Score =
bigram LM (Dirichlet-smoothed, trained on train-split books only, left and right context)
- lambda x OCR-aware edit cost (cheap for rn<->m, cl<->d, 1<->l, 0<->o, ...) + mu x ensemble vote.

Examples: `rnodern wornan` -> `modern woman`, `qu1ck brovvn` -> `quick brown`,
`hosptal` -> `hospital`, `very tred of` -> `very tired of`.

First version fixed 75 words but broke 76 (hyphenated words, unknown names, dropping letters).
Protections added: hyphenated words, confidently-read unknown words (names), case-protected codes,
numbers / money / dates / emails; deleting a glyph costs more than inserting one. After: 63 fixes
vs 19 breaks. Net effect on CER is small because the bigram LM is trained on only 15 books.

### 0.9 Final phase-0 results (held-out tests)

| Real receipts (100) | accuracy | WER | word F1 |
|---|---|---|---|
| Tesseract | 87.10% | 37.85% | 73.04% |
| RapidOCR | 90.87% | 46.83% | 62.79% |
| **Full pipeline** | **91.91%** | **25.06%** | **81.57%** |

| Synthetic v2 (450) | clean | medium | hard | overall | WER |
|---|---|---|---|---|---|
| Tesseract | 99.32% | 86.04% | 22.74% | 69.37% | 56.27% |
| Our model + enhancement | 99.36% | 95.41% | 74.55% | 89.77% | 22.41% |
| **Full pipeline** | **99.59%** | **95.83%** | 71.24% | 88.89% | **18.21%** |

| Synthetic v1 (300) | hard | overall | WER |
|---|---|---|---|
| Tesseract | 42.81% | 80.13% | 42.61% |
| Router (first system) | 63.18% | 87.05% | 24.26% |
| **Full pipeline** | **92.01%** | **97.27%** | **8.49%** |

v1 shares fonts and text style with our training data, so v2 and the real receipts are the honest
measures of generalization.

### 0.10 Lessons

1. Measure every step: the "obvious" upscaling step destroyed accuracy on noisy images.
2. Inspect the training data visually and run the base model on it before training.
3. Benchmarks must hold out fonts and text, otherwise the numbers flatter the model.
4. Know the ground-truth conventions (SROIE upper-case, IAM spaces around punctuation).
5. Balance dev sets when tuning, or the larger set silently wins.
6. Track both character and word metrics: they disagree (RapidOCR has good CER, poor word F1).

---

## Phase 1+ feasibility check (handwriting), done before starting

40 IAM test lines, punctuation-spacing normalized (IAM puts spaces around punctuation):

| System | char accuracy | WER | CPU speed |
|---|---|---|---|
| Our printed pipeline (Tesseract) | 48% | 97% | fast |
| TrOCR base handwritten | 92.45% | 25.86% | 8.5 s/line |
| GLM-OCR (1.3B VLM) | 93.17% | 23.63% | 9.4 s/line |
| Best of the two per line (oracle) | 95.50% | - | - |

GLM-OCR also read a hard synthetic page and a real receipt very well, but took 150 s for the
receipt on CPU. Speed is the main engineering problem for the next phases.

---

## Phase 0, round 2: pushing printed-text OCR further

### R2.1 Infrastructure

- HuggingFace, PyTorch and Kaggle access opened by the owner. The Kaggle token is stored only in
  `/root/.kaggle` (never committed).
- `.claude/hooks/session-start.sh`: every new cloud session installs Tesseract, CPU PyTorch and all
  Python packages and downloads the free resources (fonts, books, tessdata_best, PaddleOCR ONNX
  models, SmolLM2). Large models are not stored on GitHub (100 MB file limit; LFS is not free beyond
  1 GB); they are re-downloaded automatically and kept in the environment snapshot.
- Benchmark workers now pin ONNX Runtime and Tesseract to one thread each; before, every worker
  grabbed all cores and runs crawled.

### R2.2 A second real-world benchmark: CORD (phone photos)

- CORD-v2 (CC BY 4.0): 100 dev + 100 test receipt **photos**. Only part of each photo is annotated
  (headers blurred for privacy), so everything outside the padded word boxes is painted with the
  local paper colour (taken from the brighter half of pixels inside the boxes; feathered edges).
  Reference = annotated words grouped by `row_id`.
- First measurement exposed a weakness invisible on scans: on photos the previous full pipeline
  scored **79.45%** while RapidOCR alone scored 89.78%.

### R2.3 Why enhancement hurt phone photos

Ablation on 40 CORD dev photos (Tesseract eng_best):

| Step | CORD photos | synthetic medium | synthetic hard |
|---|---|---|---|
| none | 76.25% | 96.98% | 34.55% |
| deskew | 76.44% | - | - |
| background normalization | **57.85%** | 99.24% | 50.47% |
| normalization + contrast stretch (0.5%) | 50.52% | 99.10% | 41.73% |
| CLAHE | 54.17% | 97.92% | 1.97% |
| previous enhance (denoise + normalize) | 57.85% | 99.16% | 66.00% |

Photos of faint dot-matrix receipts have almost no sensor noise (sigma 0.2 vs 2.1 for scans), so
denoising never runs; background normalization whitens the paper but leaves the ink light grey and
reveals paper texture, which Tesseract reads as glyphs. Contrast stretching amplifies that texture.
Conclusion: enhancement must be chosen per image, and the voting must see raw and enhanced readings.

### R2.4 Newer free recognizers: PaddleOCR PP-OCRv5 / PP-OCRv6

Added via `rapidocr` v3 with models from huggingface.co/PaddlePaddle (Apache 2.0):
`ppocr6s` (PP-OCRv6 small), `ppocr6m` (PP-OCRv6 medium, released June 2026),
`ppocr5en` (PP-OCRv5 mobile with English-only recognizer). PP-OCRv5 server was dropped: 36 s per
receipt on CPU.

| Real dev sets | CORD acc | CORD WER | CORD word F1 | SROIE acc | SROIE WER |
|---|---|---|---|---|---|
| RapidOCR (PP-OCRv3) | 89.78% | 34.70% | 72.46% | 89.11% | 44.98% |
| Tesseract eng_best | 76.60% | 48.73% | 60.58% | 84.17% | 42.44% |
| PP-OCRv6 small | 89.45% | 21.92% | 86.89% | 91.60% | 22.84% |
| **PP-OCRv6 medium** | 90.02% | 18.58% | 90.72% | 92.78% | **18.93%** |
| **PP-OCRv6 medium, enhanced** | **92.01%** | **16.11%** | **91.68%** | **92.84%** | 19.47% |
| PP-OCRv5 English | 88.85% | 23.38% | 84.57% | 92.03% | 19.74% |

Unlike Tesseract, the PP-OCRv6 networks are robust to paper texture, so enhancement helps them even
on photos.

### R2.5 Neural word prediction (in progress)

`ocr_engine/neural_lm.py`: SmolLM2-360M (Apache 2.0) scores the whole line (plus the previous line
as context) for the top-k candidates of each suspicious word, combined with the OCR-aware edit cost
and engine votes. On hand-made examples it fixed every case, including ones the bigram model broke
(`fox jumps` stayed correct; `tomorow` -> `tomorrow`, `sorne` -> `some`, `tc` -> `to`). Speed and
benchmark effect still to be measured on an idle CPU.
