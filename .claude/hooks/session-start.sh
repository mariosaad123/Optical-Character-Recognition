#!/bin/bash
# Prepare a Claude Code cloud session: system packages, Python packages, free models and data.
# Idempotent: every step is skipped when its result is already present.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Tesseract OCR engine and its training tools
if ! command -v tesseract >/dev/null 2>&1 || ! command -v lstmtraining >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tesseract-ocr tesseract-ocr-eng >/dev/null
fi

# PyTorch CPU build (the default PyPI wheel pulls several GB of CUDA libraries)
if ! python3 -c "import torch" >/dev/null 2>&1; then
  pip install -q --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
pip install -q --break-system-packages -r requirements.txt

# Free resources: fonts, public-domain books, tessdata_best, PaddleOCR ONNX models, SmolLM2.
# A network hiccup must not block the session, so failures here only warn.
python3 scripts/fetch_resources.py --hf || echo "warning: some resources could not be downloaded" >&2

echo "export PYTHONPATH=\"$CLAUDE_PROJECT_DIR\"" >> "$CLAUDE_ENV_FILE"
