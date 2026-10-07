"""PaddleOCR PP-OCRv5 / PP-OCRv6 (Apache 2.0) through the `rapidocr` v3 ONNX runtime. Free, CPU-only.

Variants (models from huggingface.co/PaddlePaddle, fetched by scripts/fetch_resources.py --hf):
    ppocr6s   PP-OCRv6 small det + rec (bundled with rapidocr)
    ppocr6m   PP-OCRv6 medium det + rec
    ppocr5en  PP-OCRv5 mobile det + English-only PP-OCRv5 mobile rec
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

from .base import OCREngine
from .rapidocr_engine import _reading_order

HF_MODELS = Path(__file__).resolve().parents[2] / "data" / "hf_models"

VARIANTS = {
    "ppocr6s": None,
    "ppocr6m": ("PP-OCRv6_medium_det_onnx", "PP-OCRv6_medium_rec_onnx"),
    "ppocr5en": ("PP-OCRv5_mobile_det_onnx", "en_PP-OCRv5_mobile_rec_onnx"),
}


class PPOCREngine(OCREngine):
    def __init__(self, variant: str = "ppocr6m"):
        from rapidocr import RapidOCR

        self.name = variant
        params = {
            "Global.log_level": "error",
            # always detect (wide single-line images otherwise skip detection) and keep low-score lines:
            # the voting step weighs them by confidence instead
            "Global.width_height_ratio": -1,
            "Global.text_score": 0.3,
            # parallel benchmark workers set OCR_ORT_THREADS=1 to avoid oversubscribing the CPU
            "EngineConfig.onnxruntime.intra_op_num_threads": int(os.environ.get("OCR_ORT_THREADS", "-1")),
        }
        models = VARIANTS[variant]
        if models:
            det, rec = (HF_MODELS / m for m in models)
            for p in (det / "inference.onnx", rec / "inference.onnx", rec / "keys.txt"):
                if not p.exists():
                    raise FileNotFoundError(f"{p} (run `python scripts/fetch_resources.py --hf`)")
            params |= {
                "Det.model_path": str(det / "inference.onnx"),
                "Rec.model_path": str(rec / "inference.onnx"),
                "Rec.rec_keys_path": str(rec / "keys.txt"),
            }
        self.ocr = RapidOCR(params=params)

    def recognize(self, image: Image.Image) -> str:
        return "\n".join(" ".join(w for w, _ in line) for line in self.recognize_words(image))

    def recognize_words(self, image: Image.Image) -> list[list[tuple[str, float]]]:
        r = self.ocr(np.array(image.convert("RGB")))
        if r.boxes is None or not r.txts:
            return []
        items = [(box.tolist(), text, float(score)) for box, text, score in zip(r.boxes, r.txts, r.scores) if text.strip()]
        return _reading_order(items)
