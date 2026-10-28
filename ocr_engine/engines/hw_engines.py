"""Neural handwriting readers (free models from huggingface.co, run on CPU with PyTorch).

    trocr-base, trocr-large     microsoft/trocr-{base,large}-handwritten (MIT): a line recognizer.
                                Pages are split into lines with the PP-OCRv6 detector first.
    glm-ocr                     zai-org/GLM-OCR (MIT, 1.3B vision-language model): reads a line or a
                                whole page in one pass.

Suffix "-line" (e.g. "trocr-base-line") treats the whole image as one text line (line benchmarks).
Confidences: TrOCR reports the mean token probability of the decoded line; GLM-OCR the same.
"""

import os

import numpy as np
import torch
from PIL import Image

from .base import OCREngine
from .rapidocr_engine import _reading_order

torch.set_num_threads(int(os.environ.get("OCR_TORCH_THREADS", str(torch.get_num_threads()))))

def detokenize(text: str) -> str:
    """TrOCR learned IAM's tokenized transcripts ("it 's a good start ."); write natural text instead."""
    from bench.import_handwriting import detokenize as _detok
    return _detok(text)


TROCR = {"trocr-base": "microsoft/trocr-base-handwritten", "trocr-large": "microsoft/trocr-large-handwritten"}
_detector = None


def detect_lines(image: Image.Image) -> list[tuple[list, Image.Image]]:
    """Text-line boxes from the PP-OCRv6 medium detector, each with its (axis-aligned) crop."""
    global _detector
    if _detector is None:
        from .ppocr_engine import PPOCREngine
        _detector = PPOCREngine("ppocr6m")
    rgb = image.convert("RGB")
    r = _detector.ocr(np.array(rgb), use_rec=False, use_cls=False)
    out = []
    for box in (r.boxes if r.boxes is not None else []):
        xs, ys = box[:, 0], box[:, 1]
        pad = max(2, int((ys.max() - ys.min()) * 0.12))
        crop = rgb.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                         min(rgb.width, xs.max() + pad), min(rgb.height, ys.max() + pad)))
        if crop.width >= 8 and crop.height >= 8:
            out.append((box.tolist(), crop))
    return out


class _LineReader(OCREngine):
    line_mode = False

    def read_line(self, img: Image.Image) -> tuple[str, float]:
        raise NotImplementedError

    def recognize(self, image: Image.Image) -> str:
        return "\n".join(" ".join(w for w, _ in line) for line in self.recognize_words(image))

    def recognize_words(self, image: Image.Image) -> list[list[tuple[str, float]]]:
        if self.line_mode:
            text, conf = self.read_line(image.convert("RGB"))
            return [[(w, conf) for w in text.split()]] if text.split() else []
        items = []
        for box, crop in detect_lines(image):
            text, conf = self.read_line(crop)
            if text.strip():
                items.append((box, text, conf))
        return _reading_order(items)


class TrOCREngine(_LineReader):
    def __init__(self, name: str = "trocr-base", num_beams: int = 4):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        base = name.removesuffix("-line")
        self.name, self.line_mode, self.num_beams = name, name.endswith("-line"), num_beams
        self.proc = TrOCRProcessor.from_pretrained(TROCR[base])
        self.model = VisionEncoderDecoderModel.from_pretrained(TROCR[base]).eval()

    @torch.no_grad()
    def read_line(self, img: Image.Image) -> tuple[str, float]:
        out = self.model.generate(self.proc(images=img, return_tensors="pt").pixel_values, max_new_tokens=128,
                                  num_beams=self.num_beams, output_scores=True, return_dict_in_generate=True)
        text = self.proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        conf = float(torch.exp(out.sequences_scores[0])) if self.num_beams > 1 else 1.0
        return detokenize(text), conf


class GLMOCREngine(_LineReader):
    def __init__(self, name: str = "glm-ocr"):
        from transformers import AutoProcessor, GlmOcrForConditionalGeneration

        self.name, self.line_mode = name, True  # GLM-OCR reads whole pages itself
        self.proc = AutoProcessor.from_pretrained("zai-org/GLM-OCR")
        self.model = GlmOcrForConditionalGeneration.from_pretrained("zai-org/GLM-OCR").eval()

    @torch.no_grad()
    def recognize_words(self, image: Image.Image) -> list[list[tuple[str, float]]]:
        msgs = [{"role": "user", "content": [{"type": "image", "image": image.convert("RGB")},
                                             {"type": "text", "text": "Text Recognition:"}]}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt")
        out = self.model.generate(**inp, max_new_tokens=1024, do_sample=False, output_scores=True,
                                  return_dict_in_generate=True)
        new = out.sequences[0][inp["input_ids"].shape[1]:]
        text = self.proc.decode(new, skip_special_tokens=True).strip()
        probs = [float(torch.softmax(s[0], -1)[t]) for s, t in zip(out.scores, new)]
        conf = float(np.mean(probs)) if probs else 0.0
        return [[(w, conf) for w in line.split()] for line in text.splitlines() if line.split()]
