import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from .base import OCREngine

# The recognizer returns empty text for very wide crops; split lines wider than this (in heights).
MAX_ASPECT = 20


class RapidOCREngine(OCREngine):
    """PaddleOCR models (DBNet detection + SVTR recognition) via ONNX Runtime. Free, CPU-only."""

    name = "rapidocr"

    def __init__(self):
        # width_height_ratio=-1: always run detection (by default wide single-line images skip it and come back empty)
        self.ocr = RapidOCR(text_score=0.0, width_height_ratio=-1)

    def recognize(self, image: Image.Image) -> str:
        rgb = np.array(image.convert("RGB"))
        result, _ = self.ocr(rgb)
        if not result:
            return ""
        items = []
        for box, text, score in result:
            if float(score) < 0.5:
                text = self._recognize_wide_line(rgb, box)
            if text:
                items.append((box, text))
        return "\n".join(_reading_order(items))

    def _recognize_wide_line(self, rgb: np.ndarray, box) -> str:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        x0, x1 = max(0, int(min(xs))), min(rgb.shape[1], int(max(xs)) + 1)
        y0, y1 = max(0, int(min(ys))), min(rgb.shape[0], int(max(ys)) + 1)
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return ""
        pieces = [np.ascontiguousarray(crop[:, a:b]) for a, b in _split_at_gaps(crop, MAX_ASPECT * crop.shape[0])]
        rec, _ = self.ocr.text_recognizer(pieces)
        return " ".join(t.strip() for t, s in rec if t.strip() and float(s) >= 0.5)


def _split_at_gaps(crop: np.ndarray, max_width: int) -> list[tuple[int, int]]:
    """Cut a text line into pieces no wider than max_width, cutting at the centre of blank columns."""
    gray = crop.mean(axis=2)
    ink = (gray < gray.mean() - 0.5 * gray.std()).sum(axis=0)
    blank = ink == 0
    width = crop.shape[1]
    pieces, start = [], 0
    while width - start > max_width:
        window = blank[start + max_width // 2 : start + max_width]
        idx = np.flatnonzero(window)
        if idx.size:
            # pick the centre of the last blank run in the window
            runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
            cut = start + max_width // 2 + int(runs[-1].mean())
        else:
            cut = start + max_width
        pieces.append((start, cut))
        start = cut
    pieces.append((start, width))
    return pieces


def _reading_order(items) -> list[str]:
    """Group boxes into lines (top-to-bottom), then sort each line left-to-right."""
    boxes = []
    for box, text in items:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        boxes.append((min(ys), max(ys), min(xs), text))
    boxes.sort(key=lambda b: (b[0] + b[1]) / 2)

    lines: list[list[tuple]] = []
    for b in boxes:
        center = (b[0] + b[1]) / 2
        if lines:
            last = lines[-1]
            top = min(x[0] for x in last)
            bottom = max(x[1] for x in last)
            if top <= center <= bottom:
                last.append(b)
                continue
        lines.append([b])
    return [" ".join(x[3] for x in sorted(line, key=lambda x: x[2])) for line in lines]
