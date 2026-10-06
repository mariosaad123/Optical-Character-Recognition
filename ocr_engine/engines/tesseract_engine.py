import pytesseract
from PIL import Image

from .base import OCREngine


class TesseractEngine(OCREngine):
    """Classic LSTM-based engine (Tesseract 5). Free, CPU-only."""

    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 6):
        # psm 6 = assume a single uniform block of text
        self.config = f"--oem 1 --psm {psm}"
        self.lang = lang

    def recognize(self, image: Image.Image) -> str:
        return self.recognize_with_confidence(image)[0]

    def recognize_with_confidence(self, image: Image.Image) -> tuple[str, float]:
        """Return (text, mean word confidence in [0, 1]) from a single Tesseract pass."""
        data = pytesseract.image_to_data(image, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT)
        lines: dict[tuple, list[str]] = {}
        confs = []
        for i, word in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if conf < 0 or not word.strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            confs.append(conf / 100)
        text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
        return text, (sum(confs) / len(confs) if confs else 0.0)
