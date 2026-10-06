from pathlib import Path

import pytesseract
from PIL import Image

from .base import OCREngine

ROOT = Path(__file__).resolve().parents[2]
# our trained models are committed in models/; downloaded ones (eng_best) live in data/tessdata/
SEARCH_DIRS = [ROOT / "models", ROOT / "data" / "tessdata"]


class TesseractEngine(OCREngine):
    """Classic LSTM-based engine (Tesseract 5). Free, CPU-only.

    model: "eng" (system, fast int model), or the name of a .traineddata file in models/ or
    data/tessdata/ (e.g. "eng_best", the float model, or "eng_ft2", our fine-tuned model).
    """

    name = "tesseract"

    def __init__(self, model: str = "eng", psm: int = 6):
        # psm 6 = assume a single uniform block of text
        self.config = f"--oem 1 --psm {psm}"
        if model != "eng":
            found = [d for d in SEARCH_DIRS if (d / f"{model}.traineddata").exists()]
            if not found:
                raise FileNotFoundError(f"{model}.traineddata not in {SEARCH_DIRS} (run scripts/fetch_resources.py)")
            self.config += f' --tessdata-dir "{found[0]}"'
        self.lang = model
        self.name = "tesseract" if model == "eng" else f"tesseract-{model}"

    def recognize(self, image: Image.Image) -> str:
        return self.recognize_with_confidence(image)[0]

    def recognize_words(self, image: Image.Image) -> list[list[tuple[str, float]]]:
        """Lines of (word, confidence in [0, 1]) from a single Tesseract pass."""
        data = pytesseract.image_to_data(image, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT)
        lines: dict[tuple, list[tuple[str, float]]] = {}
        for i, word in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if conf < 0 or not word.strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append((word, conf / 100))
        return [words for _, words in sorted(lines.items())]

    def recognize_with_confidence(self, image: Image.Image) -> tuple[str, float]:
        """Return (text, mean word confidence in [0, 1])."""
        lines = self.recognize_words(image)
        confs = [c for line in lines for _, c in line]
        text = "\n".join(" ".join(w for w, _ in line) for line in lines)
        return text, (sum(confs) / len(confs) if confs else 0.0)
