from PIL import Image

from .base import OCREngine
from .rapidocr_engine import RapidOCREngine
from .tesseract_engine import TesseractEngine

# Tuned on the dev split (seed 999), never on the test split. See bench/tune_router.py.
DEFAULT_THRESHOLD = 0.56


class RouterEngine(OCREngine):
    """Fast path first: Tesseract (best on clean pages). If its confidence is low,
    the page is probably degraded, so fall back to RapidOCR (more robust to noise/blur)."""

    name = "router"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.fast = TesseractEngine()
        self._robust = None

    @property
    def robust(self) -> RapidOCREngine:
        if self._robust is None:
            self._robust = RapidOCREngine()
        return self._robust

    def recognize(self, image: Image.Image) -> str:
        text, conf = self.fast.recognize_with_confidence(image)
        if conf >= self.threshold:
            return text
        return self.robust.recognize(image)
