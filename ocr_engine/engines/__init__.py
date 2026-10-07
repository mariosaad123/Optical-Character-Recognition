from .base import OCREngine


def get_engine(name: str) -> OCREngine:
    """Lazy-import engines so a missing optional dependency only breaks that engine."""
    if name == "tesseract":
        from .tesseract_engine import TesseractEngine
        return TesseractEngine()
    if name.startswith("tesseract-"):  # e.g. tesseract-eng_best, tesseract-eng_ft
        from .tesseract_engine import TesseractEngine
        return TesseractEngine(name.split("-", 1)[1])
    if name == "pipeline":
        from ..pipeline import PipelineEngine
        return PipelineEngine()
    if name == "rapidocr":
        from .rapidocr_engine import RapidOCREngine
        return RapidOCREngine()
    if name.startswith("ppocr"):  # ppocr6s, ppocr6m, ppocr5en
        from .ppocr_engine import PPOCREngine
        return PPOCREngine(name)
    if name == "router":
        from .router_engine import RouterEngine
        return RouterEngine()
    raise ValueError(f"Unknown engine: {name}")


ENGINES = ["tesseract", "rapidocr", "router", "pipeline"]
