from .base import OCREngine


def get_engine(name: str) -> OCREngine:
    """Lazy-import engines so a missing optional dependency only breaks that engine."""
    if name == "tesseract":
        from .tesseract_engine import TesseractEngine
        return TesseractEngine()
    if name == "rapidocr":
        from .rapidocr_engine import RapidOCREngine
        return RapidOCREngine()
    if name == "router":
        from .router_engine import RouterEngine
        return RouterEngine()
    raise ValueError(f"Unknown engine: {name}")


ENGINES = ["tesseract", "rapidocr", "router"]
