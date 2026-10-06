"""The full system: enhance -> several readers -> word-level voting -> word prediction.

    image ──┬─ raw ───────┬─ Tesseract (our fine-tuned model) ─┐
            │             └─ RapidOCR (PaddleOCR models)      ├─ ROVER voting ─ word prediction ─ text
            └─ enhanced ──┬─ Tesseract (our fine-tuned model) │   (confusion      (lexicon + bigram LM
                          └─ ...                              ┘    network)        + OCR confusions)

Every reader is a "source" = (engine, preprocessing variant). Source weights and the
word-prediction parameters are tuned on the dev split (bench/tune_pipeline.py).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .ensemble import Hypothesis, combine, to_lines
from .preprocess import VARIANTS

Lines = list[list[tuple[str, float]]]
CONFIG_PATH = Path(__file__).resolve().parent / "pipeline_config.json"


@dataclass
class PipelineConfig:
    sources: dict[str, float] = field(default_factory=lambda: {
        "tesseract-eng_ft/raw": 1.0,
        "tesseract-eng_ft/enhanced": 1.0,
        "rapidocr/raw": 0.8,
    })
    use_lm: bool = True
    lm_params: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "PipelineConfig":
        """The tuned config (bench/tune_pipeline.py) if present, else defaults."""
        return cls(**json.loads(path.read_text())) if path.exists() else cls()


def _engine(name: str):
    if name.startswith("tesseract"):
        from .engines.tesseract_engine import TesseractEngine
        return TesseractEngine("eng" if name == "tesseract" else name.split("-", 1)[1])
    if name == "rapidocr":
        from .engines.rapidocr_engine import RapidOCREngine
        return RapidOCREngine()
    raise ValueError(name)


def read_source(source: str, image: Image.Image, engines: dict, variants: dict) -> Lines:
    engine_name, variant = source.split("/")
    if variant not in variants:
        variants[variant] = VARIANTS[variant](image)
    if engine_name not in engines:
        engines[engine_name] = _engine(engine_name)
    return engines[engine_name].recognize_words(variants[variant])


def fuse(readings: dict[str, Lines], cfg: PipelineConfig, predictor=None) -> str:
    """Combine per-source readings into final text (pure function, shared by live use and benchmarks)."""
    hyps = [Hypothesis(readings[s], w, s) for s, w in cfg.sources.items() if s in readings and readings[s]]
    if not hyps:
        return ""
    # pivot = the most trustworthy reading (weight x mean confidence), it decides the layout
    def trust(h: Hypothesis) -> float:
        ws = h.words
        return h.weight * (sum(c for _, c in ws) / len(ws)) * min(1.0, len(ws) / max(len(x.words) for x in hyps))
    hyps.sort(key=trust, reverse=True)
    slots = combine(hyps)
    if predictor is None or not cfg.use_lm:
        return "\n".join(" ".join(w for w, _ in line) for line in to_lines(slots))

    lines, cur = [], []
    for s in slots:
        word, share = s.best()
        if word:
            cur.append((word, share, s.candidates()))
        if s.line_break_after and cur:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    out = []
    for line in lines:
        words = [w for w, _, _ in line]
        fixed = predictor.correct_line(words, alts=[c for _, _, c in line], confs=[s for _, s, _ in line])
        out.append(" ".join(fixed))
    return "\n".join(out)


class PipelineEngine:
    name = "pipeline"

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig.load()
        self.engines: dict = {}
        self.predictor = None
        if self.cfg.use_lm:
            from .lm import WordPredictor
            self.predictor = WordPredictor(**self.cfg.lm_params)

    def recognize(self, image: Image.Image) -> str:
        variants: dict = {}
        readings = {s: read_source(s, image, self.engines, variants) for s in self.cfg.sources}
        return fuse(readings, self.cfg, self.predictor)
