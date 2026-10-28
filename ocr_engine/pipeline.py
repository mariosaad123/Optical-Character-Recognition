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
from .preprocess import VARIANTS, auto_fix, noise_level, to_gray

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
    min_conf: float = 0.0  # drop words below this confidence (noise read as glyphs)
    char_vote: bool = False  # character-level voting inside disputed word slots
    # word prediction only runs when at least this share of the confidently read words are English
    # (receipts in other languages, codes or menus must not be "corrected" into English)
    english_gate: float = 0.0
    # image-quality router: pages noisier than this (Immerkaer sigma) use the `noisy` voting config
    noise_threshold: float = 99.0
    noisy: dict | None = None
    # speed cascade: the `cheap` sources run first; if their texts agree (similarity >= tau) their vote
    # is final and the expensive sources are skipped (bench/tune_cascade.py)
    cascade: dict | None = None

    def for_image(self, sigma: float) -> "PipelineConfig":
        """The config to use for an image with this noise level (word prediction settings are shared)."""
        if self.noisy is None or sigma <= self.noise_threshold:
            return self
        return PipelineConfig(**{**self.__dict__, **self.noisy, "noisy": None})

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
    if name.startswith("ppocr"):
        from .engines.ppocr_engine import PPOCREngine
        return PPOCREngine(name)
    if name.startswith("trocr"):
        from .engines.hw_engines import TrOCREngine
        return TrOCREngine(name)
    if name.startswith("glm-ocr"):
        from .engines.hw_engines import GLMOCREngine
        return GLMOCREngine(name)
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
    def clean(lines: Lines) -> Lines:
        return [[(w, c) for w, c in line if c >= cfg.min_conf] for line in lines]

    hyps = [Hypothesis(clean(readings[s]), w, s) for s, w in cfg.sources.items() if s in readings and readings[s]]
    if not hyps:
        return ""
    hyps = [h for h in hyps if h.words]
    if not hyps:
        return ""
    # pivot = the most trustworthy reading (weight x mean confidence x coverage); it decides the layout
    longest = max(len(h.words) for h in hyps)

    def trust(h: Hypothesis) -> float:
        ws = h.words
        return h.weight * (sum(c for _, c in ws) / len(ws)) * min(1.0, len(ws) / longest)
    hyps.sort(key=trust, reverse=True)
    slots = combine(hyps)
    if predictor is None or not cfg.use_lm:
        return "\n".join(" ".join(w for w, _ in line) for line in to_lines(slots, cfg.char_vote))

    lines, cur = [], []
    for s in slots:
        word, share = s.best_by_chars() if cfg.char_vote else s.best()
        if word:
            # certainty = agreement between readers x the engines' own confidence
            cur.append((word, share * s.engine_conf(word), s.candidates()))
        if s.line_break_after and cur:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    if cfg.english_gate and english_share(lines, predictor) < cfg.english_gate:
        return "\n".join(" ".join(w for w, _, _ in line) for line in lines)
    out = []
    for line in lines:
        words = [w for w, _, _ in line]
        fixed = predictor.correct_line(words, alts=[c for _, _, c in line], confs=[s for _, s, _ in line],
                                       context=out[-1] if out else "")
        out.append(" ".join(fixed))
    return "\n".join(out)


def english_share(lines, predictor) -> float:
    """Share of confidently read alphabetic words that are in the English lexicon."""
    from .lm import LETTERS_RE, _core

    words = [_core(w) for line in lines for w, conf, _ in line if conf >= 0.8]
    words = [w for w in words if len(w) >= 3 and LETTERS_RE.match(w)]
    if not words:
        return 1.0
    return sum(predictor.lm.known(w) for w in words) / len(words)


class PipelineEngine:
    name = "pipeline"

    MODES = {"accurate": None, "balanced": 0.98, "fast": 0.85}  # cascade tau per mode

    def __init__(self, cfg: PipelineConfig | None = None, mode: str = "balanced"):
        self.cfg = cfg or PipelineConfig.load()
        self.mode = mode
        self.engines: dict = {}
        self.predictor = None
        if self.cfg.use_lm:
            from .lm import WordPredictor
            self.predictor = WordPredictor(**self.cfg.lm_params)

    def recognize(self, image: Image.Image) -> str:
        seed: dict = {}  # auto_fix's own PP-OCRv6-small reading, reused below
        image, self.last_fixes = auto_fix(image, seed)  # sideways / upside-down / inverted / dark pages
        cfg = self.cfg.for_image(noise_level(to_gray(image)))
        variants: dict = {}
        tau = self.MODES[self.mode]
        cheap = (cfg.cascade or {}).get("cheap", [])
        if tau is not None and cheap and cfg is self.cfg:
            readings = {s: seed[s] if s in seed else read_source(s, image, self.engines, variants) for s in cheap}
            if cascade_agreement(readings, cheap) >= tau:
                quick = PipelineConfig(**{**cfg.__dict__, "sources": {s: cfg.sources.get(s, 1.0) for s in cheap}})
                return fuse(readings, quick, self.predictor)
        else:
            readings = {k: v for k, v in seed.items() if k in cfg.sources}
        for s in cfg.sources:
            if s not in readings:
                readings[s] = read_source(s, image, self.engines, variants)
        return fuse(readings, cfg, self.predictor)


def cascade_agreement(readings: dict, cheap: list[str]) -> float:
    """Lowest character similarity between the cheap readers' texts (0 if any read nothing)."""
    from rapidfuzz.distance import Levenshtein

    texts = [" ".join(w for line in readings.get(s, []) for w, _ in line) for s in cheap]
    if not all(texts):
        return 0.0
    return min(Levenshtein.normalized_similarity(texts[0], t) for t in texts[1:])
