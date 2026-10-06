"""Accuracy metrics for OCR output.

CER = character edit distance / reference length
WER = word edit distance / reference word count
Lower is better; accuracy is reported as max(0, 1 - CER).
"""

from rapidfuzz.distance import Levenshtein


def normalize(text: str) -> str:
    """Collapse all whitespace (including line breaks) to single spaces."""
    return " ".join(text.split())


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)
