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


def bow_f1(reference: str, hypothesis: str) -> float:
    """Order-independent word accuracy (F1 over word multisets). Robust to reading-order
    differences, e.g. multi-column receipts where engines may read columns differently."""
    from collections import Counter

    ref, hyp = Counter(normalize(reference).split()), Counter(normalize(hypothesis).split())
    if not ref:
        return 1.0 if not hyp else 0.0
    tp = sum((ref & hyp).values())
    if tp == 0:
        return 0.0
    precision, recall = tp / sum(hyp.values()), tp / sum(ref.values())
    return 2 * precision * recall / (precision + recall)
