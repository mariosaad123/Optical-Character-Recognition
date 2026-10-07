"""Image enhancement before recognition.

Measured on the dev split (Tesseract eng_best, accuracy = 1 - CER):

    step                      medium    hard
    none                      96.98%    42.84%
    denoise (3x sigma)        96.98%    59.62%
    upscale                   95.55%  -218.16%   <- magnifies noise into fake glyphs; not used
    denoise + normalize       99.24%    65.28%
    denoise 6x + normalize    99.16%    72.25%
    denoise + deskew + norm.  99.36%    66.08%

So: adaptive denoising (strength proportional to the measured noise), projection-profile
deskew, and background normalization (shadows / uneven light). Two strengths are exposed
as separate variants so the voting step gets diverse readings.
"""

import re

import cv2
import numpy as np
from PIL import Image, ImageOps

def to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"))


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary ink mask that ignores isolated noise specks (median blur + opening)."""
    smooth = cv2.medianBlur(gray, 3)
    _, bw = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def estimate_skew(gray: np.ndarray, max_angle: float = 5.0, step: float = 0.2, min_gain: float = 1.15) -> float:
    """Projection-profile method: the right angle makes text rows sharpest (max row-sum variance).
    Returns 0 unless the best angle is clearly better than no rotation (protects noisy images)."""
    small = gray
    scale = 800 / max(gray.shape)
    if scale < 1:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bw = _ink_mask(small)
    # pad so rotating a wide, short image does not push text out of the frame
    py = int(bw.shape[1] * np.sin(np.radians(max_angle)) / 2) + 2
    bw = cv2.copyMakeBorder(bw, py, py, 2, 2, cv2.BORDER_CONSTANT, value=0)
    h, w = bw.shape
    center = (w / 2, h / 2)
    scores = {}
    for angle in np.arange(-max_angle, max_angle + 1e-9, step):
        m = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rot = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        scores[round(float(angle), 2)] = float(np.var(rot.sum(axis=1)))
    best = max(scores, key=scores.get)
    if scores[best] < min_gain * scores.get(0.0, 0.0):
        return 0.0
    return best


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    bg = int(np.median(gray))  # fill with paper colour; replicating edge pixels creates fake strokes
    return cv2.warpAffine(gray, m, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


def normalize_background(gray: np.ndarray) -> np.ndarray:
    """Divide by a smooth background estimate -> flat white paper, preserved ink contrast."""
    k = max(15, (min(gray.shape) // 8) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4)
    norm = cv2.divide(gray.astype(np.float32), bg.astype(np.float32) + 1e-3, scale=255.0)
    return np.clip(norm, 0, 255).astype(np.uint8)


def denoise(gray: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return gray
    return cv2.fastNlMeansDenoising(gray, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def noise_level(gray: np.ndarray) -> float:
    """Robust noise sigma estimate from the Laplacian (Immerkaer's method)."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    lap = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    return float(np.sqrt(np.pi / 2) * np.mean(np.abs(lap)) / 6)


def enhance(image: Image.Image, strength: float = 6.0) -> Image.Image:
    """Adaptive enhancement: denoise only when noisy, deskew only when clearly skewed."""
    gray = to_gray(image)
    sigma = noise_level(gray)
    if sigma > 2.0:
        gray = denoise(gray, strength=min(30.0, strength * sigma))
    gray = rotate(gray, estimate_skew(gray))
    return Image.fromarray(normalize_background(gray))


def binarize(image: Image.Image) -> Image.Image:
    gray = to_gray(image)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return Image.fromarray(bw)


VARIANTS = {
    "raw": lambda im: im.convert("L"),
    "enhanced": enhance,
    "soft": lambda im: enhance(im, strength=3.0),
    "binary": lambda im: binarize(enhance(im)),
}


_orient_reader = None


def _read_boxes(gray: Image.Image):
    """PP-OCRv6 small without its 180-degree line classifier: (boxes, texts, scores)."""
    global _orient_reader
    if _orient_reader is None:
        from .engines.ppocr_engine import PPOCREngine
        _orient_reader = PPOCREngine("ppocr6s", use_cls=False)
    r = _orient_reader.ocr(np.array(gray.convert("RGB")))
    if r.boxes is None or not r.txts:
        return [], [], []
    return list(r.boxes), list(r.txts), [float(x) for x in r.scores]


def _readability(texts, scores) -> float:
    """Confidently read characters: an upright page yields many, a rotated one few."""
    return sum(len(t) * s for t, s in zip(texts, scores) if s > 0.5)


def auto_fix(image: Image.Image) -> tuple[Image.Image, list[str]]:
    """Undo conditions that break every engine: light-on-dark text, very dark photos, sideways or
    upside-down pages. Each fix is applied only when clearly needed; returns (image, fixes)."""
    gray = image.convert("L")
    arr = np.asarray(gray)
    fixes = []
    if np.median(arr) < 100 and np.percentile(arr, 99.5) - np.median(arr) > 60:
        # mostly dark with bright details: light text on a dark background
        gray, arr = ImageOps.invert(gray), 255 - arr
        fixes.append("invert")
    if np.median(arr) < 120:
        # the paper itself is dark: under-exposed photo. Stretch the used range to the full range.
        lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
        arr = np.clip((arr.astype(np.float32) - lo) * 255.0 / max(1.0, hi - lo), 0, 255).astype(np.uint8)
        gray = Image.fromarray(arr)
        fixes.append("exposure")

    boxes, texts, scores = _read_boxes(gray)
    base = _readability(texts, scores)
    if boxes:
        sizes = [(np.ptp(np.asarray(b)[:, 0]), np.ptp(np.asarray(b)[:, 1])) for b in boxes]
        vertical = sum(h > 1.5 * w for w, h in sizes) > len(sizes) / 2
        mean_conf = sum(scores) / len(scores)
        fill = int(np.median(arr))
        best_rot = 0
        if vertical:
            # a sideways page: the reader auto-turns tall crops one way only, so the untouched page is no
            # fair reference; compare the two upright candidates with each other instead
            r90, r270 = (_readability(*_read_boxes(gray.rotate(r, expand=True, fillcolor=fill))[1:]) for r in (90, 270))
            best_rot = 90 if r90 >= r270 else 270
            if max(r90, r270) < 1.15 * min(r90, r270) + 5:
                best_rot = 0
        elif mean_conf < 0.85:
            r180 = _readability(*_read_boxes(gray.rotate(180, expand=True, fillcolor=fill))[1:])
            # keep the flip only when it is clearly better than leaving the page as it is
            if r180 > 1.3 * base + 5:
                best_rot = 180
        if best_rot:
            gray = gray.rotate(best_rot, expand=True, fillcolor=fill)
            fixes.append(f"rotate{best_rot}")
    return (gray if fixes else image), fixes
