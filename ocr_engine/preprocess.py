"""Image enhancement before recognition.

Each step targets one failure mode seen on the benchmark:
- upscale:   low-DPI text (Tesseract works best with ~30px capital height)
- deskew:    rotated scans / photos
- normalize: shadows and uneven lighting (divide by estimated background)
- denoise:   sensor noise, salt & pepper, JPEG artifacts
"""

import cv2
import numpy as np
from PIL import Image

TARGET_TEXT_HEIGHT = 32  # px, height of a typical lowercase+ascender glyph after scaling


def to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"))


def estimate_text_height(gray: np.ndarray) -> float:
    """Median height of glyph-sized connected components."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return 0.0
    h, w, area = stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_AREA]
    keep = (h >= 4) & (area >= 8) & (h < gray.shape[0] * 0.5) & (w < gray.shape[1] * 0.3)
    return float(np.median(h[keep])) if keep.any() else 0.0


def upscale(gray: np.ndarray, max_factor: float = 4.0) -> tuple[np.ndarray, float]:
    th = estimate_text_height(gray)
    if th <= 0:
        return gray, 1.0
    factor = float(np.clip(TARGET_TEXT_HEIGHT / th, 1.0, max_factor))
    if factor < 1.15:
        return gray, 1.0
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC), factor


def estimate_skew(gray: np.ndarray, max_angle: float = 6.0, step: float = 0.2) -> float:
    """Projection-profile method: the right angle makes text rows sharpest (max row-sum variance)."""
    small = gray
    scale = 800 / max(gray.shape)
    if scale < 1:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape
    center = (w / 2, h / 2)
    best, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-9, step):
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best, best_score = float(angle), score
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
    return cv2.warpAffine(gray, m, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


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


def enhance(image: Image.Image) -> Image.Image:
    """Adaptive enhancement: each step runs only when the image needs it."""
    gray = to_gray(image)
    sigma = noise_level(gray)
    if sigma > 4:
        gray = denoise(gray, strength=min(25.0, 1.2 * sigma))
    gray, _ = upscale(gray)
    gray = rotate(gray, estimate_skew(gray))
    gray = normalize_background(gray)
    return Image.fromarray(gray)


def binarize(image: Image.Image) -> Image.Image:
    gray = to_gray(image)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return Image.fromarray(bw)


VARIANTS = {
    "raw": lambda im: im.convert("L"),
    "enhanced": enhance,
    "binary": lambda im: binarize(enhance(im)),
}
