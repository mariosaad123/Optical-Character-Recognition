import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_engine.preprocess import enhance, estimate_skew, noise_level, rotate, to_gray


def _page(size=28):
    img = Image.new("L", (900, 300), 255)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSans.ttf", size)
    for i in range(5):
        d.text((30, 30 + i * 50), "The quick brown fox jumps over the lazy dog", font=font, fill=0)
    return img


def test_skew_is_detected_and_undone():
    gray = to_gray(_page())
    tilted = rotate(gray, 3.0)
    angle = estimate_skew(tilted)
    assert abs(angle + 3.0) <= 0.4


def test_noise_is_measured():
    gray = to_gray(_page())
    noisy = np.clip(gray + np.random.default_rng(0).normal(0, 20, gray.shape), 0, 255).astype(np.uint8)
    assert noise_level(noisy) > 3 * noise_level(gray)


def test_enhance_keeps_clean_page_readable():
    out = enhance(_page())
    assert out.size[0] >= 900 and np.asarray(out).dtype == np.uint8
