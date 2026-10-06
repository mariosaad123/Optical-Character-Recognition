import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_engine.preprocess import enhance, estimate_skew, estimate_text_height, rotate, to_gray


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


def test_text_height_estimate_is_plausible():
    h = estimate_text_height(to_gray(_page(28)))
    assert 12 <= h <= 32


def test_enhance_upscales_small_text():
    small = _page(28).resize((300, 100))
    out = enhance(small)
    assert out.width > small.width
    assert np.asarray(out).dtype == np.uint8
