from PIL import Image, ImageDraw, ImageFont, ImageOps

from ocr_engine.preprocess import auto_fix


def _page():
    img = Image.new("L", (900, 260), 245)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSans.ttf", 30)
    for i, line in enumerate(["The quick brown fox jumps over", "the lazy dog near the river bank", "and then it ran away quickly"]):
        d.text((30, 30 + i * 70), line, font=font, fill=20)
    return img


def test_upright_page_is_untouched():
    img = _page()
    out, fixes = auto_fix(img)
    assert fixes == [] and out is img


def test_inverted_page_is_restored():
    _, fixes = auto_fix(ImageOps.invert(_page()))
    assert "invert" in fixes


def test_sideways_page_is_rotated_back():
    _, fixes = auto_fix(_page().rotate(90, expand=True))
    assert fixes == ["rotate270"]
