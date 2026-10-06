"""Synthetic English OCR benchmark generator.

Renders text with many fonts and three degradation levels (clean / medium / hard).
Because we render the text ourselves, the ground truth is exact and the cost is zero.
Output is deterministic for a given --seed.

v1: random document-style word sequences, system fonts (the original benchmark).
v2: real prose from held-out public-domain books mixed with document lines, held-out font families,
    and a richer degradation model (perspective, ink bleed, paper texture, motion blur).

Usage:
    python -m bench.generate --out data/bench_en --per-level 100
    python -m bench.generate --version v2 --split test --out data/bench_en_v2 --per-level 150 --seed 2024
"""

import argparse
import io
import json
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from bench import resources

LEVELS = ["clean", "medium", "hard"]

WORDS = (
    "the of and to in is that for it as was with be by on not he this are or his from at which "
    "but have an they you were her she there one all we their can has been if more when will would "
    "who so no report invoice total amount payment customer account balance service delivery order "
    "number date address phone email company limited street avenue suite contract agreement section "
    "clause party parties hereby shall terms conditions effective period renewal notice written "
    "system model data training accuracy results analysis method performance network learning image "
    "document page table figure reference journal research university department science quick brown "
    "fox jumps over lazy dog quality control shipping warehouse product quantity unit price discount "
    "tax subtotal due receipt signature approved manager director meeting schedule project budget"
).split()

FIRST = "John Mary Ahmed Sarah David Fatima Michael Laura Omar Emily James Nadia Robert Olivia".split()
LAST = "Smith Johnson Hassan Brown Garcia Miller Saad Wilson Taylor Anderson Thomas Lee".split()
MONTHS = "January February March April May June July August September October November December".split()


def _token(rng: random.Random) -> str:
    """Mostly words, plus the things real documents are full of and OCR gets wrong."""
    r = rng.random()
    if r < 0.70:
        w = rng.choice(WORDS)
        return w.capitalize() if rng.random() < 0.12 else (w.upper() if rng.random() < 0.03 else w)
    if r < 0.76:
        return f"${rng.randint(1, 99999):,}.{rng.randint(0, 99):02d}"
    if r < 0.81:
        return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2030)}"
    if r < 0.85:
        return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, {rng.randint(1990, 2030)}"
    if r < 0.89:
        return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    if r < 0.92:
        return f"{rng.choice(FIRST).lower()}.{rng.choice(LAST).lower()}@example.com"
    if r < 0.95:
        return f"INV-{rng.randint(1000, 99999)}"
    if r < 0.98:
        return f"{rng.randint(0, 100)}%"
    return f"+1 ({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}"


def make_line(rng: random.Random, min_words=4, max_words=11) -> str:
    words = [_token(rng) for _ in range(rng.randint(min_words, max_words))]
    line = " ".join(words)
    line = line[0].upper() + line[1:]
    return line + rng.choice([".", ".", ",", ":", ";", "", "?", "!"])


ITEMS = ("COFFEE MILK TEA RICE CHICKEN BEEF FISH NOODLE BREAD SUGAR SALT WATER JUICE APPLE ORANGE BAG PEN PAPER "
         "TAPE GLUE SOAP OIL EGG CAKE SOUP SALAD STEAK BURGER FRIES PIZZA PLATE CUP BOX SET PACK MINI LARGE").split()
LABELS = ["TOTAL", "SUBTOTAL", "CASH", "CHANGE", "ROUNDING", "DISCOUNT", "TAX", "GST", "SERVICE CHARGE", "AMOUNT DUE",
          "TOTAL SALES (INCLUSIVE OF GST)", "TOTAL QTY", "NET TOTAL", "BALANCE", "PAID"]


def _money(rng: random.Random) -> str:
    return f"{rng.randint(0, 999)}.{rng.randint(0, 99):02d}"


def receipt_line(rng: random.Random) -> str:
    """Receipt-style line: prices with decimals, codes, dates, times, mostly upper case."""
    r = rng.random()
    if r < 0.30:
        name = " ".join(rng.choice(ITEMS) for _ in range(rng.randint(1, 4)))
        return f"{rng.randint(1, 9)} {name} {rng.choice(['', 'SR ', 'ZR ', 'RM '])}{_money(rng)}".replace("  ", " ")
    if r < 0.50:
        sep = rng.choice([" : ", ": ", " ", " RM "])
        return f"{rng.choice(LABELS)}{sep}{rng.choice(['', '-'])}{_money(rng)}"
    if r < 0.60:
        return (f"DATE: {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2010, 2030)} "
                f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}")
    if r < 0.68:
        return f"{rng.choice(['TEL', 'TEL:', 'FAX', 'PHONE'])} {rng.randint(0, 9)}{rng.randint(1, 9)}-{rng.randint(1000000, 9999999)}"
    if r < 0.76:
        return f"{rng.choice(['GST ID', 'GST REG NO', 'INVOICE NO', 'DOC NO', 'RECEIPT #', 'BILL NO'])}{rng.choice([': ', ' : ', ' '])}{rng.choice(['', 'CS', 'OR', 'INV'])}{rng.randint(10**5, 10**12)}"
    if r < 0.84:
        return f"{rng.choice(['SR', 'ZR', 'TX'])} {rng.choice([0, 6, 10])} {_money(rng)} {_money(rng)}"
    line = make_line(rng, 2, 7)
    return line.upper() if rng.random() < 0.6 else line


def thermal(img: Image.Image, rng: random.Random) -> Image.Image:
    """Faded thermal-printer look: patchy ink, low contrast, horizontal banding."""
    arr = np.asarray(img).astype(np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    paper = float(np.median(arr))
    ink = arr < paper - 40
    fade = cv2.GaussianBlur(np_rng.random(arr.shape).astype(np.float32), (0, 0), rng.uniform(1.0, 3.0))
    fade = (fade - fade.min()) / (np.ptp(fade) + 1e-6)
    strength = rng.uniform(0.2, 0.6)
    arr = np.where(ink, arr + (paper - arr) * strength * fade, arr)
    band = 1 + 0.06 * np.sin(np.arange(arr.shape[0]) / rng.uniform(1.5, 4))[:, None]
    return Image.fromarray(np.clip(arr * band, 0, 255).astype(np.uint8))


def find_fonts() -> list[str]:
    out = subprocess.run(["fc-list", "--format", "%{file}\n"], capture_output=True, text=True).stdout
    skip = ("emoji", "ipa", "unifont", "wqy", "opens", "loma", "japanese")
    fonts = sorted({f for f in out.splitlines() if f.lower().endswith(".ttf") and not any(s in f.lower() for s in skip)})
    if not fonts:
        raise RuntimeError("No .ttf fonts found (install fonts-liberation / fonts-dejavu).")
    return fonts


def load_font(font_path: str, size: int, rng: random.Random | None = None) -> ImageFont.FreeTypeFont:
    """Load a font; for variable fonts, pick a random named instance (weight/width) when rng is given."""
    font = ImageFont.truetype(font_path, size)
    if rng is not None:
        try:
            names = [n for n in font.get_variation_names() if b"Thin" not in n and b"Hairline" not in n]
            if names:
                font.set_variation_by_name(rng.choice(names))
        except OSError:
            pass  # not a variable font
    return font


def render(lines: list[str], font_path: str, size: int, rng: random.Random, vary: bool = False,
           pad: int | None = None) -> Image.Image:
    font = load_font(font_path, size, rng if vary else None)
    spacing = int(size * rng.uniform(0.25, 0.6))
    ink = rng.randint(0, 60)
    paper = rng.randint(225, 255)
    widths = [font.getbbox(l)[2] for l in lines]
    line_h = font.getbbox("Ag|")[3]
    if pad is None:
        pad = rng.randint(size // 2, size * 2)
    w = max(widths) + 2 * pad
    h = len(lines) * line_h + (len(lines) - 1) * spacing + 2 * pad
    img = Image.new("L", (w, h), paper)
    draw = ImageDraw.Draw(img)
    y = pad
    for line in lines:
        draw.text((pad, y), line, font=font, fill=ink)
        y += line_h + spacing
    return img


def degrade(img: Image.Image, level: str, rng: random.Random, max_angle: float | None = None) -> Image.Image:
    if level == "clean":
        return img
    strong = level == "hard"
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))

    # rotation (scan skew)
    angle = rng.uniform(-3.0, 3.0) if strong else rng.uniform(-1.0, 1.0)
    if max_angle is not None:
        angle = max(-max_angle, min(max_angle, angle))
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=int(np.median(np.asarray(img))))

    # resolution loss (low-DPI scan / phone photo)
    scale = rng.uniform(0.45, 0.65) if strong else rng.uniform(0.7, 0.9)
    small = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
    img = small.resize(img.size, Image.BILINEAR)

    # blur
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.3) if strong else rng.uniform(0.2, 0.6)))

    arr = np.asarray(img).astype(np.float32)
    # uneven lighting (shadow gradient)
    if strong or rng.random() < 0.3:
        gx = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.9, 1.1), arr.shape[1])
        arr = arr * gx[None, :]
    # sensor noise
    arr += np_rng.normal(0, 14 if strong else 6, arr.shape)
    # salt & pepper specks
    if strong:
        mask = np_rng.random(arr.shape)
        arr[mask < 0.004] = 0
        arr[mask > 0.996] = 255
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # JPEG compression artifacts
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(20, 40) if strong else rng.randint(50, 80))
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


def degrade_v2(img: Image.Image, level: str, rng: random.Random, max_angle: float | None = None) -> Image.Image:
    """v1 degradations plus perspective, ink thickness, paper texture and motion blur."""
    if level == "clean":
        return img if rng.random() < 0.7 else degrade(img, "medium", random.Random(rng.random()), max_angle)
    strong = level == "hard"
    arr = np.asarray(img)
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    bg = int(np.median(arr))

    # ink thickness: thin (erode background = dilate ink) or bleed
    if rng.random() < 0.5:
        k = np.ones((2, 2), np.uint8)
        arr = cv2.erode(arr, k) if rng.random() < 0.5 else cv2.dilate(arr, k)

    # perspective (photo of a page)
    if rng.random() < (0.7 if strong else 0.3):
        h, w = arr.shape
        d = (0.04 if strong else 0.015) * min(h, w) + (0.01 if strong else 0.004) * w
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np_rng.uniform(-d, d, src.shape).astype(np.float32)
        arr = cv2.warpPerspective(arr, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=bg)

    # motion blur (hand shake)
    if rng.random() < (0.4 if strong else 0.15):
        k = rng.choice([3, 5]) if strong else 3
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        arr = cv2.filter2D(arr, -1, cv2.warpAffine(kernel, rot, (k, k)))

    # paper texture
    if rng.random() < 0.5:
        tex = cv2.GaussianBlur(np_rng.normal(0, 1, arr.shape).astype(np.float32), (0, 0), rng.uniform(2, 6))
        arr = np.clip(arr.astype(np.float32) + tex * (10 if strong else 5), 0, 255).astype(np.uint8)

    return degrade(Image.fromarray(arr), level, rng, max_angle)


def generate_v2(out_dir: Path, per_level: int, seed: int, split: str) -> None:
    rng = random.Random(seed)
    fonts = [f for f in resources.fonts_for(split) if split == "train" or not resources.is_handwriting(f)]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for level in LEVELS:
        for i in range(per_level):
            n_lines = 1 if rng.random() < 0.3 else rng.randint(2, 8)
            if rng.random() < 0.6:
                lines = resources.prose_lines(rng, split, n_lines)
            else:
                lines = [make_line(rng) for _ in range(n_lines)]
            font = rng.choice(fonts)
            size = rng.randint(20, 40) if level != "hard" else rng.randint(16, 32)
            img = degrade_v2(render(lines, font, size, rng, vary=True), level, rng)
            name = f"{level}_{i:04d}.png"
            img.save(out_dir / name)
            manifest.append({"image": name, "level": level, "font": Path(font).name, "size": size, "text": "\n".join(lines)})
    (out_dir / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    print(f"Wrote {len(manifest)} samples ({len(fonts)} {split} fonts) to {out_dir}")


def generate(out_dir: Path, per_level: int, seed: int) -> None:
    rng = random.Random(seed)
    fonts = find_fonts()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for level in LEVELS:
        for i in range(per_level):
            n_lines = 1 if rng.random() < 0.5 else rng.randint(2, 6)
            lines = [make_line(rng) for _ in range(n_lines)]
            font = rng.choice(fonts)
            size = rng.randint(22, 40) if level != "hard" else rng.randint(18, 32)
            img = degrade(render(lines, font, size, rng), level, rng)
            name = f"{level}_{i:04d}.png"
            img.save(out_dir / name)
            manifest.append({"image": name, "level": level, "font": Path(font).name, "size": size, "text": "\n".join(lines)})
    (out_dir / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    print(f"Wrote {len(manifest)} samples ({len(fonts)} fonts) to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/bench_en"))
    p.add_argument("--per-level", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--version", choices=["v1", "v2"], default="v1")
    p.add_argument("--split", choices=["train", "dev", "test"], default="test")
    a = p.parse_args()
    if a.version == "v1":
        generate(a.out, a.per_level, a.seed)
    else:
        generate_v2(a.out, a.per_level, a.seed, a.split)


if __name__ == "__main__":
    main()
