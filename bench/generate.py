"""Synthetic English OCR benchmark generator.

Renders text with many fonts and three degradation levels (clean / medium / hard).
Because we render the text ourselves, the ground truth is exact and the cost is zero.
Output is deterministic for a given --seed.

Usage:
    python -m bench.generate --out data/bench_en --per-level 100
"""

import argparse
import io
import json
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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


def find_fonts() -> list[str]:
    out = subprocess.run(["fc-list", "--format", "%{file}\n"], capture_output=True, text=True).stdout
    skip = ("emoji", "ipa", "unifont", "wqy", "opens", "loma", "japanese")
    fonts = sorted({f for f in out.splitlines() if f.lower().endswith(".ttf") and not any(s in f.lower() for s in skip)})
    if not fonts:
        raise RuntimeError("No .ttf fonts found (install fonts-liberation / fonts-dejavu).")
    return fonts


def render(lines: list[str], font_path: str, size: int, rng: random.Random) -> Image.Image:
    font = ImageFont.truetype(font_path, size)
    spacing = int(size * rng.uniform(0.25, 0.6))
    ink = rng.randint(0, 60)
    paper = rng.randint(225, 255)
    widths = [font.getbbox(l)[2] for l in lines]
    line_h = font.getbbox("Ag|")[3]
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


def degrade(img: Image.Image, level: str, rng: random.Random) -> Image.Image:
    if level == "clean":
        return img
    strong = level == "hard"
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))

    # rotation (scan skew)
    angle = rng.uniform(-3.0, 3.0) if strong else rng.uniform(-1.0, 1.0)
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
    a = p.parse_args()
    generate(a.out, a.per_level, a.seed)


if __name__ == "__main__":
    main()
