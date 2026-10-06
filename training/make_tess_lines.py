"""Generate single-line training data for fine-tuning Tesseract's LSTM.

Each sample: <name>.png + <name>.gt.txt + <name>.lstmf (Tesseract's training format).
Uses only TRAIN fonts and TRAIN books (test/dev fonts and books are never seen).

Usage:
    python -m training.make_tess_lines --out data/train_lines --n 40000
"""

import argparse
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path

from bench import resources
from bench.generate import degrade_v2, make_line, receipt_line, render, thermal

LEVEL_MIX = [("clean", 0.25), ("medium", 0.35), ("hard", 0.40)]


def _level(rng: random.Random) -> str:
    r, acc = rng.random(), 0.0
    for lvl, p in LEVEL_MIX:
        acc += p
        if r < acc:
            return lvl
    return "hard"


def make_one(args) -> str | None:
    i, seed, out, style = args
    rng = random.Random(seed * 1_000_003 + i)
    fonts = resources.fonts_for("train")
    if style == "receipt":
        text = receipt_line(rng)
    elif rng.random() < 0.6:
        lines = resources.prose_lines(rng, "train", 1)
        text = lines[0] if lines else make_line(rng)
    else:
        text = make_line(rng, 2, 9)
    text = text.strip()
    if not text:
        return None
    font = rng.choice(fonts)
    if resources.is_handwriting(font) and rng.random() < 0.7:
        font = rng.choice(fonts)
    level = _level(rng)
    size = rng.randint(18, 40) if level != "hard" else rng.randint(16, 32)
    try:
        # Tight crop: in raw-line mode (psm 13) Tesseract scales the whole image to the network's
        # input height, so wide margins would shrink the text and ruin training.
        pad = rng.randint(max(2, size // 8), max(3, size // 3))
        img = render([text], font, size, rng, vary=True, pad=pad)
        if style == "receipt" and rng.random() < 0.6:
            img = thermal(img, rng)
        img = degrade_v2(img, level, rng, max_angle=0.6)
    except Exception:
        return None
    base = out / f"l{i:06d}"
    img.save(f"{base}.png")
    Path(f"{base}.gt.txt").write_text(text + "\n")
    # whole-line box file in WordStr format (same as tesstrain's generate_line_box.py)
    w, h = img.size
    Path(f"{base}.box").write_text(f"WordStr 0 0 {w} {h} 0 #{text}\n\t 0 0 {w} {h} 0\n")
    r = subprocess.run(
        ["tesseract", f"{base}.png", str(base), "--psm", "13", "lstm.train"],
        capture_output=True, text=True,
    )
    lstmf = Path(f"{base}.lstmf")
    return str(lstmf) if r.returncode == 0 and lstmf.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/train_lines"))
    p.add_argument("--n", type=int, default=40000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-frac", type=float, default=0.02)
    p.add_argument("--style", choices=["general", "receipt"], default="general")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    with Pool(a.workers) as pool:
        files = [f for f in pool.imap(make_one, [(i, a.seed, a.out, a.style) for i in range(a.n)], chunksize=64) if f]
    n_eval = max(50, int(len(files) * a.eval_frac))
    (a.out / "list.eval").write_text("\n".join(files[:n_eval]) + "\n")
    (a.out / "list.train").write_text("\n".join(files[n_eval:]) + "\n")
    print(f"{len(files)} lines ({len(files) - n_eval} train / {n_eval} eval) in {a.out}")


if __name__ == "__main__":
    main()
