"""Measure each preprocessing variant x Tesseract model on a (dev) set.

Usage:
    python -m bench.ablate_preprocess --data data/dev_en_v2 --models eng eng_best
"""

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from bench.metrics import cer
from ocr_engine.engines.tesseract_engine import TesseractEngine
from ocr_engine.preprocess import VARIANTS


def _run(args):
    data, s, models = args
    img = Image.open(data / s["image"])
    out = []
    for vname, fn in VARIANTS.items():
        v = fn(img)
        for m in models:
            out.append((m, vname, s["level"], cer(s["text"], TesseractEngine(m).recognize(v))))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/dev_en_v2"))
    p.add_argument("--models", nargs="+", default=["eng", "eng_best"])
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    samples = [json.loads(l) for l in (a.data / "manifest.jsonl").read_text().splitlines() if l.strip()]
    agg = defaultdict(list)
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(_run, [(a.data, s, a.models) for s in samples]):
            for m, v, lvl, c in res:
                agg[(m, v, lvl)].append(c)
                agg[(m, v, "all")].append(c)
    print("| model | variant | clean | medium | hard | all |\n|---|---|---|---|---|---|")
    for m in a.models:
        for v in VARIANTS:
            cells = [f"{(1 - sum(agg[(m, v, l)]) / len(agg[(m, v, l)])) * 100:.2f}%" for l in ["clean", "medium", "hard", "all"]]
            print(f"| {m} | {v} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
