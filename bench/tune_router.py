"""Pick the router's Tesseract-confidence threshold on a separate dev split.

Usage:
    python -m bench.generate --out data/dev_en --per-level 50 --seed 999
    python -m bench.tune_router --data data/dev_en
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from bench.metrics import cer
from ocr_engine.engines.rapidocr_engine import RapidOCREngine
from ocr_engine.engines.tesseract_engine import TesseractEngine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/dev_en"))
    a = p.parse_args()

    samples = [json.loads(l) for l in (a.data / "manifest.jsonl").read_text().splitlines() if l.strip()]
    tess, rapid = TesseractEngine(), RapidOCREngine()
    rows = []
    for s in samples:
        img = Image.open(a.data / s["image"])
        t_text, conf = tess.recognize_with_confidence(img)
        rows.append((conf, cer(s["text"], t_text), cer(s["text"], rapid.recognize(img))))

    best = None
    print("threshold | mean CER | % sent to robust engine")
    for t in [x / 100 for x in range(50, 100, 2)]:
        errs = [tc if conf >= t else rc for conf, tc, rc in rows]
        mean = sum(errs) / len(errs)
        routed = sum(conf < t for conf, _, _ in rows) / len(rows)
        print(f"{t:.2f} | {mean:.4f} | {routed:.0%}")
        if best is None or mean < best[1]:
            best = (t, mean)
    print(f"\nbest threshold: {best[0]:.2f} (dev CER {best[1]:.4f})")


if __name__ == "__main__":
    main()
