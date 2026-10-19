"""Measure the average time of each reading source on this CPU (single process, all cores).

Usage:
    python -m bench.time_sources --data data/dev_en_v2 data/real_sroie_dev data/real_cord_dev --per-set 6
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from ocr_engine.pipeline import CONFIG_PATH, read_source
from ocr_engine.preprocess import auto_fix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", type=Path, required=True)
    p.add_argument("--per-set", type=int, default=6)
    p.add_argument("--out", type=Path, default=Path("results/source_timings.json"))
    a = p.parse_args()
    cfg = json.loads(CONFIG_PATH.read_text())
    sources = sorted(set(cfg["sources"]) | set((cfg.get("noisy") or {}).get("sources", {})))
    images = []
    for d in a.data:
        ms = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip()]
        images += [Image.open(d / m["image"]) for m in ms[:: max(1, len(ms) // a.per_set)][: a.per_set]]
    engines: dict = {}
    for src in sources:  # warm up: load every model once
        read_source(src, images[0], engines, {})
    totals = {s: 0.0 for s in sources} | {"auto_fix": 0.0}
    for img in images:
        t = time.time(); auto_fix(img); totals["auto_fix"] += time.time() - t
        for src in sources:
            t = time.time(); read_source(src, img, engines, {}); totals[src] += time.time() - t
    out = {k: round(v / len(images), 3) for k, v in totals.items()}
    a.out.write_text(json.dumps(out, indent=2))
    for k, v in sorted(out.items(), key=lambda kv: kv[1]):
        print(f"{k:34s} {v:6.2f}s")


if __name__ == "__main__":
    main()
