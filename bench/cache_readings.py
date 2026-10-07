"""Run every (engine, variant) source once per image and cache the word-level readings.

Tuning and ablations then run on the cache in seconds instead of hours.

Usage:
    python -m bench.cache_readings --data data/dev_en_v2 --sources tesseract/raw rapidocr/raw ...
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # one thread per Tesseract call; we parallelize over images
os.environ.setdefault("OCR_ORT_THREADS", "1")  # same for ONNX Runtime engines

from PIL import Image  # noqa: E402

from ocr_engine.pipeline import read_source  # noqa: E402
from ocr_engine.preprocess import auto_fix  # noqa: E402

_engines: dict = {}


def _run(args):
    data, s, sources = args
    img = Image.open(data / s["image"])
    if os.environ.get("OCR_AUTOFIX", "1") == "1":
        img, _ = auto_fix(img)  # same as PipelineEngine.recognize
    variants: dict = {}
    return s["image"], {src: read_source(src, img, _engines, variants) for src in sources}


def _save(path: Path, cache: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    cache_path = a.data / "readings.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    samples = [json.loads(l) for l in (a.data / "manifest.jsonl").read_text().splitlines() if l.strip()]
    todo = [(a.data, s, [src for src in a.sources if src not in cache.get(s["image"], {})]) for s in samples]
    todo = [t for t in todo if t[2]]
    with Pool(a.workers) as pool:
        for i, (name, res) in enumerate(pool.imap_unordered(_run, todo, chunksize=2), 1):
            cache.setdefault(name, {}).update(res)
            if i % 25 == 0:  # save progress so an interrupted run can resume
                _save(cache_path, cache)
                print(f"{i}/{len(todo)}", flush=True)
    _save(cache_path, cache)
    print(f"cached {len(a.sources)} sources for {len(samples)} images -> {cache_path}")


if __name__ == "__main__":
    main()
