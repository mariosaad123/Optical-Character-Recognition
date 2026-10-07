"""Real training lines from the CORD-v2 TRAIN split (receipt photos, CC BY 4.0).

Each annotated row (`row_id`) becomes one line crop with its exact transcript, written as
<name>.png + <name>.gt.txt (+ .box/.lstmf for Tesseract) and listed in lines.jsonl for
neural recognizers. Dev/test receipts (validation/test splits) are never used.

Usage:
    python -m training.make_cord_lines --out data/train_cord_lines
"""

import argparse
import io
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image

from bench.import_cord import fetch, words_of


def _lines(gt: dict) -> list[tuple[str, tuple[int, int, int, int]]]:
    rows: dict[int, list[dict]] = {}
    for w in words_of(gt):
        rows.setdefault(w["row"], []).append(w)
    out = []
    for ws in rows.values():
        ws.sort(key=lambda w: w["box"][0])
        x0 = min(w["box"][0] for w in ws)
        y0 = min(w["box"][1] for w in ws)
        x1 = max(w["box"][2] for w in ws)
        y1 = max(w["box"][3] for w in ws)
        out.append((" ".join(w["text"] for w in ws), (x0, y0, x1, y1)))
    return out


def _tess(base: Path) -> str | None:
    img = Image.open(f"{base}.png")
    w, h = img.size
    text = Path(f"{base}.gt.txt").read_text().strip()
    Path(f"{base}.box").write_text(f"WordStr 0 0 {w} {h} 0 #{text}\n\t 0 0 {w} {h} 0\n")
    r = subprocess.run(["tesseract", f"{base}.png", str(base), "--psm", "13", "lstm.train"],
                       capture_output=True, text=True, check=False)
    lstmf = Path(f"{base}.lstmf")
    return str(lstmf) if r.returncode == 0 and lstmf.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/train_cord_lines"))
    p.add_argument("--cache", type=Path, default=Path("data/external/cord"))
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    records, bases = [], []
    for parquet in fetch("train", a.cache, parts=4):
        for _, row in pd.read_parquet(parquet).iterrows():
            gt = json.loads(row["ground_truth"])
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            for k, (text, (x0, y0, x1, y1)) in enumerate(_lines(gt)):
                h = y1 - y0
                if h < 8 or x1 - x0 < 8 or not text.strip():
                    continue
                pad = max(2, h // 6)
                crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))
                name = f"cord_{gt['meta']['image_id']:04d}_{k:02d}"
                crop.save(a.out / f"{name}.png")
                (a.out / f"{name}.gt.txt").write_text(text + "\n")
                records.append({"image": f"{name}.png", "text": text})
                bases.append(a.out / name)
    (a.out / "lines.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with Pool(a.workers) as pool:
        files = [f for f in pool.imap(_tess, bases, chunksize=32) if f]
    (a.out / "list.train").write_text("\n".join(files) + "\n")
    print(f"{len(records)} real lines ({len(files)} Tesseract samples) in {a.out}")


if __name__ == "__main__":
    main()
