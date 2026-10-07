"""Build REAL-WORLD benchmarks from CORD-v2 receipt photos (CC BY 4.0).

CORD annotates only part of the text on each photo (store headers are blurred for privacy, some
lines are left out). To score page-level OCR fairly, everything outside the annotated word boxes
(padded) is painted with the local paper colour, so the reference covers all readable text.
Reference text = annotated words grouped by `row_id`, rows top-to-bottom, words left-to-right.

Usage:
    python -m bench.import_cord            # validation -> data/real_cord_dev, test -> data/real_cord_test
"""

import argparse
import io
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

URL = "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/refs%2Fconvert%2Fparquet/default/{split}/{part:04d}.parquet"
SPLITS = {"validation": "dev", "test": "test"}


def fetch(split: str, out_dir: Path, parts: int = 1) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for part in range(parts):
        p = out_dir / f"{split}-{part:04d}.parquet"
        if not p.exists():
            urllib.request.urlretrieve(URL.format(split=split, part=part), p)
        paths.append(p)
    return paths


def words_of(gt: dict) -> list[dict]:
    out = []
    for line in gt.get("valid_line", []):
        for w in line["words"]:
            q = w["quad"]
            xs, ys = [q["x1"], q["x2"], q["x3"], q["x4"]], [q["y1"], q["y2"], q["y3"], q["y4"]]
            if w["text"].strip():
                out.append({"text": w["text"].strip(), "row": w["row_id"], "box": (min(xs), min(ys), max(xs), max(ys))})
    return out


def reference_text(words: list[dict]) -> str:
    rows: dict[int, list[dict]] = {}
    for w in words:
        rows.setdefault(w["row"], []).append(w)
    ordered = sorted(rows.values(), key=lambda ws: np.mean([(w["box"][1] + w["box"][3]) / 2 for w in ws]))
    return "\n".join(" ".join(w["text"] for w in sorted(ws, key=lambda w: w["box"][0])) for ws in ordered)


def mask_unannotated(img: Image.Image, words: list[dict]) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).copy()
    h, w = arr.shape[:2]
    keep = np.zeros((h, w), np.uint8)
    for wd in words:
        x0, y0, x1, y1 = wd["box"]
        pad = max(3, int(0.35 * (y1 - y0)))
        keep[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)] = 1
    inside = arr[keep.astype(bool)]
    lum = inside.mean(axis=1)
    # paper = the brighter half of the pixels inside the word boxes (ink is the darker minority)
    paper = np.median(inside[lum >= np.median(lum)], axis=0) if len(inside) else np.array([235, 235, 235])
    # feather the mask so box borders don't become fake strokes
    alpha = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 2.0)[..., None]
    out = arr.astype(np.float32) * alpha + paper.astype(np.float32) * (1 - alpha)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def build(split: str, out: Path, cache: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for p in fetch(split, cache):
        df = pd.read_parquet(p)
        for i, row in df.iterrows():
            gt = json.loads(row["ground_truth"])
            words = words_of(gt)
            if not words:
                continue
            name = f"cord_{split}_{gt['meta']['image_id']:04d}.png"
            img = Image.open(io.BytesIO(row["image"]["bytes"]))
            mask_unannotated(img, words).save(out / name)
            manifest.append({"image": name, "level": "real", "text": reference_text(words)})
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    return len(manifest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("data/external/cord"))
    a = p.parse_args()
    for split, name in SPLITS.items():
        n = build(split, Path(f"data/real_cord_{name}"), a.cache)
        print(f"{split}: {n} receipts -> data/real_cord_{name}")


if __name__ == "__main__":
    main()
