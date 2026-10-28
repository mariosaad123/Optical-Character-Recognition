"""Build the handwriting benchmarks (phase 1).

GNHK (GoodNotes Handwriting Kollection, CC BY 4.0): English handwriting photographed with phones by
people in many countries. Each word has a polygon, a line index and a type (H handwritten, P printed);
special tokens mark things that are not plain text (%math%, %SC% scribbles, %NA% illegible, ...).

  hw_gnhk_pages_{dev,test}  whole pages; everything that is not an annotated plain-text word (including
                            special-token regions and unannotated areas) is painted with the paper colour
  hw_gnhk_lines_{dev,test}  one crop per text line that contains only plain-text words

IAM (research use only, never committed, evaluation only):
  hw_iam_{dev,test}         line images; IAM's tokenized transcripts ("it 's a good start .") are
                            de-tokenized ("it's a good start.") so natural spacing is not punished

Splits: GNHK test pages -> test; GNHK train pages -> a fixed dev subset; every remaining GNHK train
page is reserved for training (listed in data/hw_gnhk_train_pages.json) and never evaluated on.

Usage:
    python -m bench.import_handwriting
"""

import argparse
import io
import json
import random
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

GNHK = "https://huggingface.co/datasets/radhakrishnadeshpande/gnhk-pagelevel/resolve/refs%2Fconvert%2Fparquet/default/{split}/{part:04d}.parquet"
IAM = "https://huggingface.co/datasets/Teklia/IAM-line/resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
SPECIAL = re.compile(r"^%.*%$")
MAX_SIDE = 2048  # phone photos are ~4000 px wide; every reader downsizes anyway


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def detokenize(text: str) -> str:
    """IAM transcripts separate punctuation and clitics with spaces; join them back."""
    text = re.sub(r" ([.,;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[]) ", r"\1", text)
    text = re.sub(r" (['’](s|t|re|ve|ll|d|m|S|T|RE|VE|LL|D|M))\b", r"\1", text)
    text = re.sub(r'" (.*?) "', r'"\1"', text)
    return " ".join(text.split())


def _box(poly: dict) -> tuple[int, int, int, int]:
    xs = [poly[f"x{i}"] for i in range(4)]
    ys = [poly[f"y{i}"] for i in range(4)]
    return min(xs), min(ys), max(xs), max(ys)


def _paper_fill(arr: np.ndarray, keep: np.ndarray) -> np.ndarray:
    inside = arr[keep.astype(bool)]
    if not len(inside):
        return arr
    lum = inside.mean(axis=1)
    paper = np.median(inside[lum >= np.median(lum)], axis=0)
    alpha = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 3.0)[..., None]
    return np.clip(arr.astype(np.float32) * alpha + paper * (1 - alpha), 0, 255).astype(np.uint8)


def gnhk_page(img: Image.Image, words: list[dict]) -> tuple[Image.Image, str, list[tuple[Image.Image, str]]]:
    """Masked page, its reference text, and clean line crops."""
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    keep = np.zeros((h, w), np.uint8)
    plain = [wd for wd in words if wd["text"].strip() and not SPECIAL.match(wd["text"].strip())]
    for wd in plain:
        pts = np.array([[wd["polygon"][f"x{i}"], wd["polygon"][f"y{i}"]] for i in range(4)], np.int32)
        x0, y0, x1, y1 = _box(wd["polygon"])
        pad = max(4, int(0.15 * (y1 - y0)))
        cv2.fillPoly(keep, [pts], 1)
        keep[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)] = 1
    masked = _paper_fill(arr, keep)

    lines: dict[int, list[dict]] = {}
    for wd in words:
        lines.setdefault(wd["line_idx"], []).append(wd)
    ordered = sorted(lines.items(), key=lambda kv: np.mean([(_box(w["polygon"])[1] + _box(w["polygon"])[3]) / 2 for w in kv[1]]))
    ref_lines, crops = [], []
    for _, ws in ordered:
        ws = sorted(ws, key=lambda wd: _box(wd["polygon"])[0])
        clean = [wd for wd in ws if wd in plain]
        if clean:
            ref_lines.append(" ".join(wd["text"].strip() for wd in clean))
        if clean and len(clean) == len(ws):  # line crops only for lines without special tokens
            boxes = [_box(wd["polygon"]) for wd in ws]
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            pad = max(6, (y1 - y0) // 8)
            # only this line's own words stay visible: neighbouring lines that reach into the crop are
            # painted over, so a line reader is never asked to read text that is not in the reference
            own = np.zeros((h, w), np.uint8)
            for wd in ws:
                bx0, by0, bx1, by1 = _box(wd["polygon"])
                p2 = max(3, (by1 - by0) // 10)
                own[max(0, by0 - p2):min(h, by1 + p2), max(0, bx0 - p2):min(w, bx1 + p2)] = 1
            sl = (slice(max(0, y0 - pad), min(h, y1 + pad)), slice(max(0, x0 - pad), min(w, x1 + pad)))
            region = _paper_fill(arr[sl], own[sl])
            if region.shape[0] >= 16 and region.shape[1] >= 16:  # skip degenerate boxes
                crops.append((Image.fromarray(region), ref_lines[-1]))
    page = Image.fromarray(masked)
    page.thumbnail((MAX_SIDE, MAX_SIDE))
    return page, "\n".join(ref_lines), crops


def _write(out: Path, items: list[tuple[str, Image.Image, str]], level: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, img, text in items:
        img.save(out / name)
        manifest.append({"image": name, "level": level, "text": text})
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    print(f"{len(manifest):5d} samples -> {out}")


def build_gnhk(cache: Path, dev_pages: int, seed: int) -> None:
    rows = {"test": [], "train": []}
    for split, parts in (("test", 1), ("train", 2)):
        for part in range(parts):
            df = pd.read_parquet(fetch(GNHK.format(split=split, part=part), cache / f"gnhk_{split}_{part}.parquet"))
            rows[split] += list(df.itertuples(index=False))
    train = sorted(rows["train"], key=lambda r: r.filename)
    random.Random(seed).shuffle(train)
    dev, reserved = train[:dev_pages], train[dev_pages:]
    Path("data").mkdir(exist_ok=True)
    Path("data/hw_gnhk_train_pages.json").write_text(json.dumps(sorted(r.filename for r in reserved)))
    for name, split_rows in (("dev", dev), ("test", rows["test"])):
        pages, lines = [], []
        for r in split_rows:
            img = Image.open(io.BytesIO(r.image["bytes"]))
            page, text, crops = gnhk_page(img, json.loads(r.annotations))
            stem = Path(r.filename).stem
            if text.strip():
                pages.append((f"{stem}.jpg", page, text))
            lines += [(f"{stem}_{i:02d}.png", c, t) for i, (c, t) in enumerate(crops)]
        _write(Path(f"data/hw_gnhk_pages_{name}"), [(n, i.convert("RGB"), t) for n, i, t in pages], "gnhk_page")
        _write(Path(f"data/hw_gnhk_lines_{name}"), lines, "gnhk_line")


def build_iam(cache: Path, n_dev: int, n_test: int, seed: int) -> None:
    for split, name, n in (("validation", "dev", n_dev), ("test", "test", n_test)):
        df = pd.read_parquet(fetch(IAM.format(split=split), cache / f"iam_{split}.parquet"))
        idx = sorted(random.Random(seed).sample(range(len(df)), min(n, len(df))))
        items = [(f"iam_{split}_{i:05d}.png", Image.open(io.BytesIO(df.image[i]["bytes"])).convert("RGB"),
                  detokenize(df.text[i])) for i in idx]
        _write(Path(f"data/hw_iam_{name}"), items, "iam_line")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("data/external/handwriting"))
    p.add_argument("--dev-pages", type=int, default=40)
    p.add_argument("--iam-dev", type=int, default=300)
    p.add_argument("--iam-test", type=int, default=600)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    build_gnhk(a.cache, a.dev_pages, a.seed)
    build_iam(a.cache, a.iam_dev, a.iam_test, a.seed)


if __name__ == "__main__":
    main()
