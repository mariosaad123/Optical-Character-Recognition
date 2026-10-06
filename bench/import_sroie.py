"""Build a REAL-WORLD benchmark from ICDAR-2019 SROIE scanned receipts.

The data is used locally for evaluation only (research license) and is never committed.
Box transcripts are grouped into text lines (top-to-bottom, left-to-right) to form the reference.

Usage:
    python -m bench.import_sroie --n 150 --dev 50
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

REPO = "https://github.com/zzzDavid/ICDAR-2019-SROIE.git"


def fetch(checkout: Path, n: int) -> None:
    if not checkout.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(checkout)], check=True)
    names = sorted(subprocess.run(["git", "-C", str(checkout), "ls-tree", "-r", "--name-only", "HEAD", "data/img"],
                                  capture_output=True, text=True, check=True).stdout.split())[:n]
    paths = names + [p.replace("data/img/", "data/box/").replace(".jpg", ".csv") for p in names]
    subprocess.run(["git", "-C", str(checkout), "sparse-checkout", "set", "--no-cone", *["/" + p for p in paths]], check=True)


def boxes_to_text(csv_text: str) -> str:
    boxes = []
    for line in csv_text.splitlines():
        parts = line.strip().split(",", 8)
        if len(parts) < 9 or not parts[8].strip():
            continue
        xs, ys = [int(v) for v in parts[0:8:2]], [int(v) for v in parts[1:8:2]]
        boxes.append((min(ys), max(ys), min(xs), parts[8].strip()))
    boxes.sort(key=lambda b: (b[0] + b[1]) / 2)
    lines: list[list] = []
    for b in boxes:
        c = (b[0] + b[1]) / 2
        if lines and min(x[0] for x in lines[-1]) <= c <= max(x[1] for x in lines[-1]):
            lines[-1].append(b)
        else:
            lines.append([b])
    return "\n".join(" ".join(x[3] for x in sorted(l, key=lambda x: x[2])) for l in lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkout", type=Path, default=Path("data/external/sroie"))
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--dev", type=int, default=50)
    a = p.parse_args()
    fetch(a.checkout, a.n)
    imgs = sorted((a.checkout / "data" / "img").glob("*.jpg"))[: a.n]
    for split, chunk in (("dev", imgs[: a.dev]), ("test", imgs[a.dev:])):
        out = Path(f"data/real_sroie_{split}")
        out.mkdir(parents=True, exist_ok=True)
        manifest = []
        for img in chunk:
            box = a.checkout / "data" / "box" / (img.stem + ".csv")
            if not box.exists():
                continue
            shutil.copy(img, out / img.name)
            text = boxes_to_text(box.read_text(encoding="utf-8", errors="ignore"))
            # SROIE transcripts are upper-cased regardless of the printed case -> compare case-insensitively
            manifest.append({"image": img.name, "level": "real", "text": text, "case_insensitive": True})
        (out / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
        print(f"{split}: {len(manifest)} receipts -> {out}")


if __name__ == "__main__":
    main()
