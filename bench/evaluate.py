"""Run OCR engines on a benchmark and report accuracy per degradation level.

Usage:
    python -m bench.evaluate --data data/bench_en --engines tesseract rapidocr
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

from bench.metrics import bow_f1, cer, wer
from ocr_engine import ENGINES, get_engine


def evaluate(data_dir: Path, engine_names: list[str], limit: int | None):
    samples = [json.loads(l) for l in (data_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    if limit:
        samples = samples[:limit]
    summary, predictions = {}, {}
    for name in engine_names:
        engine = get_engine(name)
        per_level = defaultdict(lambda: {"cer": 0.0, "wer": 0.0, "bow": 0.0, "n": 0})
        preds, t0 = [], time.perf_counter()
        for s in samples:
            hyp = engine.recognize(Image.open(data_dir / s["image"]))
            c, w, b = cer(s["text"], hyp), wer(s["text"], hyp), bow_f1(s["text"], hyp)
            for key in (s["level"], "all"):
                per_level[key]["cer"] += c
                per_level[key]["wer"] += w
                per_level[key]["bow"] += b
                per_level[key]["n"] += 1
            preds.append({"image": s["image"], "ref": s["text"], "hyp": hyp, "cer": round(c, 4)})
        elapsed = time.perf_counter() - t0
        summary[name] = {
            lvl: {k: v[k] / v["n"] for k in ("cer", "wer", "bow")} | {"n": v["n"]} for lvl, v in per_level.items()
        }
        summary[name]["sec_per_image"] = elapsed / max(1, len(samples))
        predictions[name] = preds
        print(f"[{name}] done in {elapsed:.1f}s")
    return summary, predictions


def to_markdown(summary: dict) -> str:
    levels = [l for l in ("clean", "medium", "hard", "real") if any(l in s for s in summary.values())] + ["all"]
    head = "| Engine | " + " | ".join(f"{l} acc (1-CER)" for l in levels) + " | all WER | all word F1 | sec/img |"
    rows = [head, "|" + "---|" * (len(levels) + 4)]
    for name, s in summary.items():
        accs = " | ".join(f"{max(0.0, 1 - s[l]['cer']) * 100:.2f}%" if l in s else "-" for l in levels)
        rows.append(f"| {name} | {accs} | {s['all']['wer'] * 100:.2f}% | {s['all']['bow'] * 100:.2f}% | {s.get('sec_per_image', 0):.2f} |")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/bench_en"))
    p.add_argument("--engines", nargs="+", default=ENGINES)
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args()

    summary, predictions = evaluate(a.data, a.engines, a.limit)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (a.out / "predictions.json").write_text(json.dumps(predictions, indent=1))
    table = to_markdown(summary)
    (a.out / "RESULTS.md").write_text(f"# Benchmark results: {a.data.name}\n\n{table}\n")
    print(table)


if __name__ == "__main__":
    main()
