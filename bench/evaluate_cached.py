"""Evaluate single sources and the full pipeline from cached readings (bench/cache_readings.py).

The pipeline output is produced by the same `fuse()` used at runtime, so the numbers match a live run.

Usage:
    python -m bench.evaluate_cached --data data/bench_en_v2 --out results/v2
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bench.evaluate import to_markdown
from bench.metrics import bow_f1, cer, wer
from ocr_engine.pipeline import PipelineConfig, fuse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sources", nargs="*", help="single sources to report (default: all cached)")
    a = p.parse_args()

    cache = json.loads((a.data / "readings.json").read_text())
    samples = [json.loads(l) for l in (a.data / "manifest.jsonl").read_text().splitlines() if l.strip()]
    cfg = PipelineConfig.load()
    predictor = None
    if cfg.use_lm:
        from ocr_engine.lm import WordPredictor
        predictor = WordPredictor(**cfg.lm_params)
    no_lm = PipelineConfig(sources=cfg.sources, use_lm=False)

    systems = {s: PipelineConfig(sources={s: 1.0}, use_lm=False) for s in (a.sources or sorted(next(iter(cache.values()))))}
    systems["pipeline (voting only)"] = no_lm
    systems["PIPELINE (voting + word prediction)"] = cfg

    summary, preds = {}, defaultdict(list)
    for name, c in systems.items():
        agg = defaultdict(lambda: defaultdict(float))
        for s in samples:
            hyp = fuse(cache[s["image"]], c, predictor if c.use_lm else None)
            m = {"cer": cer(s["text"], hyp), "wer": wer(s["text"], hyp), "bow": bow_f1(s["text"], hyp)}
            for key in (s["level"], "all"):
                for k, v in m.items():
                    agg[key][k] += v
                agg[key]["n"] += 1
            preds[name].append({"image": s["image"], "ref": s["text"], "hyp": hyp, "cer": round(m["cer"], 4)})
        summary[name] = {lvl: {k: v[k] / v["n"] for k in ("cer", "wer", "bow")} | {"n": v["n"]} for lvl, v in agg.items()}

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps({"config": cfg.__dict__, "results": summary}, indent=2))
    (a.out / "predictions.json").write_text(json.dumps(preds, indent=1))
    table = to_markdown(summary)
    (a.out / "RESULTS.md").write_text(f"# {a.data.name}\n\n{table}\n")
    print(table)


if __name__ == "__main__":
    main()
