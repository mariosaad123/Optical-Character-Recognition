"""Tune the pipeline on the DEV split using cached readings, then save the config.

1. score every single source
2. search source subsets + weights for the voting step
3. grid-search the word-prediction parameters on top of the best voting config

Usage:
    python -m bench.tune_pipeline --data data/dev_en_v2 data/dev_en --out ocr_engine/pipeline_config.json
"""

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

from bench.metrics import cer
from ocr_engine.pipeline import PipelineConfig, fuse

_ctx = {}
MIN_CONFS = (0.0, 0.1, 0.2, 0.3, 0.4)


def load(dirs: list[Path]):
    items = []
    for d in dirs:
        cache = json.loads((d / "readings.json").read_text())
        for l in (d / "manifest.jsonl").read_text().splitlines():
            if l.strip():
                s = json.loads(l)
                items.append((s["text"], cache[s["image"]]))
    return items


def _score(cfg_dict) -> float:
    cfg = PipelineConfig(**cfg_dict)
    predictor = None
    if cfg.use_lm:
        from ocr_engine.lm import WordPredictor
        predictor = WordPredictor(**cfg.lm_params)
    items = _ctx["items"]
    return sum(cer(ref, fuse(r, cfg, predictor)) for ref, r in items) / len(items)


def _init(items):
    _ctx["items"] = items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("ocr_engine/pipeline_config.json"))
    p.add_argument("--max-sources", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    items = load(a.data)
    sources = sorted({s for _, r in items for s in r})

    with Pool(a.workers, initializer=_init, initargs=(items,)) as pool:
        single = dict(zip(sources, pool.map(_score, [{"sources": {s: 1.0}, "use_lm": False} for s in sources])))
        # confidence filter, chosen on the best single source
        top = min(single, key=single.get)
        mc_scores = dict(zip(MIN_CONFS, pool.map(_score, [{"sources": {top: 1.0}, "use_lm": False, "min_conf": m} for m in MIN_CONFS])))
        min_conf = min(mc_scores, key=mc_scores.get)
        print(f"min_conf on {top}: {mc_scores} -> {min_conf}")
        print("single sources (dev CER):")
        for s, c in sorted(single.items(), key=lambda kv: kv[1]):
            print(f"  {s:32s} {c:.4f}")

        ranked = sorted(sources, key=single.get)[:6]
        configs = []
        for k in range(2, a.max_sources + 1):
            for subset in itertools.combinations(ranked, k):
                for scale in (0.0, 1.0, 2.0):  # weight = (1 - CER)^scale*... sharper with higher scale
                    w = {s: round((1 - single[s]) ** (1 + 4 * scale), 3) for s in subset}
                    for mc in {0.0, min_conf}:
                        configs.append({"sources": w, "use_lm": False, "min_conf": mc})
        scores = pool.map(_score, configs)
        best_vote = min(zip(scores, configs), key=lambda x: x[0])
        best_single = min(single.values())
        print(f"best voting config: {best_vote[1]['sources']} CER {best_vote[0]:.4f} (best single {best_single:.4f})")
        base = best_vote[1] if best_vote[0] < best_single else {"sources": {top: 1.0}, "min_conf": min_conf}

        grid = [
            {**base, "use_lm": True, "lm_params": {"lam": lam, "mu": mu, "margin": mg}}
            for lam in (2.0, 3.0, 4.0, 6.0) for mu in (1.0, 3.0) for mg in (1.0, 3.0, 5.0)
        ]
        lm_scores = pool.map(_score, grid)
    no_lm = min(best_vote[0], best_single)
    best_lm = min(zip(lm_scores, grid), key=lambda x: x[0])
    print(f"best word-prediction config: {best_lm[1]['lm_params']} CER {best_lm[0]:.4f} (without: {no_lm:.4f})")
    final = best_lm[1] if best_lm[0] < no_lm else {**base, "use_lm": False}
    a.out.write_text(json.dumps(final, indent=2))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
