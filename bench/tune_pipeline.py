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

from bench.metrics import score
from ocr_engine.pipeline import PipelineConfig, fuse

_ctx = {}
MIN_CONFS = (0.0, 0.1, 0.2, 0.3, 0.4)


def load(dirs: list[Path]):
    """Per dataset: list of (sample, readings). Datasets are weighted equally when scoring."""
    groups = []
    for d in dirs:
        cache = json.loads((d / "readings.json").read_text())
        items = []
        for l in (d / "manifest.jsonl").read_text().splitlines():
            if l.strip():
                s = json.loads(l)
                items.append((s, cache[s["image"]]))
        groups.append(items)
    return groups


def _score(cfg_dict) -> float:
    cfg = PipelineConfig(**cfg_dict)
    predictor = None
    if cfg.use_lm:
        from ocr_engine.lm import WordPredictor
        predictor = WordPredictor(**cfg.lm_params)
    groups = _ctx["items"]
    # macro average: synthetic and real data count equally regardless of their sizes
    return sum(sum(score(s, fuse(r, cfg, predictor))["cer"] for s, r in g) / len(g) for g in groups) / len(groups)


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
    sources = sorted(set.intersection(*[{s for _, r in g for s in r} for g in items]))

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

        ranked = sorted(sources, key=single.get)[:7]
        configs = []
        for k in range(2, a.max_sources + 1):
            for subset in itertools.combinations(ranked, k):
                for scale in (0.0, 2.0):  # weight = (1 - CER)^(1 + 4 * scale): sharper with higher scale
                    w = {s: round((1 - single[s]) ** (1 + 4 * scale), 3) for s in subset}
                    for cv in (False, True):
                        configs.append({"sources": w, "use_lm": False, "min_conf": min_conf, "char_vote": cv})
        scores = pool.map(_score, configs)
        best_vote = min(zip(scores, configs), key=lambda x: x[0])
        best_single = min(single.values())
        print(f"best voting config: {best_vote[1]['sources']} CER {best_vote[0]:.4f} (best single {best_single:.4f})")
        base = best_vote[1] if best_vote[0] < best_single else {"sources": {top: 1.0}, "min_conf": min_conf}
        print("top voting configs:")
        for sc, cfg in sorted(zip(scores, configs), key=lambda x: x[0])[:5]:
            print(f"  {sc:.4f} char_vote={cfg['char_vote']} {cfg['sources']}")

        grid = [
            {**base, "use_lm": True, "lm_params": {"lam": lam, "mu": 3.0, "margin": mg, "keep_unknown": ku}}
            for lam in (3.0, 6.0, 10.0) for mg in (1.0, 3.0) for ku in (0.5, 0.7, 0.85)
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
