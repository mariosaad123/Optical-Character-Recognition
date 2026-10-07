"""Tune the image-quality router on dev data.

Images whose measured noise (Immerkaer sigma) exceeds a threshold are read with a separate
"noisy" voting config; all others keep the main config. Searches the threshold and the noisy
config (source subsets + weights) on the dev splits, synthetic and real weighted equally.

Usage:
    python -m bench.tune_quality_router --data data/dev_en_v2 data/real_sroie_dev data/real_cord_dev
"""

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from bench.metrics import score
from bench.tune_pipeline import load
from ocr_engine.pipeline import CONFIG_PATH, PipelineConfig, fuse
from ocr_engine.preprocess import noise_level, to_gray

_ctx = {}


def _init(groups, sigmas, main):
    _ctx.update(groups=groups, sigmas=sigmas, main=main)


def _macro(threshold: float, noisy: dict | None) -> float:
    main = PipelineConfig(**{**_ctx["main"], "use_lm": False})
    alt = PipelineConfig(**{**noisy, "use_lm": False}) if noisy else main
    total = 0.0
    for g, sig in zip(_ctx["groups"], _ctx["sigmas"]):
        total += sum(score(s, fuse(r, alt if sg > threshold else main))["cer"] for (s, r), sg in zip(g, sig)) / len(g)
    return total / len(_ctx["groups"])


def _job(args):
    return _macro(*args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    groups = load(a.data)
    sigmas = [[noise_level(to_gray(Image.open(d / s["image"]))) for s, _ in g] for d, g in zip(a.data, groups)]
    saved = json.loads(CONFIG_PATH.read_text())
    main_cfg = {k: v for k, v in saved.items() if k not in ("noisy", "noise_threshold")}
    sources = sorted(set.intersection(*[{s for _, r in g for s in r} for g in groups]))

    configs = [None]
    for k in (1, 2, 3):
        for subset in itertools.combinations(sources, k):
            for w in ({s: 1.0 for s in subset},):
                configs.append({"sources": w, "char_vote": main_cfg.get("char_vote", False)})
    jobs = [(t, c) for t in (2.5, 3.0, 3.5, 4.0) for c in configs]
    with Pool(a.workers, initializer=_init, initargs=(groups, sigmas, main_cfg)) as pool:
        base = pool.apply(_macro, (99.0, None))
        scores = pool.map(_job, jobs, chunksize=4)
    best = min(zip(scores, jobs), key=lambda x: x[0])
    print(f"main config only: dev macro CER {base:.4f}")
    for sc, (t, c) in sorted(zip(scores, jobs), key=lambda x: x[0])[:5]:
        print(f"  {sc:.4f} threshold={t} noisy={c['sources'] if c else None}")
    if best[0] < base and best[1][1]:
        t, c = best[1]
        saved["noise_threshold"], saved["noisy"] = t, c
        CONFIG_PATH.write_text(json.dumps(saved, indent=2))
        print(f"saved router: noise > {t} -> {c['sources']}")
    else:
        print("router does not help; config unchanged")


if __name__ == "__main__":
    main()
