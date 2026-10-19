"""Tune the speed cascade on cached dev readings.

The cheap readers run first; when they agree closely (character similarity of their texts >= tau)
their vote is final, otherwise the expensive readers are added. For each tau we report dev macro
CER and the average reading time per page (using per-source timings measured on this CPU).

Usage:
    python -m bench.tune_cascade --data data/dev_en_v2 data/real_sroie_dev data/real_cord_dev \\
        --timings results/source_timings.json
"""

import argparse
import json
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from bench.metrics import score
from bench.tune_pipeline import load
from ocr_engine.pipeline import CONFIG_PATH, PipelineConfig, fuse


def text_of(lines) -> str:
    return " ".join(w for line in lines for w, _ in line)


def agreement(readings: dict, cheap: list[str]) -> float:
    texts = [text_of(readings.get(s, [])) for s in cheap]
    if not all(texts):
        return 0.0
    return min(Levenshtein.normalized_similarity(texts[0], t) for t in texts[1:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", type=Path, required=True)
    p.add_argument("--timings", type=Path, required=True)
    p.add_argument("--cheap", nargs="+", required=True, help="sources that always run")
    a = p.parse_args()
    groups = load(a.data)
    timing = json.loads(a.timings.read_text())
    saved = json.loads(CONFIG_PATH.read_text())
    full = PipelineConfig(**{**saved, "use_lm": False, "noisy": None})
    cheap_cfg = PipelineConfig(**{**saved, "use_lm": False, "noisy": None,
                                  "sources": {s: full.sources.get(s, 1.0) for s in a.cheap}})
    cost_cheap = sum(timing[s] for s in a.cheap)
    cost_full = cost_cheap + sum(timing[s] for s in full.sources if s not in a.cheap)
    print(f"cost: cheap {cost_cheap:.2f}s, full {cost_full:.2f}s per page")
    rows = []
    for tau in (1.01, 0.99, 0.98, 0.97, 0.95, 0.9, 0.85, 0.8, 0.0):
        cers, costs = [], []
        for g in groups:
            c, t = [], []
            for s, r in g:
                use_cheap = agreement(r, a.cheap) >= tau
                c.append(score(s, fuse(r, cheap_cfg if use_cheap else full))["cer"])
                t.append(cost_cheap if use_cheap else cost_full)
            cers.append(sum(c) / len(c))
            costs.append(sum(t) / len(t))
        rows.append((tau, sum(cers) / len(cers), sum(costs) / len(costs)))
        print(f"tau {tau:4.2f}: dev macro CER {rows[-1][1]:.4f}  avg time {rows[-1][2]:.2f}s/page")


if __name__ == "__main__":
    main()
