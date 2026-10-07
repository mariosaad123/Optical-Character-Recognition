# Project notes for Claude

- Read `docs/PROJECT_JOURNAL.md` first: it is the project's memory (phases, decisions, numbers,
  mistakes and fixes). **Append to it after every experiment or decision**, with numbers. The owner
  will build a report for a company from it.
- Talk to the owner in Egyptian Arabic. Keep code, comments, commits and docs in English.
- Work phase by phase; finish a phase completely before moving on (owner's decision).
- Zero cost: free models, free data, free compute only.
- Never keep a change without measuring it: tune on dev splits, report on held-out test splits.
- `data/` is not committed (regenerate with `scripts/fetch_resources.py`, `bench/generate.py`,
  `bench/import_sroie.py`). Our trained models live in `models/`.
- Run `python -m pytest tests` before committing.
