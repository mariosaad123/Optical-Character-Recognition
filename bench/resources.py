"""Fonts and text sources, with strict train / dev / test separation.

Test fonts and test books are never used for training or tuning, so test numbers measure
generalization to unseen typefaces and unseen text.
"""

import random
import re
import subprocess
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# Font families held out for evaluation only.
TEST_FAMILIES = {
    "literata", "merriweather", "sourcesans3", "ibmplexsans", "ibmplexmono",
    "cabin", "tinos", "robotoslab", "oswald", "spectral",
}
DEV_FAMILIES = {"lora", "publicsans", "firamono", "asap"}
HANDWRITING = {"caveat", "patrickhand", "kalam", "indieflower", "architectsdaughter"}

TEST_BOOKS = ["carroll-alice.txt", "austen-persuasion.txt", "chesterton-thursday.txt"]
DEV_BOOKS = ["bryant-stories.txt", "burgess-busterbrown.txt"]


@lru_cache
def system_fonts() -> list[str]:
    out = subprocess.run(["fc-list", "--format", "%{file}\n"], capture_output=True, text=True).stdout
    skip = ("emoji", "ipa", "unifont", "wqy", "opens", "loma", "japanese")
    return sorted({f for f in out.splitlines() if f.lower().endswith(".ttf") and not any(s in f.lower() for s in skip)})


@lru_cache
def fonts_for(split: str) -> list[str]:
    """Font files for a split. System fonts go to train only."""
    root = DATA / "fonts"
    if not root.exists():
        raise RuntimeError("Fonts missing: run `python scripts/fetch_resources.py` first.")
    fams = {d.name: sorted(str(p) for p in d.glob("*.ttf")) for d in root.iterdir() if d.is_dir()}
    if split == "test":
        names = TEST_FAMILIES
    elif split == "dev":
        names = DEV_FAMILIES
    else:
        names = set(fams) - TEST_FAMILIES - DEV_FAMILIES
    files = [f for n in sorted(names) for f in fams.get(n, [])]
    if split == "train":
        files += system_fonts()
    return files


def is_handwriting(font_path: str) -> bool:
    return Path(font_path).parent.name in HANDWRITING


_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "—": "--", "–": "-"})


@lru_cache
def book_words(split: str) -> list[list[str]]:
    """Each book as a list of words (whitespace-normalized, ASCII punctuation)."""
    root = DATA / "corpus" / "gutenberg"
    all_books = sorted(p.name for p in root.glob("*.txt"))
    if split == "test":
        names = TEST_BOOKS
    elif split == "dev":
        names = DEV_BOOKS
    else:
        names = [b for b in all_books if b not in TEST_BOOKS + DEV_BOOKS]
    books = []
    for n in names:
        text = (root / n).read_text(encoding="latin-1").translate(_QUOTES)
        text = re.sub(r"[^\x20-\x7e\s]", "", text)
        books.append(text.split())
    return books


def prose_lines(rng: random.Random, split: str, n_lines: int) -> list[str]:
    """A coherent paragraph from a book, wrapped to a random line width."""
    words = rng.choice(book_words(split))
    width = rng.randint(30, 85)
    start = rng.randint(0, len(words) - 400)
    lines, cur = [], ""
    for w in words[start:]:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            if len(lines) == n_lines:
                break
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    return lines
