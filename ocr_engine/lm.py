"""Word prediction: fix misread or missing characters using a lexicon and a language model.

For each suspicious word we generate candidates (dictionary words within edit distance 2,
plus readings proposed by other engines), and pick the one maximizing

    log P(candidate | previous word, next word)        # language model (context)
  - lambda * OCR_edit_cost(observed -> candidate)      # visual plausibility
  + mu * ensemble vote for the candidate               # what the engines saw

The edit cost knows typical OCR confusions (rn<->m, cl<->d, 1<->l, 0<->o, ...), so
"rnodern" -> "modern" is cheap while "modern" -> "madden" is expensive.

Free data only: word frequencies from `wordfreq`; bigrams from public-domain Gutenberg
books in the TRAIN split (test books are never used).
"""

import math
import pickle
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from rapidfuzz.distance import Levenshtein

CACHE = Path(__file__).resolve().parent.parent / "data" / "lm_cache.pkl"
DIRICHLET = 50.0  # bigram smoothing strength
EXTRA_GLYPH_COST = 1.4

# (seen, truth) pairs that OCR confuses; cost in [0, 1] instead of 1.
CONFUSIONS = {
    ("rn", "m"): 0.2, ("m", "rn"): 0.3, ("cl", "d"): 0.3, ("d", "cl"): 0.4, ("vv", "w"): 0.2, ("w", "vv"): 0.3,
    ("li", "h"): 0.4, ("h", "li"): 0.4, ("ri", "n"): 0.4, ("in", "m"): 0.4, ("ni", "m"): 0.4, ("iii", "m"): 0.4,
    ("1", "l"): 0.2, ("l", "1"): 0.3, ("1", "i"): 0.3, ("i", "1"): 0.4, ("I", "l"): 0.1, ("l", "I"): 0.1,
    ("0", "o"): 0.2, ("o", "0"): 0.3, ("5", "s"): 0.3, ("s", "5"): 0.4, ("8", "B"): 0.3, ("6", "b"): 0.4,
    ("e", "c"): 0.4, ("c", "e"): 0.4, ("a", "o"): 0.5, ("o", "a"): 0.5, ("u", "v"): 0.4, ("v", "u"): 0.4,
    ("n", "u"): 0.5, ("u", "n"): 0.5, ("h", "b"): 0.5, ("b", "h"): 0.5, ("i", "l"): 0.3, ("l", "i"): 0.3,
    ("t", "f"): 0.5, ("f", "t"): 0.5, ("i", "j"): 0.5, ("j", "i"): 0.5, ("y", "v"): 0.5, ("g", "q"): 0.5,
    ("!", "l"): 0.3, ("|", "l"): 0.2, ("|", "I"): 0.2, ("ii", "u"): 0.4, ("fi", "h"): 0.5, ("tl", "d"): 0.5,
}
_CONF_MAX = max(len(a) for a, _ in CONFUSIONS)

WORD_RE = re.compile(r"^([^A-Za-z0-9]*)(.*?)([^A-Za-z0-9]*)$")
LETTERS_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)?$")


@lru_cache(maxsize=500_000)
def ocr_edit_cost(seen: str, truth: str) -> float:
    """Weighted edit distance with cheap OCR-typical substitutions (multi-char aware)."""
    n, m = len(seen), len(truth)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = float(i)
    for j in range(1, m + 1):
        dp[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a, b = seen[i - 1], truth[j - 1]
            sub = 0.0 if a == b else (0.6 if a.lower() == b.lower() else CONFUSIONS.get((a, b), 1.0))
            # an extra glyph in the reading (dp[i-1][j]) is rarer in OCR than a missing or misread one
            best = min(dp[i - 1][j] + EXTRA_GLYPH_COST, dp[i][j - 1] + 1.0, dp[i - 1][j - 1] + sub)
            for la in range(1, _CONF_MAX + 1):
                for lb in range(1, _CONF_MAX + 1):
                    if (la > 1 or lb > 1) and la <= i and lb <= j:
                        c = CONFUSIONS.get((seen[i - la:i], truth[j - lb:j]))
                        if c is not None:
                            best = min(best, dp[i - la][j - lb] + c)
            dp[i][j] = best
    return dp[n][m]


class LanguageModel:
    def __init__(self, vocab_size: int = 80000):
        if CACHE.exists():
            with open(CACHE, "rb") as f:
                state = pickle.load(f)
            if state.get("vocab_size") == vocab_size and state.get("version") == 2:
                self.__dict__.update(state)
                return
        self._build(vocab_size)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _build(self, vocab_size: int) -> None:
        from wordfreq import top_n_list, word_frequency

        from bench.resources import book_words

        self.vocab_size = vocab_size
        self.version = 2
        words = [w for w in top_n_list("en", vocab_size) if LETTERS_RE.match(w)]
        self.unigram = {w: math.log(word_frequency(w, "en") + 1e-9) for w in words}
        self.floor = min(self.unigram.values()) - 2.0

        # bigrams from TRAIN books only
        bigrams: Counter = Counter()
        left: Counter = Counter()
        for book in book_words("train"):
            toks = [_core(w).lower() for w in book]
            for a, b in zip(toks, toks[1:]):
                if a and b:
                    bigrams[(a, b)] += 1
                    left[a] += 1
        self.bigram = dict(bigrams)
        self.left = dict(left)
        for (a, b) in self.bigram:  # make every word seen in the books a known word
            for w in (a, b):
                if w not in self.unigram and LETTERS_RE.match(w):
                    self.unigram[w] = self.floor + 1.0

        # SymSpell-style delete index (edit distance <= 2)
        index: dict[str, list[str]] = defaultdict(list)
        for w in self.unigram:
            for d in _deletes(w, 2):
                index[d].append(w)
        self.index = dict(index)

    def known(self, w: str) -> bool:
        return w.lower() in self.unigram

    def _p_uni(self, w: str) -> float:
        return math.exp(self.unigram.get(w, self.floor))

    def _p_next(self, prev: str, w: str) -> float:
        """Dirichlet-smoothed bigram: observed pairs dominate after frequent words, else unigram."""
        return (self.bigram.get((prev, w), 0) + DIRICHLET * self._p_uni(w)) / (self.left.get(prev, 0) + DIRICHLET)

    def logp(self, w: str, prev: str | None, nxt: str | None) -> float:
        """log P(w | prev) + log [P(next | w) / P(next)]: how well w fits between its neighbours."""
        w = w.lower()
        score = math.log(self._p_next(prev.lower(), w)) if prev else self.unigram.get(w, self.floor)
        if nxt:
            n = nxt.lower()
            score += math.log(self._p_next(w, n) / self._p_uni(n))
        return score

    def neighbors(self, w: str) -> set[str]:
        w = w.lower()
        max_d = 1 if len(w) <= 3 else 2  # short words have hundreds of distance-2 neighbours
        out = set()
        for d in _deletes(w, max_d):
            for cand in self.index.get(d, ()):
                if Levenshtein.distance(w, cand) <= max_d:
                    out.add(cand)
        return out


def _deletes(w: str, depth: int) -> set[str]:
    out, frontier = {w}, {w}
    for _ in range(depth):
        frontier = {f[:i] + f[i + 1:] for f in frontier for i in range(len(f))}
        out |= frontier
    return out


def _core(token: str) -> str:
    m = WORD_RE.match(token)
    return m.group(2) if m else token


def _apply_case(src: str, word: str) -> str:
    if src.isupper() and len(src) > 1:
        return word.upper()
    if src[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def _protected(core: str) -> bool:
    """Numbers, money, dates, emails, IDs, acronyms: never 'correct' with a dictionary."""
    if not core or any(ch.isdigit() for ch in core) and sum(ch.isalpha() for ch in core) < 2:
        return True
    if "@" in core or "/" in core or "$" in core or "%" in core:
        return True
    if core.isupper() and len(core) <= 5:
        return True
    if "-" in core:  # hyphenated compounds (to-morrow, shell-fish) are rarely in a lexicon
        return True
    return False


@lru_cache(maxsize=1)
def get_lm() -> LanguageModel:
    return LanguageModel()


@lru_cache(maxsize=200_000)
def _cached_neighbors(w: str) -> frozenset:
    return frozenset(get_lm().neighbors(w)) if 2 <= len(w) <= 20 else frozenset()


class WordPredictor:
    """Rescores every word of an OCR result. Parameters are tuned on the dev split."""

    def __init__(self, lam: float = 4.0, mu: float = 3.0, margin: float = 1.0, max_cost: float = 2.0,
                 keep_unknown: float = 0.8):
        self.lm = get_lm()
        self.lam, self.mu, self.margin, self.max_cost = lam, mu, margin, max_cost
        self.keep_unknown = keep_unknown  # unknown words read this confidently are names / rare words: keep
        self._neighbors = _cached_neighbors

    def suspect(self, token: str, conf: float, alts_i: list[tuple[str, float]] | None):
        """Return (pre, core, post, alt) if the word deserves a second look, else None."""
        m = WORD_RE.match(token)
        pre, core, post = (m.group(1), m.group(2), m.group(3)) if m else ("", token, "")
        if _protected(core):
            return None
        if core.isupper() and conf >= 0.5:
            return None  # headings, names, codes, non-English words on receipts: leave unless very unsure
        alt = {}
        for a, share in (alts_i or []):
            ac = _core(a)
            if ac and LETTERS_RE.match(ac):
                alt[ac.lower()] = max(alt.get(ac.lower(), 0.0), share)
        if self.lm.known(core) and conf >= 0.9 and len(alt) <= 1:
            return None
        if not self.lm.known(core) and conf >= self.keep_unknown and len(alt) <= 1:
            return None
        return pre, core, post, alt

    def candidates(self, core: str, alt: dict) -> set[str]:
        cands = set(alt) | self._neighbors(core.lower())
        cands.add(core.lower())
        return cands

    def channel(self, core: str, c: str, alt: dict) -> float:
        """Visual plausibility + engine votes (higher is better), shared with the neural predictor."""
        cost = 0.0 if c == core.lower() else ocr_edit_cost(core.lower(), c)
        s = -self.lam * cost + self.mu * alt.get(c, 0.0)
        if not self.lm.known(c):
            s -= 6.0
        return s

    def correct_line(self, words: list[str], alts: list[list[tuple[str, float]]] | None = None,
                     confs: list[float] | None = None, context: str = "") -> list[str]:
        """words: OCR words; alts[i]: (reading, vote share) alternatives from the ensemble."""
        out = list(words)
        for i, token in enumerate(words):
            conf = confs[i] if confs else 1.0
            sus = self.suspect(token, conf, alts[i] if alts else None)
            if sus is None:
                continue
            pre, core, post, alt = sus
            prev = _core(out[i - 1]) if i else None
            nxt = _core(words[i + 1]) if i + 1 < len(words) else None
            obs_known = self.lm.known(core)

            def score(c: str) -> float:
                return self.lm.logp(c, prev, nxt) + self.channel(core, c, alt)

            best = max(self.candidates(core, alt), key=score)
            if best == core.lower():
                continue
            cost = ocr_edit_cost(core.lower(), best)
            if cost > self.max_cost and best not in alt:
                continue
            # keep a real word unless the alternative is clearly better (avoid over-correction)
            if obs_known and score(best) - score(core.lower()) < self.margin * conf:
                continue
            out[i] = pre + _apply_case(core, best) + post
        return out
