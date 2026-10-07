"""Word-level ensemble (ROVER-style voting).

Several hypotheses (different engines and/or differently pre-processed images) are aligned
word-by-word into a confusion network; each slot then holds the competing readings with
their accumulated votes. The best reading per slot wins, and the full candidate lists are
kept so the language model can use them later.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein

Words = list[tuple[str, float]]  # (word, confidence in [0, 1])


@dataclass
class Hypothesis:
    lines: list[Words]
    weight: float = 1.0
    source: str = ""

    @property
    def words(self) -> Words:
        return [w for line in self.lines for w in line]


@dataclass
class Slot:
    votes: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    confs: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))  # engine confidences per reading
    voters: float = 0.0  # total weight of hypotheses that voted in this slot
    line_break_after: bool = False
    readings: list[tuple[str, float]] = field(default_factory=list)  # every (reading, vote), for char voting

    def add(self, word: str, conf: float, weight: float) -> None:
        # A reading counts more when its engine is reliable (weight) and confident (conf).
        vote = weight * (0.3 + 0.7 * conf)
        self.votes[word] += vote
        self.confs[word].append(conf)
        self.voters += weight
        self.readings.append((word, vote))

    def best_by_chars(self, max_share: float = 0.75) -> tuple[str, float]:
        """Character-level vote inside the slot: when readers disagree, align every reading to the
        winning word and vote per character position, so e.g. "60,000" / "00.000" / "60.000" can
        yield the right word even if no single reader got it entirely right."""
        word, share = self.best()
        reads = [(w, v) for w, v in self.readings if w]
        if not word or share >= max_share or len(reads) < 3:
            return word, share
        total = sum(v for _, v in reads)
        cols = [defaultdict(float) for _ in word]      # per pivot character: votes for its replacement
        ins = [defaultdict(float) for _ in range(len(word) + 1)]  # insertions before each position
        for w, v in reads:
            for tag, i1, i2, j1, j2 in Levenshtein.opcodes(word, w):
                if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
                    for k in range(i2 - i1):
                        cols[i1 + k][w[j1 + k]] += v
                else:  # delete, insert or uneven replace: pivot chars dropped, reading's chars inserted
                    for k in range(i1, i2):
                        cols[k][""] += v
                    if j2 > j1:
                        ins[i1][w[j1:j2]] += v
        out = []
        for i in range(len(word) + 1):
            if ins[i]:
                piece, v = max(ins[i].items(), key=lambda kv: kv[1])
                if v > total / 2:
                    out.append(piece)
            if i < len(word):
                out.append(max(cols[i].items(), key=lambda kv: kv[1])[0])
        merged = "".join(out)
        return (merged or word), share

    def best(self) -> tuple[str, float]:
        word, score = max(self.votes.items(), key=lambda kv: kv[1])
        total = sum(self.votes.values())
        return word, (score / total if total else 0.0)

    def engine_conf(self, word: str) -> float:
        cs = self.confs.get(word)
        return sum(cs) / len(cs) if cs else 0.0

    def candidates(self) -> list[tuple[str, float]]:
        total = sum(self.votes.values()) or 1.0
        return sorted(((w, v / total) for w, v in self.votes.items() if w), key=lambda kv: -kv[1])


def _sub_cost(a: str, b: str) -> float:
    return 0.0 if a == b else Levenshtein.normalized_distance(a.lower(), b.lower())


def align(ref: list[str], hyp: list[str]) -> list[tuple[int | None, int | None]]:
    """Word alignment minimizing (insert=1, delete=1, substitute=char-level distance)."""
    n, m = len(ref), len(hyp)
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == j == 0:
                continue
            best, op = inf, 0
            if i and j:
                c = dp[i - 1][j - 1] + _sub_cost(ref[i - 1], hyp[j - 1])
                if c < best:
                    best, op = c, 1
            if i and dp[i - 1][j] + 1 < best:
                best, op = dp[i - 1][j] + 1, 2
            if j and dp[i][j - 1] + 1 < best:
                best, op = dp[i][j - 1] + 1, 3
            dp[i][j], back[i][j] = best, op
    pairs, i, j = [], n, m
    while i or j:
        op = back[i][j]
        if op == 1:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif op == 2:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return pairs[::-1]


def combine(hyps: list[Hypothesis]) -> list[Slot]:
    """Build a confusion network on top of the pivot (first) hypothesis."""
    hyps = [h for h in hyps if h.words] or hyps[:1]
    if not hyps or not hyps[0].words:
        return []
    pivot = hyps[0]
    slots: list[Slot] = []
    for li, line in enumerate(pivot.lines):
        for wi, (w, c) in enumerate(line):
            s = Slot()
            s.add(w, c, pivot.weight)
            s.line_break_after = wi == len(line) - 1 and li < len(pivot.lines) - 1
            slots.append(s)
    pivot_words = [w for w, _ in pivot.words]
    # insertions relative to the pivot live in extra slots keyed by the pivot index they follow
    inserted: dict[int, list[Slot]] = defaultdict(list)

    for h in hyps[1:]:
        hw = h.words
        last_ref, ins_run = -1, 0
        for ri, hi in align(pivot_words, [w for w, _ in hw]):
            if ri is not None:
                last_ref, ins_run = ri, 0
                if hi is None:
                    slots[ri].add("", 0.5, h.weight)
                else:
                    slots[ri].add(*hw[hi], h.weight)
            else:
                run = inserted[last_ref]
                if ins_run == len(run):
                    run.append(Slot())
                run[ins_run].add(*hw[hi], h.weight)
                ins_run += 1

    total_weight = sum(h.weight for h in hyps)
    out: list[Slot] = []
    for extra in inserted.get(-1, []):
        _fill_empty(extra, total_weight)
        out.append(extra)
    for i, s in enumerate(slots):
        out.append(s)
        for extra in inserted.get(i, []):
            _fill_empty(extra, total_weight)
            out.append(extra)
    return out


def _fill_empty(slot: Slot, total_weight: float) -> None:
    """Hypotheses that did not produce this inserted word implicitly vote for 'nothing here'."""
    missing = total_weight - slot.voters
    if missing > 1e-9:
        slot.add("", 0.5, missing)


def to_lines(slots: list[Slot], char_vote: bool = False) -> list[Words]:
    lines: list[Words] = [[]]
    for s in slots:
        w, c = s.best_by_chars() if char_vote else s.best()
        if w:
            lines[-1].append((w, c))
        if s.line_break_after:
            lines.append([])
    return [l for l in lines if l]
