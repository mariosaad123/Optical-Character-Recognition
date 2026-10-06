from ocr_engine.ensemble import Hypothesis, align, combine, to_lines


def H(text, conf=0.9, weight=1.0):
    return Hypothesis([[(w, conf) for w in line.split()] for line in text.split("\n")], weight)


def text(slots):
    return "\n".join(" ".join(w for w, _ in line) for line in to_lines(slots))


def test_align_identical():
    assert align(["a", "b"], ["a", "b"]) == [(0, 0), (1, 1)]


def test_majority_fixes_single_error():
    out = combine([H("the quick brovvn fox"), H("the quick brown fox"), H("the qulck brown fox")])
    assert text(out) == "the quick brown fox"


def test_missing_word_recovered_by_majority():
    out = combine([H("the brown fox"), H("the quick brown fox"), H("the quick brown fox")])
    assert text(out) == "the quick brown fox"


def test_spurious_word_dropped():
    out = combine([H("the quick brown fox"), H("the quick ~ brown fox", conf=0.2), H("the quick brown fox")])
    assert text(out) == "the quick brown fox"


def test_line_breaks_follow_pivot():
    out = combine([H("hello world\nsecond line"), H("hello world second line")])
    assert text(out) == "hello world\nsecond line"
