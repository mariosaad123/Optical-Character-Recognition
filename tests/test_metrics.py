from bench.metrics import cer, wer


def test_perfect_match():
    assert cer("Hello world", "Hello world") == 0.0
    assert wer("Hello world", "Hello world") == 0.0


def test_whitespace_is_normalized():
    assert cer("Hello\nworld", "Hello   world") == 0.0


def test_single_char_error():
    assert cer("abcd", "abed") == 0.25


def test_word_error():
    assert wer("the quick brown fox", "the quick brawn fox") == 0.25


def test_empty_reference():
    assert cer("", "") == 0.0
    assert cer("", "x") == 1.0


def test_bow_f1_ignores_order():
    from bench.metrics import bow_f1
    assert bow_f1("a b c", "c b a") == 1.0
    assert abs(bow_f1("a b c d", "a b") - 2 / 3) < 1e-9


def test_score_case_insensitive_flag():
    from bench.metrics import score
    assert score({"text": "TOTAL 80.91", "case_insensitive": True}, "Total 80.91")["cer"] == 0.0
    assert score({"text": "TOTAL 80.91"}, "Total 80.91")["cer"] > 0
