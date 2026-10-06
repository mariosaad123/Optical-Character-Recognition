from ocr_engine.pipeline import PipelineConfig, fuse


def R(text, conf=0.9):
    return [[(w, conf) for w in line.split()] for line in text.split("\n")]


def test_fuse_votes_across_sources():
    readings = {"a/raw": R("the qulck brown fox"), "b/raw": R("the quick brovvn fox"), "c/raw": R("the quick brown fox")}
    cfg = PipelineConfig(sources={"a/raw": 1.0, "b/raw": 1.0, "c/raw": 1.0}, use_lm=False)
    assert fuse(readings, cfg) == "the quick brown fox"


def test_fuse_handles_empty_sources():
    cfg = PipelineConfig(sources={"a/raw": 1.0, "b/raw": 1.0}, use_lm=False)
    assert fuse({"a/raw": [], "b/raw": R("hello world")}, cfg) == "hello world"
    assert fuse({"a/raw": [], "b/raw": []}, cfg) == ""
