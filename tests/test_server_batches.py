import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_pick_batch():
    src = (ROOT / "server.py").read_text()
    start = src.index("def pick_batch(")
    end = src.index("\ndef ", start + 10)
    namespace = {"re": re, "SPLIT_LONG_BATCHES": True, "LONG_PROMPT_MARKER": "This block is long.", "LONG_MAX_WAIT": 600.0}
    exec(src[start:end], namespace)
    return namespace["pick_batch"]


def waiting(kinds, start=0.0):
    return [({"prompt": "This block is long. x" if k == "L" else "short"}, None, start + i) for i, k in enumerate(kinds)]


def test_short_requests_do_not_wait_for_long_ones():
    pick = load_pick_batch()
    w = waiting("LSSLSS")
    assert pick(w, now=10.0, batch_size=16) == [1, 2, 4, 5]


def test_long_requests_run_together_once_alone_full_or_old():
    pick = load_pick_batch()
    assert pick(waiting("LL"), now=10.0, batch_size=16) == [0, 1]
    assert pick(waiting("LLS"), now=10.0, batch_size=2) == [0, 1]
    assert pick(waiting("LSS"), now=700.0, batch_size=16) == [0]


def test_batches_are_capped_and_split_can_be_switched_off():
    pick = load_pick_batch()
    assert pick(waiting("S" * 20), now=1.0, batch_size=16) == list(range(16))
    assert pick(waiting("LSL"), now=1.0, batch_size=16, split=False) == [0, 1, 2]


def test_marker_matches_the_function_reply_instruction():
    utils = (ROOT / "src" / "llm_utils.py").read_text()
    assert re.search(r"This block is long\.", utils)
