import ast
import re
import textwrap
from pathlib import Path

import torch
from transformers import LogitsProcessor

ROOT = Path(__file__).resolve().parents[1]


def load(name="closed_code_block"):
    src = (ROOT / "server.py").read_text()
    start = src.index("def parses_as_code(")
    end = src.index("def runaway(")
    namespace = {"ast": ast, "re": re, "textwrap": textwrap, "torch": torch, "LogitsProcessor": LogitsProcessor,
                 "_LANG_TAG": re.compile(r"\s*[A-Za-z0-9_+-]*\s*")}
    exec(src[start:end], namespace)
    return namespace[name]


def test_prose_after_an_empty_fence_is_not_a_closed_block():
    closed = load()
    text = "I'll change it:\n```python\n```\nHere is the new training code\nwith one change\n```python\ndef train_angles(qc"
    assert not closed(text)


def test_a_finished_code_block_is_closed():
    closed = load()
    assert closed("```python\ndef f(x):\n    return x\n```")
    assert closed("```python\n```def f(x):\n    return x\n```")


def test_an_open_block_is_not_closed():
    closed = load()
    assert not closed("```python\ndef f(x):\n    return x\n")


class FakeTokenizer:
    def __init__(self, texts):
        self.texts = texts

    def batch_decode(self, ids, skip_special_tokens=True):
        return self.texts


def test_has_code_needs_real_code():
    has_code = load("has_code")
    assert not has_code("```python\n```")
    assert not has_code("I'll change the training block:\n```python\n```\nHere it is")
    assert has_code("```python\ndef f(x):\n    return x\n```")
    assert has_code("def f(x):\n    return x\n")


def test_an_answer_without_code_cannot_end():
    CodeBeforeEnd = load("CodeBeforeEnd")
    texts = ["```python\n```", "```python\ndef f(x):\n    return x\n```"]
    proc = CodeBeforeEnd(FakeTokenizer(texts), end_ids=[5, 7, None], every=1)
    ids = torch.zeros((2, 4), dtype=torch.long)
    scores = proc(ids, torch.zeros((2, 10)))
    assert torch.isinf(scores[0, [5, 7]]).all() and not torch.isinf(scores[0, [0, 1, 6]]).any()
    assert not torch.isinf(scores[1]).any()


def test_a_row_stays_free_to_end_once_it_has_code():
    CodeBeforeEnd = load("CodeBeforeEnd")
    tok = FakeTokenizer(["```python\ndef f(x):\n    return x\n```"])
    proc = CodeBeforeEnd(tok, end_ids=[5], every=1)
    proc(torch.zeros((1, 4), dtype=torch.long), torch.zeros((1, 10)))
    tok.texts = ["```python\ndef f(x):\n    return x\n``` and then prose"]
    scores = proc(torch.zeros((1, 5), dtype=torch.long), torch.zeros((1, 10)))
    assert not torch.isinf(scores).any()
