import ast
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    src = (ROOT / "server.py").read_text()
    start = src.index("def parses_as_code(")
    end = src.index("def runaway(")
    namespace = {"ast": ast, "re": re, "textwrap": textwrap, "_LANG_TAG": re.compile(r"\s*[A-Za-z0-9_+-]*\s*")}
    exec(src[start:end], namespace)
    return namespace["closed_code_block"]


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
