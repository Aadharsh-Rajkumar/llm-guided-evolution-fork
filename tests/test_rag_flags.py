import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import llm_crossover
import llm_mutation
from llm_utils import merge_function_reply
from qiskit_rag import RAG_FLAG_ENV, REUPLOAD_NOTE, build_rag_section, rag_flags

SEED = ROOT / "sota" / "QuantumVQC" / "network.py"
BLOCKS = SEED.read_text().split("# --OPTION--")
TEMPLATES = sorted((ROOT / "templates" / "QuantumVQC" / "Normal").glob("*.txt"))
CROSSOVER = sorted((ROOT / "templates" / "CrossOver").glob("*.txt"))


@pytest.fixture
def failure_log(tmp_path):
    path = tmp_path / "failures.jsonl"
    rows = [{"gene": f"g{i}", "gen": 1, "block": b, "op": "mutation", "error": f"TypeError: bad {i}"}
            for i, b in enumerate(range(1, len(BLOCKS)))]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


def set_flags(monkeypatch, **on):
    for name, env in RAG_FLAG_ENV.items():
        monkeypatch.setenv(env, "1" if on.get(name) else "0")


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.stem)
@pytest.mark.parametrize("block", range(1, len(BLOCKS)))
def test_all_off_mutation_prompt_equals_no_rag(monkeypatch, failure_log, template, block):
    set_flags(monkeypatch)
    text = template.read_text()
    with_rag = llm_mutation.build_prompt(text, BLOCKS[block], block, True, failure_log)
    without = llm_mutation.build_prompt(text, BLOCKS[block], block, False, failure_log)
    assert with_rag[0] == without[0]
    assert with_rag[1] == [] and with_rag[2] == []


@pytest.mark.parametrize("template", CROSSOVER, ids=lambda p: p.stem)
@pytest.mark.parametrize("block", range(1, len(BLOCKS)))
def test_all_off_crossover_prompt_equals_no_rag(monkeypatch, failure_log, template, block):
    set_flags(monkeypatch)
    text = template.read_text()
    x, y = BLOCKS[block], BLOCKS[block] + "\n# variant\n"
    assert (llm_crossover.build_prompt(text, x, y, block, True, failure_log)[0]
            == llm_crossover.build_prompt(text, x, y, block, False, failure_log)[0])


def test_crossover_flag_off_keeps_crossover_prompt_plain(monkeypatch, failure_log):
    set_flags(monkeypatch, retrieval=True, pin_api=True, pin_gate_budget=True, contract=True,
              error_feedback=True, reupload_gating=True)
    text = CROSSOVER[0].read_text()
    x, y = BLOCKS[4], BLOCKS[4] + "\n# variant\n"
    assert (llm_crossover.build_prompt(text, x, y, 4, True, failure_log)[0]
            == llm_crossover.build_prompt(text, x, y, 4, False, failure_log)[0])


@pytest.mark.parametrize("component, marker", [
    ("retrieval", "BM25="), ("pin_api", "qiskit_imports_api.md; pinned"),
    ("pin_gate_budget", "gate_budget.md; pinned"), ("contract", "Block contract"),
    ("error_feedback", "crashed with these errors")])
def test_each_component_alone(monkeypatch, failure_log, component, marker):
    set_flags(monkeypatch, **{component: True})
    text = TEMPLATES[0].read_text()
    prompt = llm_mutation.build_prompt(text, BLOCKS[4], 4, True, failure_log)[0]
    plain = llm_mutation.build_prompt(text, BLOCKS[4], 4, False, failure_log)[0]
    assert marker in prompt and prompt.startswith(plain) and prompt != plain
    others = [m for c, m in [("retrieval", "BM25="), ("pin_api", "qiskit_imports_api.md; pinned"),
                             ("pin_gate_budget", "gate_budget.md; pinned"), ("contract", "Block contract"),
                             ("error_feedback", "crashed with these errors")] if c != component]
    assert not any(m in prompt for m in others)


def test_reupload_note_only_for_accuracy_goals():
    flags = {name: True for name in RAG_FLAG_ENV}
    for block in BLOCKS[1:]:
        _, sources = build_rag_section("Make one structural mutation to this block", block, flags=flags)
        assert REUPLOAD_NOTE not in sources
    seen = [REUPLOAD_NOTE in build_rag_section("Improve this Qiskit block", b, flags=flags)[1] for b in BLOCKS[1:]]
    assert any(seen)
    ungated = dict(flags, reupload_gating=False)
    seen = [REUPLOAD_NOTE in build_rag_section("Make one structural mutation", b, flags=ungated)[1]
            for b in BLOCKS[1:]]
    assert any(seen)


def test_flags_default_on(monkeypatch):
    for env in RAG_FLAG_ENV.values():
        monkeypatch.delenv(env, raising=False)
    assert all(rag_flags().values())


def test_function_reply_merge_keeps_unreturned_functions():
    parent = BLOCKS[6]
    reply = ("def train_angles(qc, x_params, w_params, X, y, X_val=None, y_val=None,\n"
             "                 initial_weights=None, optimizer=\"cobyla\"):\n    return None\n")
    merged = merge_function_reply(parent, reply)
    assert "def local_weight_search" in merged and "def adam_shift" in merged
    assert merged.count("def train_angles") == 1 and "return None" in merged
    full = "\n".join(l for l in parent.splitlines() if not l.startswith("MAX_ITER"))
    assert "MAX_ITER = " in merge_function_reply(parent, full)
