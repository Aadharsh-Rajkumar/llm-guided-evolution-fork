import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from src.qiskit_rag import (PINNED_NOTE, build_rag_section, recent_failures,
                            retrieve, retrieve_context)


def test_structural_query_retrieves_gate_topology_guidance():
    results = retrieve("change CNOT entanglement topology and add a CZ gate", top_k=2)
    assert results
    # two_qubit_gates.md (added 2026-10-08) is the more specific match for a CZ query;
    # either gate-topology note is a correct top hit.
    assert results[0].source in ("structural_gates.md", "two_qubit_gates.md")
    assert "cx" in results[0].text.lower()


def test_weight_query_distinguishes_parameter_binding_from_topology():
    results = retrieve("ParameterVector assign_parameters optimize rotation weights", top_k=2)
    assert results
    assert results[0].source == "parameters_and_binding.md"


def test_formatted_context_has_provenance():
    context = retrieve_context("Qiskit Statevector readout probabilities", top_k=1)
    assert "statevector_readout.md" in context
    assert "Reference:" in context


def test_template_no_longer_carries_static_retrieval(monkeypatch):
    import run_improved

    monkeypatch.setattr(
        run_improved,
        "PROMPT_GLOB",
        "templates/QuantumVQC/Normal/structural_mutation.txt",
    )
    prompt, mutation_type = run_improved.generate_template(
        0, 0, [], run_improved.SOTA_ROOT, run_improved.SEED_NETWORK,
        run_improved.ROOT_DIR, "llama3",
    )
    assert mutation_type == "structural_mutation"
    assert "Do not change only feature-encoding constants" in prompt
    assert "Retrieved Qiskit API guidance" not in prompt   # the LLM job adds it per block
    assert "Mujoco RL hard rules" not in prompt


TRAINING_BLOCK = """
MAX_ITER = 178

def train_angles(qc, x_params, w_params, X, y, X_val=None, y_val=None,
                 initial_weights=None, optimizer="cobyla"):
    return minimize(lambda w: cross_entropy(qc, x_params, w_params, w, X, y), w0, method="COBYLA").x
"""


def test_block_query_retrieves_block_specific_note():
    text, sources = build_rag_section("Improve this block", TRAINING_BLOCK)
    assert sources[0] == PINNED_NOTE
    assert "weight_training.md" in sources[1:]
    assert "def train_angles(qc, x_params, w_params" in text   # contract parsed from the block
    assert "module constant MAX_ITER" in text


def test_failure_feedback_pulls_import_note_and_is_block_specific(tmp_path):
    log = tmp_path / "failures.jsonl"
    rows = [
        {"block": 4, "error": "ImportError: cannot import name 'ParameterVector' from 'qiskit' (/x/qiskit/__init__.py)"},
        {"block": 2, "error": "AttributeError: 'QuantumCircuit' object has no attribute 'cnot'"},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert recent_failures(log, 4) == ["ImportError: cannot import name 'ParameterVector' from 'qiskit'"]
    text, _ = build_rag_section("Improve this block", "def build_circuit():\n    pass",
                                recent_failures(log, 4))
    assert "do not repeat them" in text and "ParameterVector" in text
    assert "qc.cnot" in text            # pinned API note is always present


def test_mutation_job_appends_rag_only_when_enabled(tmp_path, monkeypatch):
    import src.llm_mutation as mutation

    seed = Path(__file__).resolve().parents[1] / "sota" / "QuantumVQC" / "network.py"
    template = tmp_path / "t.txt"
    template.write_text("Improve this block.\n```python\n{}\n```\n")
    prompts = []
    monkeypatch.setattr(mutation, "generate_augmented_code",
                        lambda txt, *a, **k: prompts.append(txt) or "ERROR")
    for rag in (False, True):
        out = tmp_path / f"child_{rag}.py"
        mutation.augment_network(str(seed), str(out), str(template), qiskit_rag=rag)
        meta = json.loads(Path(f"{out}.rag.json").read_text())
        assert meta["rag"] is rag and 1 <= meta["block"] <= 6
    assert "Retrieved Qiskit API guidance" not in prompts[0]
    assert "Retrieved Qiskit API guidance" in prompts[1]
