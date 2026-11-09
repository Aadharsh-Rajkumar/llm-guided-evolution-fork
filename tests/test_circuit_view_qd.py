import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import vlm_utils

SEED = ROOT / "sota" / "QuantumVQC" / "network.py"


def load_seed():
    spec = importlib.util.spec_from_file_location("seed_for_test", SEED)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_circuit_view_off_adds_nothing(tmp_path):
    assert vlm_utils.circuit_note(str(SEED), str(tmp_path), view="off") == ""


def test_ascii_view_draws_the_whole_parent_circuit(tmp_path):
    note = vlm_utils.circuit_note(str(SEED), str(tmp_path), view="ascii")
    assert "q_7" in note and "x[0]" in note and "w[0]" in note
    assert "```" not in note
    assert (tmp_path / "network_ascii.txt").exists()


def test_ascii_view_never_breaks_a_mutation(tmp_path):
    bad = tmp_path / "network_bad.py"
    bad.write_text("raise RuntimeError('broken parent')\n")
    assert vlm_utils.circuit_note(str(bad), str(tmp_path), view="ascii") == ""


def test_fingerprint_ignores_comments_but_not_behaviour():
    mod = load_seed()
    X_tr = mod.load_data()[0]
    qc, xp, wp = mod.build_circuit()
    assert mod.fingerprint(qc, xp, wp, X_tr) == mod.fingerprint(qc, xp, wp, X_tr)
    qc2, xp2, wp2 = mod.build_circuit()
    qc2.cz(0, 1)
    assert mod.fingerprint(qc2, xp2, wp2, X_tr) != mod.fingerprint(qc, xp, wp, X_tr)


def test_seen_fingerprint_is_not_trained_again(tmp_path):
    mod = load_seed()
    X_tr = mod.load_data()[0]
    qc, xp, wp = mod.build_circuit()
    seen = tmp_path / "seen.txt"
    seen.write_text(mod.fingerprint(qc, xp, wp, X_tr) + "\n")
    env = dict(os.environ, LLMGE_SEEN_FINGERPRINTS=str(seen), QVQC_OBJECTIVES="ce,wlive,cesd")
    out = subprocess.run([sys.executable, str(SEED), "--gene-id", "dup", "--out-dir", str(tmp_path), "--repr", ""],
                         env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    metrics = json.loads((tmp_path / "dup_metrics.json").read_text())
    assert metrics["duplicate"] is True
    assert (tmp_path / "dup_results.txt").read_text().split(", ") == ["inf", "inf", "inf"]


def test_qd_cell_clamps_to_the_grid():
    src = (ROOT / "run_improved.py").read_text()
    start = src.index("def qd_cell(coords):")
    end = src.index("def qd_update(")
    namespace = {"QD_NONLOCAL_GATE_RANGE": (0.0, 24.0), "QD_DEPTH_RANGE": (0.0, 36.0), "QD_GRID_SIZE": 6}
    exec(src[start:end], namespace)
    qd_cell = namespace["qd_cell"]
    assert qd_cell((0, 0)) == (0, 0)
    assert qd_cell((6, 11)) == (1, 1)
    assert qd_cell((56, 84)) == (5, 5)
    assert qd_cell((-3, -1)) == (0, 0)
