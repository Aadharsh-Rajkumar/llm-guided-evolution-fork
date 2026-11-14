import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "sota" / "QuantumVQC" / "network.py"
SEED_DRAW = """    rng = np.random.default_rng(SEED)
    weights = (rng.uniform(0, 2 * np.pi, len(w_params))
               if initial_weights is None else np.asarray(initial_weights, dtype=float))
"""


def run_variant(tmp_path, name, old=None, new=None, budget=12):
    src = SEED.read_text()
    if old is not None:
        assert src.count(old) == 1
        src = src.replace(old, new)
    path = tmp_path / f"network_{name}.py"
    path.write_text(src)
    env = dict(os.environ, QVQC_OBJECTIVES="ce,wlive,cesd")
    env.pop("LLMGE_SEEN_FINGERPRINTS", None)
    out = subprocess.run([sys.executable, str(path), "--gene-id", name, "--out-dir", str(tmp_path),
                          "--repr", "", "--train-budget", str(budget), "--n-starts", "3"],
                         env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    metrics = json.loads((tmp_path / f"{name}_metrics.json").read_text())
    objs = [float(v) for v in (tmp_path / f"{name}_results.txt").read_text().split(",")[:3]]
    return metrics, objs


def test_seed_starts_differ_and_keep_their_spread(tmp_path):
    metrics, objs = run_variant(tmp_path, "seed", budget=178)
    assert not metrics["ignored_start_angles"] and not metrics["no_skill"]
    assert len({tuple(s["weights"]) for s in metrics["starts"]}) == 3
    assert objs[2] == metrics["val_cross_entropy_sd"]


def test_hard_coded_seed_that_honours_initial_weights_is_neutralised(tmp_path):
    metrics, objs = run_variant(tmp_path, "rng42", SEED_DRAW, SEED_DRAW.replace("default_rng(SEED)", "default_rng(42)"))
    assert not metrics["ignored_start_angles"]
    assert len({tuple(s["weights"]) for s in metrics["starts"]}) == 3


def test_training_that_ignores_the_start_angles_gets_the_worst_spread(tmp_path):
    fixed = ("    weights = np.random.default_rng(42).uniform(0, 2 * np.pi, len(w_params))\n")
    metrics, objs = run_variant(tmp_path, "fixed", SEED_DRAW, fixed)
    assert metrics["ignored_start_angles"]
    assert metrics["val_cross_entropy_sd"] == 0.0
    assert objs[2] == 1.0


def test_majority_class_circuit_gets_the_worst_size(tmp_path):
    old = """    p = np.asarray(probs[:N_CLASSES], dtype=float)
    total = p.sum()
    if total <= 0:
        return np.full(N_CLASSES, 1.0 / N_CLASSES)
    return p / total
"""
    new = """    p = np.asarray(probs[:N_CLASSES], dtype=float)
    share = p[1] / max(p.sum(), 1e-12)
    return np.array([0.49 - 0.48 * share, 0.51 + 0.48 * share])
"""
    metrics, objs = run_variant(tmp_path, "majority", old, new)
    assert metrics["val_accuracy"] <= metrics["majority_rate"] + 1e-9
    assert metrics["no_skill"]
    assert objs[1] == 500.0 and objs[2] == 1.0
