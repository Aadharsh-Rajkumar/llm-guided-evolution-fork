# --PROMPT LOG--

"""
Base seed algorithm: a variational quantum classifier in Qiskit.

This is individual zero. Evolution mutates the blocks below the first
OPTION marker (see the separators further down); everything above it is fixed
scaffolding and is never handed to the LLM.

Contract with the LLM-GE harness:
  python seed_vqc.py --gene-id <id> --out-dir <dir>
writes "<obj1>, <obj2>" to <out_dir>/<gene_id>_results.txt, matching
run_improved.py check4results(). Both objectives are MINIMIZED:
  obj1 = validation cross-entropy   (task performance)
  obj2 = weighted gate cost         (two-qubit gates cost 5x, per hardware error rates)
A crash, an invalid circuit, or a missing file leaves the harness to assign
INVALID_FITNESS_MAX, which is the intended failure path - do not catch broadly
and report a fake score.

Fitness is computed on VALIDATION. Test is never touched here.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import Statevector
# Pre-imported so mutated blocks can reach for them; the import region is not evolvable.
import math
import random
import itertools

SEED = 42
N_QUBITS = 4
N_CLASSES = 3
READOUT = [0, 1]          # ceil(log2(3)) = 2, EXAQC's readout sizing
TWO_QUBIT_COST = 5.0      # two-qubit error rates are ~an order of magnitude worse
MAX_GATE_COST = 500.0     # cost ceiling; runaway circuits are not interesting


def load_data():
    """Fixed split, shared by every individual. Scaler is fit on TRAIN ONLY.

    Test is held out at test_size=0.3 / random_state=42 so it stays identical to
    every number recorded before evolution existed. Validation is carved out of
    the training portion, so nothing about the test set moves.
    """
    X, y = load_iris(return_X_y=True)
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_full, y_tr_full, test_size=0.25, random_state=SEED, stratify=y_tr_full)

    scaler = MinMaxScaler(feature_range=(0.0, np.pi)).fit(X_tr)
    return (scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te),
            y_tr, y_va, y_te)


def gate_count(qc):
    """Objective 2. Plain gate count, the "# Gates" column EXAQC reports."""
    return float(sum(1 for inst in qc.data
                     if inst.operation.name not in ("barrier", "measure")))


def weighted_gate_cost(qc):
    """Hardware-weighted count. Kept for reference; obj2 uses gate_count()."""
    cost = 0.0
    for inst in qc.data:
        if inst.operation.name in ("barrier", "measure"):
            continue
        cost += TWO_QUBIT_COST if inst.operation.num_qubits >= 2 else 1.0
    return cost



# ---------------------------------------------------------------------------
# Training budget. Every individual gets the same amount of compute, and the
# count is kept HERE rather than in the training block, which stays evolvable.
# A variant may rewrite the optimizer, the initial angles, the batching, any of
# it - but every training pass runs the circuit through forward(), so the budget
# is enforced from a place the LLM cannot edit. Counting simulations rather than
# optimizer steps keeps mini-batch training honest: fewer samples per step buys
# more steps, not more compute.
TRAIN_BUDGET_EVALS = 178        # full passes over the training set, measured in §5
_PASSES = 0                     # forward() calls made while training
_PASS_LIMIT = None              # set in main() once the training set size is known
_COUNTING = False               # only training counts, scoring afterwards does not
_BEST = [float("inf"), None]    # best (training loss, weights) seen while training


class BudgetSpent(Exception):
    """Raised inside forward() when an individual has used its training budget."""


def forward(qc, x_params, w_params, x_vals, w_vals):
    """Bind one sample plus the weights and return class probabilities."""
    global _PASSES
    if _COUNTING:
        _PASSES += 1
        if _PASS_LIMIT is not None and _PASSES > _PASS_LIMIT:
            raise BudgetSpent(f"training budget of {TRAIN_BUDGET_EVALS} passes is spent")
    bound = qc.assign_parameters(
        {**dict(zip(x_params, x_vals)), **dict(zip(w_params, w_vals))})
    probs = Statevector(bound).probabilities(READOUT)
    return readout_probabilities(probs)


def cross_entropy(qc, xp, wp, w_vals, X, y):
    eps = 1e-10
    total = 0.0
    for xi, yi in zip(X, y):
        p = forward(qc, xp, wp, xi, w_vals)
        total -= np.log(p[yi] + eps)
    loss = total / len(X)
    # Remember the best angles seen while training, so an individual that runs out
    # of budget mid-search still gets scored on its best work rather than dying.
    if _COUNTING and loss < _BEST[0]:
        _BEST[0], _BEST[1] = loss, np.array(w_vals, dtype=float)
    return loss


def accuracy(qc, xp, wp, w_vals, X, y):
    correct = sum(int(np.argmax(forward(qc, xp, wp, xi, w_vals)) == yi)
                  for xi, yi in zip(X, y))
    return correct / len(X)


def write_results(out_dir, gene_id, obj1, obj2, extra=None):
    """run_improved.py reads the first two values; anything after is diagnostic."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{gene_id}_results.txt")
    with open(path, "w") as f:
        f.write(f"{obj1}, {obj2}" + ("" if extra is None else f", {extra}"))
    return path


def count_dead_gates(qc, x_params, w_params, w_vals, tol=1e-2):
    """§7: EXAQC's champion circuits carried dead gates - R(0) no-ops, H.H = I.
    Report them alongside accuracy instead of silently inheriting the pathology.

    Only TRAINABLE rotations are assessed. Binding the inputs too would score
    every RY(x[i]) encoding gate as dead whenever that feature happens to be 0,
    which says nothing about the architecture - "is this gate dead" is ill-posed
    for a gate whose angle changes with every sample. Leaving x symbolic makes
    float() raise on those, and the except-branch skips them.
    """
    bound = qc.assign_parameters(dict(zip(w_params, w_vals)))
    dead = rotations = 0
    for inst in bound.data:
        if not inst.operation.params:
            continue
        try:
            angle = float(inst.operation.params[0])
        except (TypeError, ValueError):
            continue
        rotations += 1
        r = abs(angle) % (2 * np.pi)
        if min(r, 2 * np.pi - r) < tol:
            dead += 1
    return dead, rotations


def emit_representations(qc, out_dir, gene_id, arms=("qasm", "ascii", "image")):
    """H2 ablation (§1): three views of the SAME circuit object.

    Rendered UNBOUND so parameter names stay visible - that is the architecture
    the LLM is being asked to edit, and it maps onto the OPTION blocks below.
    Protected on purpose: an individual must not be able to rewrite how it is
    presented to the LLM.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = {}

    if "qasm" in arms:
        from qiskit.qasm3 import dumps          # qc.qasm() was removed in qiskit 2.x
        path = os.path.join(out_dir, f"{gene_id}_circuit.qasm")
        with open(path, "w") as f:
            f.write(dumps(qc))
        written["qasm"] = path

    if "ascii" in arms:
        path = os.path.join(out_dir, f"{gene_id}_circuit.txt")
        with open(path, "w") as f:
            f.write(str(qc.draw("text")))
        written["ascii"] = path

    if "image" in arms:
        import matplotlib
        matplotlib.use("Agg")                   # PACE has no display; must precede pyplot
        import matplotlib.pyplot as plt
        path = os.path.join(out_dir, f"{gene_id}_circuit.png")
        fig = qc.draw("mpl")                    # needs pylatexenc
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written["image"] = path

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-id", default="seed")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--repr", default="qasm,ascii,image",
                    help="H2 arms to emit; empty string to skip")
    args = ap.parse_args()

    t0 = time.perf_counter()
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_data()

    qc, xp, wp = build_circuit()
    if len(wp) == 0:
        raise ValueError("circuit has no trainable parameters")

    global _COUNTING, _PASS_LIMIT
    _PASS_LIMIT = TRAIN_BUDGET_EVALS * len(X_tr)
    _COUNTING = True
    try:
        w = train_angles(qc, xp, wp, X_tr, y_tr)
    except BudgetSpent:
        if _BEST[1] is None:
            raise                      # trained without using cross_entropy: no angles to keep
        w = _BEST[1]
    finally:
        _COUNTING = False

    # Both objectives MINIMIZE and both are what EXAQC's Table 1 reports: accuracy
    # (as its error, so lower is better) and a plain gate count. Accuracy is scored
    # on VALIDATION, never test - EXAQC reports test accuracy but we only touch test
    # once, at the end. Validation is 27 samples, so obj1 moves in steps of 1/27 and
    # ties are expected; the cross-entropy rides along as a third value so we can see
    # how much resolution we are giving up.
    val_acc = accuracy(qc, xp, wp, w, X_va, y_va)
    obj1 = 1.0 - val_acc
    obj2 = min(gate_count(qc), MAX_GATE_COST)
    val_ce = cross_entropy(qc, xp, wp, w, X_va, y_va)

    if not np.isfinite(val_ce):
        raise ValueError(f"non-finite validation loss: {val_ce}")

    path = write_results(args.out_dir, args.gene_id, obj1, obj2, extra=val_ce)
    print(f"gene {args.gene_id}")
    print(f"  qubits {qc.num_qubits}  depth {qc.depth()}  params {len(wp)}")
    print(f"  obj1 val error {obj1:.4f} ({val_acc:.1%} acc)   obj2 gates {obj2:.0f}"
          f"   [val cross-entropy {val_ce:.4f}, weighted cost {weighted_gate_cost(qc):.0f}]")
    print(f"  train acc {accuracy(qc, xp, wp, w, X_tr, y_tr):.1%}"
          f"   val acc {accuracy(qc, xp, wp, w, X_va, y_va):.1%}")

    dead, rotations = count_dead_gates(qc, xp, wp, w)
    print(f"  dead gates {dead}/{rotations} rotations at |angle| < 0.01")

    # Every metric we might want to plot, so a Pareto front can be redrawn on any
    # pair afterwards without re-running anything. The harness never reads this
    # file; obj1 and obj2 in the results file are what selection uses.
    two_qubit = sum(1 for inst in qc.data
                    if inst.operation.num_qubits >= 2
                    and inst.operation.name not in ("barrier", "measure"))
    metrics = {
        "gene_id": args.gene_id,
        "obj1_val_error": obj1, "obj2_gates": obj2,
        "val_accuracy": val_acc, "val_cross_entropy": val_ce,
        "train_accuracy": accuracy(qc, xp, wp, w, X_tr, y_tr),
        "train_cross_entropy": cross_entropy(qc, xp, wp, w, X_tr, y_tr),
        "gates": gate_count(qc), "weighted_gate_cost": weighted_gate_cost(qc),
        "two_qubit_gates": two_qubit, "depth": qc.depth(),
        "n_qubits": qc.num_qubits, "n_params": len(wp),
        "train_passes": _PASSES, "train_evals": _PASSES / max(len(X_tr), 1),
        "budget_evals": TRAIN_BUDGET_EVALS, "budget_spent": _PASSES >= (_PASS_LIMIT or 0),
        "dead_rotations": dead, "rotations": rotations,
        "seconds": time.perf_counter() - t0,
    }
    mpath = os.path.join(args.out_dir, f"{args.gene_id}_metrics.json")
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=1, sort_keys=True)
    print(f"  metrics -> {mpath}   (used {metrics['train_evals']:.0f} of "
          f"{TRAIN_BUDGET_EVALS} training evals)")

    arms = tuple(a for a in args.repr.split(",") if a)
    if arms:
        for arm, rpath in emit_representations(qc, args.out_dir, args.gene_id, arms).items():
            print(f"  repr [{arm}] -> {rpath}")

    print(f"  wrote {path}   ({time.perf_counter() - t0:.1f}s)")


# --OPTION--
# -- NOTE --
# Feature encoding: how the 4 classical Iris features become rotation angles.
# Data is already scaled to [0, pi]. A single rotation per feature is the
# baseline; data re-uploading (repeating this map between variational layers)
# is a known way to raise expressivity and is deliberately NOT used here.
# -- NOTE --
def build_feature_map(qc, x_params):
    for i in range(N_QUBITS):
        qc.ry(x_params[i], i)


# --OPTION--
# Variational layer: the trainable rotations applied to every qubit.
def build_variational_layer(qc, w_params, offset):
    for i in range(N_QUBITS):
        qc.ry(w_params[offset + 2 * i], i)
        qc.rz(w_params[offset + 2 * i + 1], i)
    return offset + 2 * N_QUBITS


# --OPTION--
# Entanglement topology. A CNOT ring is the baseline; linear, all-to-all and
# hardware-native couplings are all reasonable alternatives.
def build_entanglement(qc):
    for i in range(N_QUBITS):
        qc.cx(i, (i + 1) % N_QUBITS)


# --OPTION--
# Circuit assembly. Depth lives here.
N_LAYERS = 2

def build_circuit():
    n_weights = N_LAYERS * 2 * N_QUBITS
    x_params = ParameterVector("x", N_QUBITS)
    w_params = ParameterVector("w", n_weights)

    qc = QuantumCircuit(N_QUBITS)
    build_feature_map(qc, x_params)
    offset = 0
    for _ in range(N_LAYERS):
        offset = build_variational_layer(qc, w_params, offset)
        build_entanglement(qc)
    return qc, list(x_params), list(w_params)


# --OPTION--
# Readout: marginal over READOUT qubits -> class probabilities.
# EXAQC's scheme throws away the |11> mass and renormalizes the rest. That
# discarded amplitude is a known wart, kept here for faithfulness.
def readout_probabilities(probs):
    p = np.asarray(probs[:N_CLASSES], dtype=float)
    total = p.sum()
    if total <= 0:
        return np.full(N_CLASSES, 1.0 / N_CLASSES)
    return p / total


# --OPTION--
# Training the angles. COBYLA is gradient-free and cheap; Adam via
# qiskit-machine-learning + TorchConnector is the EXAQC-faithful alternative.
MAX_ITER = 178   # 178 is where the loss is within 1% of its final value

def train_angles(qc, x_params, w_params, X, y):
    rng = np.random.default_rng(SEED)
    w0 = rng.uniform(0, 2 * np.pi, len(w_params))

    def objective(w):
        return cross_entropy(qc, x_params, w_params, w, X, y)

    res = minimize(objective, w0, method="COBYLA",
                   options={"maxiter": MAX_ITER, "disp": False})
    return res.x


# -- NOTE --
# Entry point. Mutating this block breaks the results contract and the
# individual will score INVALID_FITNESS_MAX.
# -- NOTE --
if __name__ == "__main__":
    main()
