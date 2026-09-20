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


def weighted_gate_cost(qc):
    """Objective 2. Hardware-weighted gate count, not raw depth."""
    cost = 0.0
    for inst in qc.data:
        if inst.operation.name in ("barrier", "measure"):
            continue
        cost += TWO_QUBIT_COST if inst.operation.num_qubits >= 2 else 1.0
    return cost


def forward(qc, x_params, w_params, x_vals, w_vals):
    """Bind one sample plus the weights and return class probabilities."""
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
    return total / len(X)


def accuracy(qc, xp, wp, w_vals, X, y):
    correct = sum(int(np.argmax(forward(qc, xp, wp, xi, w_vals)) == yi)
                  for xi, yi in zip(X, y))
    return correct / len(X)


def write_results(out_dir, gene_id, obj1, obj2):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{gene_id}_results.txt")
    with open(path, "w") as f:
        f.write(f"{obj1}, {obj2}")
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

    w = train_angles(qc, xp, wp, X_tr, y_tr)

    obj1 = cross_entropy(qc, xp, wp, w, X_va, y_va)
    obj2 = min(weighted_gate_cost(qc), MAX_GATE_COST)

    if not np.isfinite(obj1):
        raise ValueError(f"non-finite validation loss: {obj1}")

    path = write_results(args.out_dir, args.gene_id, obj1, obj2)
    print(f"gene {args.gene_id}")
    print(f"  qubits {qc.num_qubits}  depth {qc.depth()}  params {len(wp)}")
    print(f"  obj1 val cross-entropy {obj1:.4f}   obj2 weighted gate cost {obj2:.1f}")
    print(f"  train acc {accuracy(qc, xp, wp, w, X_tr, y_tr):.1%}"
          f"   val acc {accuracy(qc, xp, wp, w, X_va, y_va):.1%}")

    dead, rotations = count_dead_gates(qc, xp, wp, w)
    print(f"  dead gates {dead}/{rotations} rotations at |angle| < 0.01")

    arms = tuple(a for a in args.repr.split(",") if a)
    if arms:
        for arm, rpath in emit_representations(qc, args.out_dir, args.gene_id, arms).items():
            print(f"  repr [{arm}] -> {rpath}")

    print(f"  wrote {path}   ({time.perf_counter() - t0:.1f}s)")

# ---------------------------------------------------------------------------
# DESIGN NOTES (kept here, above the first marker, so they are never sent to the
# LLM as editable text - the blocks below should present code, not prose)
#
# Encoding      One RY per feature. Data re-uploading, repeating the encoding
#               between variational layers, is a well-established way to raise
#               expressivity and EXAQC does not use it - a candidate edit.
# Entanglement  A CNOT ring is the baseline. Linear, all-to-all and
#               hardware-native couplings are all reasonable alternatives, and
#               they trade accuracy against obj2 (two-qubit gates cost 5x).
# Readout       EXAQC's scheme: marginal over qubits 0,1, first 3 basis states,
#               renormalized. This discards the |11> mass. Faithful, but a wart.
# Training      COBYLA is gradient-free and cheap. Adam via
#               qiskit-machine-learning was measured at 11x the cost for no
#               accuracy gain, so COBYLA stays.
# ---------------------------------------------------------------------------

# --OPTION--
# Encoding: classical features -> rotation angles. Inputs are pre-scaled to [0, pi].
def build_feature_map(qc, x_params):
    for i in range(N_QUBITS):
        qc.ry(x_params[i], i)

# --OPTION--
# Trainable rotations.
def build_variational_layer(qc, w_params, offset):
    for i in range(N_QUBITS):
        qc.ry(w_params[offset + 2 * i], i)
        qc.rz(w_params[offset + 2 * i + 1], i)
    return offset + 2 * N_QUBITS

# --OPTION--
# Entangling topology.
def build_entanglement(qc):
    for i in range(N_QUBITS):
        qc.cx(i, (i + 1) % N_QUBITS)

# --OPTION--
# Circuit assembly and depth.
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
# Readout: measured probabilities -> class probabilities.
def readout_probabilities(probs):
    p = np.asarray(probs[:N_CLASSES], dtype=float)
    total = p.sum()
    if total <= 0:
        return np.full(N_CLASSES, 1.0 / N_CLASSES)
    return p / total

# --OPTION--
# Angle optimisation.
MAX_ITER = 300

def train_angles(qc, x_params, w_params, X, y):
    rng = np.random.default_rng(SEED)
    w0 = rng.uniform(0, 2 * np.pi, len(w_params))

    def objective(w):
        return cross_entropy(qc, x_params, w_params, w, X, y)

    res = minimize(objective, w0, method="COBYLA",
                   options={"maxiter": MAX_ITER, "disp": False})
    return res.x


if __name__ == "__main__":
    main()

