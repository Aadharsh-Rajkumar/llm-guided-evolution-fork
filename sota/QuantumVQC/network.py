"""VQC seed individual — LLM-GE quantum sub-team.
Dataset: Breast Cancer (Wisconsin), 30 features -> 8 via PCA.
"""
import argparse
import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

SEED = 42
N_QUBITS  = 8
N_CLASSES = 2
READOUT   = [0]
MAX_ITER  = 300

def load_data():
    X, y = load_breast_cancer(return_X_y=True)
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_full, y_tr_full, test_size=0.25, random_state=SEED, stratify=y_tr_full)
    pca = PCA(n_components=N_QUBITS, random_state=SEED).fit(X_tr)
    X_tr, X_va, X_te = pca.transform(X_tr), pca.transform(X_va), pca.transform(X_te)
    scaler = MinMaxScaler(feature_range=(0.0, np.pi)).fit(X_tr)
    return (scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te),
            y_tr, y_va, y_te)

def cross_entropy(probs, labels):
    eps = 1e-10
    return float(np.mean([-np.log(probs[i][labels[i]] + eps) for i in range(len(labels))]))

def gate_count(qc):
    one_q = sum(1 for _, qargs, _ in qc.data if len(qargs) == 1)
    two_q = sum(1 for _, qargs, _ in qc.data if len(qargs) == 2)
    return one_q + 5 * two_q

# --OPTION--
def encode_features(qc, x_params):
    for i in range(N_QUBITS):
        qc.ry(x_params[i], i)

# --OPTION--
def variational_layer(qc, w_params, idx):
    for q in range(N_QUBITS):
        qc.ry(w_params[idx], q); idx += 1
        qc.rz(w_params[idx], q); idx += 1
    return idx

# --OPTION--
def entangle(qc):
    for q in range(N_QUBITS):
        qc.cx(q, (q + 1) % N_QUBITS)

# --OPTION--
N_LAYERS = 2
def build_circuit():
    x_params = ParameterVector('x', N_QUBITS)
    w_params = ParameterVector('w', N_LAYERS * N_QUBITS * 2)
    qc = QuantumCircuit(N_QUBITS)
    encode_features(qc, x_params)
    idx = 0
    for _ in range(N_LAYERS):
        idx = variational_layer(qc, w_params, idx)
        entangle(qc)
    return qc, x_params, w_params

# --OPTION--
def readout(sv):
    probs = sv.probabilities(qargs=READOUT)[:N_CLASSES]
    return probs / probs.sum()

# --OPTION--
def optimize(qc, xp, wp, X_tr, y_tr):
    rng = np.random.default_rng(SEED)
    w0 = rng.uniform(-np.pi, np.pi, len(wp))
    def obj(w):
        probs = []
        for x in X_tr:
            bind = dict(zip(xp, x)); bind.update(dict(zip(wp, w)))
            sv = Statevector.from_instruction(qc.assign_parameters(bind))
            probs.append(readout(sv))
        return cross_entropy(probs, y_tr)
    res = minimize(obj, w0, method='COBYLA', options={'maxiter': MAX_ITER})
    return res.x

def evaluate(qc, xp, wp, w_star, X, y):
    probs = []
    for x in X:
        bind = dict(zip(xp, x)); bind.update(dict(zip(wp, w_star)))
        sv = Statevector.from_instruction(qc.assign_parameters(bind))
        probs.append(readout(sv))
    return cross_entropy(probs, y)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-network', type=str, default="network")
    return parser.parse_args()

def main():
    args = get_args()
    try:
        gene_id = args.network.split('network_')[1]
    except IndexError:
        gene_id = 'seed'

    X_tr, X_va, X_te, y_tr, y_va, y_te = load_data()
    qc, xp, wp = build_circuit()
    w_star = optimize(qc, xp, wp, X_tr, y_tr)

    obj1 = evaluate(qc, xp, wp, w_star, X_va, y_va)
    obj2 = gate_count(qc)

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f'{gene_id}_results.csv'), 'w') as f:
        f.write("obj1,obj2\n")
        f.write(f"{obj1},{obj2}\n")

    print(f"obj1={obj1:.4f} obj2={obj2}")
    print("job done")

if __name__ == "__main__":
    main()