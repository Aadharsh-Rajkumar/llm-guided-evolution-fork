"""Isolate where local and PACE diverge on a supposedly deterministic seed."""
import hashlib, importlib.util, sys
import numpy as np, scipy, sklearn

spec = importlib.util.spec_from_file_location("seed", "seed_vqc.py")
m = importlib.util.module_from_spec(spec); sys.modules["seed"] = m
sys.argv = ["seed"]; spec.loader.exec_module(m)

h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

print(f"numpy   {np.__version__}")
print(f"scipy   {scipy.__version__}")
print(f"sklearn {sklearn.__version__}")
import qiskit; print(f"qiskit  {qiskit.__version__}")
print()

X_tr, X_va, X_te, y_tr, y_va, y_te = m.load_data()
print(f"split sizes      train={len(y_tr)} val={len(y_va)} test={len(y_te)}")
print(f"y_tr hash        {h(y_tr)}")
print(f"y_va hash        {h(y_va)}")
print(f"X_tr hash        {h(np.round(X_tr, 10))}")
print()

qc, xp, wp = m.build_circuit()
w0 = np.random.default_rng(m.SEED).uniform(0, 2*np.pi, len(wp))
print(f"w0 hash          {h(np.round(w0, 12))}")
print(f"w0[:3]           {np.round(w0[:3], 8)}")
print(f"loss at w0       {m.cross_entropy(qc, xp, wp, w0, X_tr, y_tr):.10f}")
print()

w = m.train_angles(qc, xp, wp, X_tr, y_tr)
print(f"final train loss {m.cross_entropy(qc, xp, wp, w, X_tr, y_tr):.10f}")
print(f"final val loss   {m.cross_entropy(qc, xp, wp, w, X_va, y_va):.10f}")
print(f"w_final hash     {h(np.round(w, 8))}")
