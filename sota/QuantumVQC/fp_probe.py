"""Is the divergence floating-point/architecture level, below printed precision?"""
import importlib.util, platform, sys
import numpy as np

spec = importlib.util.spec_from_file_location("seed", "seed_vqc.py")
m = importlib.util.module_from_spec(spec); sys.modules["seed"] = m
sys.argv = ["seed"]; spec.loader.exec_module(m)

print(f"machine   {platform.machine()}")
print(f"processor {platform.processor() or 'n/a'}")
print(f"system    {platform.system()}")
try:
    import numpy.__config__ as cfg
    blas = cfg.get_info("blas_opt") if hasattr(cfg, "get_info") else {}
    print(f"blas      {blas.get('libraries', numpy.show_config and 'see below')}")
except Exception:
    pass
print(f"simd      {np.show_runtime and ''}", end="")
try:
    rt = np.show_runtime
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rt()
    line = [l for l in buf.getvalue().splitlines() if "found" in l.lower() or "baseline" in l.lower()]
    print(" | ".join(x.strip()[:90] for x in line[:2]))
except Exception:
    print("n/a")

X_tr, X_va, X_te, y_tr, y_va, y_te = m.load_data()
qc, xp, wp = m.build_circuit()
w0 = np.random.default_rng(m.SEED).uniform(0, 2*np.pi, len(wp))

# FULL precision - repr() of a float shows all 17 significant digits
loss0 = m.cross_entropy(qc, xp, wp, w0, X_tr, y_tr)
print()
print(f"loss at w0 (full)  {loss0!r}")
print(f"loss at w0 (hex)   {loss0.hex()}")

p = m.forward(qc, xp, wp, X_tr[0], w0)
print(f"forward[0] (hex)   {[x.hex() for x in p]}")
