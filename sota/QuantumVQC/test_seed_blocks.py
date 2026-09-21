import subprocess, sys, os

def test_markers():
    with open("network.py") as f:
        n = f.read().count("# --OPTION--")
    assert n == 6, f"expected 6 markers, found {n}"
    print("marker check: OK")

def test_roundtrip():
    r = subprocess.run([sys.executable, "train.py", "-network", "models.network"],
                        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".")
    assert "job done" in r.stdout, r.stdout + r.stderr
    result_file = os.path.join("results", "seed_results.csv")
    assert os.path.exists(result_file), "seed_results.csv not written"
    print("roundtrip check: OK")

if __name__ == "__main__":
    test_markers()
    test_roundtrip()