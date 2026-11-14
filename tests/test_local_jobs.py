import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

RUNNER = ROOT / "local_sbatch.sh"


def job(tmp_path, name, body, limit="0:00:30", kind="evaluateGene"):
    path = tmp_path / f"{name}.sh"
    path.write_text(f"#!/bin/bash\n#SBATCH --job-name={kind}\n#SBATCH -t {limit}\n"
                    f"#SBATCH --output={tmp_path}/slurm-%j.out\n{body}\n")
    return path


def submit(tmp_path, path, slots="2"):
    env = {"PATH": "/usr/bin:/bin", "LLMGE_LOCAL_EVAL_SLOTS": slots, "LLMGE_LOCAL_SLOT_DIR": str(tmp_path / "slots")}
    start = time.time()
    out = subprocess.run([str(RUNNER), str(path)], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and time.time() - start < 5
    return out.stdout.split("job ")[-1].strip()


def wait_for(path, text, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if path.exists() and text in path.read_text():
            return path.read_text()
        time.sleep(0.5)
    raise AssertionError(f"{text!r} never appeared in {path}")


def test_runner_returns_an_id_and_writes_the_slurm_style_log(tmp_path):
    jid = submit(tmp_path, job(tmp_path, "ok", "echo job done"))
    log = wait_for(tmp_path / f"slurm-{jid}.out", "Begin Slurm Epilog")
    assert "job done" in log and "exit 0" in log


def test_runner_reports_time_limits_like_slurm(tmp_path):
    jid = submit(tmp_path, job(tmp_path, "slow", "sleep 20", limit="0:00:01"))
    log = wait_for(tmp_path / f"slurm-{jid}.out", "Begin Slurm Epilog")
    assert "DUE TO TIME LIMIT" in log


def test_runner_never_runs_more_jobs_than_slots(tmp_path):
    ids = [submit(tmp_path, job(tmp_path, f"j{i}", "date +%s.%N; sleep 2; date +%s.%N"), slots="1") for i in range(3)]
    spans = []
    for jid in ids:
        log = wait_for(tmp_path / f"slurm-{jid}.out", "Begin Slurm Epilog", timeout=30)
        stamps = [float(line) for line in log.splitlines() if line[:1].isdigit()]
        spans.append((stamps[0], stamps[1]))
    spans.sort()
    assert all(b[0] >= a[1] for a, b in zip(spans, spans[1:]))


def test_server_pool_spreads_requests(tmp_path, monkeypatch):
    import llm_utils
    pool = tmp_path / "pool.log"
    pool.write_text("host-a:8201\n\nhost-b:8202\n")
    monkeypatch.setattr(llm_utils, "HOSTNAME_DIR", str(pool))
    assert {llm_utils.get_llm_server_hostname() for _ in range(60)} == {"host-a:8201", "host-b:8202"}
    pool.write_text("host-a:8201\n")
    assert llm_utils.get_llm_server_hostname() == "host-a:8201"
