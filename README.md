## LLM-Guided Evolution for Variational Quantum Classifiers

<p align="left">
  <img src="./assets/header.png" alt="">
</p>

This branch uses LLM-Guided Evolution (LLM-GE) to evolve Qiskit variational quantum
circuits for Breast Cancer Wisconsin classification. Llama-3.3-70B edits one block of
the seed circuit at a time (feature map, variational layer, entanglement, circuit
assembly, readout, training), and NSGA-II or MAP-Elites selects on validation
cross-entropy, live weighted gate count and the spread of cross-entropy across training
starts.

### Layout

- `sota/QuantumVQC/` - the seed circuit (`network.py`, `seed_vqc.py`) and its evaluation adapter (`train.py`)
- `src/` - mutation and crossover jobs, the Qiskit RAG retriever (`qiskit_rag.py`), the circuit view (`vlm_utils.py`) and LLM utilities
- `src/cfg/constants_QuantumVQC.py` - run settings (`src/cfg/constants.py` links to it)
- `docs/qiskit_rag/` - the notes RAG retrieves from; `docs/rag_placebo/` - neutral notes for the placebo control
- `templates/QuantumVQC/`, `templates/CrossOver/` - mutation and crossover prompts
- `server.py`, `server.sh` - the Llama server
- `quantum_island.sbatch`, `quantum_pair.sbatch` - one run, or one seed with several arms from a shared generation 0
- `tests/` - unit tests for the harness

### Switches

| variable | effect |
|---|---|
| `LLMGE_QISKIT_RAG` | Qiskit RAG on (1) or off (0) |
| `LLMGE_RAG_*` | one flag per RAG component; `LLMGE_RAG_PLACEBO=1` swaps the notes for neutral text of the same length |
| `LLMGE_CIRCUIT_VIEW` | `off`, `ascii` or `gemma` view of the parent circuit |
| `LLMGE_QD_MODE`, `LLMGE_QD_DEDUP` | MAP-Elites parent selection and duplicate skipping |
| `LLMGE_LOCAL_JOBS`, `LLMGE_LOCAL_EVAL` | run LLM calls and evaluations inside the run's own Slurm job |

### Paper

The method follows "LLM Guided Evolution - The Automation of Models Advancing Models"
(`assets/paper/`).
