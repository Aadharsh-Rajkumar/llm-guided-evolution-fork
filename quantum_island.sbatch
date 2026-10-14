#!/bin/bash
#SBATCH --job-name=qvqc_island
#SBATCH -p pace-cpu
#SBATCH -t 16:00:00
#SBATCH --mem 16G
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/qvqc_island-%x-%j.out
# One QuantumVQC evolution island (the orchestrator only; it needs no GPU).
# The LLM server must be started separately:
#     SUBMIT_ISLAND_CONTROLLER=0 sbatch server.sh
#
# RAG vs no-RAG from the SAME initial population (2026-10-13). Step 1 builds and
# evaluates generation 0 once (no RAG) and stops; step 2 copies it into both arms,
# which also share LLMGE_RANDOM_SEED, so they differ only in the RAG prompts:
#     sbatch -J init  --export=ALL,RUN=qvqc_init_s1,LLMGE_INIT_ONLY=1,LLMGE_QISKIT_RAG=0 quantum_island.sbatch
#     sbatch -J rag   -d afterok:<init> --export=ALL,RUN=qvqc_rag_s1,INIT_FROM=qvqc_init_s1,LLMGE_QISKIT_RAG=1 quantum_island.sbatch
#     sbatch -J norag -d afterok:<init> --export=ALL,RUN=qvqc_norag_s1,INIT_FROM=qvqc_init_s1,LLMGE_QISKIT_RAG=0 quantum_island.sbatch
# Another trial = another init RUN name and LLMGE_RANDOM_SEED.
# Resubmitting with the same RUN resumes from that run's last checkpoint.
cd "$SLURM_SUBMIT_DIR"
module load uv
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

RUN=${RUN:-qvqc_rag}
export LLMGE_MIGRATION_GEN=${LLMGE_MIGRATION_GEN:-0}       # no 5-generation island stop
export LLMGE_NUM_GENERATIONS=${LLMGE_NUM_GENERATIONS:-30}
export LLMGE_START_POPULATION=${LLMGE_START_POPULATION:-48}
export LLMGE_POPULATION=${LLMGE_POPULATION:-24}
export LLMGE_NUM_ELITES=${LLMGE_NUM_ELITES:-16}
export LLMGE_RANDOM_SEED=${LLMGE_RANDOM_SEED:-1}
export LLMGE_QISKIT_RAG=${LLMGE_QISKIT_RAG:-1}
CKPT="quantum_runs/$RUN/checkpoints"
GLOBAL="quantum_runs/$RUN/global_data"

if [ -n "$INIT_FROM" ] && ! ls "$CKPT"/checkpoint_gen_*.pkl >/dev/null 2>&1; then
    SRC="quantum_runs/$INIT_FROM"
    if [ ! -f "$SRC/checkpoints/checkpoint_gen_0.pkl" ]; then
        echo "INIT_FROM=$INIT_FROM has no generation-0 checkpoint"; exit 1
    fi
    mkdir -p "$CKPT" "$GLOBAL"
    cp "$SRC/checkpoints/checkpoint_gen_0.pkl" "$CKPT/"
    cp "$SRC/global_data/global_gen_0.pkl" "$GLOBAL/"
    [ -f "$SRC/global_data/failures.jsonl" ] && cp "$SRC/global_data/failures.jsonl" "$GLOBAL/"
    echo "starting from the shared generation 0 of $INIT_FROM"
fi
echo "run=$RUN rag=$LLMGE_QISKIT_RAG gens=$LLMGE_NUM_GENERATIONS pop=$LLMGE_START_POPULATION/$LLMGE_POPULATION" \
     "elites=$LLMGE_NUM_ELITES seed=$LLMGE_RANDOM_SEED init_only=${LLMGE_INIT_ONLY:-0} host=$(hostname)"

.venv/bin/python -u run_improved.py \
    --checkpoints "$CKPT" \
    --global_path "$GLOBAL" \
    --llm_model llama3 \
    --prompt_group QuantumVQC/Normal
