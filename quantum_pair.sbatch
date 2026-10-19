#!/bin/bash
#SBATCH --job-name=qvqc_pair
#SBATCH -p pace-cpu
#SBATCH -t 14:00:00
#SBATCH --mem 32G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/qvqc_pair-%x-%j.out
# One trial of the RAG vs no-RAG study in a single job (2026-10-18): build and evaluate
# a shared generation 0 for LLMGE_RANDOM_SEED=$SEED, then run both arms from it side by
# side. One job instead of three keeps the per-user submit cap (50) for the LLM and
# evaluation jobs the arms submit. Each step runs quantum_island.sbatch as a script, so
# the run directories, logs and resume behaviour are the same as separate island jobs:
#     sbatch -J pair_s2 --export=ALL,SEED=2,LLMGE_HOSTNAME_FILE=$PWD/run_job_outputs/server/hostname_A.log quantum_pair.sbatch
# Resubmitting with the same SEED resumes: the init is skipped once its generation-0
# checkpoint exists and each arm continues from its last checkpoint.
# Optional: TAG (run-name suffix, default s$SEED), CANCEL_JOBS (e.g. this lane's LLM
# servers, cancelled when both arms have ended).
cd "$SLURM_SUBMIT_DIR"
: "${SEED:?set SEED}"
TAG=${TAG:-s$SEED}
export LLMGE_RANDOM_SEED=$SEED
# The study's three objectives. Set any other list in the submitting shell, not in
# --export: sbatch --export splits on commas, so "QVQC_OBJECTIVES=ce,wgates" arrives as "ce".
export QVQC_OBJECTIVES=${QVQC_OBJECTIVES:-ce,wgates,cesd}
LOGS=run_job_outputs/islands
echo "pair $TAG seed=$SEED objectives=${QVQC_OBJECTIVES:-ce,wgates} server file=${LLMGE_HOSTNAME_FILE:-hostname.log} host=$(hostname)"

if [ ! -f "quantum_runs/qvqc_init_$TAG/checkpoints/checkpoint_gen_0.pkl" ]; then
    RUN=qvqc_init_$TAG LLMGE_INIT_ONLY=1 LLMGE_QISKIT_RAG=0 \
        bash quantum_island.sbatch >> "$LOGS/qvqc_pair-init_$TAG-$SLURM_JOB_ID.out" 2>&1 || exit 1
fi
RUN=qvqc_rag_$TAG INIT_FROM=qvqc_init_$TAG LLMGE_QISKIT_RAG=1 \
    bash quantum_island.sbatch >> "$LOGS/qvqc_pair-rag_$TAG-$SLURM_JOB_ID.out" 2>&1 &
RAG=$!
RUN=qvqc_norag_$TAG INIT_FROM=qvqc_init_$TAG LLMGE_QISKIT_RAG=0 \
    bash quantum_island.sbatch >> "$LOGS/qvqc_pair-norag_$TAG-$SLURM_JOB_ID.out" 2>&1 &
NORAG=$!
wait $RAG; echo "rag arm exited $?"
wait $NORAG; echo "no-rag arm exited $?"
for j in $CANCEL_JOBS; do scancel "$j" && echo "cancelled $j"; done
