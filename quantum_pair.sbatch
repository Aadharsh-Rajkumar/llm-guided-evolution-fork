#!/bin/bash
#SBATCH --job-name=qvqc_pair
#SBATCH -p pace-cpu,ice-cpu,ice-bw-cpu
#SBATCH -t 18:00:00
#SBATCH --mem 48G
#SBATCH -c 8
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
export LLMGE_LOCAL_JOBS=${LLMGE_LOCAL_JOBS:-1}
# The study's three objectives. Set any other list in the submitting shell, not in
# --export: sbatch --export splits on commas, so "QVQC_OBJECTIVES=ce,wgates" arrives as "ce".
export QVQC_OBJECTIVES=${QVQC_OBJECTIVES:-ce,wlive,cesd}
LOGS=run_job_outputs/islands
echo "pair $TAG seed=$SEED objectives=${QVQC_OBJECTIVES:-ce,wgates} server file=${LLMGE_HOSTNAME_FILE:-hostname.log} host=$(hostname)"
env | grep -E '^(LLMGE_RAG_|LLMGE_LONG_OUTPUT|LLMGE_FUNCTION_REPLY|LLM_OUTPUT_TOKENS|LLMGE_CIRCUIT_VIEW|LLMGE_QD_|LLMGE_OUTPUT_LINE|LLMGE_LOCAL_)' | sort
echo "arms: ${ARMS:-rag norag} merged=${MERGED:-0} view=${MERGED_VIEW:-ascii}"

if [ ! -f "quantum_runs/qvqc_init_$TAG/checkpoints/checkpoint_gen_0.pkl" ]; then
    RUN=qvqc_init_$TAG LLMGE_INIT_ONLY=1 LLMGE_QISKIT_RAG=0 \
        bash quantum_island.sbatch >> "$LOGS/qvqc_pair-init_$TAG-$SLURM_JOB_ID.out" 2>&1 || exit 1
fi
VIEW=${MERGED_VIEW:-ascii}
ARMS=${ARMS:-rag norag$([ "${MERGED:-0}" = "1" ] && echo " merged")}
arm_env() {
    case "$1" in
        baseline)      echo "0 off 0 0 0" ;;
        allon)         echo "1 $VIEW 1 1 0" ;;
        norag)         echo "0 off 0 0 0" ;;
        rag)           echo "1 off 0 0 0" ;;
        placebo)       echo "1 off 0 0 1" ;;
        merged)        echo "1 $VIEW 1 1 0" ;;
        merged_norag)  echo "0 $VIEW 1 1 0" ;;
        merged_noview) echo "1 off 1 1 0" ;;
        merged_noqd)   echo "1 $VIEW 0 0 0" ;;
        *) return 1 ;;
    esac
}
PIDS=()
for arm in $ARMS; do
    spec=$(arm_env "$arm") || { echo "unknown arm $arm"; exit 1; }
    read -r rag view qd dedup placebo <<< "$spec"
    RUN=qvqc_${arm}_$TAG INIT_FROM=qvqc_init_$TAG LLMGE_QISKIT_RAG=$rag LLMGE_RAG_PLACEBO=$placebo \
        LLMGE_CIRCUIT_VIEW=$view LLMGE_QD_MODE=$qd LLMGE_QD_DEDUP=$dedup \
        bash quantum_island.sbatch >> "$LOGS/qvqc_pair-${arm}_$TAG-$SLURM_JOB_ID.out" 2>&1 &
    PIDS+=("$arm:$!")
done
for entry in "${PIDS[@]}"; do
    wait "${entry#*:}"; echo "${entry%%:*} arm exited $?"
done
for j in $CANCEL_JOBS; do scancel "$j" && echo "cancelled $j"; done
