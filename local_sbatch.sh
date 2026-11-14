#!/bin/bash
script=$1
[ -f "$script" ] || { echo "local_sbatch: no such script: $script" >&2; exit 1; }
id="$(date +%s%N)$((RANDOM % 10))"
out=$(sed -n 's/^#SBATCH --output=//p' "$script" | head -1)
out=${out:-run_job_outputs/evolution/slurm-%j.out}
out=${out//%j/$id}
name=$(sed -n 's/^#SBATCH --job-name=//p' "$script" | head -1)
limit=$(sed -n 's/^#SBATCH -t //p' "$script" | head -1)
IFS=: read -r h m s <<< "${limit:-1:00:00}"
secs=$(( 10#${h:-1} * 3600 + 10#${m:-0} * 60 + 10#${s:-0} ))
[ "$name" != llm_oper ] && [ "${LLMGE_LOCAL_EVAL:-1}" = 0 ] && exec sbatch "$script"
case "$name" in
    llm_oper) kind=llm; slots=${LLMGE_LOCAL_LLM_SLOTS:-16} ;;
    *) kind=eval; slots=${LLMGE_LOCAL_EVAL_SLOTS:-$(( ${SLURM_CPUS_PER_TASK:-8} > 4 ? ${SLURM_CPUS_PER_TASK:-8} - 2 : 2 ))} ;;
esac
dir=${LLMGE_LOCAL_SLOT_DIR:-${TMPDIR:-/tmp}/llmge_slots_${SLURM_JOB_ID:-$USER}}/$kind
mkdir -p "$dir" "$(dirname "$out")"
(
    exec < /dev/null > /dev/null 2>&1
    while :; do
        for i in $(seq 1 "$slots"); do
            exec 9> "$dir/$i"
            flock -n 9 && break 2
            exec 9>&-
        done
        sleep 2
    done
    echo "local job $id ($kind slot $i) on $(hostname) at $(date +%FT%T)" > "$out"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        timeout --signal=TERM --kill-after=30 "$secs" bash "$script" >> "$out" 2>&1 9>&-
    rc=$?
    [ "$rc" -eq 124 ] && echo "CANCELLED AT $(date +%FT%T) DUE TO TIME LIMIT" >> "$out"
    echo "Begin Slurm Epilog: local job exit $rc at $(date +%FT%T)" >> "$out"
) &
disown
echo "Submitted batch job $id"
