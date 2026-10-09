#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C "H100"
# GPUs on these nodes reported "busy or unavailable" on 2026-10-03 (quantum-seed run 6060968).
#SBATCH --exclude=atl1-1-03-010-10-0,atl1-1-03-011-13-0,atl1-1-03-011-18-0
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --output=run_job_outputs/server/slurm-%j.out
echo "launching LLM Server"

# Optional chained submission count to work around walltime limits
COUNT=${1:-1}

hostname

module load cuda
module load uv

# Make sure CUDA can see all GPUs
export CUDA_VISIBLE_DEVICES=0,1
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE=$(pwd)"/hostname.log"


# Log the island controller setting for debugging
echo "SUBMIT_ISLAND_CONTROLLER=${SUBMIT_ISLAND_CONTROLLER:-<not set>}"

# Default behavior: START island controller unless explicitly disabled
if [ "${SUBMIT_ISLAND_CONTROLLER:-1}" = "1" ]; then
    # Submit the paired island-controller job from here so the two stay in sync.
    echo "Submitting island controller (count=$COUNT)"
    sbatch island_controller.sbatch "$COUNT" "$SLURM_JOB_ID"
else
    echo "Skipping island controller submission (SUBMIT_ISLAND_CONTROLLER=${SUBMIT_ISLAND_CONTROLLER})"
fi

# Publish the hostname only once the model is loaded and answering. A relay server
# queued to start before this one's 8 h GPU limit (QOSMaxGRESMinutesPerJob) then
# takes over without clients ever being pointed at a server that is still loading:
#     sbatch --begin=now+7hours --export=ALL,SUBMIT_ISLAND_CONTROLLER=0 server.sh
uv run python -m uvicorn server:app --host $SERVER_HOSTNAME --port 8169 --workers 1 &
SERVER_PID=$!
until curl -sf "http://$SERVER_HOSTNAME:8169/" > /dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "server exited before becoming ready"; exit 1; fi
    sleep 15
done
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Server ready; wrote hostname to $HOSTNAME_FILE"
wait $SERVER_PID
