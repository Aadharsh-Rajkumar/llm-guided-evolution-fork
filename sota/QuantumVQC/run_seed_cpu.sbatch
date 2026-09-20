#!/bin/bash
#SBATCH --job-name=vqc_seed
#SBATCH -t 00:20:00
#SBATCH -A coc
#SBATCH -q coe-ice
#SBATCH -p pace-cpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=8G
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# CPU-only on purpose. The seed is exact statevector simulation on 4 qubits -
# 16 amplitudes. It needs no GPU, and queueing it behind GPU demand would cost
# hours per generation for nothing. Do NOT copy the GPU template from
# PYTHON_BASH_SCRIPT_TEMPLATE for this seed.

echo "host: $(hostname)"
echo "start: $(date)"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_guided_env

cd "$SLURM_SUBMIT_DIR"
python test_seed_blocks.py || { echo "BLOCK STRUCTURE TEST FAILED"; exit 1; }
echo
python seed_vqc.py --gene-id pace_smoke --out-dir results

echo "end: $(date)"
