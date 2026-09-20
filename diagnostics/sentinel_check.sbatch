#!/bin/bash
#SBATCH --job-name=evaluateGene
#SBATCH -t 8:00:00
#SBATCH -A coc
#SBATCH -q coe-ice
#SBATCH -p coe-gpu
#SBATCH --gres=gpu:1
#SBATCH -C "H100"
# Pinned to ONE microarchitecture on purpose. coe-gpu spans H100 (Sapphire
# Rapids) and H200 nodes; different CPU generations dispatch different SIMD
# kernels in numpy/BLAS, which changes floating-point rounding, which sends
# COBYLA down a different path. Measured spread across node types was ~1
# validation sample. Letting genomes scatter across both would bake the
# queue into the fitness. 19 H100 nodes available, so this costs nothing.
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
echo "Launching Python Evaluation"
hostname

# Account/QOS/partition are required on ICE - the stock template omitted them and
# every submission failed with "Invalid qos specification". coe-gpu is the GPU
# partition this QOS is allowed on; pace-gpu permits only pace-ice/pace-grade.
module load cuda
export SCRATCH_DIR="/storage/ice1/9/7/rdesai317/scratch.cache"
export UV_CACHE_DIR="$SCRATCH_DIR/uv"
export UV_PYTHON_INSTALL_DIR="$SCRATCH_DIR/uv_python"
source /home/hice1/rdesai317/scratch/Quantam/llm-guided-evolution-fork/.venv/bin/activate

# Run Python script
python sota/QuantumVQC/train.py -network models.network_sentineltest
