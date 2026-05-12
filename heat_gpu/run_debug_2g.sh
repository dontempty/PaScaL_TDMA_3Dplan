#!/bin/bash
#SBATCH -J pascal_dbg
#SBATCH -p cas_v100nv_4
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --comment etc
set -e
module purge
module load nvhpc/25.11_cuda12
export CUDA_LIBDIR="${NVHPC_ROOT}/cuda/lib64"
export UCX_MEMTYPE_CACHE=n
export CUDA_LAUNCH_BLOCKING=1
cd "${SLURM_SUBMIT_DIR}"
EXE="$(pwd)/build/bin/heat_gpu.out"
echo "host=$(hostname) | gpus=$(nvidia-smi -L | wc -l)"
INP="$(pwd)/run/heat_gpu_order_0.25/PARA_INPUT_2gpu_64.txt"
mpirun --bind-to none -np 2 "${EXE}" "${INP}"
