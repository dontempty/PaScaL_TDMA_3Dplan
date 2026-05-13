#!/bin/bash
#SBATCH -J heat_debug_ucx
#SBATCH -p amd_a100nv_8
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --comment etc

# ============================================================
#  MPI / UCX transport diagnostic for heat_gpu solver MPI.
#  Goal: confirm whether device-pointer MPI_Alltoallv is using
#  cuda_ipc / cuda_copy fast path or falling back to host staging.
# ============================================================
set -e

module purge
module load nvhpc/25.11_cuda12
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
export CUDA_LIBDIR="${NVHPC_ROOT}/cuda/lib64"

# Verbosity — print which transport UCX selects for each peer pair.
export UCX_LOG_LEVEL=info
export OMPI_MCA_pml_ucx_verbose=3
export OMPI_MCA_pml=ucx

if [ -n "${SLURM_SUBMIT_DIR}" ]; then cd "${SLURM_SUBMIT_DIR}"; fi
PROJ="$(pwd)"
mkdir -p log results

echo "=============================================="
echo " host    : $(hostname)"
echo " job_id  : ${SLURM_JOB_ID}"
echo " gpus    : $(nvidia-smi -L | wc -l) visible"
echo "=============================================="

# --- 1) UCX available transports ---
echo
echo "================ ucx_info -d ================="
ucx_info -d 2>&1 | head -100 || true

# --- 2) Quick build (incremental) ---
USE_CUDA=1 CUDA_ARCH=80 make lib       > /dev/null
USE_CUDA=1 CUDA_ARCH=80 make -C heat_gpu > /dev/null

EXE="${PROJ}/build/bin/heat_gpu.out"
INP="${PROJ}/run/strong_balanced/PARA_INPUT_221.txt"   # NP=4 (2,2,1)

echo
echo "================ heat_gpu NP=4 (verbose) ================="
# Single timestep so the log is short — kill via Tmax=2*dt
# (but our PARA_INPUT_221 has Tmax=0.05/dt=0.001 → 50 steps; that's fine for short test)
mpirun --bind-to none -np 4 "${EXE}" "${INP}" 2>&1 | head -200

echo "[done]"
