#!/bin/bash
#SBATCH -J heat_balanced_8
#SBATCH -p amd_a100nv_8
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --comment etc

# ============================================================
#  Balanced strong scaling: NP=8 (2,2,2)
#  Grid N=512^3, rho=0.25
#  Submit from /scratch/x3319a05/PaScaL_TDMA :
#      sbatch heat_gpu/run_balanced_8.sh
# ============================================================
set -e

module purge
module load nvhpc/25.11_cuda12
# Force UCX to use CUDA transports (cuda_copy, cuda_ipc) — without this, OpenMPI
# may fall back to host-staged paths on device pointers for collectives.
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
export CUDA_LIBDIR="${NVHPC_ROOT}/cuda/lib64"

if [ -n "${SLURM_SUBMIT_DIR}" ]; then cd "${SLURM_SUBMIT_DIR}"; fi
PROJ="$(pwd)"

RESDIR="${PROJ}/results/${SLURM_JOB_ID:-local}"
mkdir -p log "${RESDIR}"

echo "[build] make clean (always — avoid stale artifacts from prior runs)"
USE_CUDA=1 CUDA_ARCH=80 make clean
echo "[build] make lib (USE_CUDA=1 CUDA_ARCH=80)"
USE_CUDA=1 CUDA_ARCH=80 make lib
echo "[build] make -C heat_gpu"
USE_CUDA=1 CUDA_ARCH=80 make -C heat_gpu

EXE="${PROJ}/build/bin/heat_gpu.out"
RUN_DIR="${PROJ}/run/strong_balanced"
LOG="${RESDIR}/run.txt"

NP=8
TAG=222
INP="${RUN_DIR}/PARA_INPUT_${TAG}.txt"
[ -f "${INP}" ] || { echo "[ERROR] missing ${INP}" >&2; exit 1; }

export TIMING_CSV="${RESDIR}/timing_513_${TAG}.csv"

echo "================================================================" | tee -a "${LOG}"
echo " Balanced strong scaling (N=512^3, rho=0.25), NP=${NP} (${TAG})" | tee -a "${LOG}"
echo " RESDIR=${RESDIR}"                                               | tee -a "${LOG}"
echo " CSV   =${TIMING_CSV}"                                           | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"

T0=$(date +%s)
mpirun --bind-to none -np ${NP} "${EXE}" "${INP}" 2>&1 | tee -a "${LOG}"
T1=$(date +%s)
echo "[wall] NP=${NP} decomp=${TAG}  $((T1-T0))s" | tee -a "${LOG}"

echo "[done] balanced NP=${NP} complete, log: ${LOG}"
