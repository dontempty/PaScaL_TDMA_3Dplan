#!/bin/bash
#SBATCH -J heat_strong_8
#SBATCH -p amd_a100nv_8
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --comment etc

# ============================================================
#  PaScaL_TDMA heat_gpu — strong scaling (rho=0.25, N=512^3)
#  Runs NP=8 on an 8-GPU single-node allocation (long queue wait).
#  Submit from /scratch/x3319a05/PaScaL_TDMA :
#      sbatch heat_gpu/run_strong_8.sh
# ============================================================
set -e

module purge
module load nvhpc/25.11_cuda12

export UCX_MEMTYPE_CACHE=n
export CUDA_LIBDIR="${NVHPC_ROOT}/cuda/lib64"

if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
PROJ="$(pwd)"
mkdir -p log results

echo "=============================================="
echo " host    : $(hostname)"
echo " date    : $(date '+%F %T')"
echo " job_id  : ${SLURM_JOB_ID:-interactive}"
echo " gpus    : $(nvidia-smi -L 2>/dev/null | wc -l) visible"
echo "=============================================="

# --- Build (cheap if already built) --------------------------------------
echo "[build] make lib (USE_CUDA=1 CUDA_ARCH=80)"
USE_CUDA=1 CUDA_ARCH=80 make lib
echo "[build] make -C heat_gpu"
USE_CUDA=1 CUDA_ARCH=80 make -C heat_gpu

EXE="${PROJ}/build/bin/heat_gpu.out"
RUN_DIR="${PROJ}/run/strong_0.25"
LOG="${PROJ}/results/strong_8_${SLURM_JOB_ID:-local}.txt"

[ -x "${EXE}" ] || { echo "[ERROR] missing binary ${EXE}" >&2; exit 1; }

NP=8
INP="${RUN_DIR}/PARA_INPUT_${NP}.txt"
[ -f "${INP}" ] || { echo "[ERROR] missing input ${INP}" >&2; exit 1; }

unset TIMING_CSV   # let the binary auto-name: results/timing_<N>_<npxnpynpz>.csv

echo "================================================================" | tee -a "${LOG}"
echo " Strong scaling (rho=0.25, N=512^3), NP=${NP}"                    | tee -a "${LOG}"
echo " EXE  : ${EXE}"                                                   | tee -a "${LOG}"
echo " INP  : ${INP}"                                                   | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"

T0=$(date +%s)
mpirun --bind-to none -np ${NP} "${EXE}" "${INP}" 2>&1 | tee -a "${LOG}"
T1=$(date +%s)
echo "[wall] NP=${NP}  $((T1-T0))s" | tee -a "${LOG}"

echo "[done] strong scaling NP=${NP} complete, log: ${LOG}"
