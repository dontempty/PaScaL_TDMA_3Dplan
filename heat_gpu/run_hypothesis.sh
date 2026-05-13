#!/bin/bash
#SBATCH -J heat_hyp_test
#SBATCH -p amd_a100nv_8
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --comment etc

# ============================================================
#  Hypothesis test: is NP=4's modified_thomas slowdown
#  (a) GPU under-saturation due to smaller n_sys, OR
#  (b) multi-process host/driver contention?
#
#  Two runs:
#    (1) NP=4 (2,2,1) on 512^3 grid  → solver_x n_sys=131072 (baseline)
#    (2) NP=4 (2,2,1) on 512x512x1024 grid → solver_x n_sys=262144 (matches NP=2)
#
#  If (2) is ~10.7ms → under-saturation. If (2) is ~12.8ms → multi-proc contention.
# ============================================================
set -e

module purge
module load nvhpc/25.11_cuda12
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
export CUDA_LIBDIR="${NVHPC_ROOT}/cuda/lib64"

if [ -n "${SLURM_SUBMIT_DIR}" ]; then cd "${SLURM_SUBMIT_DIR}"; fi
PROJ="$(pwd)"

RESDIR="${PROJ}/results/${SLURM_JOB_ID:-local}"
mkdir -p log "${RESDIR}"

echo "[build] make clean"
USE_CUDA=1 CUDA_ARCH=80 make clean
echo "[build] make lib + heat_gpu"
USE_CUDA=1 CUDA_ARCH=80 make lib
USE_CUDA=1 CUDA_ARCH=80 make -C heat_gpu

EXE="${PROJ}/build/bin/heat_gpu.out"
RUN_DIR="${PROJ}/run/strong_balanced"
LOG="${RESDIR}/run.txt"

echo "================================================================" | tee -a "${LOG}"
echo " Hypothesis test — NP=4 with two grid sizes"                       | tee -a "${LOG}"
echo "   (1) 512^3        : solver_x n_sys=131072"                       | tee -a "${LOG}"
echo "   (2) 512x512x1024 : solver_x n_sys=262144 (matches NP=2)"        | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"

# Run 1: standard 512^3 NP=4 → solver_x n_sys=131072
export TIMING_CSV="${RESDIR}/timing_513_221_small.csv"
echo                                                                     | tee -a "${LOG}"
echo "=== (1) NP=4 standard (512^3) ==="                                 | tee -a "${LOG}"
T0=$(date +%s)
mpirun --bind-to none -np 4 "${EXE}" "${RUN_DIR}/PARA_INPUT_221.txt" 2>&1 | tee -a "${LOG}"
T1=$(date +%s)
echo "[wall] (1) $((T1-T0))s" | tee -a "${LOG}"

# Run 2: bigZ grid NP=4 → solver_x n_sys=262144 (same as NP=2 baseline)
export TIMING_CSV="${RESDIR}/timing_513_221_bigZ.csv"
echo                                                                     | tee -a "${LOG}"
echo "=== (2) NP=4 bigZ (512x512x1024) ==="                              | tee -a "${LOG}"
T0=$(date +%s)
mpirun --bind-to none -np 4 "${EXE}" "${RUN_DIR}/PARA_INPUT_221_bigZ.txt" 2>&1 | tee -a "${LOG}"
T1=$(date +%s)
echo "[wall] (2) $((T1-T0))s" | tee -a "${LOG}"

echo "[done] log: ${LOG}"
