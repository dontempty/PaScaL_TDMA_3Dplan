#!/bin/bash
#SBATCH -J heat_balanced_124
#SBATCH -p amd_a100nv_8
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --comment etc

# ============================================================
#  Balanced strong scaling: NP=1 (1,1,1), NP=2 (2,1,1), NP=4 (2,2,1)
#  Grid N=512^3, rho=0.25
#  Submit from /scratch/x3319a05/PaScaL_TDMA :
#      sbatch heat_gpu/run_balanced_124.sh
# ============================================================
set -e

module purge
module load nvhpc/25.11_cuda12
# Match PaScaL_TDMA_F MPI runtime flags exactly (see
# PaScaL_TDMA_F/run/PERFORMANCE_NP1to8.md):
#   --mca pml ucx / --mca osc ucx → force UCX PML/OSC (CUDA-aware path).
#   --mca coll ^hcoll              → disable HCOLL; HCOLL has a known device-
#                                    buffer bug in alltoall that triggers host
#                                    staging (visible in nsys as 1+ GB of D2H
#                                    + H2D per timestep at NP=4).
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_coll=^hcoll
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
export CUDA_LIBDIR="${NVHPC_ROOT}/cuda/lib64"

if [ -n "${SLURM_SUBMIT_DIR}" ]; then cd "${SLURM_SUBMIT_DIR}"; fi
PROJ="$(pwd)"

# Per-jobid result directory (avoids overwriting previous runs).
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

[ -x "${EXE}" ] || { echo "[ERROR] missing binary ${EXE}" >&2; exit 1; }

echo "================================================================" | tee -a "${LOG}"
echo " Balanced strong scaling (N=512^3, rho=0.25)"                     | tee -a "${LOG}"
echo " NP=1 (1,1,1),  NP=2 (2,1,1),  NP=4 (2,2,1)"                      | tee -a "${LOG}"
echo " RESDIR=${RESDIR}"                                                | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"

for TAG in 111 211 221; do
    case "${TAG}" in
        111) NP=1 ;;  211) NP=2 ;;  221) NP=4 ;;
    esac
    INP="${RUN_DIR}/PARA_INPUT_${TAG}.txt"
    [ -f "${INP}" ] || { echo "[WARN] missing ${INP}" | tee -a "${LOG}"; continue; }

    export TIMING_CSV="${RESDIR}/timing_513_${TAG}.csv"
    echo                                                                | tee -a "${LOG}"
    echo "=== balanced NP=${NP}  decomp=${TAG} ==="                     | tee -a "${LOG}"
    echo "    CSV: ${TIMING_CSV}"                                       | tee -a "${LOG}"
    T0=$(date +%s)
    mpirun --bind-to none -np ${NP} "${EXE}" "${INP}" 2>&1 | tee -a "${LOG}"
    T1=$(date +%s)
    echo "[wall] NP=${NP} decomp=${TAG}  $((T1-T0))s" | tee -a "${LOG}"
done

echo "[done] balanced NP=1,2,4 complete, log: ${LOG}"
