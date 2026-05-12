#!/bin/bash
#SBATCH -J pascal_gpu_conv
#SBATCH -p amd_a100nv_8
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --comment etc

# ============================================================
#  PaScaL_TDMA GPU convergence (rho=0.25, dt=dx^2, T_final fixed)
#  Sweep: 4 decompositions × 4 grid sizes = 16 runs in one job.
#
#  Submit from /scratch/x3319a05/PaScaL_TDMA :
#      sbatch heat_gpu/run_convergence_a100.sh
# ============================================================

set -e

# --- KISTI Neuron environment ---------------------------------------------
# nvhpc ships its own HPC-X OpenMPI + mpicxx + nvcc, so loading only nvhpc
# is sufficient (loading cudampi/* in addition fails because it expects an
# external UCX prefix that is not set on this node).
module purge
module load nvhpc/25.11_cuda12

# CUDA-aware MPI is enabled by default in the HPC-X bundled OpenMPI.
export UCX_MEMTYPE_CACHE=n

# CUDA runtime lib path (nvhpc provides libcudart but the link rule defaults
# to /usr/local/cuda/lib64 which does not exist on this node).
export CUDA_LIBDIR="${NVHPC_ROOT}/cuda/lib64"

# --- Resolve project root --------------------------------------------------
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
echo " modules : $(module list 2>&1 | tail -n +2)"
echo " cwd     : $PROJ"
echo "=============================================="

# --- Build ----------------------------------------------------------------
echo "[build] make lib (USE_CUDA=1 CUDA_ARCH=80)"
USE_CUDA=1 CUDA_ARCH=80 make lib
echo "[build] make -C heat_gpu"
USE_CUDA=1 CUDA_ARCH=80 make -C heat_gpu

EXE="${PROJ}/build/bin/heat_gpu.out"
RUN_DIR="${PROJ}/run/heat_gpu_order_0.25"
RESULT="${PROJ}/results/convergence_${SLURM_JOB_ID:-local}.txt"

[ -x "${EXE}" ] || { echo "[ERROR] missing binary ${EXE}" >&2; exit 1; }
echo
echo "================================================================"
echo " Convergence sweep (rho=0.25, T_final=Tmax fixed, dt=dx^2)"
echo " EXE  : ${EXE}"
echo " RUN  : ${RUN_DIR}"
echo " LOG  : ${RESULT}"
echo "================================================================"
: > "${RESULT}"

# --- Sweep: 4 decompositions × 4 grid sizes ------------------------------
for LABEL in 1gpu 2gpu 4gpu 8gpu; do
    case "${LABEL}" in
        1gpu) NP=1 ;;  2gpu) NP=2 ;;  4gpu) NP=4 ;;  8gpu) NP=8 ;;
    esac

    for N in 64 128 256 512; do
        INP="${RUN_DIR}/PARA_INPUT_${LABEL}_${N}.txt"
        if [ ! -f "${INP}" ]; then
            echo "[WARN] missing ${INP}, skip" | tee -a "${RESULT}"
            continue
        fi

        echo                                                       | tee -a "${RESULT}"
        echo "=== ${LABEL}  (NP=${NP})  N=${N} ==="                | tee -a "${RESULT}"
        T0=$(date +%s)
        mpirun --bind-to none -np ${NP} "${EXE}" "${INP}" 2>&1 | tee -a "${RESULT}"
        T1=$(date +%s)
        echo "[time] ${LABEL} N=${N}  ${SECONDS}s elapsed, this run $((T1-T0))s" \
            | tee -a "${RESULT}"
    done
done

echo
echo "[done] all 16 runs complete, log: ${RESULT}"
