#!/bin/bash
#  PaScaL_TDMA GPU convergence — one NP-decomposition × all 4 grid sizes.
#  Submit with:
#    NP=1  sbatch -p cas_v100nv_4 --gres=gpu:1 -J pascal_gpu_1g  heat_gpu/run_one_np.sh
#    NP=2  sbatch -p cas_v100nv_4 --gres=gpu:2 -J pascal_gpu_2g  heat_gpu/run_one_np.sh
#    NP=4  sbatch -p cas_v100nv_4 --gres=gpu:4 -J pascal_gpu_4g  heat_gpu/run_one_np.sh
#    NP=8  sbatch -p cas_v100nv_8 --gres=gpu:8 -J pascal_gpu_8g  heat_gpu/run_one_np.sh
#
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --comment etc

set -e
module purge
module load nvhpc/25.11_cuda12
export CUDA_LIBDIR="${NVHPC_ROOT}/cuda/lib64"

# --- CUDA-aware MPI (HPC-X / UCX) ----------------------------------------
# Force UCX PML; enable cuda transports so device pointers stay on-GPU
# through Isend/Irecv (no D2H staging). cuda_ipc handles intra-node GPU
# peers; gdr_copy needs GPUDirect RDMA (may not be available on V100 PCIe
# nodes but UCX silently falls back).
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib,uct
export OMPI_MCA_pml_ucx_opal_cuda=1
export OMPI_MCA_opal_cuda_support=true
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc,gdr_copy
export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_THRESH=8192

if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
PROJ="$(pwd)"
mkdir -p log results

: "${NP:?set NP=1|2|4|8 env}"
case "${NP}" in
    1) LABEL=1gpu ;;
    2) LABEL=2gpu ;;
    4) LABEL=4gpu ;;
    8) LABEL=8gpu ;;
    *) echo "[ERR] unsupported NP=${NP}" >&2; exit 1 ;;
esac

echo "=============================================="
echo " host    : $(hostname)"
echo " job_id  : ${SLURM_JOB_ID}"
echo " label   : ${LABEL}  (NP=${NP})"
echo " gpus    : $(nvidia-smi -L | wc -l) visible"
echo "=============================================="

# Build (cheap idempotent rebuild — uses prebuilt obj if up to date).
echo "[build] USE_CUDA=1 CUDA_ARCH=70 make lib"
USE_CUDA=1 CUDA_ARCH=70 make lib >/dev/null
USE_CUDA=1 CUDA_ARCH=70 make -C heat_gpu >/dev/null

EXE="${PROJ}/build/bin/heat_gpu.out"
RUN_DIR="${PROJ}/run/heat_gpu_order_0.25"
RESULT="${PROJ}/results/convergence_${LABEL}_${SLURM_JOB_ID}.txt"
: > "${RESULT}"

for N in 64 128 256 512; do
    INP="${RUN_DIR}/PARA_INPUT_${LABEL}_${N}.txt"
    echo                                                | tee -a "${RESULT}"
    echo "=== ${LABEL}  (NP=${NP})  N=${N} ==="         | tee -a "${RESULT}"
    T0=$(date +%s)
    mpirun --bind-to none -np ${NP} "${EXE}" "${INP}" 2>&1 | tee -a "${RESULT}"
    T1=$(date +%s)
    echo "[time] ${LABEL} N=${N}  $((T1-T0))s"          | tee -a "${RESULT}"
done
echo "[done] -> ${RESULT}"
