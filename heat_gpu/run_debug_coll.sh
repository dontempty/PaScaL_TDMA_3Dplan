#!/bin/bash
#SBATCH -J heat_debug_coll
#SBATCH -p amd_a100nv_8
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --comment etc

set -e
module purge
module load nvhpc/25.11_cuda12
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
export CUDA_LIBDIR="${NVHPC_ROOT}/cuda/lib64"

# Force UCC (GPU-aware collective backend) and disable hcoll.
# UCC typically has a CUDA-optimized Alltoallv path that goes through
# cuda_ipc directly, unlike OpenMPI's coll/tuned which is host-tuned.
export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_coll_ucc_priority=100
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_pml=ucx

# Tell UCC to also use CUDA transports.
export UCC_TLS=cuda,nccl,ucp
export UCC_CL_BASIC_TLS=cuda,nccl,ucp

# Verbose so we can confirm path
export UCC_LOG_LEVEL=info

if [ -n "${SLURM_SUBMIT_DIR}" ]; then cd "${SLURM_SUBMIT_DIR}"; fi
PROJ="$(pwd)"
mkdir -p log results

USE_CUDA=1 CUDA_ARCH=80 make lib       > /dev/null
USE_CUDA=1 CUDA_ARCH=80 make -C heat_gpu > /dev/null

EXE="${PROJ}/build/bin/heat_gpu.out"
INP="${PROJ}/run/strong_balanced/PARA_INPUT_221.txt"

echo "================ ompi_info coll ================="
ompi_info --param coll all --level 9 2>&1 | grep -E "^\s*MCA coll" | head -30 || true

echo "================ heat_gpu NP=4 with UCC ================="
T0=$(date +%s)
mpirun --bind-to none -np 4 "${EXE}" "${INP}" 2>&1 | tee /tmp/heat_ucc_$$.log | head -40
T1=$(date +%s)
echo "[wall] NP=4 with UCC: $((T1-T0))s"

# Show step 25 timing
echo
echo "================ NP=4 step 25 timing (rank 0) ================="
awk -F',' '$1=="0" && $2=="25"' "${PROJ}/results/timing_513_221.csv" 2>/dev/null

echo "[done]"
