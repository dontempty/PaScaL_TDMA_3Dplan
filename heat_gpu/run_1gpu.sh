#!/bin/bash
#SBATCH -J heat_gpu_1gpu
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH -o /shared/home/wel1come1234/workspace/PaScaL_TDMA_C/heat_gpu/results/%x_%j.out
#SBATCH -e /shared/home/wel1come1234/workspace/PaScaL_TDMA_C/heat_gpu/results/%x_%j.err

module purge
module load nvhpc/23.7
source /opt/nvidia/hpc-sdk/Linux_x86_64/23.7/comm_libs/12.2/hpcx/hpcx-2.15/hpcx-init.sh
hpcx_load

cd /shared/home/wel1come1234/workspace/PaScaL_TDMA_C
mkdir -p heat_gpu/results

# Use CUDA-aware MPI
export OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
export UCX_MEMTYPE_CACHE=n

EXE=./build/bin/heat_gpu.out
RUN=./run/heat_gpu_order

for N in 64 128 256 512; do
    echo "================================================================"
    echo "  1-GPU run: Nx=Ny=Nz=$N"
    echo "================================================================"
    mpirun -np 1 $EXE $RUN/PARA_INPUT_1gpu_$N.txt
done
