#!/bin/bash
#SBATCH -J heat_gpu_2gpu_512
#SBATCH -p batch
#SBATCH -w gpu02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH -t 60:00
#SBATCH -o /shared/home/wel1come1234/workspace/PaScaL_TDMA_C/heat_gpu/results/%x_%j.out
#SBATCH -e /shared/home/wel1come1234/workspace/PaScaL_TDMA_C/heat_gpu/results/%x_%j.err

module purge
module load nvhpc/23.7
source /opt/nvidia/hpc-sdk/Linux_x86_64/23.7/comm_libs/12.2/hpcx/hpcx-2.15/hpcx-init.sh
hpcx_load

cd /shared/home/wel1come1234/workspace/PaScaL_TDMA_C
mkdir -p heat_gpu/results

mpirun -np 2 ./build/bin/heat_gpu.out ./run/heat_gpu_order/PARA_INPUT_2gpu_512.txt
