#!/bin/bash
#SBATCH -J channel_flow
#SBATCH -p batch
#SBATCH -w cpu02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH -o results/%x_%j.out
#SBATCH -e results/%x_%j.err

module purge
module load nvhpc/23.7

# Read NP from input.dat
NP=$(grep -E '^\s*NP\s' channel/input.dat | awk '{print $2}')
NP=${NP:-1}  # default to 1 if not found

echo "Running channel flow with $NP MPI processes"
cd /shared/home/wel1come1234/workspace/TDMA/PaScaL_TDMAv2/channel && \
    mpirun -np $NP ../build/bin/channel.out
