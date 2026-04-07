#!/bin/bash
#SBATCH -J filtered_tdma
#SBATCH -p batch
#SBATCH -w cpu02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH -o results/%x_%j.out
#SBATCH -e results/%x_%j.err

module purge
module load nvhpc/23.7

mkdir -p results
mpirun -np 8 ./build/bin/a.out ./run/PARA_INPUT.txt
