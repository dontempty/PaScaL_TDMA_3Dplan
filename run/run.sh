#!/bin/bash
#SBATCH -J test_kdh
#SBATCH -p batch
#SBATCH -w cpu05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -o results/%x_%j.out
#SBATCH -e results/%x_%j.err
#SBATCH --comment xxx

module purge 
module load nvhpc/23.7

mpirun -np 64 ./a.out ./PARA_INPUT.txt