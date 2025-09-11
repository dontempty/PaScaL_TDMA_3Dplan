#!/bin/bash
#SBATCH -J test_kdh
#SBATCH -p batch
#SBATCH -w cpu02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # 물리 코어 수만큼 랭크
#SBATCH -o results/%x_%j.out
#SBATCH -e results/%x_%j.err
#SBATCH --comment xxx

module purge 
module load nvhpc/23.7

mpirun -np 1 ./a.out ./PARA_INPUT.txt