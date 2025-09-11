#!/bin/bash
#SBATCH -J test_kdh
#SBATCH -p batch
#SBATCH -w cpu01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2      # 코어 64개 = 랭크 64개
#SBATCH --cpus-per-task=1         # 코어 1개를 각 랭크에 할당
#SBATCH --ntasks-per-core=1
#SBATCH --hint=multithread        # 코어의 HT 모두 허용
#SBATCH -o results/%x_%j.out
#SBATCH -e results/%x_%j.err
#SBATCH --comment xxx

module purge 
module load nvhpc/23.7

# OpenMP: 코어의 2 HW 스레드에 배치
export OMP_NUM_THREADS=2
export OMP_PLACES=cores
export OMP_PROC_BIND=close

srun -n 2 --cpus-per-task=2 --cpu-bind=cores ./a.out ./PARA_INPUT.txt