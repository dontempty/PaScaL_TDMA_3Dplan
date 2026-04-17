#!/bin/bash
#SBATCH -J test_kdh_all      # Job 이름 (구분을 위해 scaling 추가)
#SBATCH -p batch                 # 파티션 이름
#SBATCH -w cpu06                 # 실행할 노드 이름
#SBATCH --nodes=1                # 항상 1개의 노드만 사용
#SBATCH --ntasks-per-node=16     # 루프에서 사용할 최대 프로세스 개수를 할당
#SBATCH -o results/%x_%j.out     # 표준 출력 파일
#SBATCH -e results/%x_%j.err     # 표준 에러 파일
#SBATCH --comment xxx

# --- 환경 설정 ---
echo "Loading modules..."
module purge 
module load nvhpc/23.7
echo "Modules loaded."
echo ""

# --- 실행할 프로세스 개수 목록 ---
# 이 배열의 값을 수정하여 원하는 테스트 케이스를 실행할 수 있습니다.
PROCESS_COUNTS=(1 2 4 8 16)

RHO="0.25"
BUILD_DIR="build/bin"
# INPUT_DIR="order_para_input"
INPUT_DIR="strong"

# --- 각 프로세스 개수에 대해 루프 실행 ---s
for NP in "${PROCESS_COUNTS[@]}"
do
    echo "========================================="
    echo "RUNNING WITH $NP PROCESSES"s
    echo "========================================="
    
    INPUT_FILE="./run/${INPUT_DIR}_${RHO}/PARA_INPUT_${NP}.txt"
    
    # 실행 전, 해당 프로세스 개수에 맞는 입력 파일이 있는지 확인
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: Input file '$INPUT_FILE' not found. Skipping this run."
        echo ""
        continue # 파일이 없으면 이 단계는 건너뛰고 다음 루프로 이동
    fi

    # mpirun 명령어 실행
    # -np 플래그와 입력 파일 이름을 현재 루프의 $NP 값으로 설정
    mpirun -np $NP ./${BUILD_DIR}/a.out "$INPUT_FILE"
    
    # srun -n $NP \
    #  --ntasks-per-node=$NP \
    #  ./${BUILD_DIR}/a.out "$INPUT_FILE"
    #  # --cpu-bind=map_cpu:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
     

    # Order of accuracy
    # case "$NP" in
    #     1|2|4)
    #         mpirun -np 8 ./${BUILD_DIR}/a.out "$INPUT_FILE"
    #         ;;
    #     8|16)
    #         mpirun -np 16 ./${BUILD_DIR}/a.out "$INPUT_FILE"
    #         ;;
    #     *)
    #         echo "Unsupported NP value: $NP"
    #         exit 1
    #         ;;
    # esac
    
    echo "Finished run with $NP processes."
    echo "" # 출력 파일의 가독성을 위해 빈 줄 추가

done

echo "All scaling tests are complete."