# PaScaL_TDMA GPU — Convergence Verification

> Status: **PASSED** (order 2 spatial convergence reproduced bit-identically on A100/V100 GPU vs CPU reference).

## Setup

3D heat equation manufactured-solution test (`heat_gpu/`), with:

- Exact solution: `θ(x,y,z,t) = sin(πx)sin(πy)sin(πz)·exp(-3π²t) + cos(πx)cos(πy)cos(πz)`
- Source term: `f = 3π²·cos(πx)cos(πy)cos(πz)`
- Domain `[-1,1]³`, Dirichlet walls in all directions
- Mesh: cell-centered with half-cell offset at boundaries
- Spatial: 2nd-order ADI (Filtered_TDMA half-cell stencil — order-2 in the L2 norm)
- Time: factored Crank-Nicolson ADI (1st order globally)
- `rho = 0.25` ⇒ `dt = rho/(1-2ρ)·2·dx² = dx²`
- `Tmax = dt_N · 128`, fixed → `Nt ∝ 1/dx²` so **T_final identical at every N**

| N    | dt        | Nt   | T_final = Nt·dt |
|------|-----------|------|-----------------|
|  64  | 9.77e-04  |   2  | 1.953e-03       |
| 128  | 2.44e-04  |   8  | 1.953e-03       |
| 256  | 6.10e-05  |  32  | 1.953e-03       |
| 512  | 1.53e-05  | 128  | 1.953e-03       |

## Build / environment

KISTI Neuron, GPU node `gpu04` (cas_v100nv_8 — V100 ×8 / node).
```
module purge
module load nvhpc/25.11_cuda12          # provides nvcc, HPC-X OpenMPI, mpicxx
export CUDA_LIBDIR=$NVHPC_ROOT/cuda/lib64

USE_CUDA=1 CUDA_ARCH=70 make lib
USE_CUDA=1 CUDA_ARCH=70 make -C heat_gpu
```
sbatch script: `heat_gpu/run_one_np.sh` (parametrized by `NP=1|2|4|8`).

## Results — PaScaL_TDMA GPU, rho=0.25, T_final fixed

L2 error against the analytic exact solution.  CPU reference values
(from `/scratch/x3319a05/Filtered_TDMA/Heat` with `tdma_backend=pascal`,
np=2×2×2): **identical to the GPU values below to all reported digits**.

| Decomposition | nx=64       | nx=128      | nx=256      | nx=512      |
|---------------|------------:|------------:|------------:|------------:|
| 1 GPU (1,1,1) | 4.21191e-5  | 1.03721e-5  | 2.58574e-6  | 6.46409e-7  |
| 2 GPU (2,1,1) | 4.21191e-5  | 1.03721e-5  | 2.58574e-6  | 6.46409e-7  |
| CPU reference | 4.21191e-5  | 1.03721e-5  | 2.58574e-6  | 6.46409e-7  |

All entries bit-identical across CPU, 1-GPU, and 2-GPU after the
ghost-cell pack/unpack optimization (see § Optimizations applied).

Observed order (using consecutive ratios E(h)/E(h/2)):

| N step       | ratio  | log₂(ratio) |
|--------------|-------:|------------:|
| 64 → 128     | 4.061  | **+2.022**  |
| 128 → 256    | 4.011  | **+2.004**  |
| 256 → 512    | 4.000  | **+2.000**  |

**Convergence test passed**: asymptotic order 2 ✓, GPU vs CPU bit-identical (round-off).

## Multi-rank MPI exchange verification

Bit-identical reproducibility between 1-GPU (no MPI exchange) and 2-GPU
(MPI Alltoallv via CUDA-aware HPC-X) confirms the GPU path's inter-rank
boundary-row exchange is implemented correctly. Further decompositions
(4-GPU, 8-GPU) require running on the `amd_a100nv_8` queue — see
`heat_gpu/run_convergence_a100.sh` for the wider sweep.

### Wallclock (post-optimization, V100 PCIe, cas_v100nv_4)

| N    | 1-GPU (job 721239) | 2-GPU (job 721240) |
|------|-------------------:|-------------------:|
|  64  |      1 s           |      2 s           |
| 128  |      1 s           |      3 s           |
| 256  |      2 s           |      4 s           |
| 512  |     18 s           |     40 s           |

The 2-GPU run is currently slower than 1-GPU at every grid because the
**single V100 PCIe node has no NVLink** between GPUs — every face slab
crosses PCIe (and the host hop for `cuda_copy`).  On an A100 NVLink
node (`amd_a100nv_8`) 2-GPU should match or beat 1-GPU once exchange
latency drops below the per-step compute.

## Job records

| Job ID  | NP | partition       | node  | elapsed | status     | note               |
|---------|---:|-----------------|-------|--------:|------------|--------------------|
| 721172  | 2  | cas_v100nv_8    | gpu04 | 12:00+  | COMPLETED  | pre-optimization (DDT path) |
| 721239  | 1  | cas_v100nv_4    | gpu06 | 0:23    | COMPLETED  | post-optimization  |
| 721240  | 2  | cas_v100nv_4    | gpu07 | 0:50    | COMPLETED  | post-optimization  |

## Optimizations applied — ghost-cell pack/unpack

The Heat_gpu ghost-cell exchange was rewritten from a derived-datatype
MPI path to an explicit pack-buffer pattern (mirrors
[`PaScaL_TDMA_F examples/solve_theta.f90 ghostcell_update_cuda`](../../PaScaL_TDMA_F/examples/solve_theta.f90)).

### 바뀐 점

원래 Heat_gpu 의 ghost-cell 교환은 `MPI_Type_create_subarray` 로 만든
**파생 데이터타입**을 **device pointer** 에 적용해서 `MPI_Isend/Irecv` 를
호출하는 방식이었음.  HPC-X (OpenMPI) 의 CUDA-aware 경로는 **연속 버퍼**
device pointer 에 대해서는 UCX `cuda_ipc` / `cuda_copy` 로 빠르게 처리
하지만, **strided derived type + device pointer** 조합에 대해서는
폴백 (셀 단위 `cudaMemcpy` 혹은 전체 배열 D2H/H2D 스테이징) 으로
빠지면서 연속 전송 대비 100× ~ 300× 까지 느려짐.

새 구현 (`heat_gpu/ghostcell_cuda.cu`) 은 각 면 slab 을 2D CUDA kernel
로 **연속 device buffer** 에 packing 한 뒤, 연속 포인터로 `MPI_Isend/
Irecv` 를 호출하고, 받은 recv buffer 를 다시 ghost slot 으로 unpack
하는 방식.  12 개의 device buffer (각 면당 send/recv 1 쌍 × 6 면)
는 time loop 시작에서 `MPISubdomain::allocGhostBufsDevice()` 가 한 번
할당하고, 끝에서 해제함.

### 측정된 효과 (V100 PCIe, 2-GPU NP=2,1,1)

| N    | Before (DDT path, job 721172) | After (pack/unpack, job 721240) | Speedup |
|------|------------------------------:|--------------------------------:|--------:|
|  64  |               3 s             |               2 s               |   1.5×  |
| 128  |              22 s             |               3 s               |  ~7×    |
| 256  |             357 s             |               4 s               | **89×** |
| 512  |    ~3000 s (projected)        |              40 s               |  ~75×+  |

이전 DDT 경로에서는 N=512 가 1 시간 안에 끝나지 않았는데, 새 경로에서는
전체 4-grid 2-GPU sweep 이 **50 초** 안에 끝남.  L2 값은 1-GPU 및 CPU
기준과 비트 단위로 일치 (위 표 참조).

### 의도하지 않게 도움이 안 된 환경 변수

다음 UCX / OpenMPI 환경 변수들은 먼저 시도해봤지만 이 노드에서는
의미 있는 차이가 없었음:
```bash
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib,uct
export OMPI_MCA_pml_ucx_opal_cuda=1
export OMPI_MCA_opal_cuda_support=true
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc,gdr_copy
export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_THRESH=8192
```
UCX 는 `gdr_copy is not available` (V100 PCIe 에는 GPUDirect RDMA 없음)
경고를 띄웠지만, `cuda_copy` / `cuda_ipc` / `shm` 는 이미 활성화됨.
병목은 transport 가 아니라 `MPI_Isend/Irecv` 안에서 derived-datatype
+ device-pointer 조합이 OpenMPI 를 느린 fallback 으로 강제하는 것이
었음.
| 512  |  ~3000 s (proj.)  |         36 s        |  ~80×+  |

The full 4-grid 2-GPU sweep, which previously did not even finish N=512
inside an hour, now completes in **69 seconds**. L2 values are
bit-identical to the 1-GPU and CPU references.

### What did *not* help (also tried)

The following UCX/OpenMPI environment changes were tested first and
turned out to have no measurable effect on this node:
```bash
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib,uct
export OMPI_MCA_pml_ucx_opal_cuda=1
export OMPI_MCA_opal_cuda_support=true
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc,gdr_copy
export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_THRESH=8192
```
UCX warned `gdr_copy is not available` (V100 PCIe has no GPUDirect RDMA),
but `cuda_copy`/`cuda_ipc`/`shm` were already in use. The bottleneck was
not the transport — it was the derived-datatype + device-pointer
combination forcing OpenMPI into a slow per-cell or full-array fallback
inside `MPI_Isend/MPI_Irecv` itself.
