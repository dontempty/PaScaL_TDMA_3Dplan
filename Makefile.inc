# ============================================================
#  Compiler / flag configuration for PaScaL_TDMAv2
# ============================================================

# ---------- nvhpc ----------
# CXX       := mpicxx
# CXXFLAGS  := -O3 -std=c++17 -fPIC -tp=icelake -fast

# ---------- g++ (default — matches Filtered_TDMA build) ----------
CXX       := mpicxx
CXXFLAGS  := -O3 -std=c++17 -fPIC -march=native

# ---------- intel ----------
# CXX       := mpiicpx
# CXXFLAGS  := -O3 -std=c++17 -g -fPIC -fp-model precise -march=native

# ============================================================
#  CUDA / GPU build (set USE_CUDA=1 to build the GPU variant)
# ============================================================
USE_CUDA  ?= 0

# nvcc; -ccbin uses the MPI C++ wrapper so MPI headers / link are consistent.
NVCC      := nvcc
# Override as needed (e.g., 70 V100, 80 A100, 90 H100).
CUDA_ARCH ?= 80
NVCCFLAGS := -O3 -std=c++17 -Xcompiler -fPIC -arch=sm_$(CUDA_ARCH) -ccbin $(CXX)
CUDA_LDFLAGS := -lcudart
