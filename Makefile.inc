CXX       := mpicxx

# nvhpc
CXXFLAGS  := -O3 -std=c++17 -g -fPIC
CXXFLAGS  += -Kieee -tp=icelake
CXXFLAGS  +=-mp # openmp

# g++
# CXXFLAGS  := -O3 -std=c++17 -g -fPIC
# CXXFLAGS  += -fno-fast-math -ffp-contract=off
# CXXFLAGS  +=-fopenmp # openmp
