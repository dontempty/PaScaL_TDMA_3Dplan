# nvhpc
# CXX       := mpicxx
# CXXFLAGS  := -O0 -std=c++17 -g -fPIC
# CXXFLAGS  += -Kieee -tp=icelake
# CXXFLAGS  += -Minfo=all 
# run: make lib 2> ./minfo.log

# g++
# CXXFLAGS  := -O3 -std=c++17 -g -fPIC
# CXXFLAGS  += -fno-fast-math -ffp-contract=off
# CXXFLAGS  +=-fopenmp # openmp

# intel
CXX       := mpiicpx
CXXFLAGS  := -O3 -std=c++17 -g -fPIC
# CXXFLAGS  := -O3 -std=c++17 -g -fPIC -fopenmp -fp-model precise -march=native
