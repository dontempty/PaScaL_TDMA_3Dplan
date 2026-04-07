# ============================================================
#  Compiler / flag configuration for PaScaL_TDMAv2
# ============================================================

# ---------- nvhpc (default) ----------
CXX       := mpicxx
CXXFLAGS  := -O1 -std=c++17 -g -fPIC
CXXFLAGS  += -tp=icelake

# ---------- g++ ----------
# CXX       := mpicxx
# CXXFLAGS  := -O3 -std=c++17 -g -fPIC
# CXXFLAGS  += -fno-fast-math -ffp-contract=off

# ---------- intel ----------
# CXX       := mpiicpx
# CXXFLAGS  := -O3 -std=c++17 -g -fPIC -fp-model precise -march=native
