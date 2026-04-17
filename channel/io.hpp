#ifndef IO_HPP
#define IO_HPP

#include <mpi.h>
#include <vector>
#include <string>
#include "params.hpp"
#include "grid.hpp"

// -------- input parsing --------
void read_input(const char* filename, SimParams& p);

// -------- Tecplot ASCII output --------

// Write instantaneous 3D field (u,v,w,p) to Tecplot ASCII file.
// Each rank writes its own z-slab; rank 0 writes the header.
// filename example: "output/field_000100.dat"
void write_tecplot_field(const std::string& filename,
                         const SimParams& p,
                         const Grid& g,
                         const std::vector<double>& u,
                         const std::vector<double>& v,
                         const std::vector<double>& w,
                         const std::vector<double>& pr,
                         int kstart, int nz_local,
                         int myrank, int nprocs,
                         MPI_Comm comm);

// Write 1D z-profile statistics to Tecplot ASCII file (rank 0 only).
void write_tecplot_stats(const std::string& filename,
                         const SimParams& p,
                         const Grid& g,
                         const std::vector<double>& U_mean,
                         const std::vector<double>& W_mean,
                         const std::vector<double>& u_rms,
                         const std::vector<double>& v_rms,
                         const std::vector<double>& w_rms,
                         const std::vector<double>& uw_stress,
                         const std::vector<double>& P_mean,
                         int step);

#endif // IO_HPP
