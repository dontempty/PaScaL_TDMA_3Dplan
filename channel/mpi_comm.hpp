#ifndef MPI_COMM_HPP
#define MPI_COMM_HPP

#include <mpi.h>
#include "params.hpp"

// ============================================================
//  ZSlab — z-direction 1D slab decomposition
//
//  Global cells: 0 .. Nz-1
//  This rank owns cells: kstart .. kend  (nz_local cells)
//  Local array k-index:
//    k=0           : bottom ghost (from lo_rank or wall)
//    k=1..nz_local : interior cells  (global kstart..kend)
//    k=nz_local+1  : top ghost    (from hi_rank or wall)
//
//  Total local z-size (with halos): nz_local + 2
// ============================================================
struct ZSlab {
    int myrank, nprocs;
    int kstart, kend;   // global cell indices (0-based)
    int nz_local;       // kend - kstart + 1
    int lo_rank;        // rank below  (-1 if bottom wall)
    int hi_rank;        // rank above  (-1 if top wall)
    MPI_Comm comm;
};

void init_zslab(int Nz, int myrank, int nprocs, MPI_Comm comm, ZSlab& slab);

// Halo exchange for a cell-center array (Nx*Ny*(nz_local+2)).
// k=0 and k=nz_local+1 are filled from neighbors (or wall ghost).
// lo_wall / hi_wall: set to true for the bottom/top rank.
void halo_exchange(double* arr, int Nx, int Ny, const ZSlab& slab);

// Same exchange but for w-velocity faces (Nx*Ny*(nz_local+2)).
// w at the wall face is set to 0 (no-slip) instead of halo from neighbor.
void halo_exchange_w(double* w, int Nx, int Ny, const ZSlab& slab);

#endif // MPI_COMM_HPP
