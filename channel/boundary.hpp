#ifndef BOUNDARY_HPP
#define BOUNDARY_HPP

#include "params.hpp"
#include "mpi_comm.hpp"

// ============================================================
//  Boundary conditions
//
//  apply_bc_uv:
//    u, v are at cell centers (staggered in x/y, same count).
//    No-slip ghost is already set by halo_exchange for wall ranks.
//    This function additionally zeroes the wall-face values of w.
//
//  apply_bc_w:
//    w is at z-faces.  Physical wall faces (k=1 for bottom rank,
//    k=nzl for top rank) are set to 0 directly.
// ============================================================

// Enforce w=0 at wall faces (only effective on boundary ranks)
void apply_bc_w(double* w, int Nx, int Ny, const ZSlab& slab);

// Enforce no-slip for u and v via ghost cell reflection.
// This is already done inside halo_exchange; this function
// re-applies it explicitly (e.g. after intermediate velocity step).
void apply_bc_uv(double* u, double* v, int Nx, int Ny, const ZSlab& slab);

#endif // BOUNDARY_HPP
