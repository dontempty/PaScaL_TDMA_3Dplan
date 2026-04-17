#include "boundary.hpp"

// ============================================================
//  apply_bc_w — set w=0 at physical wall faces
//
//  w array layout: w[k * Nx*Ny + j*Nx + i]
//    k=0          : ghost below bottom wall (not a physical face)
//    k=1          : bottom wall face  (global face kstart if bottom rank)
//    k=nzl        : top face of last interior cell
//    k=nzl+1      : ghost above top wall (not a physical face)
//
//  For the bottom rank (lo_rank == -1):
//    global face index of k=1 is kstart = 0 → bottom wall → w=0
//  For the top rank (hi_rank == -1):
//    global face index of k=nzl+1 is kend+1 = Nz → top wall → w=0
// ============================================================
void apply_bc_w(double* w, int Nx, int Ny, const ZSlab& slab) {
    const int plane = Nx * Ny;
    const int nzl   = slab.nz_local;

    if (slab.lo_rank < 0) {
        // Bottom wall face: k=1 local = global face 0
        for (int n = 0; n < plane; ++n)
            w[1 * plane + n] = 0.0;
    }
    if (slab.hi_rank < 0) {
        // Top wall face: k=nzl+1 local = global face Nz
        for (int n = 0; n < plane; ++n)
            w[(nzl + 1) * plane + n] = 0.0;
    }
}

// ============================================================
//  apply_bc_uv — ghost cell no-slip for u and v
//  Mirrors the halo_exchange wall case: ghost = -interior.
// ============================================================
void apply_bc_uv(double* u, double* v, int Nx, int Ny, const ZSlab& slab) {
    const int plane = Nx * Ny;
    const int nzl   = slab.nz_local;

    if (slab.lo_rank < 0) {
        for (int n = 0; n < plane; ++n) {
            u[0 * plane + n] = -u[1 * plane + n];
            v[0 * plane + n] = -v[1 * plane + n];
        }
    }
    if (slab.hi_rank < 0) {
        for (int n = 0; n < plane; ++n) {
            u[(nzl + 1) * plane + n] = -u[nzl * plane + n];
            v[(nzl + 1) * plane + n] = -v[nzl * plane + n];
        }
    }
}
