#include "mpi_comm.hpp"
#include <cstring>

// ============================================================
//  init_zslab
// ============================================================
void init_zslab(int Nz, int myrank, int nprocs, MPI_Comm comm, ZSlab& slab) {
    slab.myrank  = myrank;
    slab.nprocs  = nprocs;
    slab.comm    = comm;

    // Even distribution; last rank gets any remainder
    int base  = Nz / nprocs;
    int extra = Nz % nprocs;

    slab.kstart = myrank * base + (myrank < extra ? myrank : extra);
    slab.kend   = slab.kstart + base + (myrank < extra ? 1 : 0) - 1;
    slab.nz_local = slab.kend - slab.kstart + 1;

    slab.lo_rank = (myrank > 0)          ? myrank - 1 : -1;
    slab.hi_rank = (myrank < nprocs - 1) ? myrank + 1 : -1;
}

// ============================================================
//  halo_exchange — generic cell-center array
//  Array layout: arr[k * Nx*Ny + j*Nx + i]
//    k=0             : bottom ghost
//    k=1..nz_local   : interior
//    k=nz_local+1    : top ghost
//
//  For wall ranks: ghost cell reflects interior (no-slip):
//    u_ghost = -u_first_interior
// ============================================================
void halo_exchange(double* arr, int Nx, int Ny, const ZSlab& slab) {
    const int plane = Nx * Ny;
    const int nzl   = slab.nz_local;
    MPI_Status st;

    // Bottom ghost: send k=1 to lo_rank, recv from lo_rank into k=0
    if (slab.lo_rank >= 0) {
        MPI_Sendrecv(arr +  1      * plane, plane, MPI_DOUBLE, slab.lo_rank, 10,
                     arr +  0      * plane, plane, MPI_DOUBLE, slab.lo_rank, 11,
                     slab.comm, &st);
    } else {
        // Bottom wall: no-slip ghost  → u_ghost = -u[k=1]
        for (int n = 0; n < plane; ++n)
            arr[0 * plane + n] = -arr[1 * plane + n];
    }

    // Top ghost: send k=nzl to hi_rank, recv from hi_rank into k=nzl+1
    if (slab.hi_rank >= 0) {
        MPI_Sendrecv(arr +  nzl      * plane, plane, MPI_DOUBLE, slab.hi_rank, 11,
                     arr + (nzl + 1) * plane, plane, MPI_DOUBLE, slab.hi_rank, 10,
                     slab.comm, &st);
    } else {
        // Top wall: no-slip ghost  → u_ghost = -u[k=nzl]
        for (int n = 0; n < plane; ++n)
            arr[(nzl + 1) * plane + n] = -arr[nzl * plane + n];
    }
}

// ============================================================
//  halo_exchange_w — w-velocity faces
//  w[k=0] corresponds to face at kstart-1 (or bottom wall face).
//  w[k=1..nzl] are the nzl local faces owned by this rank.
//  w[k=nzl+1] corresponds to the face above (top of last cell).
//
//  No-slip: w=0 at physical walls (faces k_global=0 and k_global=Nz).
// ============================================================
void halo_exchange_w(double* w, int Nx, int Ny, const ZSlab& slab) {
    const int plane = Nx * Ny;
    const int nzl   = slab.nz_local;
    MPI_Status st;

    // Bottom: send face k=1 down, receive face k=0 from below
    if (slab.lo_rank >= 0) {
        MPI_Sendrecv(w +  1      * plane, plane, MPI_DOUBLE, slab.lo_rank, 20,
                     w +  0      * plane, plane, MPI_DOUBLE, slab.lo_rank, 21,
                     slab.comm, &st);
    } else {
        // Bottom wall face: w = 0
        for (int n = 0; n < plane; ++n)
            w[0 * plane + n] = 0.0;
    }

    // Top: send face k=nzl up, receive face k=nzl+1 from above
    if (slab.hi_rank >= 0) {
        MPI_Sendrecv(w +  nzl      * plane, plane, MPI_DOUBLE, slab.hi_rank, 21,
                     w + (nzl + 1) * plane, plane, MPI_DOUBLE, slab.hi_rank, 20,
                     slab.comm, &st);
    } else {
        // Top wall face: w = 0
        for (int n = 0; n < plane; ++n)
            w[(nzl + 1) * plane + n] = 0.0;
    }
}
