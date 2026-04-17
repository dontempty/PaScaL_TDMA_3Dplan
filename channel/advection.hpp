#ifndef ADVECTION_HPP
#define ADVECTION_HPP

#include <vector>
#include "params.hpp"
#include "grid.hpp"
#include "mpi_comm.hpp"

// ============================================================
//  Adams-Bashforth 2nd-order advection
//
//  N(u) = -(u·∇)u  computed on staggered grid.
//
//  compute_advection_uvw:
//    Fills Nu, Nv, Nw with the advective fluxes at time n.
//    u, v, w include ghost cells (k=0 and k=nzl+1).
//
//  adams_bashforth:
//    u^* = u + dt * (1.5*N_new - 0.5*N_old)  [AB2]
//    For the first step (first_step=true) use Euler:
//    u^* = u + dt * N_new
//
//  Arrays: [k][j][i] = k*(Nx*Ny) + j*Nx + i
//    k=0,nzl+1 are ghosts; interior k=1..nzl
// ============================================================

void compute_advection_uvw(
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    std::vector<double>& Nu,
    std::vector<double>& Nv,
    std::vector<double>& Nw,
    const SimParams& p,
    const Grid& g,
    const ZSlab& slab);

void adams_bashforth(
    std::vector<double>& phi,       // in/out: u (or v,w)
    const std::vector<double>& N_new,
    const std::vector<double>& N_old,
    double dt,
    int Nx, int Ny, int nz_local,
    bool first_step);

#endif // ADVECTION_HPP
