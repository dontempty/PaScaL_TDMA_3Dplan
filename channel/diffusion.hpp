#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP

#include <vector>
#include "params.hpp"
#include "grid.hpp"
#include "mpi_comm.hpp"
#include "pascal_tdma_many.hpp"

// ============================================================
//  ADI diffusion solver (Crank-Nicolson, 3D operator splitting)
//
//  Solves the CN implicit diffusion equation:
//    (I - ν·dt/2 · ∇²) u* = u_tilde + ν·dt/2 · ∇² u_tilde
//
//  Using three sequential 1D implicit sweeps (Z → Y → X):
//    Step 1 (Z): (I - ν·dt/2·∂²/∂z²) q  = rhs   [non-uniform, wall BC]
//    Step 2 (Y): (I - ν·dt/2·∂²/∂y²) r  = q     [uniform, periodic]
//    Step 3 (X): (I - ν·dt/2·∂²/∂x²) u* = r     [uniform, periodic]
//
//  MPI decomposition: 1D z-slab.
//    Z sweep: PaScaLTDMAMany distributed across slab ranks (n_sys = Nx*Ny)
//    Y sweep: local cyclic TDMA (n_sys = Nx*nzl, n_row = Ny)
//    X sweep: local cyclic TDMA (n_sys = Ny*nzl, n_row = Nx)
//
//  Wall BC (z):
//    u,v : anti-symmetric ghost  (ghost = -interior) → fold into diagonal
//    w   : zero ghost            (ghost = 0)         → off-diagonal zeroed
// ============================================================

class ADIDiffusion {
public:
    // eps_tdma: unused (kept for API compatibility with Filtered_TDMAv2 variant)
    ADIDiffusion(const SimParams& p, const Grid& g,
                 const ZSlab& slab, double eps_tdma = 0.0);
    ~ADIDiffusion();

    // Compute ν*dt/2 * ∇²phi (full 3D, physical space) and ADD to rhs.
    void add_explicit_laplacian(const std::vector<double>& phi,
                                std::vector<double>& rhs,
                                double nu_dt_half,
                                bool is_w) const;

    // Solve (I - ν*dt/2 * ∇²) u* = rhs  via 3D ADI (Z→Y→X).
    void adi_solve(std::vector<double>& rhs,
                   double nu, double dt, bool is_w);

    void set_dt(double dt) { dt_ = dt; }

    double time_z() const { return time_z_; }

private:
    const SimParams& p_;
    const Grid&      g_;
    const ZSlab&     slab_;
    double           dt_;

    // --- Z sweep: PaScaLTDMAMany, distributed across slab ranks ---
    // n_sys = Nx*Ny  (one system per (i,j) pair)
    // n_row = nz_local per rank (passed at solve time)
    PaScaLTDMAMany* ptdma_z_;

    // --- Y sweep: PaScaLTDMAMany with MPI_COMM_SELF (fully local) ---
    // n_sys = Nx*nzl,  n_row = Ny  (periodic)
    PaScaLTDMAMany* ptdma_y_;

    // --- X sweep: PaScaLTDMAMany with MPI_COMM_SELF (fully local) ---
    // n_sys = Ny*nzl,  n_row = Nx  (periodic)
    PaScaLTDMAMany* ptdma_x_;

    double time_z_ = 0.0;

    // --- Coefficient and RHS buffers ---
    // Z buffers: [nzl * Nx*Ny]
    std::vector<double> Az_, Bz_, Cz_, Dz_;
    // Y buffers: [Ny * (Nx*nzl)]  — layout: [j_row * n_sys_y + (kl-1)*Nx + i]
    std::vector<double> Ay_, By_, Cy_, Dy_;
    // X buffers: [Nx * (Ny*nzl)]  — layout: [i_row * n_sys_x + (kl-1)*Ny + j]
    std::vector<double> Ax_, Bx_, Cx_, Dx_;
};

// Convenience wrapper: compute explicit Laplacian + ADI solve.
//   phi in : u_tilde  (u^n + dt*advection + dt*forcing)
//   phi out: u*
void adi_diffuse(
    std::vector<double>& phi,
    ADIDiffusion& solver,
    double nu, double dt,
    bool is_w);

#endif // DIFFUSION_HPP
