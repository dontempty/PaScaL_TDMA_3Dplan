#include "diffusion.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

// Periodic index wrap
static inline int pbc(int i, int N) { return (i + N) % N; }

// ============================================================
//  Constructor
// ============================================================
ADIDiffusion::ADIDiffusion(const SimParams& p, const Grid& g,
                           const ZSlab& slab, double /*eps_tdma*/)
    : p_(p), g_(g), slab_(slab), dt_(p.dt)
{
    const int Nx  = p.Nx, Ny = p.Ny;
    const int nzl = slab.nz_local;

    // --- Z sweep: PaScaLTDMAMany, distributed across slab ranks ---
    // n_sys = Nx*Ny (all (i,j) pairs), n_row passed at solve time
    ptdma_z_ = new PaScaLTDMAMany(
        Nx * Ny,
        slab.myrank, slab.nprocs, slab.comm);

    // --- Y sweep: local cyclic (MPI_COMM_SELF, nprocs=1) ---
    // n_sys = Nx*nzl,  n_row = Ny
    ptdma_y_ = new PaScaLTDMAMany(
        Nx * nzl,
        0, 1, MPI_COMM_SELF);

    // --- X sweep: local cyclic (MPI_COMM_SELF, nprocs=1) ---
    // n_sys = Ny*nzl,  n_row = Nx
    ptdma_x_ = new PaScaLTDMAMany(
        Ny * nzl,
        0, 1, MPI_COMM_SELF);

    // --- Allocate buffers ---
    Az_.resize(nzl * Nx * Ny);  Bz_.resize(nzl * Nx * Ny);
    Cz_.resize(nzl * Nx * Ny);  Dz_.resize(nzl * Nx * Ny);

    Ay_.resize(Ny * Nx * nzl);  By_.resize(Ny * Nx * nzl);
    Cy_.resize(Ny * Nx * nzl);  Dy_.resize(Ny * Nx * nzl);

    Ax_.resize(Nx * Ny * nzl);  Bx_.resize(Nx * Ny * nzl);
    Cx_.resize(Nx * Ny * nzl);  Dx_.resize(Nx * Ny * nzl);
}

ADIDiffusion::~ADIDiffusion() {
    delete ptdma_z_;
    delete ptdma_y_;
    delete ptdma_x_;
}

// ============================================================
//  add_explicit_laplacian — full 3D Laplacian in physical space
//  rhs += nu_dt_half * (d²/dx² + d²/dy² + d²/dz²) phi
// ============================================================
void ADIDiffusion::add_explicit_laplacian(
    const std::vector<double>& phi,
    std::vector<double>& rhs,
    double nu_dt_half,
    bool is_w) const
{
    const int Nx = p_.Nx, Ny = p_.Ny;
    const int nzl = slab_.nz_local;
    const double dx2 = p_.dx * p_.dx, dy2 = p_.dy * p_.dy;

    auto IDX = [&](int i, int j, int kl) {
        return kl * Nx * Ny + j * Nx + i;
    };

    for (int kl = 1; kl <= nzl; ++kl) {
        int kg = slab_.kstart + kl - 1;

        // z-direction FD coefficients (non-uniform)
        double dz_m  = g_.dzf[kg];       // zc[kg]   - zc[kg-1]
        double dz_p  = g_.dzf[kg + 1];   // zc[kg+1] - zc[kg]
        double dz_c  = g_.dz[kg];        // cell height

        double az = 1.0 / (dz_c * dz_m);
        double cz = 1.0 / (dz_c * dz_p);
        double bz = -(az + cz);

        for (int j = 0; j < Ny; ++j) {
            int jm = pbc(j - 1, Ny), jp = pbc(j + 1, Ny);
            for (int i = 0; i < Nx; ++i) {
                int im = pbc(i - 1, Nx), ip = pbc(i + 1, Nx);

                // x-direction (uniform, periodic)
                double d2x = (phi[IDX(ip, j, kl)] - 2.0 * phi[IDX(i, j, kl)]
                            + phi[IDX(im, j, kl)]) / dx2;

                // y-direction (uniform, periodic)
                double d2y = (phi[IDX(i, jp, kl)] - 2.0 * phi[IDX(i, j, kl)]
                            + phi[IDX(i, jm, kl)]) / dy2;

                // z-direction (non-uniform, ghost values set by halo/BC)
                double d2z = az * phi[IDX(i, j, kl - 1)]
                           + bz * phi[IDX(i, j, kl)]
                           + cz * phi[IDX(i, j, kl + 1)];

                rhs[IDX(i, j, kl)] += nu_dt_half * (d2x + d2y + d2z);
            }
        }
    }
}

// ============================================================
//  adi_solve — solve (I - ν·dt/2 · ∇²) u* = rhs
//
//  3D ADI factorisation (Z → Y → X):
//    (I - ν·dt/2·∂²/∂z²)(I - ν·dt/2·∂²/∂y²)(I - ν·dt/2·∂²/∂x²) u* = rhs
//
//  Step 1 (Z): non-uniform grid, wall BC, PaScaLTDMAMany
//  Step 2 (Y): uniform grid, periodic, local cyclic TDMA
//  Step 3 (X): uniform grid, periodic, local cyclic TDMA
// ============================================================
void ADIDiffusion::adi_solve(
    std::vector<double>& rhs,
    double nu, double dt, bool is_w)
{
    const int Nx  = p_.Nx, Ny = p_.Ny;
    const int nzl = slab_.nz_local;
    const int plane = Nx * Ny;
    const double nu_dt_half = nu * dt * 0.5;

    // ======================================================================
    // Step 1 — Z sweep
    //   (I - ν·dt/2·∂²/∂z²) q = rhs
    //   Non-uniform z, wall BC:
    //     u,v: anti-symmetric ghost → fold into diagonal
    //     w  : zero ghost           → off-diagonal = 0 at walls
    // ======================================================================

    for (int kl = 1; kl <= nzl; ++kl) {
        int kloc = kl - 1;
        int kg   = slab_.kstart + kl - 1;

        double dz_m = g_.dzf[kg];
        double dz_p = g_.dzf[kg + 1];
        double dz_c = g_.dz[kg];

        double az_raw = 1.0 / (dz_c * dz_m);   // > 0
        double cz_raw = 1.0 / (dz_c * dz_p);   // > 0

        // Original (interior) implicit coefficients
        double az_impl = -nu_dt_half * az_raw;  // < 0
        double cz_impl = -nu_dt_half * cz_raw;  // < 0

        bool is_bot = (slab_.lo_rank < 0 && kl == 1);
        bool is_top = (slab_.hi_rank < 0 && kl == nzl);

        double Az_k = (is_bot) ? 0.0 : az_impl;
        double Cz_k = (is_top) ? 0.0 : cz_impl;

        // For u,v: ghost = -interior → fold az_impl into diagonal at walls
        // For w:   ghost = 0 → B unchanged
        double bz_extra = 0.0;
        if (is_bot && !is_w) bz_extra -= az_impl;  // += nu_dt_half*az_raw
        if (is_top && !is_w) bz_extra -= cz_impl;  // += nu_dt_half*cz_raw

        // B = 1 + nu_dt_half*(az_raw+cz_raw) + bz_extra  (uses original az/cz)
        double Bz_k = 1.0 - az_impl - cz_impl + bz_extra;

        const double* rhs_k = rhs.data() + kl * plane;
        double* Az = Az_.data() + kloc * plane;
        double* Bz = Bz_.data() + kloc * plane;
        double* Cz = Cz_.data() + kloc * plane;
        double* Dz = Dz_.data() + kloc * plane;

        for (int s = 0; s < plane; ++s) {
            Az[s] = Az_k;
            Bz[s] = Bz_k;
            Cz[s] = Cz_k;
            Dz[s] = rhs_k[s];
        }
    }

    // Solve all Nx*Ny z-tridiagonal systems simultaneously
    ptdma_z_->solve(Az_.data(), Bz_.data(), Cz_.data(), Dz_.data(),
                    plane, nzl);

    // Unpack result back into rhs (intermediate q)
    for (int kl = 1; kl <= nzl; ++kl) {
        int kloc = kl - 1;
        const double* Dz = Dz_.data() + kloc * plane;
        double* rhs_k    = rhs.data() + kl * plane;
        for (int s = 0; s < plane; ++s)
            rhs_k[s] = Dz[s];
    }

    // ======================================================================
    // Step 2 — Y sweep
    //   (I - ν·dt/2·∂²/∂y²) r = q
    //   Uniform periodic y, fully local.
    //   n_sys = Nx*nzl,  n_row = Ny
    //   Buffer layout: [j_row * (Nx*nzl) + (kl-1)*Nx + i]
    // ======================================================================
    {
        const double ay   = nu_dt_half / (p_.dy * p_.dy);
        const double By_k = 1.0 + 2.0 * ay;
        const int    n_sys_y = Nx * nzl;

        for (int kl = 1; kl <= nzl; ++kl) {
            int kloc = kl - 1;
            for (int j = 0; j < Ny; ++j) {
                const double* rhs_kj = rhs.data() + kl * plane + j * Nx;
                double* Dy = Dy_.data() + j * n_sys_y + kloc * Nx;
                double* Ay = Ay_.data() + j * n_sys_y + kloc * Nx;
                double* By = By_.data() + j * n_sys_y + kloc * Nx;
                double* Cy = Cy_.data() + j * n_sys_y + kloc * Nx;
                for (int i = 0; i < Nx; ++i) {
                    Dy[i] = rhs_kj[i];
                    Ay[i] = -ay;
                    By[i] = By_k;
                    Cy[i] = -ay;
                }
            }
        }

        ptdma_y_->solve_cyclic(Ay_.data(), By_.data(), Cy_.data(),
                               Dy_.data(), n_sys_y, Ny);

        for (int kl = 1; kl <= nzl; ++kl) {
            int kloc = kl - 1;
            for (int j = 0; j < Ny; ++j) {
                const double* Dy  = Dy_.data() + j * n_sys_y + kloc * Nx;
                double*       rhs_kj = rhs.data() + kl * plane + j * Nx;
                for (int i = 0; i < Nx; ++i)
                    rhs_kj[i] = Dy[i];
            }
        }
    }

    // ======================================================================
    // Step 3 — X sweep
    //   (I - ν·dt/2·∂²/∂x²) u* = r
    //   Uniform periodic x, fully local.
    //   n_sys = Ny*nzl,  n_row = Nx
    //   Buffer layout: [i_row * (Ny*nzl) + (kl-1)*Ny + j]
    // ======================================================================
    {
        const double ax   = nu_dt_half / (p_.dx * p_.dx);
        const double Bx_k = 1.0 + 2.0 * ax;
        const int    n_sys_x = Ny * nzl;

        for (int kl = 1; kl <= nzl; ++kl) {
            int kloc = kl - 1;
            for (int j = 0; j < Ny; ++j) {
                const double* rhs_kj = rhs.data() + kl * plane + j * Nx;
                for (int i = 0; i < Nx; ++i) {
                    int base = i * n_sys_x + kloc * Ny + j;
                    Dx_[base] = rhs_kj[i];
                    Ax_[base] = -ax;
                    Bx_[base] = Bx_k;
                    Cx_[base] = -ax;
                }
            }
        }

        ptdma_x_->solve_cyclic(Ax_.data(), Bx_.data(), Cx_.data(),
                               Dx_.data(), n_sys_x, Nx);

        for (int kl = 1; kl <= nzl; ++kl) {
            int kloc = kl - 1;
            for (int j = 0; j < Ny; ++j) {
                double* rhs_kj = rhs.data() + kl * plane + j * Nx;
                for (int i = 0; i < Nx; ++i)
                    rhs_kj[i] = Dx_[i * n_sys_x + kloc * Ny + j];
            }
        }
    }
}

// ============================================================
//  adi_diffuse — convenience wrapper
//
//  CN scheme: (I - ν·dt/2·∇²) u* = u_tilde + ν·dt/2·∇²(u_tilde)
//
//  phi in : u_tilde  (u^n + dt*advection + dt*forcing)
//  phi out: u*
// ============================================================
void adi_diffuse(
    std::vector<double>& phi,
    ADIDiffusion& solver,
    double nu, double dt,
    bool is_w)
{
    double nu_dt_half = nu * dt * 0.5;
    std::vector<double> rhs(phi);
    solver.add_explicit_laplacian(phi, rhs, nu_dt_half, is_w);
    solver.adi_solve(rhs, nu, dt, is_w);
    phi = std::move(rhs);
}
