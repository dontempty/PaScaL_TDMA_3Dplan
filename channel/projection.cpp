#include "projection.hpp"
#include "boundary.hpp"
#include "mpi_comm.hpp"
#include <cstring>
#include <algorithm>

static inline int pbc(int i, int N) { return (i + N) % N; }

// ============================================================
//  Constructor
// ============================================================
ProjectionSolver::ProjectionSolver(const SimParams& p, const Grid& g,
                                   const ZSlab& slab, double eps_tdma)
    : p_(p), g_(g), slab_(slab)
{
    const int Nx = p.Nx, Ny = p.Ny;
    const int nzl = slab.nz_local;
    const int sz  = Nx * Ny * (nzl + 2);

    diff_solver_ = new ADIDiffusion(p, g, slab, eps_tdma);
    poisson_     = new PoissonFFT(p, g, slab);

    // Bulk velocity target: Poiseuille U_b = (-dpdx)*h²/(3*nu)
    // In non-dim mode with dpdx = -3*nu/h²  →  U_b_target = 1.0
    {
        const double h = 0.5 * p.Lz;
        U_b_target_ = (-p.dpdx) * h * h / (3.0 * p.nu);
    }
    dpdx_eff_ = p.dpdx;  // initial estimate = Poiseuille dpdx
    last_U_b_ = U_b_target_;

    Nu_old_.assign(sz, 0.0);  Nv_old_.assign(sz, 0.0);  Nw_old_.assign(sz, 0.0);
    Nu_new_.assign(sz, 0.0);  Nv_new_.assign(sz, 0.0);  Nw_new_.assign(sz, 0.0);
    u_star_.assign(sz, 0.0);  v_star_.assign(sz, 0.0);   w_star_.assign(sz, 0.0);
    phi_.assign(sz, 0.0);
}

ProjectionSolver::~ProjectionSolver() {
    delete diff_solver_;
    delete poisson_;
}

// ============================================================
//  velocity_correction
//    u^{n+1} = u^* - dt * dφ/dx
//    w^{n+1} = w^* - dt * dφ/dz  (non-uniform z, at face)
// ============================================================
void ProjectionSolver::velocity_correction(std::vector<double>& u,
                                            std::vector<double>& v,
                                            std::vector<double>& w,
                                            const std::vector<double>& phi,
                                            double dt) const
{
    const int Nx = p_.Nx, Ny = p_.Ny;
    const int nzl = slab_.nz_local;
    const double inv_dx = 1.0 / p_.dx, inv_dy = 1.0 / p_.dy;

    for (int kl = 1; kl <= nzl; ++kl) {
        for (int j = 0; j < Ny; ++j) {
            int jp = pbc(j + 1, Ny);
            for (int i = 0; i < Nx; ++i) {
                int ip = pbc(i + 1, Nx);
                int idx = kl * Nx * Ny + j * Nx + i;

                // u at x-face (i+1/2, j, k): dφ/dx at (i+1/2) = (phi[i+1] - phi[i]) / dx
                u[idx] -= dt * (phi[kl * Nx * Ny + j * Nx + ip] - phi[idx]) * inv_dx;

                // v at y-face (i, j+1/2, k): dφ/dy at (j+1/2) = (phi[j+1] - phi[j]) / dy
                v[idx] -= dt * (phi[kl * Nx * Ny + jp * Nx + i] - phi[idx]) * inv_dy;
            }
        }
    }

    // w at z-face: w[kl] is face between cell kl-1 and kl (global: between kg-1 and kg)
    // dφ/dz at face kl = (phi[kl] - phi[kl-1]) / dzf[kl]
    for (int kl = 1; kl <= nzl; ++kl) {
        int kg = slab_.kstart + kl - 1;
        if (kg == 0) continue;  // bottom wall face: w already 0
        double inv_dzf = 1.0 / g_.dzf[kg];
        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i) {
                int idx = kl * Nx * Ny + j * Nx + i;
                w[idx] -= dt * (phi[idx] - phi[(kl - 1) * Nx * Ny + j * Nx + i])
                             * inv_dzf;
            }
    }
    // Top wall face
    if (slab_.hi_rank < 0) {
        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i)
                w[(nzl + 1) * Nx * Ny + j * Nx + i] = 0.0;
    }
}

// ============================================================
//  step — one full projection time step
// ============================================================
double ProjectionSolver::step(std::vector<double>& u,
                               std::vector<double>& v,
                               std::vector<double>& w,
                               std::vector<double>& pr,
                               double dt,
                               bool first_step)
{
    const int Nx = p_.Nx, Ny = p_.Ny;
    const int nzl = slab_.nz_local;
    const int plane = Nx * Ny;

    diff_solver_->set_dt(dt);

    // ======== Step 1a: Advection ========
    compute_advection_uvw(u, v, w, Nu_new_, Nv_new_, Nw_new_, p_, g_, slab_);

    // Copy u,v,w to u_star then apply AB2 advection
    u_star_ = u;
    v_star_ = v;
    w_star_ = w;

    adams_bashforth(u_star_, Nu_new_, Nu_old_, dt, Nx, Ny, nzl, first_step);
    adams_bashforth(v_star_, Nv_new_, Nv_old_, dt, Nx, Ny, nzl, first_step);
    adams_bashforth(w_star_, Nw_new_, Nw_old_, dt, Nx, Ny, nzl, first_step);

    // ======== Step 1b: Body force (streamwise pressure gradient) ========
    // CONST_DPDX:    dpdx_eff_ == p_.dpdx  (constant throughout run)
    // CONST_FLOWRATE: dpdx_eff_ is updated each step to maintain U_b == U_b_target_
    for (int kl = 1; kl <= nzl; ++kl)
        for (int n = 0; n < plane; ++n)
            u_star_[kl * plane + n] += dt * (-dpdx_eff_);

    // ======== Step 1c: CN-ADI diffusion ========
    // Add explicit Laplacian half and then do implicit sweeps
    adi_diffuse(u_star_, *diff_solver_, p_.nu, dt, false);
    adi_diffuse(v_star_, *diff_solver_, p_.nu, dt, false);
    adi_diffuse(w_star_, *diff_solver_, p_.nu, dt, true);

    // Re-apply BCs after diffusion
    apply_bc_w(w_star_.data(), Nx, Ny, slab_);
    apply_bc_uv(u_star_.data(), v_star_.data(), Nx, Ny, slab_);
    halo_exchange(u_star_.data(), Nx, Ny, slab_);
    halo_exchange(v_star_.data(), Nx, Ny, slab_);
    halo_exchange_w(w_star_.data(), Nx, Ny, slab_);

    // ======== Step 2: Pressure Poisson ========
    poisson_->solve(u_star_, v_star_, w_star_, phi_, pr, dt);

    // Exchange phi ghost cells across MPI ranks before velocity correction.               
    // phi[kl=0] at non-bottom ranks is needed for w correction at kl=1:
    //   w[kl=1] -= dt * (phi[1] - phi[0]) / dzf[kg]
    // Without this exchange phi[0] stays 0 → wrong w correction → div blowup.
    // Wall ghost cells (lo/hi_rank < 0) are not accessed in velocity_correction
    // so the anti-symmetric reflection applied by halo_exchange is harmless.
    halo_exchange(phi_.data(), Nx, Ny, slab_);

    // ======== Step 3: Velocity correction ========
    velocity_correction(u_star_, v_star_, w_star_, phi_, dt);

    // Update velocities
    u = u_star_;
    v = v_star_;
    w = w_star_;

    // ======== CONST_FLOWRATE: enforce target bulk velocity ========
    // After the projection, the volume-averaged U_b may deviate from target
    // (turbulent drag > laminar → U_b < 1).  Apply a uniform correction
    // delta_u to all streamwise velocity cells, equivalent to an extra body
    // force delta_u/dt for this step.  Update dpdx_eff_ accordingly so the
    // next step uses the corrected estimate as its body-force starting point.
    if (p_.forcing_type == 1) {
        double sum_local = 0.0;
        for (int kl = 1; kl <= nzl; ++kl) {
            const int    kg  = slab_.kstart + kl - 1;
            const double dzk = g_.dz[kg];
            for (int n = 0; n < plane; ++n)
                sum_local += u[kl * plane + n] * dzk;
        }
        double sum_global;
        MPI_Allreduce(&sum_local, &sum_global, 1, MPI_DOUBLE, MPI_SUM, slab_.comm);
        last_U_b_ = sum_global / (static_cast<double>(Nx * Ny) * p_.Lz);
        const double delta_u = U_b_target_ - last_U_b_;

        for (int kl = 1; kl <= nzl; ++kl)
            for (int n = 0; n < plane; ++n)
                u[kl * plane + n] += delta_u;

        // Effective dpdx this step = previous estimate minus the extra impulse
        //   total body force = (-dpdx_eff_) + delta_u/dt
        //   → dpdx_eff_new   = dpdx_eff_ - delta_u/dt
        dpdx_eff_ -= delta_u / dt;
    }

    // Re-apply BCs
    apply_bc_w(w.data(), Nx, Ny, slab_);
    apply_bc_uv(u.data(), v.data(), Nx, Ny, slab_);
    halo_exchange(u.data(), Nx, Ny, slab_);
    halo_exchange(v.data(), Nx, Ny, slab_);
    halo_exchange_w(w.data(), Nx, Ny, slab_);

    // Rotate advection history
    Nu_old_ = Nu_new_;
    Nv_old_ = Nv_new_;
    Nw_old_ = Nw_new_;

    // Return max divergence for monitoring
    return poisson_->max_divergence(u, v, w);
}
