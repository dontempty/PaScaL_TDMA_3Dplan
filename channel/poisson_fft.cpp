#include "poisson_fft.hpp"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

static inline int pbc(int i, int N) { return (i + N) % N; }

// ============================================================
//  Constructor
// ============================================================
PoissonFFT::PoissonFFT(const SimParams& p, const Grid& g, const ZSlab& slab)
    : p_(p), g_(g), slab_(slab)
{
    const int Nx = p.Nx, Ny = p.Ny, Nz = p.Nz;
    Nk_      = Nx / 2 + 1;
    n_modes_ = Nk_ * Ny;

    // Per-plane real/complex buffers
    real_buf_.resize(Ny * Nx, 0.0);
    cplx_buf_.resize(2 * Ny * Nk_, 0.0);   // interleaved re/im

    // 2D r2c plan: transform [Ny][Nx] → [Ny][Nk_]
    plan_fwd_ = fftw_plan_dft_r2c_2d(
        Ny, Nx, real_buf_.data(),
        reinterpret_cast<fftw_complex*>(cplx_buf_.data()), FFTW_ESTIMATE);
    plan_inv_ = fftw_plan_dft_c2r_2d(
        Ny, Nx,
        reinterpret_cast<fftw_complex*>(cplx_buf_.data()),
        real_buf_.data(), FFTW_ESTIMATE);

    // Global z-profile buffers (interleaved re/im)
    rhs_hat_global_.resize(2 * n_modes_ * Nz, 0.0);
    phi_hat_global_.resize(2 * n_modes_ * Nz, 0.0);

    lambda_xy_.resize(n_modes_);
    z_a_.resize(Nz);
    z_b_base_.resize(Nz);
    z_c_.resize(Nz);

    build_lambda_xy();
    build_z_coeffs();
}

PoissonFFT::~PoissonFFT() {
    fftw_destroy_plan(plan_fwd_);
    fftw_destroy_plan(plan_inv_);
}

// ============================================================
//  build_lambda_xy
// ============================================================
void PoissonFFT::build_lambda_xy() {
    const int Nx = p_.Nx, Ny = p_.Ny;
    for (int ky = 0; ky < Ny; ++ky) {
        for (int kx = 0; kx < Nk_; ++kx) {
            double lx = -4.0 / (p_.dx * p_.dx)
                      * std::pow(std::sin(M_PI * kx / Nx), 2.0);
            double ly = -4.0 / (p_.dy * p_.dy)
                      * std::pow(std::sin(M_PI * ky / Ny), 2.0);
            lambda_xy_[ky * Nk_ + kx] = lx + ly;
        }
    }
}

// ============================================================
//  build_z_coeffs — Neumann BC at both walls
// ============================================================
void PoissonFFT::build_z_coeffs() {
    const int Nz = p_.Nz;
    for (int k = 0; k < Nz; ++k) {
        double dz_m = g_.dzf[k];
        double dz_p = g_.dzf[k + 1];
        double dz_c = g_.dz[k];

        double a_k = 1.0 / (dz_c * dz_m);
        double c_k = 1.0 / (dz_c * dz_p);

        if (k == 0) {
            z_a_[k]      = 0.0;
            z_c_[k]      = c_k;
            z_b_base_[k] = -c_k;   // Neumann: b = -(a+c)+a = -c
        } else if (k == Nz - 1) {
            z_a_[k]      = a_k;
            z_c_[k]      = 0.0;
            z_b_base_[k] = -a_k;   // Neumann: b = -(a+c)+c = -a
        } else {
            z_a_[k]      = a_k;
            z_b_base_[k] = -(a_k + c_k);
            z_c_[k]      = c_k;
        }
    }
}

// ============================================================
//  thomas_solve_z — Thomas algorithm for one (kx,ky) mode
//  phi_hat_all: interleaved [re0,im0, re1,im1, ...] per mode per k
//  Index: 2*(mode_idx + k*n_modes_) + {0=re, 1=im}
// ============================================================
void PoissonFFT::thomas_solve_z(std::vector<double>& phi_hat_all,
                                 int mode_idx, double lxy) const
{
    const int Nz = p_.Nz;
    const bool is_singular = (mode_idx == 0 && std::abs(lxy) < 1.0e-14);

    if (is_singular) {
        // Pin phi[0] = 0 to break singularity (same as MPM-STD reference)
        // Fall through to the normal Thomas solve below with modified row 0
    }

    std::vector<double> b(Nz), dr(Nz), di(Nz), cr(Nz);
    for (int k = 0; k < Nz; ++k) {
        b[k]  = z_b_base_[k] + lxy;
        dr[k] = rhs_hat_global_[2 * (mode_idx + k * n_modes_)];
        di[k] = rhs_hat_global_[2 * (mode_idx + k * n_modes_) + 1];
    }

    if (is_singular) {
        // Replace row 0 with: 1*phi[0] = 0  (a=0, b=1, c=0, rhs=0)
        b[0]  = 1.0;
        dr[0] = 0.0;
        di[0] = 0.0;
        // z_c_[0] is used below in cr[0] = z_c_[0]/b[0], but we need c=0
        // So we handle cr[0] separately after the general init
    }

    // Forward sweep
    cr[0]  = is_singular ? 0.0 : z_c_[0] / b[0];
    dr[0] /= b[0];
    di[0] /= b[0];
    for (int k = 1; k < Nz; ++k) {
        double m = b[k] - z_a_[k] * cr[k - 1];
        cr[k]  = z_c_[k] / m;
        dr[k]  = (dr[k] - z_a_[k] * dr[k - 1]) / m;
        di[k]  = (di[k] - z_a_[k] * di[k - 1]) / m;
    }

    // Back substitution
    phi_hat_all[2 * (mode_idx + (Nz - 1) * n_modes_)]     = dr[Nz - 1];
    phi_hat_all[2 * (mode_idx + (Nz - 1) * n_modes_) + 1] = di[Nz - 1];
    for (int k = Nz - 2; k >= 0; --k) {
        int base = 2 * (mode_idx + k * n_modes_);
        phi_hat_all[base]     = dr[k] - cr[k] * phi_hat_all[2 * (mode_idx + (k + 1) * n_modes_)];
        phi_hat_all[base + 1] = di[k] - cr[k] * phi_hat_all[2 * (mode_idx + (k + 1) * n_modes_) + 1];
    }
}

// ============================================================
//  compute_divergence
// ============================================================
void PoissonFFT::compute_divergence(const std::vector<double>& u_star,
                                     const std::vector<double>& v_star,
                                     const std::vector<double>& w_star,
                                     std::vector<double>& div_arr) const
{
    const int Nx = p_.Nx, Ny = p_.Ny;
    const int nzl = slab_.nz_local;
    const double inv_dx = 1.0 / p_.dx, inv_dy = 1.0 / p_.dy;

    for (int kl = 1; kl <= nzl; ++kl) {
        int kg = slab_.kstart + kl - 1;
        double inv_dz = 1.0 / g_.dz[kg];
        for (int j = 0; j < Ny; ++j) {
            int jm = pbc(j - 1, Ny);
            for (int i = 0; i < Nx; ++i) {
                int im = pbc(i - 1, Nx);
                double du = (u_star[kl * Nx * Ny + j * Nx + i]
                           - u_star[kl * Nx * Ny + j * Nx + im]) * inv_dx;
                double dv = (v_star[kl * Nx * Ny + j * Nx + i]
                           - v_star[kl * Nx * Ny + jm * Nx + i]) * inv_dy;
                double dw = (w_star[(kl + 1) * Nx * Ny + j * Nx + i]
                           - w_star[ kl      * Nx * Ny + j * Nx + i]) * inv_dz;
                div_arr[kl * Nx * Ny + j * Nx + i] = du + dv + dw;
            }
        }
    }
}

double PoissonFFT::max_divergence(const std::vector<double>& u_star,
                                   const std::vector<double>& v_star,
                                   const std::vector<double>& w_star) const
{
    const int Nx = p_.Nx, Ny = p_.Ny, nzl = slab_.nz_local;
    std::vector<double> div(Nx * Ny * (nzl + 2), 0.0);
    compute_divergence(u_star, v_star, w_star, div);

    double local_max = 0.0;
    for (int kl = 1; kl <= nzl; ++kl)
        for (int n = 0; n < Nx * Ny; ++n) {
            double val = std::abs(div[kl * Nx * Ny + n]);
            if (val > local_max) local_max = val;
        }

    double global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, slab_.comm);
    return global_max;
}

// ============================================================
//  solve
// ============================================================
void PoissonFFT::solve(const std::vector<double>& u_star,
                        const std::vector<double>& v_star,
                        const std::vector<double>& w_star,
                        std::vector<double>& phi,
                        std::vector<double>& pr,
                        double dt)
{
    const int Nx = p_.Nx, Ny = p_.Ny, Nz = p_.Nz;
    const int nzl = slab_.nz_local;
    const double inv_dt = 1.0 / dt;
    const double inv_NxNy = 1.0 / (Nx * Ny);

    // --- Step 1: rhs = div(u*) / dt ---
    std::vector<double> rhs(Nx * Ny * (nzl + 2), 0.0);
    compute_divergence(u_star, v_star, w_star, rhs);
    for (int kl = 1; kl <= nzl; ++kl)
        for (int n = 0; n < Nx * Ny; ++n)
            rhs[kl * Nx * Ny + n] *= inv_dt;

    // --- Step 2: 2D FFT for each local z-plane ---
    std::fill(rhs_hat_global_.begin(), rhs_hat_global_.end(), 0.0);
    for (int kl = 1; kl <= nzl; ++kl) {
        int kg = slab_.kstart + kl - 1;

        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i)
                real_buf_[j * Nx + i] = rhs[kl * Nx * Ny + j * Nx + i];

        fftw_execute(plan_fwd_);

        // Store into rhs_hat_global at z-level kg
        // cplx_buf_: interleaved [re,im] of n_modes_ complex values
        for (int m = 0; m < n_modes_; ++m) {
            rhs_hat_global_[2 * (m + kg * n_modes_)]     = cplx_buf_[2 * m];
            rhs_hat_global_[2 * (m + kg * n_modes_) + 1] = cplx_buf_[2 * m + 1];
        }
    }

    // --- Step 3: MPI_Allreduce to gather full z profile ---
    std::vector<double> rhs_hat_tmp(2 * n_modes_ * Nz, 0.0);
    // Copy only local contribution (other z-levels stay 0)
    for (int kl = 1; kl <= nzl; ++kl) {
        int kg = slab_.kstart + kl - 1;
        for (int m = 0; m < n_modes_; ++m) {
            rhs_hat_tmp[2 * (m + kg * n_modes_)]     = rhs_hat_global_[2 * (m + kg * n_modes_)];
            rhs_hat_tmp[2 * (m + kg * n_modes_) + 1] = rhs_hat_global_[2 * (m + kg * n_modes_) + 1];
        }
    }
    MPI_Allreduce(rhs_hat_tmp.data(), rhs_hat_global_.data(),
                  2 * n_modes_ * Nz, MPI_DOUBLE, MPI_SUM, slab_.comm);

    // --- Step 4: Thomas solve in z for each mode ---
    std::fill(phi_hat_global_.begin(), phi_hat_global_.end(), 0.0);
    for (int m = 0; m < n_modes_; ++m)
        thomas_solve_z(phi_hat_global_, m, lambda_xy_[m]);

    // --- Step 5: IFFT to recover φ on local z-planes ---
    for (int kl = 1; kl <= nzl; ++kl) {
        int kg = slab_.kstart + kl - 1;

        for (int m = 0; m < n_modes_; ++m) {
            cplx_buf_[2 * m]     = phi_hat_global_[2 * (m + kg * n_modes_)];
            cplx_buf_[2 * m + 1] = phi_hat_global_[2 * (m + kg * n_modes_) + 1];
        }

        fftw_execute(plan_inv_);

        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i)
                phi[kl * Nx * Ny + j * Nx + i] = real_buf_[j * Nx + i] * inv_NxNy;
    }

    // --- Step 6: Update pressure ---
    for (int kl = 1; kl <= nzl; ++kl)
        for (int n = 0; n < Nx * Ny; ++n)
            pr[kl * Nx * Ny + n] += phi[kl * Nx * Ny + n];
}
