#ifndef POISSON_FFT_HPP
#define POISSON_FFT_HPP

#include <vector>
#include <complex>
#include <fftw3.h>
#include "params.hpp"
#include "grid.hpp"
#include "mpi_comm.hpp"

// ============================================================
//  PoissonFFT — pressure Poisson solver
//
//  Solves: ∇²φ = rhs  (= div(u*)/dt)
//
//  Algorithm:
//    1. 2D FFT (r2c) in x-y for each local z-plane
//    2. MPI_Allgather φ_hat to reconstruct full z-profile
//    3. Thomas algorithm in z for each (kx,ky) Fourier mode
//    4. 2D IFFT (c2r) to recover φ(x,y,z)
//    5. p += φ
//
//  x, y: periodic, uniform → FFT
//  z   : non-uniform, Neumann (dφ/dz=0) at walls
//        (kx=0,ky=0) mode: pure Neumann → φ_hat[0,0,0]=0 pinned
// ============================================================

class PoissonFFT {
public:
    PoissonFFT(const SimParams& p, const Grid& g, const ZSlab& slab);
    ~PoissonFFT();

    // Solve ∇²φ = div(u*)/dt and update pressure: pr += φ
    // u_star, v_star, w_star include ghost cells (k=0, nzl+1)
    void solve(const std::vector<double>& u_star,
               const std::vector<double>& v_star,
               const std::vector<double>& w_star,
               std::vector<double>& phi,
               std::vector<double>& pr,
               double dt);

    // Compute divergence of (u_star, v_star, w_star) into div_arr
    void compute_divergence(const std::vector<double>& u_star,
                            const std::vector<double>& v_star,
                            const std::vector<double>& w_star,
                            std::vector<double>& div_arr) const;

    // Return max |div(u_star)| across all ranks (for convergence check)
    double max_divergence(const std::vector<double>& u_star,
                          const std::vector<double>& v_star,
                          const std::vector<double>& w_star) const;

private:
    const SimParams& p_;
    const Grid&      g_;
    const ZSlab&     slab_;

    int Nk_;           // = Nx/2 + 1 (complex modes in x after r2c)
    int n_modes_;      // = Nk_ * Ny   (total complex modes per z-plane)

    // FFTW plans (2D in-place: batch over z via plan_many)
    fftw_plan plan_fwd_;   // r2c  [Ny][Nx] → [Ny][Nx/2+1]
    fftw_plan plan_inv_;   // c2r  [Ny][Nx/2+1] → [Ny][Nx]

    // Work arrays (per z-plane)
    // Complex: flat double[2*N] = [re0,im0, re1,im1, ...], cast to fftw_complex*
    std::vector<double> real_buf_;         // size Ny*Nx
    std::vector<double> cplx_buf_;         // size 2*Ny*Nk_

    // Global z-profile buffers (full Nz, gathered)
    // Interleaved re/im: index = 2*(m + k*n_modes_) + {0=re,1=im}
    std::vector<double> rhs_hat_global_;   // 2 * n_modes_ * Nz
    std::vector<double> phi_hat_global_;   // 2 * n_modes_ * Nz

    // Per-mode eigenvalues: lambda_xy[kx][ky]
    std::vector<double> lambda_xy_;   // n_modes_

    // z-Thomas coefficients (only Dirichlet/Neumann at walls)
    // Stored as lower (a), main (b_base), upper (c) of size Nz
    std::vector<double> z_a_, z_b_base_, z_c_;

    void build_lambda_xy();
    void build_z_coeffs();
    void thomas_solve_z(std::vector<double>& phi_hat_all,
                        int mode_idx, double lambda_xy_val) const;
};

#endif // POISSON_FFT_HPP
