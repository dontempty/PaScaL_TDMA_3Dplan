// GPU heat-equation ADI solver. Mirrors PaScaL_TDMA_F/examples/solve_theta.f90.
// Layout: d_rhs[ci] with ci = kk*iy*ix + jj*ix + ii (ii fastest).

#include "solve_theta.hpp"
#include "../src/pascal_tdma_many_cuda.hpp"
#include "stencil_coeffs.hpp"
#include "index.hpp"
#include "timing_csv.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr,                                               \
                         "CUDA error %s at %s:%d: %s\n",                       \
                         cudaGetErrorName(_err), __FILE__, __LINE__,           \
                         cudaGetErrorString(_err));                            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

namespace {

constexpr double D_PI = 3.14159265358979323846;

__device__ inline void d_stencil(double dt, double dd, int lb, int rb,
                                 double& a, double& b, double& c) {
    double base = dt / (2.0 * dd * dd);
    a = base * ( 1.0 + (5.0/3.0) * lb + (1.0/3.0) * rb);
    b = base * (-2.0 -        2.0 * lb -        2.0 * rb);
    c = base * ( 1.0 + (1.0/3.0) * lb + (5.0/3.0) * rb);
}

__device__ inline std::size_t idx_full_d(int i, int j, int k, int nx, int ny) {
    return ((std::size_t)k * ny + (std::size_t)j) * nx + (std::size_t)i;
}

__global__ void rhs_kernel(double* __restrict__ d_rhs,
                           const double* __restrict__ d_theta,
                           const double* __restrict__ dmx,
                           const double* __restrict__ dmy,
                           const double* __restrict__ dmz,
                           const int* __restrict__ x_lb, const int* __restrict__ x_rb,
                           const int* __restrict__ y_lb, const int* __restrict__ y_rb,
                           const int* __restrict__ z_lb, const int* __restrict__ z_rb,
                           const double* __restrict__ x_sub,
                           const double* __restrict__ y_sub,
                           const double* __restrict__ z_sub,
                           int nx, int ny, int nz, double dt) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;
    int ix = nx - 2, iy = ny - 2, iz = nz - 2;
    if (ii >= ix || jj >= iy || kk >= iz) return;
    int i = ii + 1, j = jj + 1, k = kk + 1;

    double sxa, sxb, sxc;  d_stencil(dt, dmx[i], x_lb[i], x_rb[i], sxa, sxb, sxc);
    double sya, syb, syc;  d_stencil(dt, dmy[j], y_lb[j], y_rb[j], sya, syb, syc);
    double sza, szb, szc;  d_stencil(dt, dmz[k], z_lb[k], z_rb[k], sza, szb, szc);

    double t   = d_theta[idx_full_d(i,   j,   k,   nx, ny)];
    double tip = d_theta[idx_full_d(i+1, j,   k,   nx, ny)];
    double tim = d_theta[idx_full_d(i-1, j,   k,   nx, ny)];
    double tjp = d_theta[idx_full_d(i,   j+1, k,   nx, ny)];
    double tjm = d_theta[idx_full_d(i,   j-1, k,   nx, ny)];
    double tkp = d_theta[idx_full_d(i,   j,   k+1, nx, ny)];
    double tkm = d_theta[idx_full_d(i,   j,   k-1, nx, ny)];

    double src = dt * 3.0 * D_PI * D_PI * cos(D_PI*x_sub[i])
                                        * cos(D_PI*y_sub[j])
                                        * cos(D_PI*z_sub[k]);

    double rhs = (sxc * tip + sxa * tim)
               + (syc * tjp + sya * tjm)
               + (szc * tkp + sza * tkm)
               + (1.0 + sxb + syb + szb) * t
               + src;

    std::size_t ci = ((std::size_t)kk * iy + jj) * ix + ii;
    d_rhs[ci] = rhs;
}

__global__ void z_boundary_kernel(double* __restrict__ d_rhs,
                                  const double* __restrict__ theta_z_left,
                                  const double* __restrict__ theta_z_right,
                                  const int* __restrict__ z_lb_flag,
                                  const int* __restrict__ z_rb_flag,
                                  const double* __restrict__ dmx,
                                  const double* __restrict__ dmy,
                                  const double* __restrict__ dmz,
                                  const int* __restrict__ x_lb, const int* __restrict__ x_rb,
                                  const int* __restrict__ y_lb, const int* __restrict__ y_rb,
                                  int nx, int ny, int nz, double dt) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = nx - 2, iy = ny - 2, iz = nz - 2;
    if (ii >= ix || jj >= iy) return;
    int i = ii + 1, j = jj + 1;

    double sxa, sxb, sxc;  d_stencil(dt, dmx[i], x_lb[i], x_rb[i], sxa, sxb, sxc);
    double sya, syb, syc;  d_stencil(dt, dmy[j], y_lb[j], y_rb[j], sya, syb, syc);

    // Left wall (k=1 interior cell touches k=0 wall via theta_z_left_sub)
    {
        double dz0 = dmz[0];
        double coef = (dt * 0.5 / (dz0*dz0)) * (1.0 + 5.0/3.0);
        double accum = 0.0;
        for (int dj = -1; dj <= 1; ++dj) {
            double cy = (dj == -1) ? -sya : (dj == 0) ? (1.0 - syb) : -syc;
            // theta_z_left_sub uses idx_ij(i, j+dj, nx) = (j+dj)*nx + i
            std::size_t off    = (std::size_t)(j+dj) * nx + i;
            std::size_t off_ip = (std::size_t)(j+dj) * nx + (i+1);
            std::size_t off_im = (std::size_t)(j+dj) * nx + (i-1);
            accum += coef * cy *
                     (-sxa        * theta_z_left[off_im]
                     + (1.0-sxb) * theta_z_left[off]
                     - sxc        * theta_z_left[off_ip])
                     * (double)z_lb_flag[1];
        }
        std::size_t ci = ((std::size_t)0 * iy + jj) * ix + ii;     // k=1 → kk=0
        d_rhs[ci] += accum;
    }
    // Right wall (k=nz-2 interior cell touches k=nz-1 wall)
    {
        double dzN = dmz[nz-1];
        double coef = (dt * 0.5 / (dzN*dzN)) * (1.0 + 5.0/3.0);
        double accum = 0.0;
        for (int dj = -1; dj <= 1; ++dj) {
            double cy = (dj == -1) ? -sya : (dj == 0) ? (1.0 - syb) : -syc;
            std::size_t off    = (std::size_t)(j+dj) * nx + i;
            std::size_t off_ip = (std::size_t)(j+dj) * nx + (i+1);
            std::size_t off_im = (std::size_t)(j+dj) * nx + (i-1);
            accum += coef * cy *
                     (-sxa        * theta_z_right[off_im]
                     + (1.0-sxb) * theta_z_right[off]
                     - sxc        * theta_z_right[off_ip])
                     * (double)z_rb_flag[nz-2];
        }
        std::size_t ci = ((std::size_t)(iz-1) * iy + jj) * ix + ii;
        d_rhs[ci] += accum;
    }
}

__global__ void y_boundary_kernel(double* __restrict__ d_rhs,
                                  const double* __restrict__ theta_y_left,
                                  const double* __restrict__ theta_y_right,
                                  const int* __restrict__ y_lb_flag,
                                  const int* __restrict__ y_rb_flag,
                                  const double* __restrict__ dmx,
                                  const double* __restrict__ dmy,
                                  const int* __restrict__ x_lb, const int* __restrict__ x_rb,
                                  int nx, int ny, int nz, double dt) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int kk = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = nx - 2, iy = ny - 2, iz = nz - 2;
    if (ii >= ix || kk >= iz) return;
    int i = ii + 1, k = kk + 1;

    double sxa, sxb, sxc; d_stencil(dt, dmx[i], x_lb[i], x_rb[i], sxa, sxb, sxc);

    // theta_y_*_sub uses idx_ik(i, k, nx) = k*nx + i
    std::size_t off    = (std::size_t)k * nx + i;
    std::size_t off_ip = (std::size_t)k * nx + (i+1);
    std::size_t off_im = (std::size_t)k * nx + (i-1);

    {
        double dy0 = dmy[0];
        double coef = (dt * 0.5 / (dy0*dy0)) * (1.0 + 5.0/3.0);
        std::size_t ci = ((std::size_t)kk * iy + 0) * ix + ii;     // j=1 → jj=0
        d_rhs[ci] += coef * (double)y_lb_flag[1] *
                     (-sxa * theta_y_left[off_im]
                     + (1.0-sxb) * theta_y_left[off]
                     - sxc * theta_y_left[off_ip]);
    }
    {
        double dyN = dmy[ny-1];
        double coef = (dt * 0.5 / (dyN*dyN)) * (1.0 + 5.0/3.0);
        std::size_t ci = ((std::size_t)kk * iy + (iy-1)) * ix + ii;
        d_rhs[ci] += coef * (double)y_rb_flag[ny-2] *
                     (-sxa * theta_y_right[off_im]
                     + (1.0-sxb) * theta_y_right[off]
                     - sxc * theta_y_right[off_ip]);
    }
}

__global__ void x_boundary_kernel(double* __restrict__ d_rhs,
                                  const double* __restrict__ theta_x_left,
                                  const double* __restrict__ theta_x_right,
                                  const int* __restrict__ x_lb_flag,
                                  const int* __restrict__ x_rb_flag,
                                  const double* __restrict__ dmx,
                                  int nx, int ny, int nz, double dt) {
    int jj = blockIdx.x * blockDim.x + threadIdx.x;
    int kk = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = nx - 2, iy = ny - 2, iz = nz - 2;
    if (jj >= iy || kk >= iz) return;
    int j = jj + 1, k = kk + 1;

    // theta_x_*_sub uses idx_jk(j, k, ny) = k*ny + j
    std::size_t off = (std::size_t)k * ny + j;

    {
        double dx0 = dmx[0];
        double coef = (dt * 0.5 / (dx0*dx0)) * (1.0 + 5.0/3.0);
        std::size_t ci = ((std::size_t)kk * iy + jj) * ix + 0;     // i=1 → ii=0
        d_rhs[ci] += coef * (double)x_lb_flag[1] * theta_x_left[off];
    }
    {
        double dxN = dmx[nx-1];
        double coef = (dt * 0.5 / (dxN*dxN)) * (1.0 + 5.0/3.0);
        std::size_t ci = ((std::size_t)kk * iy + jj) * ix + (ix-1);
        d_rhs[ci] += coef * (double)x_rb_flag[nx-2] * theta_x_right[off];
    }
}

// Build Z-LHS. Layout: d_X[(kk*iy + jj)*ix + ii], ii fastest (same as d_rhs).
__global__ void build_lhs_z_kernel(double* __restrict__ d_A,
                                   double* __restrict__ d_B,
                                   double* __restrict__ d_C,
                                   double* __restrict__ d_D,
                                   const double* __restrict__ d_rhs,
                                   const double* __restrict__ dmz,
                                   const int* __restrict__ z_lb,
                                   const int* __restrict__ z_rb,
                                   int ix, int iy, int iz, double dt) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= ix || jj >= iy || kk >= iz) return;
    int k = kk + 1;

    double sa, sb, sc;  d_stencil(dt, dmz[k], z_lb[k], z_rb[k], sa, sb, sc);

    std::size_t off = ((std::size_t)kk * iy + jj) * ix + ii;
    d_A[off] = -sa;
    d_B[off] = 1.0 - sb;
    d_C[off] = -sc;
    d_D[off] = d_rhs[off];
}

// Copy Z-solution back into rhs (same layout as the build).
__global__ void copy_z_to_rhs(double* __restrict__ d_rhs,
                              const double* __restrict__ d_D,
                              std::size_t n) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_rhs[idx] = d_D[idx];
}

// Build Y-LHS. Layout: d_X[(jj*iz + kk)*ix + ii], ii fastest; j is the row axis.
__global__ void build_lhs_y_kernel(double* __restrict__ d_A,
                                   double* __restrict__ d_B,
                                   double* __restrict__ d_C,
                                   double* __restrict__ d_D,
                                   const double* __restrict__ d_rhs,
                                   const double* __restrict__ dmy,
                                   const int* __restrict__ y_lb,
                                   const int* __restrict__ y_rb,
                                   int ix, int iy, int iz, double dt) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= ix || jj >= iy || kk >= iz) return;
    int j = jj + 1;

    double sa, sb, sc;  d_stencil(dt, dmy[j], y_lb[j], y_rb[j], sa, sb, sc);

    std::size_t off_y   = ((std::size_t)jj * iz + kk) * ix + ii;
    std::size_t off_rhs = ((std::size_t)kk * iy + jj) * ix + ii;
    d_A[off_y] = -sa;
    d_B[off_y] = 1.0 - sb;
    d_C[off_y] = -sc;
    d_D[off_y] = d_rhs[off_rhs];
}

// Copy Y-solution back into rhs (transposing j↔k).
__global__ void copy_y_to_rhs(double* __restrict__ d_rhs,
                              const double* __restrict__ d_D,
                              int ix, int iy, int iz) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= ix || jj >= iy || kk >= iz) return;
    std::size_t off_y   = ((std::size_t)jj * iz + kk) * ix + ii;
    std::size_t off_rhs = ((std::size_t)kk * iy + jj) * ix + ii;
    d_rhs[off_rhs] = d_D[off_y];
}

// Build X-LHS. Layout: d_X[(ii*iz + kk)*iy + jj], jj fastest; i is the row axis.
// d_rhs has ii fastest, d_X has jj fastest — stage d_rhs through a shared
// tile so reads coalesce on ii and writes coalesce on jj. +1 padding avoids
// 32-way bank conflicts on the transposed tile read.
__global__ void build_lhs_x_kernel(double* __restrict__ d_A,
                                   double* __restrict__ d_B,
                                   double* __restrict__ d_C,
                                   double* __restrict__ d_D,
                                   const double* __restrict__ d_rhs,
                                   const double* __restrict__ dmx,
                                   const int* __restrict__ x_lb,
                                   const int* __restrict__ x_rb,
                                   int ix, int iy, int iz, double dt) {
    constexpr int TILE = 32;
    __shared__ double tile[TILE][TILE + 1];

    int ii_b = blockIdx.x * TILE;
    int jj_b = blockIdx.y * TILE;
    int kk   = blockIdx.z;
    if (kk >= iz) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load d_rhs into shared tile, coalesced in ii (= tx).
    {
        int ii_g = ii_b + tx;
        int jj_g = jj_b + ty;
        if (ii_g < ix && jj_g < iy) {
            tile[ty][tx] = d_rhs[((std::size_t)kk * iy + jj_g) * ix + ii_g];
        }
    }
    __syncthreads();

    // Write d_X with tx re-mapped to jj for coalesced writes.
    {
        int jj_g = jj_b + tx;
        int ii_g = ii_b + ty;
        if (ii_g < ix && jj_g < iy) {
            int i = ii_g + 1;
            double sa, sb, sc;
            d_stencil(dt, dmx[i], x_lb[i], x_rb[i], sa, sb, sc);
            std::size_t off_x = ((std::size_t)ii_g * iz + kk) * iy + jj_g;
            d_A[off_x] = -sa;
            d_B[off_x] = 1.0 - sb;
            d_C[off_x] = -sc;
            d_D[off_x] = tile[tx][ty];
        }
    }
}

__global__ void update_theta_kernel(double* __restrict__ d_theta,
                                    const double* __restrict__ d_D,
                                    int ix, int iy, int iz,
                                    int nx, int ny) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= ix || jj >= iy || kk >= iz) return;
    int i = ii + 1, j = jj + 1, k = kk + 1;
    std::size_t off_x = ((std::size_t)ii * iz + kk) * iy + jj;
    d_theta[idx_full_d(i, j, k, nx, ny)] = d_D[off_x];
}

template <class T>
T* alloc_and_copy(const std::vector<T>& src) {
    T* p = nullptr;
    std::size_t bytes = sizeof(T) * src.size();
    CUDA_CHECK(cudaMalloc(&p, bytes));
    CUDA_CHECK(cudaMemcpy(p, src.data(), bytes, cudaMemcpyHostToDevice));
    return p;
}

inline dim3 grid3(int nx, int ny, int nz, dim3 block) {
    return dim3((nx + block.x - 1) / block.x,
                (ny + block.y - 1) / block.y,
                (nz + block.z - 1) / block.z);
}

inline dim3 grid2(int nx, int ny, dim3 block) {
    return dim3((nx + block.x - 1) / block.x,
                (ny + block.y - 1) / block.y, 1);
}

} // namespace

SolveTheta::SolveTheta(const GlobalParams& params,
                       const MPITopology& topo,
                       MPISubdomain& sub)
    : params_(params), topo_(topo), sub_(sub) {}

void SolveTheta::profile(std::vector<double>& theta) {
    int nx_full = sub_.nx_sub + 1;
    int ny_full = sub_.ny_sub + 1;
    int nz_full = sub_.nz_sub + 1;
    int ix = nx_full - 2;
    int iy = ny_full - 2;
    int iz = nz_full - 2;
    int max_iter = params_.Nt;
    double dt    = params_.dt;

    auto cx = topo_.commX();
    auto cy = topo_.commY();
    auto cz = topo_.commZ();

    int my_rank;  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        std::cout << "[Tmax] = " << dt * max_iter
                  << " | [max_iter] = " << max_iter << "\n";
        std::cout << "[nx] = " << ix << " | [ny] = " << iy
                  << " | [nz] = " << iz << "\n";
        double beta = dt / 2.0 / (sub_.dmz_sub[1] * sub_.dmz_sub[1]);
        std::cout << "[rho] = " << beta / (1.0 + 2.0 * beta) << "\n";
    }

    std::size_t full_n  = (std::size_t)nx_full * ny_full * nz_full;
    std::size_t inner_n = (std::size_t)ix * iy * iz;

    sub_.allocGhostBufsDevice();

    double* d_theta = nullptr; CUDA_CHECK(cudaMalloc(&d_theta, sizeof(double) * full_n));
    double* d_rhs   = nullptr; CUDA_CHECK(cudaMalloc(&d_rhs,   sizeof(double) * inner_n));
    double* d_A     = nullptr; CUDA_CHECK(cudaMalloc(&d_A,     sizeof(double) * inner_n));
    double* d_B     = nullptr; CUDA_CHECK(cudaMalloc(&d_B,     sizeof(double) * inner_n));
    double* d_C     = nullptr; CUDA_CHECK(cudaMalloc(&d_C,     sizeof(double) * inner_n));
    double* d_D     = nullptr; CUDA_CHECK(cudaMalloc(&d_D,     sizeof(double) * inner_n));

    double* d_dmx     = alloc_and_copy(sub_.dmx_sub);
    double* d_dmy     = alloc_and_copy(sub_.dmy_sub);
    double* d_dmz     = alloc_and_copy(sub_.dmz_sub);
    double* d_x_sub   = alloc_and_copy(sub_.x_sub);
    double* d_y_sub   = alloc_and_copy(sub_.y_sub);
    double* d_z_sub   = alloc_and_copy(sub_.z_sub);
    int*    d_x_lb    = alloc_and_copy(sub_.theta_x_left_index);
    int*    d_x_rb    = alloc_and_copy(sub_.theta_x_right_index);
    int*    d_y_lb    = alloc_and_copy(sub_.theta_y_left_index);
    int*    d_y_rb    = alloc_and_copy(sub_.theta_y_right_index);
    int*    d_z_lb    = alloc_and_copy(sub_.theta_z_left_index);
    int*    d_z_rb    = alloc_and_copy(sub_.theta_z_right_index);
    double* d_th_xL   = alloc_and_copy(sub_.theta_x_left_sub);
    double* d_th_xR   = alloc_and_copy(sub_.theta_x_right_sub);
    double* d_th_yL   = alloc_and_copy(sub_.theta_y_left_sub);
    double* d_th_yR   = alloc_and_copy(sub_.theta_y_right_sub);
    double* d_th_zL   = alloc_and_copy(sub_.theta_z_left_sub);
    double* d_th_zR   = alloc_and_copy(sub_.theta_z_right_sub);

    CUDA_CHECK(cudaMemcpy(d_theta, theta.data(), sizeof(double) * full_n,
                          cudaMemcpyHostToDevice));

    const int bx = params_.thread_in_x_pascal;
    const int by = params_.thread_in_y_pascal;
    PaScaLTDMAManyCUDA solver_z(ix * iy, cz.myrank, cz.nprocs, cz.comm, bx, by);
    PaScaLTDMAManyCUDA solver_y(ix * iz, cy.myrank, cy.nprocs, cy.comm, bx, by);
    PaScaLTDMAManyCUDA solver_x(iy * iz, cx.myrank, cx.nprocs, cx.comm, bx, by);
    if (my_rank == 0) {
        std::cout << "[pascal_tdma block] " << bx << " x " << by
                  << " (total " << bx * by << ")\n";
    }

    // 3D kernels use (8,4,4); PaScaL TDMA uses (bx,by) from PARA_INPUT;
    // build_lhs_x uses a 32x32 tile transpose to coalesce both the d_rhs
    // read (ii-fastest) and the d_X write (jj-fastest).
    dim3 b3(8, 4, 4);
    dim3 g3 = grid3(ix, iy, iz, b3);

    constexpr int LHS_X_TILE = 32;
    dim3 b3_lhs_x(LHS_X_TILE, LHS_X_TILE, 1);
    dim3 g3_lhs_x((ix + LHS_X_TILE - 1) / LHS_X_TILE,
                  (iy + LHS_X_TILE - 1) / LHS_X_TILE,
                  iz);
    dim3 b2(16, 16, 1);
    dim3 g_xy = grid2(ix, iy, b2);
    dim3 g_xz = grid2(ix, iz, b2);
    dim3 g_yz = grid2(iy, iz, b2);

    const std::vector<std::string> event_names =
        {"rhs", "solve_z", "solve_y", "solve_x", "etc", "comm"};
    const int n_events = (int)event_names.size();
    timing_csv::timing_init(n_events, max_iter - 1, MPI_COMM_WORLD);
    std::vector<double> local_times(n_events, 0.0);

    // ev[0]→ev[1]: comm  ev[1]→ev[2]: rhs  ev[2..5]→: solve_z/y/x.
    cudaEvent_t ev[6];
    for (int i = 0; i < 6; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev[i], cudaEventBlockingSync));
    }

    for (int t_step = 0; t_step < max_iter; ++t_step) {
        MPI_Barrier(MPI_COMM_WORLD);

        cudaEventRecord(ev[0]);
        sub_.ghostcellUpdateDevice(d_theta, cx, cy, cz);
        cudaEventRecord(ev[1]);

        rhs_kernel<<<g3, b3>>>(d_rhs, d_theta,
                               d_dmx, d_dmy, d_dmz,
                               d_x_lb, d_x_rb, d_y_lb, d_y_rb, d_z_lb, d_z_rb,
                               d_x_sub, d_y_sub, d_z_sub,
                               nx_full, ny_full, nz_full, dt);
        cudaEventRecord(ev[2]);

        z_boundary_kernel<<<g_xy, b2>>>(d_rhs,
                                        d_th_zL, d_th_zR,
                                        d_z_lb, d_z_rb,
                                        d_dmx, d_dmy, d_dmz,
                                        d_x_lb, d_x_rb, d_y_lb, d_y_rb,
                                        nx_full, ny_full, nz_full, dt);
        build_lhs_z_kernel<<<g3, b3>>>(d_A, d_B, d_C, d_D, d_rhs,
                                       d_dmz, d_z_lb, d_z_rb, ix, iy, iz, dt);
        solver_z.solve(d_A, d_B, d_C, d_D, ix * iy, iz);
        {
            const int block_lin = 256;
            const int grid_lin  = (int)((inner_n + block_lin - 1) / block_lin);
            copy_z_to_rhs<<<grid_lin, block_lin>>>(d_rhs, d_D, inner_n);
        }
        cudaEventRecord(ev[3]);

        y_boundary_kernel<<<g_xz, b2>>>(d_rhs,
                                        d_th_yL, d_th_yR,
                                        d_y_lb, d_y_rb,
                                        d_dmx, d_dmy,
                                        d_x_lb, d_x_rb,
                                        nx_full, ny_full, nz_full, dt);
        build_lhs_y_kernel<<<g3, b3>>>(d_A, d_B, d_C, d_D, d_rhs,
                                       d_dmy, d_y_lb, d_y_rb, ix, iy, iz, dt);
        solver_y.solve(d_A, d_B, d_C, d_D, ix * iz, iy);
        copy_y_to_rhs<<<g3, b3>>>(d_rhs, d_D, ix, iy, iz);
        cudaEventRecord(ev[4]);

        x_boundary_kernel<<<g_yz, b2>>>(d_rhs,
                                        d_th_xL, d_th_xR,
                                        d_x_lb, d_x_rb,
                                        d_dmx,
                                        nx_full, ny_full, nz_full, dt);
        build_lhs_x_kernel<<<g3_lhs_x, b3_lhs_x>>>(d_A, d_B, d_C, d_D, d_rhs,
                                                   d_dmx, d_x_lb, d_x_rb, ix, iy, iz, dt);
        solver_x.solve(d_A, d_B, d_C, d_D, iy * iz, ix);
        update_theta_kernel<<<g3, b3>>>(d_theta, d_D, ix, iy, iz, nx_full, ny_full);
        cudaEventRecord(ev[5]);

        CUDA_CHECK(cudaEventSynchronize(ev[5]));

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev[0], ev[1]);  local_times[5] = ms * 1.0e-3;
        cudaEventElapsedTime(&ms, ev[1], ev[2]);  local_times[0] = ms * 1.0e-3;
        cudaEventElapsedTime(&ms, ev[2], ev[3]);  local_times[1] = ms * 1.0e-3;
        cudaEventElapsedTime(&ms, ev[3], ev[4]);  local_times[2] = ms * 1.0e-3;
        cudaEventElapsedTime(&ms, ev[4], ev[5]);  local_times[3] = ms * 1.0e-3;
        local_times[4] = 0.0;

        if (t_step >= 1) {
            timing_csv::timing_record(t_step, local_times, MPI_COMM_WORLD);
        }
    }

    for (int i = 0; i < 6; ++i) cudaEventDestroy(ev[i]);

    {
        char meta[256];
        std::snprintf(meta, sizeof(meta),
                      "grid=%dx%dx%d, np=%d (%d,%d,%d), dt=%10.3E, Tmax=%d, solver_kind=pascal",
                      params_.nx, params_.ny, params_.nz,
                      cx.nprocs * cy.nprocs * cz.nprocs,
                      params_.np_dim[0], params_.np_dim[1], params_.np_dim[2],
                      dt, max_iter);

        char fn[256];
        const char* env_path = std::getenv("TIMING_CSV");
        if (env_path && env_path[0] != '\0') {
            std::snprintf(fn, sizeof(fn), "%s", env_path);
        } else {
            std::snprintf(fn, sizeof(fn), "results/timing_%d_%d%d%d.csv",
                          params_.nx,
                          params_.np_dim[0], params_.np_dim[1], params_.np_dim[2]);
        }

        timing_csv::timing_save_csv(fn, event_names, meta, MPI_COMM_WORLD);
        timing_csv::timing_cleanup();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(theta.data(), d_theta, sizeof(double) * full_n,
                          cudaMemcpyDeviceToHost));

    auto safe_free = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    safe_free((void*&)d_theta); safe_free((void*&)d_rhs);
    safe_free((void*&)d_A); safe_free((void*&)d_B); safe_free((void*&)d_C); safe_free((void*&)d_D);
    safe_free((void*&)d_dmx); safe_free((void*&)d_dmy); safe_free((void*&)d_dmz);
    safe_free((void*&)d_x_sub); safe_free((void*&)d_y_sub); safe_free((void*&)d_z_sub);
    safe_free((void*&)d_x_lb); safe_free((void*&)d_x_rb);
    safe_free((void*&)d_y_lb); safe_free((void*&)d_y_rb);
    safe_free((void*&)d_z_lb); safe_free((void*&)d_z_rb);
    safe_free((void*&)d_th_xL); safe_free((void*&)d_th_xR);
    safe_free((void*&)d_th_yL); safe_free((void*&)d_th_yR);
    safe_free((void*&)d_th_zL); safe_free((void*&)d_th_zR);
    sub_.freeGhostBufsDevice();
}
