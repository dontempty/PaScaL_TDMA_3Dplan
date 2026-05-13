#ifndef GLOBAL_PARAMS_HPP
#define GLOBAL_PARAMS_HPP

#include <array>
#include <string>

class GlobalParams {
public:
    GlobalParams() = default;
    void load(const std::string& filename);

    // Physical / numerical parameters
    double Tmax, dt;
    double rho, eps_constant;
    int Nt;

    int nx, ny, nz;
    int nxm, nym, nzm;

    double lx, ly, lz;
    double x0, xN, y0, yN, z0, zN;
    double dx, dy, dz;

    // MPI process decomposition
    std::array<int, 3> np_dim;

    // Run mode: "order" or "strong"
    std::string option;

    // CUDA thread-block dimensions for the PaScaL_TDMA solver kernels
    // (modified_thomas, tdma_many, update_solution).  Mirrors the Fortran
    // reference's `thread_in_x_pascal` / `thread_in_y_pascal` namelist.
    // Default: 128 × 1 (single warp-block).
    int thread_in_x_pascal = 128;
    int thread_in_y_pascal = 1;
};

#endif // GLOBAL_PARAMS_HPP
