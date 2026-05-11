#ifndef PROJECTION_HPP
#define PROJECTION_HPP

#include <vector>
#include "params.hpp"
#include "grid.hpp"
#include "mpi_comm.hpp"
#include "advection.hpp"
#include "diffusion.hpp"
#include "poisson_fft.hpp"

// ============================================================
//  ProjectionSolver — fractional step (projection) method
//
//  Each call to step() advances one time step:
//
//  Step 1: Compute intermediate velocity u^*
//    a) AB2 advection: u_tmp = u + dt*(1.5*N^n - 0.5*N^{n-1})
//    b) Body force:    u_tmp[u-component] += dt * (-dpdx) (x-forcing)
//    c) CN-ADI diffusion: solve (I - ν*dt/2*∇²)u^* = u_tmp + ν*dt/2*∇²u_tmp
//
//  Step 2: Pressure Poisson solve
//    ∇²φ = div(u*)/dt,  p += φ
//
//  Step 3: Velocity correction
//    u^{n+1} = u^* - dt * ∇φ
//
//  All arrays: [k*(Nx*Ny) + j*Nx + i]
//    k=0, nzl+1: ghost; k=1..nzl: interior
// ============================================================

class ProjectionSolver {
public:
    ProjectionSolver(const SimParams& p, const Grid& g, const ZSlab& slab,
                     double eps_tdma = 1.0e-2);
    ~ProjectionSolver();

    // Advance one time step.
    // Returns the max divergence after correction (for monitoring).
    double step(std::vector<double>& u,
                std::vector<double>& v,
                std::vector<double>& w,
                std::vector<double>& pr,
                double dt,
                bool first_step);

    // Effective streamwise pressure gradient this step.
    // CONST_DPDX: always equal to p.dpdx.
    // CONST_FLOWRATE: updated each step to maintain U_b = U_b_target.
    double dpdx_eff() const { return dpdx_eff_; }

    // Volume-averaged bulk velocity from the last step.
    // CONST_FLOWRATE: computed inside step(). CONST_DPDX: always 0.
    double last_U_b() const { return last_U_b_; }

    double time_z() const { return diff_solver_->time_z(); }

private:
    const SimParams&  p_;
    const Grid&       g_;
    const ZSlab&      slab_;

    ADIDiffusion* diff_solver_;
    PoissonFFT*      poisson_;

    // CONST_FLOWRATE state
    double dpdx_eff_;   // effective dpdx applied this step (updated each step)
    double U_b_target_; // target bulk velocity (= Poiseuille U_b from initial dpdx)
    double last_U_b_;   // bulk velocity computed in last step (CONST_FLOWRATE only)

    // AB2 advection history
    std::vector<double> Nu_old_, Nv_old_, Nw_old_;
    std::vector<double> Nu_new_, Nv_new_, Nw_new_;

    // Intermediate velocity
    std::vector<double> u_star_, v_star_, w_star_;
    std::vector<double> phi_;

    void velocity_correction(std::vector<double>& u,
                              std::vector<double>& v,
                              std::vector<double>& w,
                              const std::vector<double>& phi,
                              double dt) const;
};

#endif // PROJECTION_HPP
