#ifndef PARAMS_HPP
#define PARAMS_HPP

// ============================================================
//  SimParams — all simulation parameters read from input.dat
//
//  Non-dimensionalization mode (MPM-STD style):
//
//   non_dimensional = 1  (default, wall units)
//     nu   = 1 / Re
//     dpdx = dpdx_in  (default -1.0)
//     Re_tau derived: u_tau = sqrt(|dpdx|*h),  Re_tau = u_tau*h/nu
//     With standard wall units (Lz=2, dpdx=-1): Re_tau = Re
//
//   non_dimensional = 0  (dimensional / physical units)
//     nu   = mu / rho
//     dpdx = dpdx_in  (user-specified)
//     Re_tau derived from resulting u_tau
// ============================================================

struct SimParams {
    // Grid
    int    Nx, Ny, Nz;
    double Lx, Ly, Lz;
    double stretch_beta;    // tanh inflation param (0 = uniform)

    // Physics — MPM-STD style non-dimensionalization
    int    non_dimensional; // 1 = non-dim (default); 0 = dimensional
    double Re;              // Reynolds number [non_dimensional=1]:  nu = 1/Re
    double mu;              // dynamic viscosity [Pa·s] [non_dimensional=0]
    double rho;             // fluid density [kg/m³] [non_dimensional=0, default 1.0]

    // Forcing
    int    forcing_type;    // 0: CONST_DPDX, 1: CONST_FLOWRATE
    double dpdx_in;         // input pressure gradient (default -1.0)

    // Initial condition
    double perturb_amp;     // perturbation amplitude as fraction of U_cl (default 0.1)

    // Time integration
    double dt;
    double cfl_max;
    int    nstep;

    // Output
    int    nstat_start;     // step at which statistics accumulation begins
    int    nstat;           // statistics accumulation interval (steps)
    int    nout_stats;      // statistics file write interval (steps, >= nstat)
    int    nout;            // instant field output interval (steps)
    int    nmonitor;        // monitoring print interval (steps)
    int    out_stats;       // 1 = write stats files, 0 = skip
    int    out_field;       // 1 = write instant field files, 0 = skip

    // Derived (filled by read_input after parsing)
    double dx, dy;
    double nu;              // kinematic viscosity (computed from mode)
    double dpdx;            // mean pressure gradient (= dpdx_in or computed)
    double Re_tau;          // friction Re (derived: u_tau*h/nu, h=Lz/2)

    // MPI decomposition
    int    np;              // number of MPI processes (from input file)
    int    np_dim_z;        // number of z-ranks (= nprocs for 1D slab)

    // Output directory (default ".")
    // output/ and stats/ subdirs are created inside outdir.
    // Fixed-size char array so SimParams can be broadcast as raw bytes.
    char   outdir[256];
};

void read_input(const char* filename, SimParams& p);

#endif // PARAMS_HPP
