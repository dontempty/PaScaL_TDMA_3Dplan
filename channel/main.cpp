#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <sys/stat.h>
#include <errno.h>

// mkdir -p equivalent: create all intermediate directories
static void mkdirs(const std::string& path) {
    for (std::size_t pos = 1; pos <= path.size(); ++pos) {
        if (pos == path.size() || path[pos] == '/') {
            std::string sub = path.substr(0, pos);
            if (mkdir(sub.c_str(), 0755) != 0 && errno != EEXIST) {
                fprintf(stderr, "[mkdirs] Failed to create directory: %s\n", sub.c_str());
            }
        }
    }
}

#include "params.hpp"
#include "grid.hpp"
#include "mpi_comm.hpp"
#include "boundary.hpp"
#include "projection.hpp"
#include "statistics.hpp"
#include "io.hpp"

// ============================================================
//  CFL-based time step control
// ============================================================
static double compute_dt(const std::vector<double>& u,
                          const std::vector<double>& v,
                          const std::vector<double>& w,
                          const SimParams& p,
                          const Grid& g,
                          const ZSlab& slab)
{
    const int Nx = p.Nx, Ny = p.Ny;
    const int nzl = slab.nz_local;
    double inv_dt_local = 0.0;

    // advection
    for (int kl = 1; kl <= nzl; ++kl) {
        int kg = slab.kstart + kl - 1;
        double dz_min = g.dz[kg];  // cell height at this level
        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i) {
                int idx = kl * Nx * Ny + j * Nx + i;
                double cfl = std::abs(u[idx]) / p.dx
                           + std::abs(v[idx]) / p.dy
                           + std::abs(0.5 * (w[idx] + w[(kl + 1) * Nx * Ny + j * Nx + i]))
                             / dz_min;
                if (cfl > inv_dt_local) inv_dt_local = cfl;
            }
    }

    double inv_dt_max;
    MPI_Allreduce(&inv_dt_local, &inv_dt_max, 1, MPI_DOUBLE, MPI_MAX, slab.comm);

    double dt_cfl = (inv_dt_max > 1.0e-15) ? (p.cfl_max / inv_dt_max) : p.dt;
    return std::min(dt_cfl, p.dt);  // never exceed input dt
}

// ============================================================
//  compute_bulk_velocity — volume-weighted mean streamwise velocity
//  (MPI collective: use only at init or in stats/field output blocks)
// ============================================================
static double compute_bulk_velocity(
    const std::vector<double>& u,
    const SimParams& p,
    const Grid& g,
    const ZSlab& slab)
{
    const int Nx = p.Nx, Ny = p.Ny;
    const int nzl = slab.nz_local;
    const int plane = Nx * Ny;

    double sum_local = 0.0;
    for (int kl = 1; kl <= nzl; ++kl) {
        const int    kg  = slab.kstart + kl - 1;
        const double dzk = g.dz[kg];
        for (int n = 0; n < plane; ++n)
            sum_local += u[kl * plane + n] * dzk;
    }
    double sum_global;
    MPI_Allreduce(&sum_local, &sum_global, 1, MPI_DOUBLE, MPI_SUM, slab.comm);
    return sum_global / (static_cast<double>(Nx * Ny) * p.Lz);
}

// ============================================================
//  Initial condition — MPM-STD cuda_momentum_init_channel() 방식과 동일
//
//  1. 랜덤 수 생성: [0,1) → -0.5 shift → [-0.5, 0.5)
//  2. Volume-weighted 전체 평균 제거 (MPI_Allreduce)
//  3. U = Umx*(1-(z/h-1)²) + vper*Umx*rand1   (라미나 포와줄 + 노이즈)
//     V = vper*Umx*rand2
//     W = vper*Umx*rand3
//  여기서 Umx = 라미나 중심선 속도, vper = perturb_amp
// ============================================================
static void initialize(std::vector<double>& u,
                        std::vector<double>& v,
                        std::vector<double>& w,
                        std::vector<double>& pr,
                        const SimParams& p,
                        const Grid& g,
                        const ZSlab& slab)
{
    const int Nx = p.Nx, Ny = p.Ny;
    const int nzl = slab.nz_local;
    const int plane = Nx * Ny;

    // Laminar centerline velocity (MPM-STD: Umx = 1.5 in their U_b=1 non-dim)
    // Here: Umx = (-dpdx/(2*nu)) * (Lz/2)^2  (general for both modes)
    const double amp_lam = (-p.dpdx) / (2.0 * p.nu);
    const double h       = 0.5 * p.Lz;
    const double Umx     = amp_lam * h * h;

    // ----------------------------------------------------------------
    // Step 1+2: 랜덤 수 생성 및 volume-weighted 평균 제거 (MPM-STD 동일)
    //   random_number() → -0.5 shift → MPI_Allreduce mean → subtract
    // ----------------------------------------------------------------
    std::vector<double> rand1(plane * (nzl + 2), 0.0);
    std::vector<double> rand2(plane * (nzl + 2), 0.0);
    std::vector<double> rand3(plane * (nzl + 2), 0.0);

    unsigned seed = static_cast<unsigned>(slab.myrank) * 1234u + 5678u;

    for (int kl = 1; kl <= nzl; ++kl)
        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i) {
                const int idx = kl * plane + j * Nx + i;
                rand1[idx] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
                rand2[idx] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
                rand3[idx] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
            }

    // Volume-weighted mean (MPM-STD: Σ rand*dx1*dx2*dx3 / Volume)
    double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, vol_local = 0.0;
    for (int kl = 1; kl <= nzl; ++kl) {
        const int    kg = slab.kstart + kl - 1;
        const double dV = p.dx * p.dy * g.dz[kg];
        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i) {
                const int idx = kl * plane + j * Nx + i;
                sum1 += rand1[idx] * dV;
                sum2 += rand2[idx] * dV;
                sum3 += rand3[idx] * dV;
                vol_local += dV;
            }
    }
    double gsum1, gsum2, gsum3, vol_total;
    MPI_Allreduce(&sum1,      &gsum1,     1, MPI_DOUBLE, MPI_SUM, slab.comm);
    MPI_Allreduce(&sum2,      &gsum2,     1, MPI_DOUBLE, MPI_SUM, slab.comm);
    MPI_Allreduce(&sum3,      &gsum3,     1, MPI_DOUBLE, MPI_SUM, slab.comm);
    MPI_Allreduce(&vol_local, &vol_total, 1, MPI_DOUBLE, MPI_SUM, slab.comm);

    const double mean1 = gsum1 / vol_total;
    const double mean2 = gsum2 / vol_total;
    const double mean3 = gsum3 / vol_total;

    for (int kl = 1; kl <= nzl; ++kl)
        for (int n = 0; n < plane; ++n) {
            const int idx = kl * plane + n;
            rand1[idx] -= mean1;
            rand2[idx] -= mean2;
            rand3[idx] -= mean3;
        }

    // ----------------------------------------------------------------
    // Step 3: 초기조건 적용 (MPM-STD cuda_momentum_init_channel 동일)
    //   U = Umx*(1-(z/h-1)^2) + vper*Umx*rand1
    //   V = vper*Umx*rand2
    //   W = vper*Umx*rand3
    // ----------------------------------------------------------------
    const double vper = p.perturb_amp;

    // U, V (cell-centered)
    for (int kl = 1; kl <= nzl; ++kl) {
        const int    kg = slab.kstart + kl - 1;
        const double z  = g.zc[kg];
        const double xi = z / h - 1.0;
        const double U_lam = Umx * (1.0 - xi * xi);   // MPM-STD: Umx*(1-(z/h-1)^2)

        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i) {
                const int idx = kl * plane + j * Nx + i;
                u[idx]  = U_lam + vper * Umx * rand1[idx];
                v[idx]  =         vper * Umx * rand2[idx];
                pr[idx] = 0.0;
            }
    }

    // W (z-faces): vper*Umx*rand3, wall faces zeroed by apply_bc_w
    // w[kl] is at global face zf[kstart+kl]; skip wall faces (kg_face=0 or Nz)
    for (int n = 0; n < plane * (nzl + 2); ++n) w[n] = 0.0;

    for (int kl = 1; kl <= nzl; ++kl) {
        const int kg_face = slab.kstart + kl;
        if (kg_face < 1 || kg_face >= p.Nz) continue;

        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i) {
                const int idx = kl * plane + j * Nx + i;
                w[idx] = vper * Umx * rand3[idx];
            }
    }
}

// ============================================================
//  print_monitor_header / print_monitor_line
//  Columns: step, time, dt, tau_w_bot, u_tau_bot, div_max, U_b
// ============================================================
static void print_monitor_header() {
    printf("--------------------------------------------------------------\n");
    printf("%8s %10s %10s | %10s %10s | %10s %8s\n",
           "step", "t", "dt", "tau_w_bot", "u_tau_bot", "div_max", "U_b");
    printf("--------------------------------------------------------------\n");
    fflush(stdout);
}

static void print_monitor_line(int step, double t, double dt,
                                double tau_w_bot, double u_tau_bot,
                                double div_max, double U_b)
{
    printf("%8d %10.4f %10.2e | %10.6f %10.6f | %10.2e %8.5f\n",
           step, t, dt, tau_w_bot, u_tau_bot, div_max, U_b);
    fflush(stdout);
}

// ============================================================
//  main
// ============================================================
int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const char* input_file = (argc > 1) ? argv[1] : "input.dat";

    // 1) Read parameters
    SimParams p{};
    if (myrank == 0) read_input(input_file, p);
    MPI_Bcast(&p, sizeof(SimParams), MPI_BYTE, 0, MPI_COMM_WORLD);

    // 2) Build grid
    Grid g;
    build_grid(p, g);

    // 3) MPI z-slab decomposition
    ZSlab slab;
    init_zslab(p.Nz, myrank, nprocs, MPI_COMM_WORLD, slab);

    if (myrank == 0) {
        printf("==============================================\n");
        printf("  Channel Flow Solver — Projection Method\n");
        printf("==============================================\n");
        printf("  Grid:      %d x %d x %d\n", p.Nx, p.Ny, p.Nz);
        printf("  Domain:    %.4f x %.4f x %.4f\n", p.Lx, p.Ly, p.Lz);
        if (p.non_dimensional) {
            double Umx = (-p.dpdx) / (2.0 * p.nu) * (p.Lz * 0.25);  // amp_lam * h^2
            printf("  Mode:      Non-dimensional (MPM-STD bulk-velocity)\n");
            printf("  Re (bulk): %.1f   nu = %.4e\n", p.Re, p.nu);
            printf("  dpdx = %.4e  (auto: -3*nu/h^2)\n", p.dpdx);
            printf("  Umx = %.4f   U_b = %.4f   Re_tau (derived) = %.2f\n",
                   Umx, Umx * 2.0 / 3.0, p.Re_tau);
        } else {
            printf("  Mode:      Dimensional  (nu = mu/rho)\n");
            printf("  mu = %.4e   rho = %.4f\n", p.mu, p.rho);
            printf("  nu = %.6e   dpdx = %.4e   Re_tau (derived) = %.2f\n",
                   p.nu, p.dpdx, p.Re_tau);
        }
        printf("  perturb_amp = %.3f\n", p.perturb_amp);
        printf("  MPI procs: %d (z-slab)\n", nprocs);
        printf("  nstep=%d  nmonitor=%d  nstat_start=%d  nstat=%d  nout_stats=%d  nout=%d\n",
               p.nstep, p.nmonitor, p.nstat_start, p.nstat, p.nout_stats, p.nout);
        printf("  Output dir: %s/\n", p.outdir);
        printf("==============================================\n");
        fflush(stdout);
        // Create output directories (mkdir -p: handles nested paths)
        mkdirs(std::string(p.outdir) + "/output");
        mkdirs(std::string(p.outdir) + "/stats");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const int Nx  = p.Nx, Ny = p.Ny;
    const int nzl = slab.nz_local;
    const int sz  = Nx * Ny * (nzl + 2);

    // 4) Allocate field arrays
    std::vector<double> u(sz, 0.0), v(sz, 0.0), w(sz, 0.0), pr(sz, 0.0);

    // 5) Initialize
    initialize(u, v, w, pr, p, g, slab);
    apply_bc_uv(u.data(), v.data(), Nx, Ny, slab);
    apply_bc_w(w.data(), Nx, Ny, slab);
    halo_exchange(u.data(), Nx, Ny, slab);
    halo_exchange(v.data(), Nx, Ny, slab);
    halo_exchange_w(w.data(), Nx, Ny, slab);

    // ---- Begin scoped solver block (must destroy before MPI_Finalize) ----
    {

    // 6) Create solvers
    ProjectionSolver solver(p, g, slab);
    Statistics       stats(p, g, slab);

    // 6b) Initial diagnostics (MPI allowed here — one-time startup cost)
    double U_b_init = compute_bulk_velocity(u, p, g, slab);
    if (myrank == 0) {
        // tau_w_bot: rank 0 always owns the bottom wall slab (kstart=0)
        double sum = 0.0;
        for (int n = 0; n < Nx * Ny; ++n) sum += u[1 * Nx * Ny + n];
        double tau_w_init = p.nu * (sum / (Nx * Ny)) / g.zc[0];
        double u_tau_init = std::sqrt(std::abs(tau_w_init));
        printf("  [init] U_b=%.5f  tau_w=%.6f  u_tau=%.6f\n",
               U_b_init, tau_w_init, u_tau_init);
        print_monitor_header();
        fflush(stdout);

        // Create history file with header
        std::string wss_path = std::string(p.outdir) + "/stats/wss_history.dat";
        FILE* fp_wss = fopen(wss_path.c_str(), "w");
        if (fp_wss) {
            fprintf(fp_wss, "# %8s %12s %12s %12s %12s %12s %12s\n",
                    "step", "time", "dt", "tau_w_bot", "u_tau_bot", "div_max", "U_b");
            fclose(fp_wss);
        }
    }

    // 7) Time loop
    double dt = p.dt;
    double time = 0.0;
    bool first_step = true;

    for (int step = 1; step <= p.nstep; ++step) {
        // Dynamic time step
        dt = compute_dt(u, v, w, p, g, slab);

        // Advance one step
        double div_max = solver.step(u, v, w, pr, dt, first_step);
        first_step = false;
        time += dt;

        // --- Monitoring (no MPI: rank 0 computes locally) ---
        if (step % p.nmonitor == 0 && myrank == 0) {
            // tau_w_bot: rank 0 owns bottom wall slab — no MPI needed
            double sum = 0.0;
            for (int n = 0; n < Nx * Ny; ++n) sum += u[1 * Nx * Ny + n];
            double tau_w_bot = p.nu * (sum / (Nx * Ny)) / g.zc[0];
            double u_tau_bot = std::sqrt(std::abs(tau_w_bot));
            // U_b: already computed inside solver.step() for CONST_FLOWRATE
            double U_b = solver.last_U_b();

            print_monitor_line(step, time, dt, tau_w_bot, u_tau_bot, div_max, U_b);

            std::string wss_path2 = std::string(p.outdir) + "/stats/wss_history.dat";
            FILE* fp = fopen(wss_path2.c_str(), "a");
            if (fp) {
                fprintf(fp, "%10d %12.6f %12.4e %12.6f %12.6f %12.4e %12.6f\n",
                        step, time, dt, tau_w_bot, u_tau_bot, div_max, U_b);
                fclose(fp);
            }
        }

        // --- Statistics: accumulate every nstat steps from nstat_start ---
        if (step >= p.nstat_start && (step - p.nstat_start) % p.nstat == 0)
            stats.accumulate(u, v, w, pr);

        // --- Statistics: write file every nout_stats steps from nstat_start ---
        if (p.out_stats && step >= p.nstat_start
                        && (step - p.nstat_start) % p.nout_stats == 0) {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/stats/stats_%08d.dat", p.outdir, step);
            stats.write(fname, step);  // all ranks must call (MPI_Allreduce inside)
        }

        // --- Field output ---
        if (p.out_field && step % p.nout == 0) {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/output/field_%08d.dat", p.outdir, step);
            write_tecplot_field(fname, p, g, u, v, w, pr,
                                slab.kstart, nzl, myrank, nprocs, MPI_COMM_WORLD);
            if (myrank == 0) {
                printf("  [output] %s written\n", fname);
                fflush(stdout);
            }
        }
    }

    if (myrank == 0) {
        printf("----------------------------------------------------------------------"
               "----------------------------------------------\n");
        printf("\nSimulation complete.  Final stats written to %s/stats/.\n", p.outdir);
    }

    } // ---- End scoped solver block ----

    MPI_Finalize();
    return 0;
}
