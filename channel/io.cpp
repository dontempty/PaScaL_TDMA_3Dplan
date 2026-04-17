#include "io.hpp"
#include "grid.hpp"

#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <string>

// ============================================================
//  read_input — parse key-value pairs from input.dat
//
//  Non-dimensionalization follows MPM-STD convention:
//    NON_DIMENSIONAL 1  → nu = 1/Re,  dpdx = DPDX (default -1.0)
//    NON_DIMENSIONAL 0  → nu = MU/RHO, dpdx = DPDX (user-specified)
//  In both modes Re_tau is a *derived* output:
//    u_tau = sqrt(|dpdx| * h),  Re_tau = u_tau * h / nu   (h = Lz/2)
// ============================================================
void read_input(const char* filename, SimParams& p) {
    // --- Defaults ---
    strncpy(p.outdir, ".", sizeof(p.outdir) - 1);
    p.outdir[sizeof(p.outdir) - 1] = '\0';
    p.non_dimensional = 1;       // non-dim mode (wall units) by default
    p.Re              = 180.0;   // default Re (= Re_tau in wall-unit mode)
    p.mu              = 1.0e-3;  // default dynamic viscosity (unused in non-dim mode)
    p.rho             = 1.0;     // default density
    p.dpdx_in         = 0.0;     // 0 = auto (-3*nu/h^2 for non-dim, must set for dimensional)
    p.perturb_amp     = 0.05;     // 10% of U_cl as perturbation amplitude
    p.out_stats       = 1;
    p.out_field       = 1;
    p.nmonitor        = 10;
    p.nstat_start     = 0;
    p.nout_stats      = 0;  // 0 = set to nstat after parsing

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[read_input] Cannot open %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char key[128], sval[128];
    while (fscanf(fp, "%127s", key) == 1) {
        if (key[0] == '#') {
            fgets(sval, sizeof(sval), fp);
            continue;
        }
        if      (strcmp(key, "NX")              == 0) fscanf(fp, "%d",  &p.Nx);
        else if (strcmp(key, "NY")              == 0) fscanf(fp, "%d",  &p.Ny);
        else if (strcmp(key, "NZ")              == 0) fscanf(fp, "%d",  &p.Nz);
        else if (strcmp(key, "NP")              == 0) fscanf(fp, "%d",  &p.np);
        else if (strcmp(key, "LX")              == 0) fscanf(fp, "%lf", &p.Lx);
        else if (strcmp(key, "LY")              == 0) fscanf(fp, "%lf", &p.Ly);
        else if (strcmp(key, "LZ")              == 0) fscanf(fp, "%lf", &p.Lz);
        else if (strcmp(key, "STRETCH_BETA")    == 0) fscanf(fp, "%lf", &p.stretch_beta);
        // --- MPM-STD style physics parameters ---
        else if (strcmp(key, "NON_DIMENSIONAL") == 0) fscanf(fp, "%d",  &p.non_dimensional);
        else if (strcmp(key, "RE")              == 0) fscanf(fp, "%lf", &p.Re);
        else if (strcmp(key, "MU")              == 0) fscanf(fp, "%lf", &p.mu);
        else if (strcmp(key, "RHO")             == 0) fscanf(fp, "%lf", &p.rho);
        else if (strcmp(key, "DPDX")            == 0) fscanf(fp, "%lf", &p.dpdx_in);
        else if (strcmp(key, "PERTURB_AMP")     == 0) fscanf(fp, "%lf", &p.perturb_amp);
        // --- Time integration ---
        else if (strcmp(key, "DT")              == 0) fscanf(fp, "%lf", &p.dt);
        else if (strcmp(key, "CFL_MAX")         == 0) fscanf(fp, "%lf", &p.cfl_max);
        else if (strcmp(key, "NSTEP")           == 0) fscanf(fp, "%d",  &p.nstep);
        else if (strcmp(key, "NSTAT_START")     == 0) fscanf(fp, "%d",  &p.nstat_start);
        else if (strcmp(key, "NSTAT")           == 0) fscanf(fp, "%d",  &p.nstat);
        else if (strcmp(key, "NOUT_STATS")      == 0) fscanf(fp, "%d",  &p.nout_stats);
        else if (strcmp(key, "NOUT")            == 0) fscanf(fp, "%d",  &p.nout);
        else if (strcmp(key, "NMONITOR")        == 0) fscanf(fp, "%d",  &p.nmonitor);
        else if (strcmp(key, "OUT_STATS")       == 0) fscanf(fp, "%d",  &p.out_stats);
        else if (strcmp(key, "OUT_FIELD")       == 0) fscanf(fp, "%d",  &p.out_field);
        else if (strcmp(key, "OUTDIR")          == 0) {
            fscanf(fp, "%255s", p.outdir);
        }
        else if (strcmp(key, "FORCING_TYPE")    == 0) {
            fscanf(fp, "%127s", sval);
            p.forcing_type = (strcmp(sval, "CONST_FLOWRATE") == 0) ? 1 : 0;
        }
        else {
            fgets(sval, sizeof(sval), fp);
        }
    }
    fclose(fp);

    // If NOUT_STATS not set, default to same interval as NSTAT
    if (p.nout_stats == 0) p.nout_stats = p.nstat;

    // ----------------------------------------------------------------
    // Derived quantities — MPM-STD style
    // ----------------------------------------------------------------
    p.dx = p.Lx / p.Nx;
    p.dy = p.Ly / p.Ny;

    if (p.non_dimensional) {
        // MPM-STD 방식 bulk-velocity 무차원화:
        //   Re = Re_b = U_b * h / nu  (벌크 Reynolds 수)
        //   nu = 1/Re
        //   dpdx = -3*nu/h^2  (Poiseuille: 이 값이 U_b=1, Umx=3/2를 보장)
        //   → MPM-STD Umx=1.5 고정과 동일한 효과
        // DPDX 입력값이 있으면 override (단, 비표준 케이스용)
        p.nu = 1.0 / p.Re;
        const double h = 0.5 * p.Lz;
        p.dpdx = (p.dpdx_in != 0.0) ? p.dpdx_in
                                     : -3.0 * p.nu / (h * h);
    } else {
        // Dimensional mode: nu = mu / rho  (like MPM-STD Cmu = Nu when non_dimensional=false)
        p.nu   = p.mu / p.rho;
        p.dpdx = p.dpdx_in;
    }

    // Re_tau is always *derived* from the forcing and viscosity:
    //   balance: -dp/dx = tau_w / h  =>  tau_w = |dpdx| * h
    //   u_tau  = sqrt(tau_w / rho)   =>  Re_tau = u_tau * h / nu
    {
        double h     = 0.5 * p.Lz;
        double tau_w = std::abs(p.dpdx) * h;          // (rho = 1 in non-dim; otherwise /rho)
        double rho_  = p.non_dimensional ? 1.0 : p.rho;
        double u_tau = std::sqrt(tau_w / rho_);
        p.Re_tau = u_tau * h / p.nu;
    }
}

// ============================================================
//  compute_lambda2 — middle eigenvalue of symmetric 3x3 matrix
//  Uses the analytical cosine/Cardano method.
// ============================================================
static double compute_lambda2(double M11, double M22, double M33,
                               double M12, double M13, double M23)
{
    double p1 = M12*M12 + M13*M13 + M23*M23;
    if (p1 < 1e-30) {
        double e[3] = {M11, M22, M33};
        std::sort(e, e + 3);
        return e[1];
    }
    double q  = (M11 + M22 + M33) / 3.0;
    double b11 = M11 - q, b22 = M22 - q, b33 = M33 - q;
    double p2 = b11*b11 + b22*b22 + b33*b33 + 2.0*p1;
    double p  = std::sqrt(p2 / 6.0);
    double B11 = b11/p, B22 = b22/p, B33 = b33/p;
    double B12 = M12/p, B13 = M13/p, B23 = M23/p;
    double r = 0.5 * (B11*(B22*B33 - B23*B23)
                    - B12*(B12*B33 - B23*B13)
                    + B13*(B12*B23 - B22*B13));
    r = std::max(-1.0, std::min(1.0, r));
    double phi  = std::acos(r) / 3.0;
    double eig1 = q + 2.0*p*std::cos(phi);
    double eig3 = q + 2.0*p*std::cos(phi + 2.0*M_PI/3.0);
    double eig2 = 3.0*q - eig1 - eig3;
    return eig2;
}

// ============================================================
//  write_tecplot_field — each rank appends its z-slab
//  Tecplot POINT format, sequential write via rank ordering
//  Outputs: X Y Z U V W P Q Lambda2
// ============================================================
void write_tecplot_field(const std::string& filename,
                         const SimParams& p,
                         const Grid& g,
                         const std::vector<double>& u,
                         const std::vector<double>& v,
                         const std::vector<double>& w,
                         const std::vector<double>& pr,
                         int kstart, int nz_local,
                         int myrank, int nprocs,
                         MPI_Comm comm)
{
    const int Nx = p.Nx, Ny = p.Ny;
    const int plane = Nx * Ny;

    // Rank 0 writes header
    if (myrank == 0) {
        FILE* fp = fopen(filename.c_str(), "w");
        if (!fp) {
            fprintf(stderr, "[write_tecplot_field] Cannot open %s\n", filename.c_str());
            MPI_Abort(comm, 1);
        }
        fprintf(fp, "TITLE = \"Channel Flow Field\"\n");
        fprintf(fp, "VARIABLES = \"X\" \"Y\" \"Z\" \"U\" \"V\" \"W\" \"P\" \"Q\" \"Lambda2\"\n");
        fprintf(fp, "ZONE T=\"Field\", I=%d, J=%d, K=%d, DATAPACKING=POINT\n",
                Nx, Ny, p.Nz);
        fclose(fp);
    }
    MPI_Barrier(comm);

    const double inv_2dx = 0.5 / p.dx;
    const double inv_2dy = 0.5 / p.dy;

    // Extended zc for wall ghost-cell positions (mirror symmetry)
    auto zc_ext = [&](int kg) -> double {
        if (kg < 0)     return -g.zc[0];
        if (kg >= p.Nz) return 2.0*p.Lz - g.zc[p.Nz - 1];
        return g.zc[kg];
    };

    // Each rank writes its z-slab in order
    for (int r = 0; r < nprocs; ++r) {
        if (myrank == r) {
            FILE* fp = fopen(filename.c_str(), "a");
            if (!fp) {
                fprintf(stderr, "[write_tecplot_field] Cannot open %s (rank %d)\n",
                        filename.c_str(), myrank);
                MPI_Abort(comm, 1);
            }
            // k=1..nz_local are interior cells (k=0, nz_local+1 are ghosts)
            for (int kl = 1; kl <= nz_local; ++kl) {
                int kg = kstart + kl - 1;
                double z      = g.zc[kg];
                double inv_dz  = 1.0 / g.dz[kg];
                double inv_2dz = 0.5 / (zc_ext(kg + 1) - zc_ext(kg - 1));

                for (int j = 0; j < Ny; ++j) {
                    int jp = (j + 1) % Ny, jm = (j - 1 + Ny) % Ny;
                    double y = (j + 0.5) * p.dy;
                    for (int i = 0; i < Nx; ++i) {
                        int ip = (i + 1) % Nx, im = (i - 1 + Nx) % Nx;
                        double x = (i + 0.5) * p.dx;

                        int c   = kl     * plane + j  * Nx + i;
                        int cip = kl     * plane + j  * Nx + ip;
                        int cim = kl     * plane + j  * Nx + im;
                        int cjp = kl     * plane + jp * Nx + i;
                        int cjm = kl     * plane + jm * Nx + i;
                        int ckp = (kl+1) * plane + j  * Nx + i;
                        int ckm = (kl-1) * plane + j  * Nx + i;

                        double dudx = (u[cip] - u[cim]) * inv_2dx;
                        double dudy = (u[cjp] - u[cjm]) * inv_2dy;
                        double dudz = (u[ckp] - u[ckm]) * inv_2dz;

                        double dvdx = (v[cip] - v[cim]) * inv_2dx;
                        double dvdy = (v[cjp] - v[cjm]) * inv_2dy;
                        double dvdz = (v[ckp] - v[ckm]) * inv_2dz;

                        // wc at (kl, j, i±1) and (kl, j±1, i) for dwdx, dwdy
                        double wc_ip = 0.5*(w[cip] + w[(kl+1)*plane + j *Nx + ip]);
                        double wc_im = 0.5*(w[cim] + w[(kl+1)*plane + j *Nx + im]);
                        double wc_jp = 0.5*(w[cjp] + w[(kl+1)*plane + jp*Nx + i ]);
                        double wc_jm = 0.5*(w[cjm] + w[(kl+1)*plane + jm*Nx + i ]);
                        double wc    = 0.5*(w[c]   + w[ckp]);

                        double dwdx = (wc_ip - wc_im) * inv_2dx;
                        double dwdy = (wc_jp - wc_jm) * inv_2dy;
                        double dwdz = (w[ckp] - w[c]) * inv_dz;

                        double S11 = dudx, S22 = dvdy, S33 = dwdz;
                        double S12 = 0.5*(dudy + dvdx);
                        double S13 = 0.5*(dudz + dwdx);
                        double S23 = 0.5*(dvdz + dwdy);
                        double W12 = 0.5*(dudy - dvdx);
                        double W13 = 0.5*(dudz - dwdx);
                        double W23 = 0.5*(dvdz - dwdy);

                        double normS2 = S11*S11 + S22*S22 + S33*S33
                                      + 2.0*(S12*S12 + S13*S13 + S23*S23);
                        double normW2 = 2.0*(W12*W12 + W13*W13 + W23*W23);
                        double Q = 0.5*(normW2 - normS2);

                        double M11 = S11*S11 + S12*S12 + S13*S13
                                   - W12*W12 - W13*W13;
                        double M22 = S12*S12 + S22*S22 + S23*S23
                                   - W12*W12 - W23*W23;
                        double M33 = S13*S13 + S23*S23 + S33*S33
                                   - W13*W13 - W23*W23;
                        double M12 = S11*S12 + S12*S22 + S13*S23 - W13*W23;
                        double M13 = S11*S13 + S12*S23 + S13*S33 + W12*W23;
                        double M23 = S12*S13 + S22*S23 + S23*S33 - W12*W13;

                        double lam2 = compute_lambda2(M11, M22, M33, M12, M13, M23);

                        fprintf(fp, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
                                x, y, z,
                                u[c], v[c], wc, pr[c],
                                Q, lam2);
                    }
                }
            }
            fclose(fp);
        }
        MPI_Barrier(comm);
    }
}

// ============================================================
//  write_tecplot_stats — rank 0 writes 1D z-profile
// ============================================================
void write_tecplot_stats(const std::string& filename,
                         const SimParams& p,
                         const Grid& g,
                         const std::vector<double>& U_mean,
                         const std::vector<double>& W_mean,
                         const std::vector<double>& u_rms,
                         const std::vector<double>& v_rms,
                         const std::vector<double>& w_rms,
                         const std::vector<double>& uw_stress,
                         const std::vector<double>& P_mean,
                         int step)
{
    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "[write_tecplot_stats] Cannot open %s\n", filename.c_str());
        return;
    }
    fprintf(fp, "TITLE = \"Channel Flow Statistics (step=%d)\"\n", step);
    fprintf(fp, "VARIABLES = \"Z\" \"Z_plus\" \"U_mean\" \"W_mean\""
                " \"u_rms\" \"v_rms\" \"w_rms\" \"uw_stress\" \"P_mean\"\n");
    fprintf(fp, "ZONE T=\"Stats\", I=%d, J=1, K=1, DATAPACKING=POINT\n", p.Nz);

    for (int k = 0; k < p.Nz; ++k) {
        double zp = g.zc[k] * p.Re_tau;   // z+ = z * Re_tau  (wall units: h=1)
        fprintf(fp, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
                g.zc[k], zp,
                U_mean[k], W_mean[k],
                u_rms[k], v_rms[k], w_rms[k],
                uw_stress[k], P_mean[k]);
    }
    fclose(fp);
}
