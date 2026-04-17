#include "statistics.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>

// ============================================================
//  Constructor / reset
// ============================================================
Statistics::Statistics(const SimParams& p, const Grid& g, const ZSlab& slab)
    : p_(p), g_(g), slab_(slab), n_(0)
{
    const int nzl = slab.nz_local;
    U_m_.assign(nzl, 0.0);   U2_m_.assign(nzl, 0.0);
    V_m_.assign(nzl, 0.0);   V2_m_.assign(nzl, 0.0);
    Wc_m_.assign(nzl, 0.0);  Wc2_m_.assign(nzl, 0.0);
    UWc_m_.assign(nzl, 0.0); P_m_.assign(nzl, 0.0);
}

void Statistics::reset() {
    n_ = 0;
    for (auto* v : {&U_m_, &U2_m_, &V_m_, &V2_m_,
                    &Wc_m_, &Wc2_m_, &UWc_m_, &P_m_})
        std::fill(v->begin(), v->end(), 0.0);
}

// ============================================================
//  gather_to_global — pack local slice, Allreduce(SUM) → Nz array
// ============================================================
void Statistics::gather_to_global(const std::vector<double>& local,
                                   std::vector<double>& global) const
{
    const int Nz  = p_.Nz;
    const int nzl = slab_.nz_local;
    global.assign(Nz, 0.0);
    std::vector<double> tmp(Nz, 0.0);
    for (int kl = 0; kl < nzl; ++kl)
        tmp[slab_.kstart + kl] = local[kl];
    MPI_Allreduce(tmp.data(), global.data(), Nz, MPI_DOUBLE, MPI_SUM, slab_.comm);
}

// ============================================================
//  accumulate — zero MPI, pure local arithmetic
//
//  For each z-level: compute xy-mean of u, u², v, v², wc, wc², u·wc, p
//  and update running averages with Welford formula.
//  wc = 0.5*(w_face_bottom + w_face_top): face→cell-centre interpolation.
// ============================================================
void Statistics::accumulate(const std::vector<double>& u,
                             const std::vector<double>& v,
                             const std::vector<double>& w,
                             const std::vector<double>& pr)
{
    const int Nx    = p_.Nx, Ny = p_.Ny;
    const int nzl   = slab_.nz_local;
    const int plane = Nx * Ny;
    const double inv_NxNy = 1.0 / plane;

    ++n_;
    const double inv_n = 1.0 / n_;

    for (int kl = 1; kl <= nzl; ++kl) {
        double su=0, su2=0, sv=0, sv2=0, swc=0, swc2=0, suwc=0, sp=0;

        const double* uk  = u.data()  +  kl      * plane;
        const double* vk  = v.data()  +  kl      * plane;
        const double* wkb = w.data()  +  kl      * plane;   // face: bottom of cell kl
        const double* wkt = w.data()  + (kl + 1) * plane;   // face: top    of cell kl
        const double* pk  = pr.data() +  kl      * plane;

        for (int n = 0; n < plane; ++n) {
            const double ui  = uk[n];
            const double vi  = vk[n];
            const double wci = 0.5 * (wkb[n] + wkt[n]);
            const double pi  = pk[n];
            su   += ui;
            su2  += ui  * ui;
            sv   += vi;
            sv2  += vi  * vi;
            swc  += wci;
            swc2 += wci * wci;
            suwc += ui  * wci;
            sp   += pi;
        }

        const int kl0    = kl - 1;
        const double um   = su   * inv_NxNy;
        const double u2m  = su2  * inv_NxNy;
        const double vm   = sv   * inv_NxNy;
        const double v2m  = sv2  * inv_NxNy;
        const double wcm  = swc  * inv_NxNy;
        const double wc2m = swc2 * inv_NxNy;
        const double uwcm = suwc * inv_NxNy;
        const double pm   = sp   * inv_NxNy;

        U_m_[kl0]   += (um   - U_m_[kl0])   * inv_n;
        U2_m_[kl0]  += (u2m  - U2_m_[kl0])  * inv_n;
        V_m_[kl0]   += (vm   - V_m_[kl0])   * inv_n;
        V2_m_[kl0]  += (v2m  - V2_m_[kl0])  * inv_n;
        Wc_m_[kl0]  += (wcm  - Wc_m_[kl0])  * inv_n;
        Wc2_m_[kl0] += (wc2m - Wc2_m_[kl0]) * inv_n;
        UWc_m_[kl0] += (uwcm - UWc_m_[kl0]) * inv_n;
        P_m_[kl0]   += (pm   - P_m_[kl0])   * inv_n;
    }
}

// ============================================================
//  write — gather to global (8 Allreduce), then output on rank 0
// ============================================================
void Statistics::write(const std::string& filename, int step) const
{
    const int    Nz     = p_.Nz;
    const double inv_nu = 1.0 / p_.nu;

    // Gather all 8 quantities (8 Allreduce calls)
    std::vector<double> U_g, U2_g, V_g, V2_g, Wc_g, Wc2_g, UWc_g, P_g;
    gather_to_global(U_m_,   U_g);
    gather_to_global(U2_m_,  U2_g);
    gather_to_global(V_m_,   V_g);
    gather_to_global(V2_m_,  V2_g);
    gather_to_global(Wc_m_,  Wc_g);
    gather_to_global(Wc2_m_, Wc2_g);
    gather_to_global(UWc_m_, UWc_g);
    gather_to_global(P_m_,   P_g);

    // u_tau from time-averaged wall shear stress at k=0 (bottom wall cell)
    //   τ_w = ν · <U(k=0)>_tavg / z_c[0]
    const double tau_w = p_.nu * std::abs(U_g[0]) / g_.zc[0];
    const double u_tau = std::sqrt(tau_w);

    if (slab_.myrank != 0) return;

    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "[Statistics::write] Cannot open %s\n", filename.c_str());
        return;
    }

    fprintf(fp, "TITLE = \"Channel Flow Statistics (step=%d, n_samples=%d)\"\n",
            step, n_);
    fprintf(fp, "VARIABLES = \"Z\" \"Z_plus\""
                " \"U_mean\" \"W_mean\""
                " \"u_rms\" \"v_rms\" \"w_rms\""
                " \"uw_stress\" \"P_mean\"\n");
    fprintf(fp, "ZONE T=\"Stats\", I=%d, J=1, K=1, DATAPACKING=POINT\n", Nz);

    for (int k = 0; k < Nz; ++k) {
        // y+ from actual time-averaged u_tau
        const double zp    = g_.zc[k] * u_tau * inv_nu;

        // RMS and stress via 제평제: <u'²> = <u²> - <u>²
        const double u_rms = std::sqrt(std::max(U2_g[k]  - U_g[k]  * U_g[k],  0.0));
        const double v_rms = std::sqrt(std::max(V2_g[k]  - V_g[k]  * V_g[k],  0.0));
        const double w_rms = std::sqrt(std::max(Wc2_g[k] - Wc_g[k] * Wc_g[k], 0.0));
        const double uw    = UWc_g[k] - U_g[k] * Wc_g[k];

        fprintf(fp, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
                g_.zc[k], zp,
                U_g[k], Wc_g[k],
                u_rms, v_rms, w_rms,
                uw, P_g[k]);
    }
    fclose(fp);
}
