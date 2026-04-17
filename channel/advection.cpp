#include "advection.hpp"
#include <cmath>

// ============================================================
//  Helper: periodic index wrap for x and y
// ============================================================
static inline int pbc(int i, int N) {
    return (i + N) % N;
}

// ============================================================
//  compute_advection_uvw  —  DIVERGENCE (conservative) form
//
//  Staggered-grid layout (0-based, periodic x,y):
//    u[i,j,k]  at face between cell i and cell i+1 in x
//              position: x = (i+1)*dx,  y = (j+0.5)*dy,  z = zc[k]
//    v[i,j,k]  at face between cell j and cell j+1 in y
//              position: x = (i+0.5)*dx, y = (j+1)*dy,   z = zc[k]
//    w[i,j,k]  at face between cell k-1 and cell k in z
//              position: x = (i+0.5)*dx, y = (j+0.5)*dy, z = zf[k]
//    p[i,j,k]  at cell center
//              position: x = (i+0.5)*dx, y = (j+0.5)*dy, z = zc[k]
//
//  Conservative form:
//    N(u) = -[ d(uu)/dx + d(vu)/dy + d(wu)/dz ]
//    N(v) = -[ d(uv)/dx + d(vv)/dy + d(wv)/dz ]
//    N(w) = -[ d(uw)/dx + d(vw)/dy + d(ww)/dz ]
//
//  Fluxes are evaluated at control-volume (CV) faces by
//  interpolating both the transporting velocity and the
//  transported momentum to the CV face location.
//
//  Indexing: IDX(i,j,k) = k*(Nx*Ny) + j*Nx + i
//    k=0,nzl+1: ghosts   k=1..nzl: interior
// ============================================================
void compute_advection_uvw(
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    std::vector<double>& Nu,
    std::vector<double>& Nv,
    std::vector<double>& Nw,
    const SimParams& p,
    const Grid& g,
    const ZSlab& slab)
{
    const int Nx  = p.Nx, Ny = p.Ny;
    const int nzl = slab.nz_local;
    const double dx = p.dx, dy = p.dy;

    auto IDX = [&](int i, int j, int kl) {
        return kl * Nx * Ny + j * Nx + i;
    };

    // ================================================================
    //  N(u) = -[ d(uu)/dx + d(vu)/dy + d(wu)/dz ]
    //
    //  u-CV for u[i,j,kl]:
    //    x: from cell-center i  to cell-center i+1
    //    y: from y-face j       to y-face j+1   (across cell j)
    //    z: from z-face kl      to z-face kl+1  (across cell kl)
    // ================================================================
    for (int kl = 1; kl <= nzl; ++kl) {
        int kg = slab.kstart + kl - 1;
        double dz_c = g.dz[kg];

        for (int j = 0; j < Ny; ++j) {
            int jm = pbc(j - 1, Ny), jp = pbc(j + 1, Ny);
            for (int i = 0; i < Nx; ++i) {
                int im = pbc(i - 1, Nx), ip = pbc(i + 1, Nx);

                // --- d(uu)/dx ---
                // CV x-faces are at cell centers i and i+1.
                // At cell-center i+1: U = 0.5*(u[i]+u[ip]),  mom = same
                // At cell-center i  : U = 0.5*(u[im]+u[i]),  mom = same
                double U_R = 0.5 * (u[IDX(i, j, kl)] + u[IDX(ip, j, kl)]);
                double U_L = 0.5 * (u[IDX(im, j, kl)] + u[IDX(i, j, kl)]);
                double flux_uu = (U_R * U_R - U_L * U_L) / dx;

                // --- d(vu)/dy ---
                // CV y-faces are at y-face j+1 and y-face j.
                // Top face (y-face j+1):
                //   V_top = 0.5*(v[i,j] + v[ip,j])  → v interp to u's x-pos
                //   u_top = 0.5*(u[i,j] + u[i,jp])   → u interp to top face
                // Bottom face (y-face j):
                //   V_bot = 0.5*(v[i,jm] + v[ip,jm])
                //   u_bot = 0.5*(u[i,jm] + u[i,j])
                double V_top = 0.5 * (v[IDX(i, j, kl)] + v[IDX(ip, j, kl)]);
                double u_top = 0.5 * (u[IDX(i, j, kl)] + u[IDX(i, jp, kl)]);
                double V_bot = 0.5 * (v[IDX(i, jm, kl)] + v[IDX(ip, jm, kl)]);
                double u_bot = 0.5 * (u[IDX(i, jm, kl)] + u[IDX(i, j, kl)]);
                double flux_vu = (V_top * u_top - V_bot * u_bot) / dy;

                // --- d(wu)/dz ---
                // CV z-faces are at z-face kl+1 and z-face kl.
                // Upper face (z-face kl+1):
                //   W_up = 0.5*(w[i,j,kl+1] + w[ip,j,kl+1])
                //   u_up = 0.5*(u[i,j,kl] + u[i,j,kl+1])
                // Lower face (z-face kl):
                //   W_dn = 0.5*(w[i,j,kl] + w[ip,j,kl])
                //   u_dn = 0.5*(u[i,j,kl-1] + u[i,j,kl])
                double W_up = 0.5 * (w[IDX(i, j, kl + 1)] + w[IDX(ip, j, kl + 1)]);
                double u_up = 0.5 * (u[IDX(i, j, kl)] + u[IDX(i, j, kl + 1)]);
                double W_dn = 0.5 * (w[IDX(i, j, kl)] + w[IDX(ip, j, kl)]);
                double u_dn = 0.5 * (u[IDX(i, j, kl - 1)] + u[IDX(i, j, kl)]);
                double flux_wu = (W_up * u_up - W_dn * u_dn) / dz_c;

                Nu[IDX(i, j, kl)] = -(flux_uu + flux_vu + flux_wu);
            }
        }
    }

    // ================================================================
    //  N(v) = -[ d(uv)/dx + d(vv)/dy + d(wv)/dz ]
    //
    //  v-CV for v[i,j,kl]:
    //    x: from x-face i       to x-face i+1   (across cell i)
    //    y: from cell-center j  to cell-center j+1
    //    z: from z-face kl      to z-face kl+1  (across cell kl)
    // ================================================================
    for (int kl = 1; kl <= nzl; ++kl) {
        int kg = slab.kstart + kl - 1;
        double dz_c = g.dz[kg];

        for (int j = 0; j < Ny; ++j) {
            int jm = pbc(j - 1, Ny), jp = pbc(j + 1, Ny);
            for (int i = 0; i < Nx; ++i) {
                int im = pbc(i - 1, Nx), ip = pbc(i + 1, Nx);

                // --- d(uv)/dx ---
                // CV x-faces at x-face i+1 and x-face i.
                // Right face (x-face i+1):
                //   U_R = 0.5*(u[i,j] + u[i,jp])    → u interp to v's y-pos
                //   v_R = 0.5*(v[i,j] + v[ip,j])     → v interp to right face
                // Left face (x-face i):
                //   U_L = 0.5*(u[im,j] + u[im,jp])
                //   v_L = 0.5*(v[im,j] + v[i,j])
                double U_R = 0.5 * (u[IDX(i, j, kl)] + u[IDX(i, jp, kl)]);
                double v_R = 0.5 * (v[IDX(i, j, kl)] + v[IDX(ip, j, kl)]);
                double U_L = 0.5 * (u[IDX(im, j, kl)] + u[IDX(im, jp, kl)]);
                double v_L = 0.5 * (v[IDX(im, j, kl)] + v[IDX(i, j, kl)]);
                double flux_uv = (U_R * v_R - U_L * v_L) / dx;

                // --- d(vv)/dy ---
                // CV y-faces at cell-center j+1 and cell-center j.
                double V_R = 0.5 * (v[IDX(i, j, kl)] + v[IDX(i, jp, kl)]);
                double V_L = 0.5 * (v[IDX(i, jm, kl)] + v[IDX(i, j, kl)]);
                double flux_vv = (V_R * V_R - V_L * V_L) / dy;

                // --- d(wv)/dz ---
                // CV z-faces at z-face kl+1 and z-face kl.
                // Upper face:
                //   W_up = 0.5*(w[i,j,kl+1] + w[i,jp,kl+1])
                //   v_up = 0.5*(v[i,j,kl] + v[i,j,kl+1])
                // Lower face:
                //   W_dn = 0.5*(w[i,j,kl] + w[i,jp,kl])
                //   v_dn = 0.5*(v[i,j,kl-1] + v[i,j,kl])
                double W_up = 0.5 * (w[IDX(i, j, kl + 1)] + w[IDX(i, jp, kl + 1)]);
                double v_up = 0.5 * (v[IDX(i, j, kl)] + v[IDX(i, j, kl + 1)]);
                double W_dn = 0.5 * (w[IDX(i, j, kl)] + w[IDX(i, jp, kl)]);
                double v_dn = 0.5 * (v[IDX(i, j, kl - 1)] + v[IDX(i, j, kl)]);
                double flux_wv = (W_up * v_up - W_dn * v_dn) / dz_c;

                Nv[IDX(i, j, kl)] = -(flux_uv + flux_vv + flux_wv);
            }
        }
    }

    // ================================================================
    //  N(w) = -[ d(uw)/dx + d(vw)/dy + d(ww)/dz ]
    //
    //  w-CV for w[i,j,kl]:
    //    x: from x-face i       to x-face i+1   (across cell i)
    //    y: from y-face j       to y-face j+1   (across cell j)
    //    z: from cell-center kg-1 to cell-center kg
    //
    //  w[i,j,kl] is at z-face between cell (kg-1) and cell kg,
    //  where kg = slab.kstart + kl - 1.
    //  kl=1..nzl are the local interior faces.
    // ================================================================
    for (int kl = 1; kl <= nzl; ++kl) {
        int kg = slab.kstart + kl - 1;

        // w-CV spans from zc[kg-1] to zc[kg] → height = dzf[kg]
        double dz_cv = g.dzf[kg];

        for (int j = 0; j < Ny; ++j) {
            int jm = pbc(j - 1, Ny), jp = pbc(j + 1, Ny);
            for (int i = 0; i < Nx; ++i) {
                int im = pbc(i - 1, Nx), ip = pbc(i + 1, Nx);

                // --- d(uw)/dx ---
                // CV x-faces at x-face i+1 and x-face i.
                // Right face:
                //   U_R = 0.5*(u[i,j,kl-1] + u[i,j,kl])   → u interp to w's z-pos
                //   w_R = 0.5*(w[i,j,kl] + w[ip,j,kl])      → w interp to right face
                // Left face:
                //   U_L = 0.5*(u[im,j,kl-1] + u[im,j,kl])
                //   w_L = 0.5*(w[im,j,kl] + w[i,j,kl])
                double U_R = 0.5 * (u[IDX(i, j, kl - 1)] + u[IDX(i, j, kl)]);
                double w_R = 0.5 * (w[IDX(i, j, kl)] + w[IDX(ip, j, kl)]);
                double U_L = 0.5 * (u[IDX(im, j, kl - 1)] + u[IDX(im, j, kl)]);
                double w_L = 0.5 * (w[IDX(im, j, kl)] + w[IDX(i, j, kl)]);
                double flux_uw = (U_R * w_R - U_L * w_L) / dx;

                // --- d(vw)/dy ---
                // CV y-faces at y-face j+1 and y-face j.
                // Top face:
                //   V_top = 0.5*(v[i,j,kl-1] + v[i,j,kl])   → v interp to w's z-pos
                //   w_top = 0.5*(w[i,j,kl] + w[i,jp,kl])      → w interp to top face
                // Bottom face:
                //   V_bot = 0.5*(v[i,jm,kl-1] + v[i,jm,kl])
                //   w_bot = 0.5*(w[i,jm,kl] + w[i,j,kl])
                double V_top = 0.5 * (v[IDX(i, j, kl - 1)] + v[IDX(i, j, kl)]);
                double w_top = 0.5 * (w[IDX(i, j, kl)] + w[IDX(i, jp, kl)]);
                double V_bot = 0.5 * (v[IDX(i, jm, kl - 1)] + v[IDX(i, jm, kl)]);
                double w_bot = 0.5 * (w[IDX(i, jm, kl)] + w[IDX(i, j, kl)]);
                double flux_vw = (V_top * w_top - V_bot * w_bot) / dy;

                // --- d(ww)/dz ---
                // CV z-faces at cell-center kg and cell-center kg-1.
                // Upper face (cell-center kg):
                //   W_up = 0.5*(w[kl] + w[kl+1])
                // Lower face (cell-center kg-1):
                //   W_dn = 0.5*(w[kl-1] + w[kl])
                double W_up = 0.5 * (w[IDX(i, j, kl)] + w[IDX(i, j, kl + 1)]);
                double W_dn = 0.5 * (w[IDX(i, j, kl - 1)] + w[IDX(i, j, kl)]);
                double flux_ww = (W_up * W_up - W_dn * W_dn) / dz_cv;

                Nw[IDX(i, j, kl)] = -(flux_uw + flux_vw + flux_ww);
            }
        }
    }
}

// ============================================================
//  adams_bashforth — in-place AB2 update
//  phi^* = phi + dt * (1.5*N_new - 0.5*N_old)  [AB2]
//  or phi^* = phi + dt * N_new                  [Euler, first step]
// ============================================================
void adams_bashforth(
    std::vector<double>& phi,
    const std::vector<double>& N_new,
    const std::vector<double>& N_old,
    double dt,
    int Nx, int Ny, int nz_local,
    bool first_step)
{
    const int plane = Nx * Ny;
    if (first_step) {
        for (int kl = 1; kl <= nz_local; ++kl)
            for (int n = 0; n < plane; ++n) {
                int idx = kl * plane + n;
                phi[idx] += dt * N_new[idx];
            }
    } else {
        for (int kl = 1; kl <= nz_local; ++kl)
            for (int n = 0; n < plane; ++n) {
                int idx = kl * plane + n;
                phi[idx] += dt * (1.5 * N_new[idx] - 0.5 * N_old[idx]);
            }
    }
}
