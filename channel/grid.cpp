#include "grid.hpp"
#include <cmath>
#include <stdexcept>

// ============================================================
//  build_grid — generate z-coordinates with tanh stretching
// ============================================================
void build_grid(const SimParams& p, Grid& g) {
    const int Nz = p.Nz;
    const double Lz = p.Lz;
    const double beta = p.stretch_beta;

    g.zf.resize(Nz + 1);
    g.zc.resize(Nz);
    g.dz.resize(Nz);
    g.dzf.resize(Nz + 1);
    g.dx = p.dx;
    g.dy = p.dy;

    // --- z face positions ---
    if (beta < 1.0e-10) {
        // Uniform grid
        for (int k = 0; k <= Nz; ++k)
            g.zf[k] = Lz * k / Nz;
    } else {
        // tanh inflation stretching
        double tanh_beta = tanh(beta);
        for (int k = 0; k <= Nz; ++k)
            g.zf[k] = 0.5 * Lz
                    * (1.0 - tanh(beta * (1.0 - 2.0 * k / Nz)) / tanh_beta);
    }

    // --- cell centers and cell heights ---
    for (int k = 0; k < Nz; ++k) {
        g.zc[k] = 0.5 * (g.zf[k] + g.zf[k + 1]);
        g.dz[k] = g.zf[k + 1] - g.zf[k];
    }

    // --- face spacings (between adjacent cell centers) ---
    // dzf[k] = zc[k] - zc[k-1]  for k=1..Nz-1
    // Ghost entries: dzf[0] uses mirror of bottom cell,
    //               dzf[Nz] uses mirror of top cell.
    g.dzf[0] = g.dz[0];          // ghost: same as first cell height
    for (int k = 1; k < Nz; ++k)
        g.dzf[k] = g.zc[k] - g.zc[k - 1];
    g.dzf[Nz] = g.dz[Nz - 1];   // ghost: same as last cell height
}

// ============================================================
//  build_z_laplacian_coeffs
//  Second-order FD for d²φ/dz² on non-uniform cell-center grid.
//
//  For interior cell k (1 <= k <= Nz-2):
//    a[k] =  1 / (dz[k-1] * dzf[k])
//    c[k] =  1 / (dz[k]   * dzf[k])
//    b[k] = -(a[k] + c[k])
//  where dzf[k] = (dz[k-1] + dz[k]) / 2 = zc[k] - zc[k-1]
//
//  Boundary cells with no-slip ghost reflection (u_ghost = -u_0):
//    k=0 (bottom): ghost at z = -zc[0] → only c term, a absorbed into b
//    k=Nz-1 (top): ghost at z = 2Lz-zc[Nz-1] → only a term, c absorbed into b
// ============================================================
ZTriCoeffs build_z_laplacian_coeffs(const Grid& g, int Nz) {
    ZTriCoeffs tc;
    tc.a.assign(Nz, 0.0);
    tc.b.assign(Nz, 0.0);
    tc.c.assign(Nz, 0.0);

    for (int k = 0; k < Nz; ++k) {
        double dz_km = g.dzf[k];         // zc[k] - zc[k-1]  (or ghost below)
        double dz_kp = g.dzf[k + 1];     // zc[k+1] - zc[k]  (or ghost above)
        double dz_c  = g.dz[k];          // cell height

        // Standard interior coefficients
        double a_k = 1.0 / (dz_c * dz_km);
        double c_k = 1.0 / (dz_c * dz_kp);

        if (k == 0) {
            // Bottom wall: no-slip ghost → φ_ghost = -φ_0
            // a term folds into diagonal (adds to b)
            tc.a[k] = 0.0;
            tc.c[k] = c_k;
            tc.b[k] = -(a_k + c_k) + (-a_k);  // -a_k from ghost reflection
        } else if (k == Nz - 1) {
            // Top wall: no-slip ghost → φ_ghost = -φ_{Nz-1}
            tc.a[k] = a_k;
            tc.c[k] = 0.0;
            tc.b[k] = -(a_k + c_k) + (-c_k);  // -c_k from ghost reflection
        } else {
            tc.a[k] = a_k;
            tc.b[k] = -(a_k + c_k);
            tc.c[k] = c_k;
        }
    }
    return tc;
}
