#ifndef GRID_HPP
#define GRID_HPP

#include <vector>
#include "params.hpp"

// ============================================================
//  Grid — z-direction coordinates and spacing arrays
//
//  Notation (Nz cells, Nz+1 faces):
//    zf[k]   : face position,  k = 0..Nz        (Nz+1 values)
//    zc[k]   : cell-center,    k = 0..Nz-1       (Nz values)
//    dz[k]   : cell height,    dz[k] = zf[k+1]-zf[k], k=0..Nz-1
//    dzf[k]  : face spacing,   dzf[k] = zc[k]-zc[k-1], k=1..Nz-1
//              (distance between adjacent cell centers, used in
//               second-order non-uniform FD on z-faces)
// ============================================================
struct Grid {
    std::vector<double> zf;   // size Nz+1
    std::vector<double> zc;   // size Nz
    std::vector<double> dz;   // size Nz     (cell height)
    std::vector<double> dzf;  // size Nz+1   (face spacing; dzf[0] and dzf[Nz] are ghost)
    double dx, dy;
};

void build_grid(const SimParams& p, Grid& g);

// Tridiagonal coefficients for the non-uniform z second derivative
// operating on a cell-center array of size Nz with Dirichlet BC.
// Returns vectors a (lower), b (main diagonal), c (upper) of size Nz.
// Wall ghost-cell reflection is included in a[0] and c[Nz-1].
struct ZTriCoeffs {
    std::vector<double> a, b, c;
};

ZTriCoeffs build_z_laplacian_coeffs(const Grid& g, int Nz);

#endif // GRID_HPP
