#include "tdma_local.hpp"

#include <cstddef>
#include <vector>

void tdma_many(double* __restrict A,
               double* __restrict B,
               double* __restrict C,
               double* __restrict D,
               int n_sys, int n_row) {

    double* B0 = B;
    double* C0 = C;
    double* D0 = D;

    // --- Preprocess (j=0) ---
    #pragma omp simd
    for (int i = 0; i < n_sys; ++i) {
        double inv_b = 1.0 / B0[i];
        D0[i] *= inv_b;
        C0[i] *= inv_b;
    }

    // --- Forward Elimination ---
    for (int j = 1; j < n_row; ++j) {
        double*       Aj  = A + (std::size_t)j * n_sys;
        double*       Bj  = B + (std::size_t)j * n_sys;
        double*       Cj  = C + (std::size_t)j * n_sys;
        double*       Dj  = D + (std::size_t)j * n_sys;
        const double* Cjm = C + (std::size_t)(j - 1) * n_sys;
        const double* Djm = D + (std::size_t)(j - 1) * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            double r = 1.0 / (Bj[i] - Aj[i] * Cjm[i]);
            Dj[i] = r * (Dj[i] - Aj[i] * Djm[i]);
            Cj[i] = r * Cj[i];
        }
    }

    // --- Backward Substitution ---
    for (int j = n_row - 2; j >= 0; --j) {
        double*       Cj  = C + (std::size_t)j * n_sys;
        double*       Dj  = D + (std::size_t)j * n_sys;
        const double* Djp = D + (std::size_t)(j + 1) * n_sys;

        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            Dj[i] -= Cj[i] * Djp[i];
        }
    }
}

void tdma_single(std::vector<double>& a, std::vector<double>& b,
                 std::vector<double>& c, std::vector<double>& d, int n) {

    d[0] = d[0] / b[0];
    c[0] = c[0] / b[0];

    for (int i = 1; i < n; ++i) {
        double r = 1.0 / (b[i] - a[i] * c[i - 1]);
        d[i] = r * (d[i] - a[i] * d[i - 1]);
        c[i] = r * c[i];
    }

    for (int i = n - 2; i >= 0; --i) {
        d[i] -= c[i] * d[i + 1];
    }
}

void tdma_cyclic_single(std::vector<double>& a, std::vector<double>& b,
                        std::vector<double>& c, std::vector<double>& d, int n) {

    // Sherman-Morrison auxiliary vector e
    std::vector<double> e(n, 0.0);
    e[1]     = -a[1];
    e[n - 1] = -c[n - 1];

    // --- Preprocess (i=1) ---
    d[1] = d[1] / b[1];
    e[1] = e[1] / b[1];
    c[1] = c[1] / b[1];

    // --- Forward Elimination ---
    for (int i = 2; i <= n - 1; ++i) {
        double rr = 1.0 / (b[i] - a[i] * c[i - 1]);
        d[i] = rr * (d[i] - a[i] * d[i - 1]);
        e[i] = rr * (e[i] - a[i] * e[i - 1]);
        c[i] = rr * c[i];
    }

    // --- Backward Substitution ---
    for (int i = n - 2; i >= 1; --i) {
        d[i] -= c[i] * d[i + 1];
        e[i] -= c[i] * e[i + 1];
    }

    // --- Solve for d[0] using cyclic boundary ---
    d[0] = (d[0] - a[0] * d[n - 1] - c[0] * d[1])
         / (b[0] + a[0] * e[n - 1] + c[0] * e[1]);

    // --- Back-substitute d[0] ---
    for (int i = 1; i <= n - 1; ++i) {
        d[i] += d[0] * e[i];
    }
}
