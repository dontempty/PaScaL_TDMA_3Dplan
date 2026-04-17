#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <vector>
#include <string>
#include "params.hpp"
#include "grid.hpp"
#include "mpi_comm.hpp"

// ============================================================
//  Statistics — running time-averaged channel flow statistics
//
//  accumulate(): zero MPI — pure local arithmetic.
//    Per sample: compute xy-means of u, u², v, v², wc, wc², u·wc, p
//    and update local running averages with Welford formula.
//
//  write(): MPI communication happens here only (8 Allreduce calls).
//    RMS and stress use 제평제:
//      u_rms  = sqrt( <u²> - <u>² )
//      <u'w'> = <u·wc> - <u><wc>
//
//  y+ uses actual time-averaged friction velocity:
//      τ_w = ν · <U(k=0)>_tavg / z_c[0]
//      u_τ = sqrt(τ_w)
//      y+  = z_c[k] · u_τ / ν
// ============================================================

class Statistics {
public:
    Statistics(const SimParams& p, const Grid& g, const ZSlab& slab);

    void accumulate(const std::vector<double>& u,
                    const std::vector<double>& v,
                    const std::vector<double>& w,
                    const std::vector<double>& pr);

    void write(const std::string& filename, int step) const;
    void reset();
    int  n_samples() const { return n_; }

private:
    const SimParams& p_;
    const Grid&      g_;
    const ZSlab&     slab_;
    int              n_;

    // Local running time-averages of xy-plane quantities [nzl].
    // Gathered to global only in write().
    std::vector<double> U_m_;    // <u>_xy
    std::vector<double> U2_m_;   // <u²>_xy
    std::vector<double> V_m_;    // <v>_xy
    std::vector<double> V2_m_;   // <v²>_xy
    std::vector<double> Wc_m_;   // <wc>_xy   (wc = face→cell-centre interp)
    std::vector<double> Wc2_m_;  // <wc²>_xy
    std::vector<double> UWc_m_;  // <u·wc>_xy
    std::vector<double> P_m_;    // <p>_xy

    void gather_to_global(const std::vector<double>& local,
                          std::vector<double>& global) const;
};

#endif // STATISTICS_HPP
