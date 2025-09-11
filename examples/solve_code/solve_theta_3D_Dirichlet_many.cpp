// solve_theta.cpp
#include "solve_theta.hpp"
#include "../examples_lab/save.hpp"
#include "iostream"
#include "index.hpp"
#include "timer.hpp"

#include <cmath>
#include <chrono> 

// solve_theta.cpp
solve_theta::solve_theta(const GlobalParams& params,
                         const MPITopology& topo,
                         MPISubdomain& sub)
    : params(params), topo(topo), sub(sub) {}

void solve_theta::solve_theta_plan_single(std::vector<double>& theta) 
{   
    int myrank, ierr;
    int time_step;      // Current time step
    double t_curr;      // Current simulation time

    // 내가 사용할거
    int i, j, k;
    int idx;
    int ij, jk, ik, ki;
    int ijk;
    int idx_ip, idx_im;
    int idx_jp, idx_jm;
    int idx_kp, idx_km;
    double dxdx, dydy, dzdz;
    double coef_x_a, coef_x_b, coef_x_c;
    double coef_y_a, coef_y_b, coef_y_c;
    double coef_z_a, coef_z_b, coef_z_c;
    
    // Loop and index variables (그냥 셀 개수)
    int nz1 = sub.nz_sub+1; // number of cell with ghost cell in z axis 
    int ny1 = sub.ny_sub+1; // number of cell with ghost cell in y axis 
    int nx1 = sub.nx_sub+1; // number of cell with ghost cell in x axis

    // 1) 계획(plan) 객체 선언
    PaScaL_TDMA::ptdma_plan_many px_many, py_many, pz_many;

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank==0) {
        std::cout << "Start to solve" << std::endl;
    }

    auto cx = topo.commX();
    int rankx = cx.myrank;
    PaScaL_TDMA tdma_x; // ijk
    tdma_x.PaScaL_TDMA_plan_many_create(px_many, sub.ny_sub-1, cx.myrank, cx.nprocs, cx.comm);
    std::vector<double> Axx((nx1-2)*(ny1-2)), Bxx((nx1-2)*(ny1-2)), Cxx((nx1-2)*(ny1-2)), Dxx((nx1-2)*(ny1-2));

    auto cy = topo.commY();
    int ranky = cy.myrank;
    PaScaL_TDMA tdma_y; // jki
    tdma_y.PaScaL_TDMA_plan_many_create(py_many, sub.nz_sub-1, cy.myrank, cy.nprocs, cy.comm);
    std::vector<double> Ayy((ny1-2)*(nz1-2)), Byy((ny1-2)*(nz1-2)), Cyy((ny1-2)*(nz1-2)), Dyy((ny1-2)*(nz1-2));

    auto cz = topo.commZ();
    int rankz = cz.myrank;
    PaScaL_TDMA tdma_z; // kij
    tdma_z.PaScaL_TDMA_plan_many_create(pz_many, sub.nx_sub-1, cz.myrank, cz.nprocs, cz.comm);
    std::vector<double> Azz((nz1-2)*(nx1-2)), Bzz((nz1-2)*(nx1-2)), Czz((nz1-2)*(nx1-2)), Dzz((nz1-2)*(nx1-2));
    
    std::vector<double> rhs_x(nx1 * ny1 * nz1, 0.0);
    std::vector<double> rhs_y(nx1 * ny1 * nz1, 0.0);
    std::vector<double> rhs_z(nx1 * ny1 * nz1, 0.0);
    std::vector<double> theta_z(nx1 * ny1 * nz1, 0.0);
    std::vector<double> theta_y(nx1 * ny1 * nz1, 0.0);
    
    double dt = params.dt;
    int max_iter = params.Nt;
    MultiTimer timer;
    timer.start("solve_heat");
    for (int t_step=0; t_step<max_iter; ++t_step) {

        // timer.start("rhs");
        // Calculating r.h.s -----------------------------------------------------------------------------------------------------

        // rhs_x ---------------------------
        for (k=0; k<nz1; ++k) {
            for (j=0; j<ny1; ++j) {
                for (i=1; i<nx1-1; ++i) {
                    ijk    = idx_ijk(i  , j, k, nx1, ny1);
                    idx_ip = idx_ijk(i+1, j, k, nx1, ny1);
                    idx_im = idx_ijk(i-1, j, k, nx1, ny1);
                    dxdx = sub.dmx_sub[i]*sub.dmx_sub[i];

                    coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * sub.theta_x_left_index[i] + (1.0/3.0) * sub.theta_x_right_index[i] );
                    coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * sub.theta_x_left_index[i] -     (2.0) * sub.theta_x_right_index[i] );
                    coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * sub.theta_x_left_index[i] + (5.0/3.0) * sub.theta_x_right_index[i] );
                    
                    rhs_x[ijk] = (coef_x_c*theta[idx_ip] + (1.0+coef_x_b)*theta[ijk] + coef_x_a*theta[idx_im]);
                }
            }
        }

        // sub.ghostcellUpdate(rhs_x, cx, cy, cz, params);

        // rhs_y ---------------------------
        for (i=1; i<nx1-1; ++i) {
            for (k=0; k<nz1; ++k) {
                for (j=1; j<ny1-1; ++j) {
                    ijk    = idx_ijk(i, j  , k, nx1, ny1);
                    idx_jp = idx_ijk(i, j+1, k, nx1, ny1);
                    idx_jm = idx_ijk(i, j-1, k, nx1, ny1);
                    dydy = sub.dmy_sub[j]*sub.dmy_sub[j];

                    coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) * sub.theta_y_left_index[j] + (1.0/3.0) * sub.theta_y_right_index[j] );
                    coef_y_b = (dt / 2.0 / dydy) * (-2.0 -     (2.0) * sub.theta_y_left_index[j] -     (2.0) * sub.theta_y_right_index[j] );
                    coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (1.0/3.0) * sub.theta_y_left_index[j] + (5.0/3.0) * sub.theta_y_right_index[j] );
                    
                    rhs_y[ijk] = (coef_y_c*rhs_x[idx_jp] + (1.0+coef_y_b)*rhs_x[ijk] + coef_y_a*rhs_x[idx_jm]);
                }
            }
        }

        // sub.ghostcellUpdate(rhs_y, cx, cy, cz, params);

        // rhs_z ---------------------------
        for (j=1; j<ny1-1; ++j) {
            for (i=1; i<nx1-1; ++i) {
                for (k=1; k<nz1-1; ++k) {
                    ijk    = idx_ijk(i, j, k  , nx1, ny1);
                    idx_kp = idx_ijk(i, j, k+1, nx1, ny1);
                    idx_km = idx_ijk(i, j, k-1, nx1, ny1);
                    dzdz = sub.dmz_sub[k]*sub.dmz_sub[k];

                    coef_z_a = (dt / 2.0 / dzdz) * ( 1.0 + (5.0/3.0) * sub.theta_z_left_index[k] + (1.0/3.0) * sub.theta_z_right_index[k] );
                    coef_z_b = (dt / 2.0 / dzdz) * (-2.0 -     (2.0) * sub.theta_z_left_index[k] -     (2.0) * sub.theta_z_right_index[k] );
                    coef_z_c = (dt / 2.0 / dzdz) * ( 1.0 + (1.0/3.0) * sub.theta_z_left_index[k] + (5.0/3.0) * sub.theta_z_right_index[k] );
                    
                    rhs_z[ijk] = (coef_z_c*rhs_y[idx_kp] + (1.0+coef_z_b)*rhs_y[ijk] + coef_z_a*rhs_y[idx_km]);

                    rhs_z[ijk] += dt * ( 3.0 * Pi*Pi * cos(Pi*sub.x_sub[i]) * cos(Pi*sub.y_sub[j]) * cos(Pi*sub.z_sub[k]));
                }
            }
        }

        // sub.ghostcellUpdate(rhs_z, cx, cy, cz, params);
        // std::cout << "[myrank] = " << myrank << "| [rhs] elapsed: " << timer.elapsed_ms("rhs") << " ms\n";

        // Calculating A matrix ----------------------------------------------------------------
        
        // bdy(z)
        for (j=1; j<ny1-1; ++j) {

            dydy = sub.dmy_sub[j]*sub.dmy_sub[j];
            coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) * sub.theta_y_left_index[j] + (1.0/3.0) * sub.theta_y_right_index[j] );
            coef_y_b = (dt / 2.0 / dydy) * (-2.0 -     (2.0) * sub.theta_y_left_index[j] -     (2.0) * sub.theta_y_right_index[j] );
            coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (1.0/3.0) * sub.theta_y_left_index[j] + (5.0/3.0) * sub.theta_y_right_index[j] ); 

            for (i=1; i<nx1-1; ++i) {

                dxdx = sub.dmx_sub[i]*sub.dmx_sub[i];
                coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * sub.theta_x_left_index[i] + (1.0/3.0) * sub.theta_x_right_index[i] );
                coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * sub.theta_x_left_index[i] -     (2.0) * sub.theta_x_right_index[i] );
                coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * sub.theta_x_left_index[i] + (5.0/3.0) * sub.theta_x_right_index[i] );

                // k=0
                dzdz = sub.dmz_sub[0]*sub.dmz_sub[0];
                coef_z_a = (dt / 2.0 / dzdz) * ( 1.0 + (5.0/3.0) );

                idx    = idx_ij(i  , j-1, nx1);
                idx_ip = idx_ij(i+1, j-1, nx1);
                idx_im = idx_ij(i-1, j-1, nx1);
                rhs_z[idx_ijk(i, j, 1, nx1, ny1)] += (coef_z_a) * (-coef_y_a) *
                                                     (-coef_x_a*sub.theta_z_left_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_left_sub[idx] - coef_x_c*sub.theta_z_left_sub[idx_ip]) *
                                                     sub.theta_z_left_index[1];

                idx    = idx_ij(i  , j, nx1);
                idx_ip = idx_ij(i+1, j, nx1);
                idx_im = idx_ij(i-1, j, nx1);
                rhs_z[idx_ijk(i, j, 1, nx1, ny1)] += (coef_z_a) * (1.0-coef_y_b) * 
                                                     (-coef_x_a*sub.theta_z_left_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_left_sub[idx] - coef_x_c*sub.theta_z_left_sub[idx_ip]) *
                                                     sub.theta_z_left_index[1];

                idx    = idx_ij(i  , j+1, nx1);
                idx_ip = idx_ij(i+1, j+1, nx1);
                idx_im = idx_ij(i-1, j+1, nx1);
                rhs_z[idx_ijk(i, j, 1, nx1, ny1)] += (coef_z_a) * (-coef_y_c)  *
                                                     (-coef_x_a*sub.theta_z_left_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_left_sub[idx] - coef_x_c*sub.theta_z_left_sub[idx_ip]) * 
                                                     sub.theta_z_left_index[1];

                // k=nz1-1
                dzdz = sub.dmz_sub[nz1-1]*sub.dmz_sub[nz1-1];
                coef_z_c = (dt / 2.0 / dzdz) * ( 1.0 + (5.0/3.0) );

                idx    = idx_ij(i  , j-1, nx1);
                idx_ip = idx_ij(i+1, j-1, nx1);
                idx_im = idx_ij(i-1, j-1, nx1);
                rhs_z[idx_ijk(i, j, nz1-2, nx1, ny1)] += (coef_z_c) * (-coef_y_a) *
                                                         (-coef_x_a*sub.theta_z_right_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_right_sub[idx] - coef_x_c*sub.theta_z_right_sub[idx_ip]) *
                                                         sub.theta_z_right_index[nz1-2];

                idx    = idx_ij(i  , j, nx1);
                idx_ip = idx_ij(i+1, j, nx1);
                idx_im = idx_ij(i-1, j, nx1);
                rhs_z[idx_ijk(i, j, nz1-2, nx1, ny1)] += (coef_z_c) * (1.0-coef_y_b) *
                                                         (-coef_x_a*sub.theta_z_right_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_right_sub[idx] - coef_x_c*sub.theta_z_right_sub[idx_ip]) *
                                                         sub.theta_z_right_index[nz1-2];
                idx    = idx_ij(i  , j+1, nx1);
                idx_ip = idx_ij(i+1, j+1, nx1);
                idx_im = idx_ij(i-1, j+1, nx1);
                rhs_z[idx_ijk(i, j, nz1-2, nx1, ny1)] += (coef_z_c) * (-coef_y_c) *
                                                         (-coef_x_a*sub.theta_z_right_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_right_sub[idx] - coef_x_c*sub.theta_z_right_sub[idx_ip]) *
                                                         sub.theta_z_right_index[nz1-2];         
            }
        }

        // timer.start("solve_z");
        // z solve
        for (j=1; j<ny1-1; ++j) {
            for (i=1; i<nx1-1; ++i) {
                for (k=1; k<nz1-1; ++k) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    ki = idx_ki(k-1, i-1, nz1-2);
                    dzdz = sub.dmz_sub[k]*sub.dmz_sub[k];

                    coef_z_a = (dt / 2.0 / dzdz) * ( 1.0 + (5.0/3.0) * sub.theta_z_left_index[k] + (1.0/3.0) * sub.theta_z_right_index[k] );
                    coef_z_b = (dt / 2.0 / dzdz) * (-2.0 -     (2.0) * sub.theta_z_left_index[k] -     (2.0) * sub.theta_z_right_index[k] );
                    coef_z_c = (dt / 2.0 / dzdz) * ( 1.0 + (1.0/3.0) * sub.theta_z_left_index[k] + (5.0/3.0) * sub.theta_z_right_index[k] );

                    Azz[ki] = -coef_z_a;
                    Bzz[ki] = (1.0-coef_z_b);
                    Czz[ki] = -coef_z_c;
                    Dzz[ki] = rhs_z[ijk];
                }
            }
            tdma_z.PaScaL_TDMA_many_solve(pz_many, Azz, Bzz, Czz, Dzz, (nx1-2), (nz1-2));
            for (i=1; i<nx1-1; ++i) {
                for (k=1; k<nz1-1; ++k) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    ki = idx_ki(k-1, i-1, nz1-2);
                    theta_z[ijk] = Dzz[ki];
                }
            }
        }
        // std::cout << "[solve_z] elapsed: " << timer.elapsed_ms("solve_z") << " ms\n";

        // bdy(y)
        for (k=1; k<nz1-1; ++k) {
            for (i=1; i<nx1-1; ++i) {

                dxdx = sub.dmx_sub[i]*sub.dmx_sub[i];
                coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * sub.theta_x_left_index[i] + (1.0/3.0) * sub.theta_x_right_index[i] );
                coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * sub.theta_x_left_index[i] -     (2.0) * sub.theta_x_right_index[i] );
                coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * sub.theta_x_left_index[i] + (5.0/3.0) * sub.theta_x_right_index[i] );

                // j=0
                dydy = sub.dmy_sub[0]*sub.dmy_sub[0];
                coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) );

                idx    = idx_ik(i  , k, nx1);
                idx_ip = idx_ik(i+1, k, nx1);
                idx_im = idx_ik(i-1, k, nx1);
                theta_z[idx_ijk(i, 1, k, nx1, ny1)] += coef_y_a * sub.theta_y_left_index[1] * 
                                                       (-coef_x_a*sub.theta_y_left_sub[idx_im] + (1.0-coef_x_b)*sub.theta_y_left_sub[idx] - coef_x_c*sub.theta_y_left_sub[idx_ip]);

                // j=ny1-1
                dydy = sub.dmy_sub[ny1-1]*sub.dmy_sub[ny1-1];
                coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) );

                idx    = idx_ik(i  , k, nx1);
                idx_ip = idx_ik(i+1, k, nx1);
                idx_im = idx_ik(i-1, k, nx1);
                theta_z[idx_ijk(i, ny1-2, k, nx1, ny1)] += coef_y_c * sub.theta_y_right_index[ny1-2] * 
                                                           (-coef_x_a*sub.theta_y_right_sub[idx_im] + (1.0-coef_x_b)*sub.theta_y_right_sub[idx] - coef_x_c*sub.theta_y_right_sub[idx_ip]);
            }
        }

        // timer.start("solve_y");
        // y solve
        for (i=1; i<nx1-1; ++i) {
            for (k=1; k<nz1-1; ++k) {
                for (j=1; j<ny1-1; ++j) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    jk = idx_jk(j-1, k-1, ny1-2);
                    dydy = sub.dmy_sub[j]*sub.dmy_sub[j];

                    coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) * sub.theta_y_left_index[j] + (1.0/3.0) * sub.theta_y_right_index[j] );
                    coef_y_b = (dt / 2.0 / dydy) * (-2.0 -     (2.0) * sub.theta_y_left_index[j] -     (2.0) * sub.theta_y_right_index[j] );
                    coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (1.0/3.0) * sub.theta_y_left_index[j] + (5.0/3.0) * sub.theta_y_right_index[j] );

                    Ayy[jk] = -coef_y_a;
                    Byy[jk] = (1.0-coef_y_b);
                    Cyy[jk] = -coef_y_c;
                    Dyy[jk] = theta_z[ijk];
                }
            }
            tdma_y.PaScaL_TDMA_many_solve(py_many, Ayy, Byy, Cyy, Dyy, nz1-2, ny1-2);
            for (k=1; k<nz1-1; ++k) {
                for (j=1; j<ny1-1; ++j) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    jk = idx_jk(j-1, k-1, ny1-2);
                    theta_y[ijk] = Dyy[jk];
                }
            }
        }
        // std::cout << "[solve_y] elapsed: " << timer.elapsed_ms("solve_y") << " ms\n";

        // bdy(x)
        for (k=1; k<nz1-1; ++k) {
            for (j=1; j<ny1-1; ++j) {
                
                // i=0
                dxdx = sub.dmx_sub[0]*sub.dmx_sub[0];
                coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) );

                idx = idx_jk(j, k, ny1);
                theta_y[idx_ijk(1, j, k, nx1, ny1)] += coef_x_a * sub.theta_x_left_index[1] * 
                                                       sub.theta_x_left_sub[idx];

                // i=nx1-1
                dxdx = sub.dmx_sub[nx1-1]*sub.dmx_sub[nx1-1];
                coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) );

                idx = idx_jk(j, k, ny1);
                theta_y[idx_ijk(nx1-2, j, k, nx1, ny1)] += coef_x_c * sub.theta_x_right_index[nx1-2] * 
                                                           sub.theta_x_right_sub[idx];
            }
        }

        // timer.start("solve_x");
        // x solve
        for (k=1; k<nz1-1; ++k) {
            for (j=1; j<ny1-1; ++j) {
                for (i=1; i<nx1-1; ++i) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    ij = idx_ij(i-1, j-1, nx1-2);
                    dxdx = (sub.dmx_sub[i]*sub.dmx_sub[i]);

                    coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * sub.theta_x_left_index[i] + (1.0/3.0) * sub.theta_x_right_index[i] );
                    coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * sub.theta_x_left_index[i] -     (2.0) * sub.theta_x_right_index[i] );
                    coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * sub.theta_x_left_index[i] + (5.0/3.0) * sub.theta_x_right_index[i] );

                    Axx[ij] = -coef_x_a;
                    Bxx[ij] = (1.0-coef_x_b);
                    Cxx[ij] = -coef_x_c;
                    Dxx[ij] = theta_y[ijk];
                }
            }
            tdma_x.PaScaL_TDMA_many_solve(px_many, Axx, Bxx, Cxx, Dxx, ny1-2, nx1-2);
            for (j=1; j<ny1-1; ++j) {
                for (i=1; i<nx1-1; ++i) {
                    ijk = idx_ijk(i, j, k, nx1, ny1);
                    ij = idx_ij(i-1, j-1, nx1-2);
                    theta[ijk] = Dxx[ij];
                }
            }
        }
        // timer.start("solve_x");
        // tdma_x.PaScaL_TDMA_single_solve(px_single, Ax, Bx, Cx, Dx, nx1-2);
        // std::cout << "[solve_x] elapsed: " << timer.elapsed_ms("solve_x") << " ms\n";
 
        // Update ghostcells from the solutions.
        sub.ghostcellUpdate(theta, cx, cy, cz, params);

    }   // Time step end------------------------
    std::cout << "[solve_heat] elapsed: " << timer.elapsed_ms("solve_heat") << " ms\n";

    if (myrank==0) {
        std::cout << "tN = " << dt * max_iter << std::endl;
    }
    tdma_x.PaScaL_TDMA_plan_many_destroy(px_many, px_many.nprocs);
    tdma_y.PaScaL_TDMA_plan_many_destroy(py_many, py_many.nprocs);
    tdma_z.PaScaL_TDMA_plan_many_destroy(pz_many, pz_many.nprocs);
}