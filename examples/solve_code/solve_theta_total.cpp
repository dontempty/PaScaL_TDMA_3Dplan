// solve_theta.cpp
#include "solve_theta.hpp"
#include "../examples_lab/save.hpp"
#include "iostream"
#include "index.hpp"
#include "timer.hpp"
#include "Debug.hpp"

#include <string>
#include <cmath>
#include <chrono> 

// solve_theta.cpp
solve_theta::solve_theta(const GlobalParams& params,
                         const MPITopology& topo,
                         MPISubdomain& sub)
    : params(params), topo(topo), sub(sub) {}

void solve_theta::solve_theta_plan_single(std::vector<double>& theta) 
{   
    // int myrank, ierr;
    int time_step;      // Current time step
    double t_curr;      // Current simulation time

    // 내가 사용할거
    int i, j, k;
    int idx;
    int ij, jk, ik, ki;
    int ijk, jki, kij;
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

    // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // if (myrank==0) {
    //     std::cout << "Start to solve" << std::endl;
    // }

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

    // double solve_1, solve_2;
    
    double dt = params.dt;
    int max_iter = params.Nt;
    // MPI_Barrier(MPI_COMM_WORLD); // ---------------------------------------------------------------------------------
    // solve_1 = MPI_Wtime();
    for (int t_step=0; t_step<max_iter; ++t_step) {

        double rhs_x_1 = 0.0;
        double rhs_x_2 = 0.0;
        double rhs_y_1 = 0.0;
        double rhs_y_2 = 0.0;
        double rhs_z_1 = 0.0;
        double rhs_z_2 = 0.0;
        
        double bdy_z_1 = 0.0;
        double bdy_z_2 = 0.0;
        double bdy_y_1 = 0.0;
        double bdy_y_2 = 0.0;

        double bdy_x_1 = 0.0;
        double bdy_x_2 = 0.0;
        double solve_z_1 = 0.0;
        double solve_z_2 = 0.0;
        //     double make_Az_1, make_Az_2;
        //     double TDMA_z_1, TDMA_z_2;
        //         std::vector<double> time_list(5, 0.0); // preprocess, alltoall_1, tdma_many, alltoall_2, solve_remain
        //     double store_z_1, store_z_2;

        double solve_y_1 = 0.0;
        double solve_y_2 = 0.0;
        double solve_x_1 = 0.0;
        double solve_x_2 = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        // Calculating r.h.s -----------------------------------------------------------------------------------------------------

        // rhs_x ---------------------------
        rhs_x_1 = MPI_Wtime();
        for (k=0; k<nz1; ++k) {
            for (j=0; j<ny1; ++j) {
                for (i=1; i<nx1-1; ++i) {
                    ijk    = idx_ijk(i  , j, k, nx1, ny1);
                    // idx_ip = idx_ijk(i+1, j, k, nx1, ny1);
                    // idx_im = idx_ijk(i-1, j, k, nx1, ny1);
                    dxdx = sub.dmx_sub[i]*sub.dmx_sub[i];

                    coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * sub.theta_x_left_index[i] + (1.0/3.0) * sub.theta_x_right_index[i] );
                    coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * sub.theta_x_left_index[i] -     (2.0) * sub.theta_x_right_index[i] );
                    coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * sub.theta_x_left_index[i] + (5.0/3.0) * sub.theta_x_right_index[i] );
                    
                    jki = idx_jki(j, k, i, ny1, nz1);
                    rhs_x[jki] = (coef_x_c*theta[ijk+1] + (1.0+coef_x_b)*theta[ijk] + coef_x_a*theta[ijk-1]);
                }
            }
        }
        rhs_x_2 = MPI_Wtime();

        // rhs_y ---------------------------
        rhs_y_1 = MPI_Wtime();
        for (i=1; i<nx1-1; ++i) {
            for (k=0; k<nz1; ++k) {
                for (j=1; j<ny1-1; ++j) {
                    jki    = idx_jki(j  , k, i, ny1, nz1); 
                    // idx_jp = idx_jki(j+1, k, i, ny1, nz1); 
                    // idx_jm = idx_jki(j-1, k, i, ny1, nz1); 
                    dydy = sub.dmy_sub[j]*sub.dmy_sub[j];

                    coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) * sub.theta_y_left_index[j] + (1.0/3.0) * sub.theta_y_right_index[j] );
                    coef_y_b = (dt / 2.0 / dydy) * (-2.0 -     (2.0) * sub.theta_y_left_index[j] -     (2.0) * sub.theta_y_right_index[j] );
                    coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (1.0/3.0) * sub.theta_y_left_index[j] + (5.0/3.0) * sub.theta_y_right_index[j] );
                    
                    kij = idx_kij(k, i, j, nz1, nx1);
                    rhs_y[kij] = (coef_y_c*rhs_x[jki+1] + (1.0+coef_y_b)*rhs_x[jki] + coef_y_a*rhs_x[jki-1]);
                }
            }
        }
        rhs_y_2 = MPI_Wtime();

        // rhs_z ---------------------------
        rhs_z_1 = MPI_Wtime();
        for (j=1; j<ny1-1; ++j) {
            for (i=1; i<nx1-1; ++i) {
                for (k=1; k<nz1-1; ++k) {
                    kij    = idx_kij(k  , i, j, nz1, nx1);
                    // idx_kp = idx_kij(k+1, i, j, nz1, nx1);
                    // idx_km = idx_kij(k-1, i, j, nz1, nx1);
                    dzdz = sub.dmz_sub[k]*sub.dmz_sub[k];

                    coef_z_a = (dt / 2.0 / dzdz) * ( 1.0 + (5.0/3.0) * sub.theta_z_left_index[k] + (1.0/3.0) * sub.theta_z_right_index[k] );
                    coef_z_b = (dt / 2.0 / dzdz) * (-2.0 -     (2.0) * sub.theta_z_left_index[k] -     (2.0) * sub.theta_z_right_index[k] );
                    coef_z_c = (dt / 2.0 / dzdz) * ( 1.0 + (1.0/3.0) * sub.theta_z_left_index[k] + (5.0/3.0) * sub.theta_z_right_index[k] );
                    
                    rhs_z[kij] = (coef_z_c*rhs_y[kij+1] + (1.0+coef_z_b)*rhs_y[kij] + coef_z_a*rhs_y[kij-1]);

                    rhs_z[kij] += dt * ( 3.0 * Pi*Pi * cos(Pi*sub.x_sub[i]) * cos(Pi*sub.y_sub[j]) * cos(Pi*sub.z_sub[k]));
                }
            }
        }
        rhs_z_2 = MPI_Wtime();

        // Calculating A matrix ----------------------------------------------------------------

        // bdy(z)
        bdy_z_1 = MPI_Wtime();
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
                rhs_z[idx_kij(1, i, j, nz1, nx1)] += (coef_z_a) * (-coef_y_a) *
                                                     (-coef_x_a*sub.theta_z_left_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_left_sub[idx] - coef_x_c*sub.theta_z_left_sub[idx_ip]) *
                                                     sub.theta_z_left_index[1];

                idx    = idx_ij(i  , j, nx1);
                idx_ip = idx_ij(i+1, j, nx1);
                idx_im = idx_ij(i-1, j, nx1);
                rhs_z[idx_kij(1, i, j, nz1, nx1)] += (coef_z_a) * (1.0-coef_y_b) * 
                                                     (-coef_x_a*sub.theta_z_left_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_left_sub[idx] - coef_x_c*sub.theta_z_left_sub[idx_ip]) *
                                                     sub.theta_z_left_index[1];

                idx    = idx_ij(i  , j+1, nx1);
                idx_ip = idx_ij(i+1, j+1, nx1);
                idx_im = idx_ij(i-1, j+1, nx1);
                rhs_z[idx_kij(1, i, j, nz1, nx1)] += (coef_z_a) * (-coef_y_c)  *
                                                     (-coef_x_a*sub.theta_z_left_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_left_sub[idx] - coef_x_c*sub.theta_z_left_sub[idx_ip]) * 
                                                     sub.theta_z_left_index[1];

                // k=nz1-1
                dzdz = sub.dmz_sub[nz1-1]*sub.dmz_sub[nz1-1];
                coef_z_c = (dt / 2.0 / dzdz) * ( 1.0 + (5.0/3.0) );

                idx    = idx_ij(i  , j-1, nx1);
                idx_ip = idx_ij(i+1, j-1, nx1);
                idx_im = idx_ij(i-1, j-1, nx1);
                rhs_z[idx_kij(nz1-2, i, j, nz1, nx1)] += (coef_z_c) * (-coef_y_a) *
                                                         (-coef_x_a*sub.theta_z_right_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_right_sub[idx] - coef_x_c*sub.theta_z_right_sub[idx_ip]) *
                                                         sub.theta_z_right_index[nz1-2];

                idx    = idx_ij(i  , j, nx1);
                idx_ip = idx_ij(i+1, j, nx1);
                idx_im = idx_ij(i-1, j, nx1);
                rhs_z[idx_kij(nz1-2, i, j, nz1, nx1)] += (coef_z_c) * (1.0-coef_y_b) *
                                                         (-coef_x_a*sub.theta_z_right_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_right_sub[idx] - coef_x_c*sub.theta_z_right_sub[idx_ip]) *
                                                         sub.theta_z_right_index[nz1-2];
                idx    = idx_ij(i  , j+1, nx1);
                idx_ip = idx_ij(i+1, j+1, nx1);
                idx_im = idx_ij(i-1, j+1, nx1);
                rhs_z[idx_kij(nz1-2, i, j, nz1, nx1)] += (coef_z_c) * (-coef_y_c) *
                                                         (-coef_x_a*sub.theta_z_right_sub[idx_im] + (1.0-coef_x_b)*sub.theta_z_right_sub[idx] - coef_x_c*sub.theta_z_right_sub[idx_ip]) *
                                                         sub.theta_z_right_index[nz1-2];         
            }
        }
        bdy_z_2 = MPI_Wtime();

        // z solve
        // MPI_Barrier(MPI_COMM_WORLD); // ---------------------------------------------------------------------------------
        solve_z_1 = MPI_Wtime();
        for (j=1; j<ny1-1; ++j) {
            // make_Az_1 += MPI_Wtime();
            for (i=1; i<nx1-1; ++i) {
                for (k=1; k<nz1-1; ++k) {
                    kij = idx_kij(k, i, j, nz1, nx1);
                    ki = idx_ki(k-1, i-1, nz1-2);
                    dzdz = sub.dmz_sub[k]*sub.dmz_sub[k];

                    coef_z_a = (dt / 2.0 / dzdz) * ( 1.0 + (5.0/3.0) * sub.theta_z_left_index[k] + (1.0/3.0) * sub.theta_z_right_index[k] );
                    coef_z_b = (dt / 2.0 / dzdz) * (-2.0 -     (2.0) * sub.theta_z_left_index[k] -     (2.0) * sub.theta_z_right_index[k] );
                    coef_z_c = (dt / 2.0 / dzdz) * ( 1.0 + (1.0/3.0) * sub.theta_z_left_index[k] + (5.0/3.0) * sub.theta_z_right_index[k] );

                    Azz[ki] = -coef_z_a;
                    Bzz[ki] = (1.0-coef_z_b);
                    Czz[ki] = -coef_z_c;
                    Dzz[ki] = rhs_z[kij];
                }
            }
            // make_Az_2 += MPI_Wtime(); 

            // MPI_Barrier(MPI_COMM_WORLD);
            // TDMA_z_1 += MPI_Wtime();
            tdma_z.PaScaL_TDMA_many_solve(pz_many, Azz, Bzz, Czz, Dzz, (nx1-2), (nz1-2));
            // tdma_z.PaScaL_TDMA_many_solve(pz_many, Azz, Bzz, Czz, Dzz, (nx1-2), (nz1-2));
            // TDMA_z_2 += MPI_Wtime();

            // store_z_1 += MPI_Wtime();
            for (i=1; i<nx1-1; ++i) {
                for (k=1; k<nz1-1; ++k) {
                    jki = idx_jki(j, k, i, ny1, nz1);
                    ki = idx_ki(k-1, i-1, nz1-2);

                    theta_z[jki] = Dzz[ki];
                }
            }
            // store_z_2 += MPI_Wtime();

        }
        solve_z_2 = MPI_Wtime();

        // bdy(y)
        bdy_y_1 = MPI_Wtime();
        for (i=1; i<nx1-1; ++i) {

            dxdx = sub.dmx_sub[i]*sub.dmx_sub[i];
            coef_x_a = (dt / 2.0 / dxdx) * ( 1.0 + (5.0/3.0) * sub.theta_x_left_index[i] + (1.0/3.0) * sub.theta_x_right_index[i] );
            coef_x_b = (dt / 2.0 / dxdx) * (-2.0 -     (2.0) * sub.theta_x_left_index[i] -     (2.0) * sub.theta_x_right_index[i] );
            coef_x_c = (dt / 2.0 / dxdx) * ( 1.0 + (1.0/3.0) * sub.theta_x_left_index[i] + (5.0/3.0) * sub.theta_x_right_index[i] );

            for (k=1; k<nz1-1; ++k) {

                // j=0
                dydy = sub.dmy_sub[0]*sub.dmy_sub[0];
                coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) );

                idx    = idx_ki(k, i  , nz1);
                idx_ip = idx_ki(k, i+1, nz1);
                idx_im = idx_ki(k, i-1, nz1);
                theta_z[idx_jki(1, k, i, ny1, nz1)] += coef_y_a * sub.theta_y_left_index[1] * 
                                                       (-coef_x_a*sub.theta_y_left_sub[idx_im] + (1.0-coef_x_b)*sub.theta_y_left_sub[idx] - coef_x_c*sub.theta_y_left_sub[idx_ip]);

                // j=ny1-1
                dydy = sub.dmy_sub[ny1-1]*sub.dmy_sub[ny1-1];
                coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) );
                
                idx    = idx_ki(k, i  , nz1);
                idx_ip = idx_ki(k, i+1, nz1);
                idx_im = idx_ki(k, i-1, nz1);
                theta_z[idx_jki(ny1-2, k, i, ny1, nz1)] += coef_y_c * sub.theta_y_right_index[ny1-2] * 
                                                           (-coef_x_a*sub.theta_y_right_sub[idx_im] + (1.0-coef_x_b)*sub.theta_y_right_sub[idx] - coef_x_c*sub.theta_y_right_sub[idx_ip]);
            }
        }
        bdy_y_2 = MPI_Wtime();

        // y solve
        solve_y_1 = MPI_Wtime();
        for (i=1; i<nx1-1; ++i) {
            for (k=1; k<nz1-1; ++k) {
                for (j=1; j<ny1-1; ++j) {
                    jki = idx_jki(j, k, i, ny1, nz1);
                    jk = idx_jk(j-1, k-1, ny1-2);
                    dydy = sub.dmy_sub[j]*sub.dmy_sub[j];

                    coef_y_a = (dt / 2.0 / dydy) * ( 1.0 + (5.0/3.0) * sub.theta_y_left_index[j] + (1.0/3.0) * sub.theta_y_right_index[j] );
                    coef_y_b = (dt / 2.0 / dydy) * (-2.0 -     (2.0) * sub.theta_y_left_index[j] -     (2.0) * sub.theta_y_right_index[j] );
                    coef_y_c = (dt / 2.0 / dydy) * ( 1.0 + (1.0/3.0) * sub.theta_y_left_index[j] + (5.0/3.0) * sub.theta_y_right_index[j] );

                    Ayy[jk] = -coef_y_a;
                    Byy[jk] = (1.0-coef_y_b);
                    Cyy[jk] = -coef_y_c;
                    Dyy[jk] = theta_z[jki];
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
        solve_y_2 = MPI_Wtime();

        // bdy(x)
        bdy_x_1 = MPI_Wtime();
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
        bdy_x_2 = MPI_Wtime();

        // x solve
        solve_x_1 = MPI_Wtime();
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
        solve_x_2 = MPI_Wtime();

        std::vector<std::string> event_names = {
            "rhs_x", "rhs_y", "rhs_z",
            "bdy_z", "solve_z",
            "bdy_y", "solve_y",
            "bdy_x", "solve_x"
        };
        std::vector<double> local_times = {
            rhs_x_2 - rhs_x_1,
            rhs_y_2 - rhs_y_1,
            rhs_z_2 - rhs_z_1,
            bdy_z_2 - bdy_z_1,
            solve_z_2 - solve_z_1,
            bdy_y_2 - bdy_y_1,
            solve_y_2 - solve_y_1,
            bdy_x_2 - bdy_x_1,
            solve_x_2 - solve_x_1
        };

        int npxyz = params.np_dim[0] * params.np_dim[1] * params.np_dim[2];
        save_timing_data("results/t_" + std::to_string(npxyz)  + "_" + std::to_string(t_step) + ".txt", MPI_COMM_WORLD, event_names, local_times);
 
        // Update ghostcells from the solutions.
        sub.ghostcellUpdate(theta, cx, cy, cz, params);

    }   // Time step end------------------------
    // solve_2 = MPI_Wtime();

    

    // // Debug output
    // int myrank, nprocs;
    // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    // // 소수점 자리수를 정확히 보고 싶으시다 하셨으니 printf 사용(ios manipulator 불필요)
    // for (int turn = 0; turn < nprocs; ++turn) {
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     if (myrank == turn) {
    //         std::printf("[rhs_x]: %.9f\n", (rhs_x_2 - rhs_x_1));
    //         std::printf("[rhs_y]: %.9f\n", (rhs_y_2 - rhs_y_1));
    //         std::printf("[rhs_z]: %.9f\n", (rhs_z_2 - rhs_z_1));

    //         std::printf("[bdy_z]: %.9f\n", (bdy_z_2 - bdy_z_1));
    //             // std::printf("[make_Az]: %.9f\n", (make_Az_2 - make_Az_1));
    //             // std::printf("[TDMA_z]: %.9f\n", (TDMA_z_2 - TDMA_z_1));
    //                 // std::printf("[preprocess]:   %.9f\n", (time_list[0]));
    //                 // std::printf("[gather]:       %.9f\n", (time_list[1]));
    //                 // std::printf("[tdma_many]:    %.9f\n", (time_list[2]));
    //                 // std::printf("[scatter]:      %.9f\n", (time_list[3]));
    //                 // std::printf("[solve_remain]: %.9f\n", (time_list[4]));
    //         //     std::printf("[store_z]: %.9f\n", (store_z_2 - store_z_1));
    //         std::printf("[solve_z]: %.9f\n", (solve_z_2 - solve_z_1));

    //         std::printf("[bdy_y]: %.9f\n", (bdy_y_2 - bdy_y_1));
    //         std::printf("[solve_y]: %.9f\n", (solve_y_2 - solve_y_1));

    //         std::printf("[bdy_x]: %.9f\n", (bdy_x_2 - bdy_x_1));
    //         std::printf("[solve_x]: %.9f\n", (solve_x_2 - solve_x_1));

    //         // std::printf("[solve]: %.9f\n", (solve_2 - solve_1));
    //         std::fflush(stdout);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }





    // if (myrank==0) {
    //     std::cout << "tN = " << dt * max_iter << std::endl;
    // }
    tdma_x.PaScaL_TDMA_plan_many_destroy(px_many, px_many.nprocs);
    tdma_y.PaScaL_TDMA_plan_many_destroy(py_many, py_many.nprocs);
    tdma_z.PaScaL_TDMA_plan_many_destroy(pz_many, pz_many.nprocs);
}