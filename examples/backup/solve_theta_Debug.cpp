// solve_theta.cpp
#include "solve_theta.hpp"
// #include "../examples_lab/save.hpp"
#include "iostream"
#include "index.hpp"
#include "timer.hpp"
#include "Debug.hpp"

#include <string>
#include <cmath>
#include <chrono> 

// salloc -w cpu02 --nodes=1 --ntasks-per-node=64

// solve_theta.cpp
solve_theta::solve_theta(const GlobalParams& params,
                         const MPITopology& topo,
                         MPISubdomain& sub)
    : params(params), topo(topo), sub(sub) {}

void solve_theta::solve_theta_plan_single(std::vector<double>& theta) 
{   

    // Loop and index variables (그냥 셀 개수)
    int nx1 = sub.nx_sub+1; // number of cell with ghost cell in x axis 
    int ny1 = sub.ny_sub+1; // number of cell with ghost cell in y axis 
    int nz1 = sub.nz_sub+1; // number of cell with ghost cell in z axis 

    // 내가 사용할거
    int i, j, k;
    int ik;
    int ijk;

    // double dzdz;
    // double coef_z_a, coef_z_b, coef_z_c;

    // 1) 계획(plan) 객체 선언
    PaScaL_TDMA::ptdma_plan_many pz_many;

    auto cz = topo.commZ();
    // int rankz = cz.myrank;
    PaScaL_TDMA tdma_z; // kij
    tdma_z.PaScaL_TDMA_plan_many_create(pz_many, sub.nx_sub-1, cz.myrank, cz.nprocs, cz.comm);
    std::vector<double> Azz((nz1-2)*(nx1-2)), Bzz((nz1-2)*(nx1-2)), Czz((nz1-2)*(nx1-2)), Dzz((nz1-2)*(nx1-2));

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank==0) {
        std::cout << "Start | iteration" << std::endl;
        std::cout << "nx = " << nx1-2 << " | ny = " << ny1-2 << "| nz = " << nz1-2 << std::endl;
    }

    // Debug ----------------------------------------------------------
    int Debug = 1;
    // Debug ----------------------------------------------------------
    
    int max_iter = params.Nt;
    for (int t_step=0; t_step<max_iter; ++t_step) {

        if (Debug==0) {
            double solve_2 = 0.0;
            double solve_1 = 0.0;
            double solve = 0.0;

            double pack_2 = 0.0;
            double pack_1 = 0.0;
            double pack = 0.0;

            double unpack_2 = 0.0;
            double unpack_1 = 0.0;
            double unpack = 0.0;

            // solve
            MPI_Barrier(MPI_COMM_WORLD);
            
            for (j=1; j<ny1-1; ++j) {
                pack_1 = MPI_Wtime();
                for (k=1; k<nz1-1; ++k) {
                    for (i=1; i<nx1-1; ++i) {
                        ik = idx_ik(i-1, k-1, nx1-2);

                        Azz[ik] = 1.0;
                        Bzz[ik] = 4.0;
                        Czz[ik] = 1.0;
                        Dzz[ik] = 6.0 - sub.theta_z_left_index[k] - sub.theta_z_right_index[k];
                    }
                }
                pack_2 = MPI_Wtime();
                pack += (pack_2 - pack_1);

                solve_1 = MPI_Wtime();
                tdma_z.PaScaL_TDMA_many_solve(pz_many, Azz, Bzz, Czz, Dzz, (nx1-2), (nz1-2));
                solve_2 = MPI_Wtime();
                solve += solve_2 - solve_1;

                unpack_1 = MPI_Wtime();
                for (k=1; k<nz1-1; ++k) {
                    for (i=1; i<nx1-1; ++i) {
                        ijk = idx_ijk(i, j, k, nx1, ny1);
                        ik = idx_ik(i-1, k-1, nx1-2);

                        theta[ijk] = Dzz[ik];
                    }
                }
                unpack_2  = MPI_Wtime();
                unpack += unpack_2 - unpack_1;
            }

            std::vector<std::string> event_names = {
                "pack", "solve", "unpack"
            };
            std::vector<double> local_times = {
                pack, solve, unpack
            };
            int npxyz = params.np_dim[0] * params.np_dim[1] * params.np_dim[2];
            save_timing_data("results/t_" + std::to_string(npxyz)  + "_" + std::to_string(t_step) + ".txt", MPI_COMM_WORLD, event_names, local_times);
        }
        else if (Debug==1) {
            std::vector<double> time_list_g(8, 0.0);
            std::vector<double> time_list(8, 0.0);

            // solve
            MPI_Barrier(MPI_COMM_WORLD);
            
            for (j=1; j<ny1-1; ++j) {
                for (k=1; k<nz1-1; ++k) {
                    for (i=1; i<nx1-1; ++i) {
                        ik = idx_ik(i-1, k-1, nx1-2);

                        Azz[ik] = 1.0;
                        Bzz[ik] = 4.0;
                        Czz[ik] = 1.0;
                        Dzz[ik] = 6.0 - sub.theta_z_left_index[k] - sub.theta_z_right_index[k];
                    }
                }
                tdma_z.PaScaL_TDMA_many_solve_debug(pz_many, Azz, Bzz, Czz, Dzz, (nx1-2), (nz1-2), time_list);
                for (int tt=0; tt<8; ++tt) {
                    time_list_g[tt] += time_list[tt];
                }
                for (k=1; k<nz1-1; ++k) {
                    for (i=1; i<nx1-1; ++i) {
                        ijk = idx_ijk(i, j, k, nx1, ny1);
                        ik = idx_ik(i-1, k-1, nx1-2);

                        theta[ijk] = Dzz[ik];
                    }
                }
            }

            std::vector<std::string> event_names = {
                "Preprocess", "Forward", "Backrward", "Pack", "gather", "solve", "scatter", "final"
            };
            int npxyz = params.np_dim[0] * params.np_dim[1] * params.np_dim[2];
            save_timing_data("results/t_" + std::to_string(npxyz)  + "_" + std::to_string(t_step) + ".txt", MPI_COMM_WORLD, event_names, time_list_g);
        }
    }

    tdma_z.PaScaL_TDMA_plan_many_destroy(pz_many, pz_many.nprocs);
}