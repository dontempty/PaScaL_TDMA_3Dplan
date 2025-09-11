// hyper thread backup
void PaScaL_TDMA::PaScaL_TDMA_many_solve(ptdma_plan_many& plan,
                                         std::vector<double>& A,
                                         std::vector<double>& B,
                                         std::vector<double>& C,
                                         std::vector<double>& D,
                                         int n_sys, int n_row) {

    double* __restrict a0 = A.data();
    double* __restrict b0 = B.data();
    double* __restrict c0 = C.data();
    double* __restrict d0 = D.data();

    std::vector<MPI_Request> request(4);

    if (plan.nprocs == 1) {
        tdma_many(A, B, C, D, n_sys, n_row);
        return;
    }
    
    // ========================================================================
    // 변수 선언 최적화 적용
    // ========================================================================
    int i, j, idx;
    double r_inv0, r_inv1, r_inv;
    double r, denom;
    double d0_val, dn_val;

    #pragma omp parallel
    {
        // --- Preprocess ---
        // 포인터 변수 없이 베이스 포인터와 인덱스로 직접 접근
        #pragma omp for schedule(static) nowait private(idx, r_inv0, r_inv1)   
        for (j = 0; j < n_sys; ++j) {
            idx = j * n_row;

            // i=0
            r_inv0 = 1.0 / b0[idx];
            a0[idx] *= r_inv0;
            d0[idx] *= r_inv0;
            c0[idx] *= r_inv0;

            // i=1
            r_inv1 = 1.0 / b0[idx+1];
            a0[idx+1] *= r_inv1;
            d0[idx+1] *= r_inv1;
            c0[idx+1] *= r_inv1;
        }

        // --- Forward Elimination ---
        // 루프 내 포인터 선언 제거
        #pragma omp for schedule(static) nowait private(idx, i, r_inv, denom) 
        for (j = 0; j < n_sys; ++j) {
            idx = j * n_row;
            double* a = a0 + idx;
            double* b = b0 + idx;
            double* c = c0 + idx;
            double* d = d0 + idx;

            for (i = 2; i < n_row; ++i) {
                denom = std::fma(-a[i], c[i - 1], b[i]);
                r_inv = 1.0 / denom;

                d[i] = r_inv * (d[i] - a[i] * d[i - 1]);
                c[i] *= r_inv;
                a[i] = -r_inv * a[i] * a[i - 1];
            }
        }

        // --- Backward Substitution ---
        #pragma omp for schedule(static) nowait private(idx, i)
        for (j = 0; j < n_sys; ++j) {
            idx = j * n_row;
            double* a = a0 + idx;
            double* c = c0 + idx;
            double* d = d0 + idx;

            for (i = n_row - 3; i >= 1; --i) {
                d[i] -= c[i] * d[i + 1];
                a[i] -= c[i] * a[i + 1];
                c[i] *= -c[i + 1];
            }
        }

        // --- Pack ---
        #pragma omp for schedule(static) private(idx, r)
        for (j = 0; j < n_sys; ++j) {
            idx = j * n_row;
            double* a = a0 + idx;
            double* c = c0 + idx;
            double* d = d0 + idx;

            r = 1.0 / (1.0 - a[1] * c[0]);
            d[0] = r * (d[0] - c[0] * d[1]);
            a[0] *= r;
            c[0] = -r * c[0] * c[1];

            plan.A_rd[2*j + 0] = a[0];
            plan.A_rd[2*j + 1] = a[n_row - 1];
            plan.B_rd[2*j + 0] = 1.0;
            plan.B_rd[2*j + 1] = 1.0;
            plan.C_rd[2*j + 0] = c[0];
            plan.C_rd[2*j + 1] = c[n_row - 1];
            plan.D_rd[2*j + 0] = d[0];
            plan.D_rd[2*j + 1] = d[n_row - 1];
        }
    }
    
    // (MPI 통신)
    MPI_Ialltoallw(plan.A_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                   plan.A_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                   plan.ptdma_world, &request[0]);
    MPI_Ialltoallw(plan.B_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                   plan.B_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                   plan.ptdma_world, &request[1]);
    MPI_Ialltoallw(plan.C_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                   plan.C_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                   plan.ptdma_world, &request[2]);
    MPI_Ialltoallw(plan.D_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                   plan.D_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                   plan.ptdma_world, &request[3]);
    MPI_Waitall(4, request.data(), MPI_STATUSES_IGNORE);

    tdma_many(plan.A_rt, plan.B_rt, plan.C_rt, plan.D_rt, plan.n_sys_rt, plan.n_row_rt);

    MPI_Ialltoallw(plan.D_rt.data(), plan.count_recv.data(), plan.displ_recv.data(), plan.ddtype_Bs.data(),
                   plan.D_rd.data(), plan.count_send.data(), plan.displ_send.data(), plan.ddtype_Fs.data(),
                   plan.ptdma_world, &request[0]);
    MPI_Waitall(1, request.data(), MPI_STATUSES_IGNORE);

    // --- Final Local Solve ---
    #pragma omp parallel for schedule(static) private(idx, i, d0_val, dn_val)
    for (j = 0; j < n_sys; ++j) {
        idx = j * n_row;
        double* a = a0 + idx;
        double* c = c0 + idx;
        double* d = d0 + idx;

        d0_val = plan.D_rd[j*2 + 0];
        dn_val = plan.D_rd[j*2 + 1];

        d[0]       = d0_val;
        d[n_row-1] = dn_val;

        for (i = 1; i < n_row - 1; ++i) {
            d[i] -= a[i] * d0_val + c[i] * dn_val;
        }
    }
}