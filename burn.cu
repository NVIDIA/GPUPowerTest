/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <iostream>
#include <string>
#include <exception>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <curand.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <pthread.h>
#include <stdlib.h>

#define M 1000000
#define K 1000

#define CHECK_ERROR(error) \
    if (error != cudaSuccess) { \
        cout << "ERROR:" << cudaGetErrorString(error) << endl; \
        exit(-1); \
    }

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(-1);}} while(0)

using std::string;
using std::cout;
using std::endl;

#ifdef __cplusplus
extern "C" {
    void burn(int gpu, int cores, int low, int drop, double u_secs, double d_secs);
}
#endif

void *core_spin(void *target_ms) {
    timeval tod; 
    suseconds_t target_up_ms = *static_cast<suseconds_t *>(target_ms);
    gettimeofday(&tod, NULL);
    while(true) {
        if (target_up_ms <= tod.tv_sec * K + tod.tv_usec / K)  break;
        gettimeofday(&tod, NULL);
    }
    return(NULL);
}

#define CORE_SPIN() { \
    for (int thrix=0; thrix < cores; thrix++) {\
                ret = pthread_create(&tids[thrix], NULL, \
                        core_spin, (suseconds_t *) &target_up_ms);\
                if (ret != 0) {\
                    perror("burn pthread_create");\
                    exit(0);\
                } \
            }\
    }

class BurnGPU {
private:
    int roundoff(int v, int d) {
        return (v + d - 1) / d * d;
    }

    cudaError_t cuda_error {};  
    cublasStatus_t cublas_status {};
    
    float* A_up = nullptr;
    float* B_up = nullptr;
    float* C_up = nullptr;
    float* A_dn = nullptr;
    float* B_dn = nullptr;
    float* C_dn = nullptr;
    cublasComputeType_t cuCompType = CUBLAS_COMPUTE_32F_FAST_16F;
    cudaDataType_t cuDataType = CUDA_R_32F;

    float alpha = 1.0;
    float beta = 0.0;

    int cores = 0;
    int low = 0;
    int drop = 0;

// matrix dims must agree with const int ld (see below)
// for the transpose and op states
#define SEED_UP 8000
#define SEED_DN_LOW 10
#define SEED_DN_HOT 1000


    const int Mm_up = SEED_UP;
    const int Mn_up = SEED_UP;
    const int Mk_up = SEED_UP;
    const size_t As_up = SEED_UP * SEED_UP;
    const size_t Bs_up = SEED_UP * SEED_UP;
    const size_t Cs_up = SEED_UP * SEED_UP;
    const int Mm_dn = (low) ?  SEED_DN_LOW : SEED_DN_HOT;
    const int Mn_dn = (low) ?  SEED_DN_LOW : SEED_DN_HOT;
    const int Mk_dn = (low) ?  SEED_DN_LOW : SEED_DN_HOT;
    const size_t As_dn = (low) ?  SEED_DN_LOW : SEED_DN_HOT * ((low) ?  SEED_DN_LOW : SEED_DN_HOT);
    const size_t Bs_dn = (low) ?  SEED_DN_LOW : SEED_DN_HOT * ((low) ?  SEED_DN_LOW : SEED_DN_HOT);
    const size_t Cs_dn = (low) ?  SEED_DN_LOW : SEED_DN_HOT * ((low) ?  SEED_DN_LOW : SEED_DN_HOT);

    cublasLtMatmulDesc_t matmulDesc_up = NULL;
    cublasLtMatrixLayout_t Adesc_up = NULL;
    cublasLtMatrixLayout_t Bdesc_up = NULL;
    cublasLtMatrixLayout_t Cdesc_up = NULL;
    cublasLtMatmulDesc_t matmulDesc_dn = NULL;
    cublasLtMatrixLayout_t Adesc_dn = NULL;
    cublasLtMatrixLayout_t Bdesc_dn = NULL;
    cublasLtMatrixLayout_t Cdesc_dn = NULL;
    cublasLtHandle_t handle_up {};
    cublasLtHandle_t handle_dn {};
    void* workspace;
    const size_t workspaceSize = 8192 * 8192 * 4;
    const cublasOperation_t op = CUBLAS_OP_N;
    const int ld_up = SEED_UP;
    const int ld_dn = (low) ?  SEED_DN_LOW : SEED_DN_HOT;
    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    // for the square wave
    double up_seconds;
    double dn_seconds;
    int gpuid = -1;
    pthread_t* tids = 0;

    class Drop {

    public:
        const suseconds_t DROP_ZERO_TO_ONE_PER_MS = 6000;
        const suseconds_t DROP_ONE_TO_LIM_SEC_DELAY = 4;

        Drop(suseconds_t base_ms) {
            BurnGPU::Drop::lim_ms = base_ms + DROP_ZERO_TO_ONE_PER_MS;
            srand(static_cast<int>(base_ms / K));
            calc_drop(base_ms); 
        }

        void apply(suseconds_t ms) {
            if (ms >= lim_ms) {
                lim_ms = ms + DROP_ZERO_TO_ONE_PER_MS;
                calc_drop(ms);
            }
            if (drop_target_ms && ms >= drop_target_ms) {
                cout << "Drop for " << drop_delay_sec << " seconds" << endl;              
                (void) sleep(drop_delay_sec);
                drop_target_ms = 0;
            }
        }

    private:

        suseconds_t lim_ms;
        suseconds_t drop_target_ms;
        int drop_delay_sec;

        void calc_drop(suseconds_t ms){

            drop_delay_sec =  (ms / K )  % DROP_ONE_TO_LIM_SEC_DELAY;    
            drop_delay_sec = (drop_delay_sec) ? drop_delay_sec : 1;
            drop_target_ms = (suseconds_t)(rand() % 2);
            drop_target_ms = (suseconds_t)(drop_target_ms) ? ms + (rand() % DROP_ZERO_TO_ONE_PER_MS) : 0;
        }
    };

public:
    BurnGPU(int gpu, int cores, int low, int drop, double u_secs, double d_secs) : cores(cores), 
                low(low), drop(drop) {
        cudaDeviceProp devprop {};
        CHECK_ERROR(cudaSetDevice(gpu));
        CHECK_ERROR(cudaDeviceReset());
        CHECK_ERROR(cudaDeviceSynchronize());
        CHECK_ERROR(cudaGetDeviceProperties(&devprop, gpu));
        cout << "Found GPU " << gpu << " " << devprop.name << endl;
        cout << "Spinning " << cores << " CPU cores per GPU " << endl;

        if (cores) {
            tids = (pthread_t *) malloc(sizeof(pthread_t) * cores);
            if (! tids) {
                cout << "Failed to allocate memory for " << cores << " pthread_t " << endl;
                exit(-1);
            }
        }

	up_seconds = u_secs;
	dn_seconds = d_secs;
        gpuid = gpu;

        CHECK_ERROR((cudaMalloc((void**)&A_up, As_up)));
        CHECK_ERROR((cudaMalloc((void**)&B_up, Bs_up)));
        CHECK_ERROR((cudaMalloc((void**)&C_up, Cs_up)));
        CHECK_ERROR((cudaMalloc((void**)&A_dn, As_dn)));
        CHECK_ERROR((cudaMalloc((void**)&B_dn, Bs_dn)));
        CHECK_ERROR((cudaMalloc((void**)&C_dn, Cs_dn)));
        CHECK_ERROR((cudaMalloc((void**)&workspace, workspaceSize)));
    }

    void operator()() noexcept(false) {
        try {

            cublas_status = cublasLtCreate(&handle_up);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtCreate failed "
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtCreate(&handle_dn);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtCreate failed "
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatmulDescCreate(&matmulDesc_up,
                cuCompType,
                cuDataType);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatmulDescCreate failed" 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatmulDescCreate(&matmulDesc_dn,
                cuCompType,
                cuDataType);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatmulDescCreate failed" 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatmulDescSetAttribute(matmulDesc_up,
                CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatmulDescSetAttribute A failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatmulDescSetAttribute(matmulDesc_dn,
                CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatmulDescSetAttribute A failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatmulDescSetAttribute(matmulDesc_up,
                CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatmulDescSetAttribute B failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatmulDescSetAttribute(matmulDesc_dn,
                CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatmulDescSetAttribute B failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatmulDescSetAttribute(matmulDesc_up,
                CUBLASLT_MATMUL_DESC_TRANSC, &op, sizeof(op));
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatmulDescSetAttribute C failed "
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatmulDescSetAttribute(matmulDesc_dn,
                CUBLASLT_MATMUL_DESC_TRANSC, &op, sizeof(op));
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatmulDescSetAttribute C failed "
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutCreate(&Adesc_up,
                    cuDataType,
                    Mm_up,
                    Mn_up,
                    ld_up);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutCreate A failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutCreate(&Adesc_dn,
                    cuDataType,
                    Mm_dn,
                    Mn_dn,
                    ld_dn);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutCreate A failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutCreate(&Adesc_dn,
                    cuDataType,
                    Mm_dn,
                    Mn_dn,
                    ld_dn);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutCreate A failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutSetAttribute(Adesc_up,
                    CUBLASLT_MATRIX_LAYOUT_ORDER,
                    &order,
                    sizeof(cublasLtOrder_t)
                    );
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutSetAttribute A failed "
                    << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutSetAttribute(Adesc_dn,
                    CUBLASLT_MATRIX_LAYOUT_ORDER,
                    &order,
                    sizeof(cublasLtOrder_t)
                    );
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutSetAttribute A failed "
                    << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutCreate(&Bdesc_up,
                    cuDataType,
                    Mn_up,
                    Mk_up,
                    ld_up);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutCreate B failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutCreate(&Bdesc_dn,
                    cuDataType,
                    Mn_dn,
                    Mk_dn,
                    ld_dn);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutCreate B failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutSetAttribute(Bdesc_up,
                CUBLASLT_MATRIX_LAYOUT_ORDER,
                &order,
                sizeof(cublasLtOrder_t)
                );
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutSetAttribute B failed "
                    << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutSetAttribute(Bdesc_dn,
                CUBLASLT_MATRIX_LAYOUT_ORDER,
                &order,
                sizeof(cublasLtOrder_t)
                );
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutSetAttribute B failed "
                    << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutCreate(&Cdesc_up,
                    cuDataType,
                    Mk_up,
                    Mm_up,
                    ld_up);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutCreate C failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutCreate(&Cdesc_dn,
                    cuDataType,
                    Mk_dn,
                    Mm_dn,
                    ld_dn);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutCreate C failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutSetAttribute(Cdesc_up,
                CUBLASLT_MATRIX_LAYOUT_ORDER,
                &order,
                sizeof(cublasLtOrder_t)
                );
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutSetAttribute C failed "
                    << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutSetAttribute(Cdesc_dn,
                CUBLASLT_MATRIX_LAYOUT_ORDER,
                &order,
                sizeof(cublasLtOrder_t)
                );
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutSetAttribute C failed "
                    << cublas_status << endl;
                exit(-1);
            }

            curandGenerator_t prngGPU;
            CURAND_CALL(curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MRG32K3A));
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(prngGPU, 777));
            CURAND_CALL(curandGenerateUniform(prngGPU, (float *) A_up, As_up));
            CURAND_CALL(curandGenerateUniform(prngGPU, (float *) B_up, Bs_up));
            CURAND_CALL(curandGenerateUniform(prngGPU, (float *) C_up, Cs_up));
            CURAND_CALL(curandGenerateUniform(prngGPU, (float *) A_dn, As_dn));
            CURAND_CALL(curandGenerateUniform(prngGPU, (float *) B_dn, Bs_dn));
            CURAND_CALL(curandGenerateUniform(prngGPU, (float *) C_dn, Cs_dn));

            timeval tod;
            gettimeofday(&tod, NULL);
            int iterations = 1;
            /* Add one to the target up second and usleep to start on a second boundary with
            ** the target ms. set to 0; this elimnates the ms. slop comming out of the MPI barrier
            */
            BurnGPU::Drop power_drop = BurnGPU::Drop((suseconds_t)(((double) tod.tv_sec) * K));
            suseconds_t target_up_ms = (suseconds_t) (((double) tod.tv_sec + up_seconds + 1.0) * K);
            printf("GPU %2d arrival second %ld ms. %ld target_up_ms %ld\n", gpuid, tod.tv_sec,
                    tod.tv_usec / K, target_up_ms);
	    usleep(M - tod.tv_usec);
	    printf("GPU %2d %sEntering loop. Up: %3.3f seconds. Down: %3.3f seconds.\n", 
                    gpuid, (ctime(&tod.tv_sec)), up_seconds, dn_seconds);
            int ret = 0;
            CORE_SPIN();

            while (iterations++) {
                cublas_status = cublasLtMatmul(handle_up,
                    matmulDesc_up,
                    &alpha,
                    A_up,
                    Adesc_up,
                    B_up,
                    Bdesc_up,
                    &beta,
                    C_up,
                    Cdesc_up,
                    C_up,
                    Cdesc_up,
                    NULL,
                    workspace,
                    workspaceSize,
                    0);
                if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                    cout << "cublasLtMatmul UP failed "
                        << cublas_status << endl;
                    exit(-1);
                }
                CHECK_ERROR(cudaDeviceSynchronize());
                gettimeofday(&tod, NULL);
                if (drop) power_drop.apply(tod.tv_sec * K + tod.tv_usec / K);
		if (target_up_ms <= tod.tv_sec * K + tod.tv_usec / K) {
                    suseconds_t target_dn_ms =(suseconds_t) ((double) tod.tv_sec * 
                            (double) K + (double) tod.tv_usec / (double) K + dn_seconds * (double) K);
                    printf("GPU %2d dn phase begin at ms. %ld target_dn_ms %ld Iterations %-8d\n",
                            gpuid, tod.tv_sec * K + tod.tv_usec / K, target_dn_ms, iterations);
                    iterations = 1;
                    for (int i=0; i < cores; i++) {
                        pthread_join(tids[i], (void **)NULL);
                    }
                    while(iterations) {
                        cublas_status = cublasLtMatmul(handle_dn,
                            matmulDesc_dn,
                            &alpha,
                            A_dn,
                            Adesc_dn,
                            B_dn,
                            Bdesc_dn,
                            &beta,
                            C_dn,
                            Cdesc_dn,
                            C_dn,
                            Cdesc_dn,
                            NULL,
                            workspace,
                            workspaceSize,
                            0);
                        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                            cout << "cublasLtMatmul DN failed "
                                << cublas_status << endl;
                            exit(-1);
                        }
                        CHECK_ERROR(cudaDeviceSynchronize());
                        gettimeofday(&tod, NULL);
        		if (target_dn_ms <= tod.tv_sec * K + tod.tv_usec / K)  break;
                    }
		    int rtn = MPI_Barrier(MPI_COMM_WORLD);
                    gettimeofday(&tod, NULL);
                    if (drop) power_drop.apply(tod.tv_sec * K + tod.tv_usec / K);
                    target_up_ms = (suseconds_t) ((double) tod.tv_sec *
                            (double) K + (double) tod.tv_usec / (double) K + up_seconds * (double) K);
                    printf("GPU %2d up phase begin at ms. %ld target_up_ms %ld\n", 
                            gpuid, tod.tv_sec * K + tod.tv_usec / K, target_up_ms);
                    CORE_SPIN(); 
	        }
            }
            CHECK_ERROR(cudaDeviceSynchronize());

            cublas_status = cublasLtDestroy(handle_up);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtDestroy failed "
                        << cublas_status << endl;
                exit(-1);
            }
        }
        catch (std::exception& e) {
            cout << "ERROR:" << e.what() << endl;
        }
    }

    ~BurnGPU() noexcept(false) {
        free(tids);

        cublas_status = cublasLtMatrixLayoutDestroy(Adesc_up);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cout << "cublasLtMatrixLayoutDestroy A failed "
                << cublas_status << endl;
            exit(-1);
        }

        cublas_status = cublasLtMatrixLayoutDestroy(Adesc_dn);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cout << "cublasLtMatrixLayoutDestroy A failed "
                << cublas_status << endl;
            exit(-1);
        }

        cublas_status = cublasLtMatrixLayoutDestroy(Bdesc_up);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cout << "cublasLtMatrixLayoutDestroy B failed "
                << cublas_status << endl;
            exit(-1);
        }

        cublas_status = cublasLtMatrixLayoutDestroy(Bdesc_dn);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cout << "cublasLtMatrixLayoutDestroy B failed "
                << cublas_status << endl;
            exit(-1);
        }

        cublas_status = cublasLtMatrixLayoutDestroy(Cdesc_up);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cout << "cublasLtMatrixLayoutDestroy C failed "
                << cublas_status << endl;
            exit(-1);
        }

        cublas_status = cublasLtMatrixLayoutDestroy(Cdesc_dn);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cout << "cublasLtMatrixLayoutDestroy C failed "
                << cublas_status << endl;
            exit(-1);
        }

        cublas_status = cublasLtMatmulDescDestroy(matmulDesc_up);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cout << "cublasLtMatmulDescDestroy failed cublas_status "
                << cublas_status << endl;
            exit(-1);
        }

        cublas_status = cublasLtMatmulDescDestroy(matmulDesc_dn);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cout << "cublasLtMatmulDescDestroy failed cublas_status "
                << cublas_status << endl;
            exit(-1);
        }

        CHECK_ERROR(cudaFree(workspace));
        CHECK_ERROR(cudaFree(A_up));
        CHECK_ERROR(cudaFree(B_up));
        CHECK_ERROR(cudaFree(C_up));
        CHECK_ERROR(cudaFree(A_dn));
        CHECK_ERROR(cudaFree(B_dn));
        CHECK_ERROR(cudaFree(C_dn));
        CHECK_ERROR(cudaDeviceReset());
        CHECK_ERROR(cudaDeviceSynchronize());
    }

};

void burn(int gpu, int cores, int low, int drop, double u_secs, double d_secs) {
    BurnGPU *burngpu = new BurnGPU(gpu, cores, low, drop, u_secs, d_secs);
    (*burngpu)();
}


