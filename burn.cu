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
    void burn(int gpu, int u_secs, int d_secs);
}
#endif

class BurnGPU {
private:
    int roundoff(int v, int d) {
        return (v + d - 1) / d * d;
    }

    cudaError_t cuda_error {};  
    cublasStatus_t cublas_status {};
    
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    cublasComputeType_t cuCompType = CUBLAS_COMPUTE_32F_FAST_16F;
    cudaDataType_t cuDataType = CUDA_R_32F;

    float alpha = 1.0;
    float beta = 0.0;

// matrix dims must agree with const int ld (see below)
// for the transpose and op states
#define SEED 10000

    const int Mm = SEED;
    const int Mn = SEED;
    const int Mk = SEED;
    const size_t As = SEED * SEED;
    const size_t Bs = SEED * SEED;
    const size_t Cs = SEED * SEED;

    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL;
    cublasLtMatrixLayout_t Bdesc = NULL;
    cublasLtMatrixLayout_t Cdesc = NULL;
    cublasLtHandle_t handle {};
    void* workspace;
    const size_t workspaceSize = 8192 * 8192 * 4;
    const cublasOperation_t op = CUBLAS_OP_N;
    const int ld = SEED;
    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    // for the square wave
    int up_seconds;
    int down_seconds;
    int gpuid = -1;

public:
    BurnGPU(int gpu, int u_secs, int d_secs) {
        cudaDeviceProp devprop {};
        CHECK_ERROR(cudaSetDevice(gpu));
        CHECK_ERROR(cudaGetDeviceProperties(&devprop, gpu));
        cout << "Found GPU " << gpu << " " << devprop.name << endl;

	up_seconds = u_secs;
	down_seconds = d_secs;
        gpuid = gpu;

        CHECK_ERROR((cudaMalloc((void**)&A, As)));
        CHECK_ERROR((cudaMalloc((void**)&B, Bs)));
        CHECK_ERROR((cudaMalloc((void**)&C, Cs)));
        CHECK_ERROR((cudaMalloc((void**)&workspace, workspaceSize)));
    }

    void operator()() noexcept(false) {
        try {

            cublas_status = cublasLtCreate(&handle);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtCreate failed "
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatmulDescCreate(&matmulDesc,
                cuCompType,
                cuDataType);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatmulDescCreate failed" 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatmulDescSetAttribute(matmulDesc,
                CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatmulDescSetAttribute A failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatmulDescSetAttribute(matmulDesc,
                CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatmulDescSetAttribute B failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatmulDescSetAttribute(matmulDesc,
                CUBLASLT_MATMUL_DESC_TRANSC, &op, sizeof(op));
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatmulDescSetAttribute C failed "
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutCreate(&Adesc,
                    cuDataType,
                    Mm,
                    Mn,
                    ld);
            
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutCreate A failed " 
                        << cublas_status << endl;
                exit(-1);
            }
            cublas_status = cublasLtMatrixLayoutSetAttribute(Adesc,
                    CUBLASLT_MATRIX_LAYOUT_ORDER,
                    &order,
                    sizeof(cublasLtOrder_t)
                    );
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutSetAttribute A failed "
                    << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutCreate(&Bdesc,
                    cuDataType,
                    Mn,
                    Mk,
                    ld);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutCreate B failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutSetAttribute(Bdesc,
                CUBLASLT_MATRIX_LAYOUT_ORDER,
                &order,
                sizeof(cublasLtOrder_t)
                );
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutSetAttribute B failed "
                    << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutCreate(&Cdesc,
                                    cuDataType,
                    Mk,
                    Mm,
                    ld);
            if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                cout << "cublasLtMatrixLayoutCreate C failed " 
                        << cublas_status << endl;
                exit(-1);
            }

            cublas_status = cublasLtMatrixLayoutSetAttribute(Cdesc,
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
            CURAND_CALL(curandGenerateUniform(prngGPU, (float *) A, As));
            CURAND_CALL(curandGenerateUniform(prngGPU, (float *) B, Bs));
            CURAND_CALL(curandGenerateUniform(prngGPU, (float *) C, Cs));

            timeval tod;
            gettimeofday(&tod, NULL);
            int iterations = 1;
	    int usleep_time = (down_seconds * M);
	    time_t t;
	    time(&t);
            /* Add one to the target up second and usleep to start on a secpnd boundary with
            ** the target ms. set to 0; this elimnates the ms. slop comming out of the MPI barrier
            */
            time_t target_up_second = t + (time_t) up_seconds + 1;
            suseconds_t target_up_ms = 0;
            printf("GPU %2d arrival second %ld ms. %3ld target_up_second %ld ms. %ld\n", gpuid, t,
                    tod.tv_usec / K, target_up_second, target_up_ms);
	    usleep(M - tod.tv_usec);
	    printf("GPU %2d %sEntering loop. Up: %d seconds. Down: %d seconds.\n", gpuid, (ctime(&t)), up_seconds, down_seconds);
            while (iterations++) {
                cublas_status = cublasLtMatmul(handle,
                    matmulDesc,
                    &alpha,
                    A,
                    Adesc,
                    B,
                    Bdesc,
                    &beta,
                    C,
                    Cdesc,
                    C,
                    Cdesc,
                    NULL,
                    workspace,
                    workspaceSize,
                    0);
                if (cublas_status != CUBLAS_STATUS_SUCCESS) {
                    cout << "cublasLtMatmul failed "
                        << cublas_status << endl;
                    exit(-1);
                }
                CHECK_ERROR(cudaDeviceSynchronize());
                time(&t);
                gettimeofday(&tod, NULL);
		if (target_up_second <= t && target_up_ms <= (tod.tv_usec / K)) {
                    printf("GPU %2d up phase done at second %ld ms. %3ld Iterations %-8d "
                           "target_up_second %ld target_up_ms %ld\n",
                            gpuid, t, tod.tv_usec / K, iterations, target_up_second, target_up_ms);
                    iterations = 1;
		    usleep(usleep_time);
		    int rtn = MPI_Barrier(MPI_COMM_WORLD);
                    time(&t);
                    gettimeofday(&tod, NULL);
                    target_up_second = t + up_seconds;
                    target_up_ms = tod.tv_usec / K;
                    printf("GPU %2d down phase done at second %ld ms. %3ld\n", 
                            gpuid, t, tod.tv_usec / K );
	        }
            }
            CHECK_ERROR(cudaDeviceSynchronize());

            cublas_status = cublasLtDestroy(handle);
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
        cublas_status = cublasLtMatrixLayoutDestroy(Adesc);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cout << "cublasLtMatrixLayoutDestroy A failed "
                << cublas_status << endl;
            exit(-1);
        }

        cublas_status = cublasLtMatrixLayoutDestroy(Bdesc);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cout << "cublasLtMatrixLayoutDestroy B failed "
                << cublas_status << endl;
            exit(-1);
        }

        cublas_status = cublasLtMatrixLayoutDestroy(Cdesc);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cout << "cublasLtMatrixLayoutDestroy C failed "
                << cublas_status << endl;
            exit(-1);
        }

        cublas_status = cublasLtMatmulDescDestroy(matmulDesc);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cout << "cublasLtMatmulDescDestroy failed cublas_status "
                << cublas_status << endl;
            exit(-1);
        }

        CHECK_ERROR(cudaFree(workspace));
        CHECK_ERROR(cudaFree(A));
        CHECK_ERROR(cudaFree(B));
        CHECK_ERROR(cudaFree(C));
    }

};

void burn(int gpu, int u_secs, int d_secs) {
    // printf("BURN, gpu: %d, up seconds: %d, down_seconds: %d\n",gpu,u_secs,d_secs);
    BurnGPU *burngpu = new BurnGPU(gpu, u_secs, d_secs);
    (*burngpu)();
}


