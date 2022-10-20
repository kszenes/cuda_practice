#include <iostream>
#include <cuda_runtime.h>
#include "timer.h"
#include <cublas_v2.h>

#define CUDA_CHECK(val) {                                                  \
    if (val != cudaSuccess) {                                              \
        printf("Error: %s in line %d\n",cudaGetErrorName(val), __LINE__);  \
        exit(-1);                                                          \
    }                                                                      \
}

#define TILE_DIM 32

template <typename T>
void printMatrix_d(size_t, size_t, const T*);

template <typename T>
void host_gemm(
    const size_t M, const size_t N, const size_t K,
    const T alpha, const T beta, const T* A, const T* B, T* C);

template <typename T>
T rmse(const size_t, const T*, const T*);

template <typename T>
__global__
void gemmNaive(const size_t M, const size_t N, const size_t K,
               const T alpha, const T beta, const T* A, const T* B, T* C) {
    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int col = threadIdx.y + blockIdx.y * blockDim.y;
    if (row < M && col < N) {
        T tmp = 0.;
        for (size_t k = 0; k < K; k++) {
            tmp += alpha * A[row + k * M] * B[k + col * K];
        }
        C[row + col * M] = tmp + beta * C[row + col * M];
    }
}

template <typename T>
__global__ void gemmCoalesced(const size_t M, const size_t N, const size_t K,
                              const T alpha, const T beta,
                              const T* __restrict__ A,
                              const T* __restrict__ B,
                              T* __restrict__ C)
{
    __shared__ T A_s[TILE_DIM * (TILE_DIM + 1)];
    __shared__ T B_s[TILE_DIM * (TILE_DIM + 1)];

    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < M && col < N) {
        A_s[threadIdx.x + (TILE_DIM + 1) * threadIdx.y] = A[row + M * threadIdx.y];
        B_s[threadIdx.x + (TILE_DIM + 1) * threadIdx.y] = B[threadIdx.x + K * col];
        __syncthreads();

        T tmp = 0.;
        T tmpA = 0.0;
        T tmpB = 0.0;
        #pragma unroll
        for (size_t k = 0; k < K; k++) {
            tmpA = A_s[threadIdx.x + k * (TILE_DIM + 1)];
            tmpB = B_s[k + threadIdx.y * (TILE_DIM + 1)];
            tmp += tmpA * tmpB;
        }
        C[row + col * M] = alpha * tmp + beta * C[row + col * M];
    }
}

// template <typename T>
// __global__
// void gemmCoalesced(const size_t M, const size_t N, const size_t K,
//                    const T alpha, const T beta, const T* A, const T* B, T* C) {
//     const int row = threadIdx.x + blockIdx.x * blockDim.x;
//     const int col = threadIdx.y + blockIdx.y * blockDim.y;

//     __shared__ T A_shmem[TILE_DIM][TILE_DIM+1];
//     __shared__ T B_shmem[TILE_DIM][TILE_DIM+1];

//     if (row < M && col < N) {
//         A_shmem[threadIdx.y][threadIdx.x] = A[row + M * threadIdx.y];
//         B_shmem[threadIdx.x][threadIdx.y] = B[threadIdx.x  + col * K];
//         __syncthreads();

//         T  tmp = 0.0;
//         for (int k = 0; k < K; k++) {
//             tmp += alpha * A_shmem[k][threadIdx.x] * B_shmem[k][threadIdx.y];
//         }
//         C[row + col * M] = tmp + beta * C[row + col * M];
//     }
// }


template <typename T>
__global__ void sharedABMultiply(const T *a, const T* b, T *c,
                                 size_t N)
{
    __shared__ T aTile[TILE_DIM][TILE_DIM],
                 bTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    T sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
    }
    c[row*N+col] = sum;
}

int main() {
    const int runs = 1;
    const bool checkRMSE = false;
    typedef float floatType;
    #define FLOAT

    const floatType alpha = 1.0f;
    const floatType beta = 0.0f;
    const size_t size = 1 << 14 ;
    const size_t M = size;
    const size_t N = size;
    const size_t K = TILE_DIM;

    const double gigaFlops = 2 * M * N * K * 1e-9;
    const double gigaBytes = (M * K + K * N +  M * N) * sizeof(floatType) * 1e-9;
    printf("Runs: %d\n", runs);
    printf("Dimension:\n\tM: %zu\n\tN: %zu\n\tK: %zu\n", M, N, K);
    printf("Memory usage = %f GB\n", gigaBytes);

    GPUTimer timer;

    floatType *A_h, *B_h, *C_h;

    CUDA_CHECK(cudaMallocHost(&A_h, M * K * sizeof(floatType)))
    CUDA_CHECK(cudaMallocHost(&B_h, K * N * sizeof(floatType)));
    CUDA_CHECK(cudaMallocHost(&C_h, M * N * sizeof(floatType)));


    floatType *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, M * K * sizeof(floatType)));
    CUDA_CHECK(cudaMalloc(&B_d, K * N * sizeof(floatType)));
    CUDA_CHECK(cudaMalloc(&C_d, M * N * sizeof(floatType)));

    if (checkRMSE) {
        for (size_t i = 0; i < M * K; i++) {
            A_h[i] = (((floatType) rand()) / RAND_MAX - 0.5);
        }
        for (size_t i = 0; i < K * N; i++) {
            B_h[i] = (((floatType) rand()) / RAND_MAX - 0.5);
        }
        
        for (size_t i = 0; i < M * N; i++){
            C_h[i] = 0.0;
        }
        CUDA_CHECK(cudaMemcpy(A_d, A_h, M * K * sizeof(floatType), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B_d, B_h, K * N * sizeof(floatType), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(C_d, C_h, M * N * sizeof(floatType), cudaMemcpyHostToDevice));
    }

    const dim3 numThreads(TILE_DIM, TILE_DIM);
    const dim3 numBlocks(
        (M + numThreads.x - 1) / numThreads.x,
        (N + numThreads.y - 1) / numThreads.y
    );

    double cpu_time = 0.0;
    if (checkRMSE) {
        for (int iter = 0; iter < runs; iter++) {
            host_gemm(M, N, K, alpha, beta, A_h, B_h, C_h);
            timer.start();
            host_gemm(M, N, K, alpha, beta, A_h, B_h, C_h);
            cpu_time += timer.seconds() / runs;
        }
    }

    floatType *tmp_d;
    CUDA_CHECK(cudaMalloc(&tmp_d, M * N * sizeof(floatType)));
    CUDA_CHECK(cudaMemset(tmp_d, 0, M * N * sizeof(floatType)));

    // double io_time = 0.0;
    // for (int iter = 0; iter < runs; iter++) {
    //     timer.start();
    //     device_io<<<numBlocks, numThreads>>>(n, a, x_d, tmp_d);
    //     io_time += timer.seconds() / runs;
    // }

    double gpu_time = 0.0;
    for (int iter = 0; iter < runs; iter++) {
        gemmNaive<<<numBlocks, numThreads>>>(M, N, K, alpha, beta, A_d, B_d, tmp_d);
        timer.start();
        gemmNaive<<<numBlocks, numThreads>>>(M, N, K, alpha, beta, A_d, B_d, tmp_d);
        gpu_time += timer.seconds() / runs;
    }
    if (checkRMSE) {
        auto myRMSE = rmse(M * N, C_h, tmp_d);
        printf("RMSE Naive: %f\n", myRMSE);
    }

    CUDA_CHECK(cudaMemset(tmp_d, 0, M * N * sizeof(floatType)));

    double coalescedTime = 0.0;
    for (int iter = 0; iter < runs; iter++) {
        gemmCoalesced<<<numBlocks, numThreads>>>(M, N, K, alpha, beta, A_d, B_d, tmp_d);
        timer.start();
        gemmCoalesced<<<numBlocks, numThreads>>>(M, N, K, alpha, beta, A_d, B_d, tmp_d);
        coalescedTime += timer.seconds() / runs;
    }
    if (checkRMSE) {
        auto myRMSE = rmse(M * N, C_h, tmp_d);
        printf("RMSE Coalesced: %f\n", myRMSE);
    }

    double nvidiaTime = 0.0;
    // for (int iter = 0; iter < runs; iter++) {
    //     sharedABMultiply<<<numBlocks, numThreads>>>(A_d, B_d, C_d, M);
    //     timer.start();
    //     sharedABMultiply<<<numBlocks, numThreads>>>(A_d, B_d, C_d, M);
    //     nvidiaTime += timer.seconds() / runs;
    // }
    // if (checkRMSE) {
    //     auto myRMSE = rmse(M * N, C_h, tmp_d);
    //     printf("RMSE Coalesced: %f\n", myRMSE);
    // }

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t error;
    CUDA_CHECK(cudaMemset(tmp_d, 0, M * N * sizeof(floatType)));
    double cublas_time = 0.0;
    for (int iter = 0; iter < runs; iter++) {
        #if defined(DOUBLE)
        error = cublasDgemm_v2(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
            A_d, M, B_d, K, &beta, tmp_d, M
        );
        #elif defined(FLOAT)
        error = cublasSgemm_v2(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
            A_d, M, B_d, K, &beta, tmp_d, M
        );
        #endif
        timer.start();
        #if defined(DOUBLE)
        error = cublasDgemm_v2(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
            A_d, M, B_d, K, &beta, tmp_d, M
        );
        #elif defined(FLOAT)
        error = cublasSgemm_v2(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
            A_d, M, B_d, K, &beta, tmp_d, M
        );
        #endif
        cublas_time += timer.seconds() / runs;

        if (error != CUBLAS_STATUS_SUCCESS) {
            printf("Error\n");
            exit(-1);
        }
    }

    if (checkRMSE) {
        auto myRMSE = rmse(M * N, C_h, tmp_d);
        printf("RMSE cuBLAS: %f\n", myRMSE);
    }
    // printf("IO: %f sec\n", io_time);
    printf("Naive:\t%.2f GB/s; %.2f GFLOPS (%f sec) \n", gigaBytes/gpu_time, gigaFlops/gpu_time, gpu_time);
    printf("Coalesced:\t%.2f GB/s; %.2f GFLOPS (%f sec) \n", gigaBytes/coalescedTime, gigaFlops/coalescedTime, coalescedTime);
    printf("cuBLAS:\t%.2f GB/s; %.2f GFLOPS (%f sec) \n", gigaBytes/cublas_time, gigaFlops/cublas_time, cublas_time);
    printf("NVIDIA:\t%.2f GB/s; %.2f GFLOPS (%f sec) \n", gigaBytes/nvidiaTime, gigaFlops/nvidiaTime, nvidiaTime);
    if (checkRMSE) printf("CPU:\t%.2f GB/s; %.2f GFLOPS (%f sec) \n", gigaBytes/cpu_time, gigaFlops/cpu_time, cpu_time);

    cublasDestroy_v2(handle);
    CUDA_CHECK(cudaFreeHost(C_h));
    CUDA_CHECK(cudaFreeHost(A_h));
    CUDA_CHECK(cudaFreeHost(B_h));

    CUDA_CHECK(cudaFree(tmp_d));
    CUDA_CHECK(cudaFree(C_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(A_d));
}


template <typename T>
void host_gemm(const size_t M, const size_t N, const size_t K, const T alpha, const T beta, const T* A, const T* B, T* C) {
    for (size_t row = 0; row < M; row++) {
        for (size_t col = 0; col < N; col++) {
            T tmp = 0.;
            for (size_t k = 0; k < K; k++)  {
                tmp += alpha * A[row + k * M] * B[k + col * K];
            }
            C[row + col * M] = tmp + beta * C[row + col * M];
        }
    }
}

template <typename T>
T rmse(const size_t n, const T* v_ref, const T* v_d) {
    T *v_h = (T*) malloc(n * sizeof(T));
    CUDA_CHECK(cudaMemcpy(v_h, v_d, n * sizeof(T), cudaMemcpyDeviceToHost));

    T diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        // printf("Ref: %f; Dev: %f\n", v_ref[i], v_h[i]);
        diff += std::sqrt((v_ref[i] - v_h[i]) * (v_ref[i] - v_h[i]));
        // printf("mat[%zu] = %f\n", i, v_h[i]);
    }

    free(v_h);
    return diff / n;

}

template <typename T>
void printMatrix_d(const size_t rows, const size_t cols, const T* matrix_d) {
    T* matrix_h;
    CUDA_CHECK(cudaMallocHost(&matrix_h, rows * cols * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(matrix_h, matrix_d, rows * cols * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            std::cout << matrix_h[col + row * cols] << ' ';
        }
        std::cout << '\n';
    }
    CUDA_CHECK(cudaFreeHost(matrix_h));
    std::cout << '\n';
}
