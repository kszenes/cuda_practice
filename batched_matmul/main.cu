#include <iostream>
#include <cuda_runtime.h>
#include "timer.h"
#include <cublas_v2.h>
#include <utils.h>


template <typename T>
void host_strided_gemm(
    const size_t Batches, const size_t M, const size_t N, const size_t K,
    const T alpha, const T beta, const T* A, const T* B, T* C);

template <typename T>
T rmse(const size_t n, const T* v_ref, const T* v_d);


template <typename T>
void printMatrix_d(size_t, size_t, const T*);

int main() {
    CUDA_CHECK(cudaSetDevice(2));
    const int runs = 2;
    const bool checkRMSE = false;
    #define FLOAT
    #if defined(FLOAT)
    printf("Single precision\n");
    typedef float floatType;

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_32F;
    #undef TENSOR
    #elif defined(TENSOR)
    printf("Tensor float precision\n");
    typedef float floatType;
    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    #undef TENSOR
    #elif defined(DOUBLE)
    printf("Double precision\n");
    typedef double floatType;

    cudaDataType_t typeA = CUDA_R_64F;
    cudaDataType_t typeB = CUDA_R_64F;
    cudaDataType_t typeC = CUDA_R_64F;
    cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_64F;
    #undef DOUBLE
    #endif

    const floatType alpha = 1.0f;
    const floatType beta = 0.0f;
    const size_t size = (1 << 6);
    const size_t M = size * size * size;
    const size_t N = size;
    const size_t K = size;
    const size_t Batches = size;
    const size_t max_streams = 100;
    const size_t num_streams = min(Batches, max_streams);

    size_t first_batches = Batches;
    size_t second_batches = Batches;
    size_t third_batches = Batches;

    size_t first_tensor_offset = M * K;
    size_t second_tensor_offset = K * N;
    size_t third_tensor_offset = M * N;
    
    // first_tensor_offset = 0; first_batches = 1;
    second_tensor_offset = 0; second_batches = 1;
    // third_tensor_offset = 0; third_batches = 1;


    const double tflops = 2 * Batches * M * N * K * 1e-12;
    const double gigaBytes = (M * K * first_batches + K * N * second_batches
                         + M * N * third_batches) * sizeof(floatType) * 1e-9;
    printf("Runs: %d\n", runs);
    printf("Dimension:\n\tBatches: %zu\n\tM: %zu\n\tN: %zu\n\tK: %zu\n", Batches, M, N, K);
    printf("Memory usage:             %f GB\n", gigaBytes);
    printf("Arithmetic Intenisity:    %f\n", tflops * 1e3/gigaBytes);

    GPUTimer timer;

    floatType *A_h, *B_h, *C_h;

    CUDA_CHECK(cudaMallocHost(&A_h, first_batches * M * K * sizeof(floatType)));
    CUDA_CHECK(cudaMallocHost(&B_h, second_batches * K * N * sizeof(floatType)));
    CUDA_CHECK(cudaMallocHost(&C_h, third_batches * M * N * sizeof(floatType)));


    floatType *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, first_batches * M * K * sizeof(floatType)));
    CUDA_CHECK(cudaMalloc(&B_d, second_batches * K * N * sizeof(floatType)));
    CUDA_CHECK(cudaMalloc(&C_d, third_batches * M * N * sizeof(floatType)));

    for (size_t i = 0; i < first_batches * M * K; i++) {
        A_h[i] = (((floatType) rand()) / RAND_MAX - 0.5);
    }
    for (size_t i = 0; i < second_batches * K * N; i++) {
        B_h[i] = (((floatType) rand()) / RAND_MAX - 0.5);
    }
    
    for (size_t i = 0; i < third_batches * M * N; i++){
        C_h[i] = 0.0;
    }
    CUDA_CHECK(cudaMemcpy(A_d, A_h, first_batches * M * K * sizeof(floatType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, second_batches * K * N * sizeof(floatType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C_h, third_batches * M * N * sizeof(floatType), cudaMemcpyHostToDevice));

    double cpu_time = 0.0;
    if (checkRMSE) {
        for (int iter = 0; iter < runs; iter++) {
            host_strided_gemm(Batches, M, N, K, alpha, beta, A_h, B_h, C_h);
            timer.start();
            host_strided_gemm(Batches, M, N, K, alpha, beta, A_h, B_h, C_h);
            cpu_time += timer.seconds() / runs;
        }
    }


    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t error;
    double cublas_strided_time = 0.0;
    for (int iter = 0; iter < runs; iter++) {
        error = cublasGemmStridedBatchedEx(
                    handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N, 
                    M, N, K, 
                    &alpha, 
                    A_d, typeA, M, first_tensor_offset,
                    B_d, typeB, K, second_tensor_offset,
                    &beta, 
                    C_d, typeC, M, third_tensor_offset,
                    Batches,
                    cublasComputeType,
                    CUBLAS_GEMM_DEFAULT); // warmup
        timer.start();
        error = cublasGemmStridedBatchedEx(
                    handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N, 
                    M, N, K, 
                    &alpha, 
                    A_d, typeA, M, first_tensor_offset,
                    B_d, typeB, K, second_tensor_offset,
                    &beta, 
                    C_d, typeC, M, third_tensor_offset,
                    Batches,
                    cublasComputeType,
                    CUBLAS_GEMM_DEFAULT); // warmup
        cublas_strided_time += timer.seconds() / runs;

        if (error != CUBLAS_STATUS_SUCCESS) {
            printf("Error\n");
            exit(-1);
        }
    }

    if (checkRMSE) {
        auto myRMSE = rmse(Batches * M * N, C_h, C_d);
        printf("RMSE BatchedStrided: %f\n", myRMSE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.start();
    cudaStream_t streams[num_streams];
    for (size_t iter = 0; iter < num_streams; iter++) {
        CUDA_CHECK(cudaStreamCreate(&streams[iter]));
    }
    auto stream_create_time = timer.seconds();

    for (size_t iter = 0; iter < Batches; iter++) {
        cublasSetStream(handle, streams[iter % num_streams]);
        error = cublasGemmEx(
                    handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N, 
                    M, N, K, 
                    &alpha, 
                    A_d + iter * (first_tensor_offset), typeA, M,
                    B_d + iter * (second_tensor_offset), typeB, K,
                    &beta, 
                    C_d + iter * (third_tensor_offset), typeC, M,
                    cublasComputeType,
                    CUBLAS_GEMM_DEFAULT); // warmup

        if (error != CUBLAS_STATUS_SUCCESS) {
            printf("Error\n");
            exit(-1);
        }
    }
    double cublas_looped_time = 0.0;
    timer.start();
    for (size_t iter = 0; iter < Batches; iter++) {
        cublasSetStream(handle, streams[iter % num_streams]);
        error = cublasGemmEx(
                    handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N, 
                    M, N, K, 
                    &alpha, 
                    A_d + iter * (first_tensor_offset), typeA, M,
                    B_d + iter * (second_tensor_offset), typeB, K,
                    &beta, 
                    C_d + iter * (third_tensor_offset), typeC, M,
                    cublasComputeType,
                    CUBLAS_GEMM_DEFAULT); // warmup

        if (error != CUBLAS_STATUS_SUCCESS) {
            printf("Error\n");
            exit(-1);
        }
    }
    cublas_looped_time = timer.seconds();

    timer.start();
    for (size_t iter = 0; iter < num_streams; iter++) {
        CUDA_CHECK(cudaStreamDestroy(streams[iter]));
    }
    auto stream_destroy_time = timer.seconds();


    if (checkRMSE) {
        auto myRMSE = rmse(Batches * M * N, C_h, C_d);
        printf("RMSE Looped: %f\n", myRMSE);
    }
    auto looped_tot_time = cublas_looped_time + stream_create_time + stream_destroy_time;
    printf("Stream: creation %f sec ; destruction %f sec\n", stream_create_time, stream_destroy_time);
    printf("cuBLAS Batched Strided:      %.2f GB/s; %.2f TFLOP/s (%f sec) \n", gigaBytes/cublas_strided_time, tflops/cublas_strided_time, cublas_strided_time);
    printf("cuBLAS Looped:               %.2f GB/s; %.2f TFLOP/s (%f sec) \n", gigaBytes/cublas_looped_time, tflops/cublas_looped_time, cublas_looped_time);
    printf("cuBLAS Looped (with stream): %.2f GB/s; %.2f TFLOP/s (%f sec) \n", gigaBytes/looped_tot_time, tflops/looped_tot_time, looped_tot_time);
    if (checkRMSE) printf("CPU:\t%.2f GB/s; %.2f TFLOP/s (%f sec) \n", gigaBytes/cpu_time, tflops/cpu_time, cpu_time);

    cublasDestroy_v2(handle);
    CUDA_CHECK(cudaFreeHost(C_h));
    CUDA_CHECK(cudaFreeHost(A_h));
    CUDA_CHECK(cudaFreeHost(B_h));

    CUDA_CHECK(cudaFree(C_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(A_d));
}


template <typename T>
void host_strided_gemm(
    const size_t Batches, const size_t M, const size_t N, const size_t K,
    const T alpha, const T beta, const T* A, const T* B, T* C)
{
    for (size_t batch = 0; batch < Batches; batch++) {
        for (size_t row = 0; row < M; row++) {
            for (size_t col = 0; col < N; col++) {
                T tmp = 0.;
                for (size_t k = 0; k < K; k++)  {
                    tmp += alpha * A[row + k * M + batch * K * M]
                            * B[k + col * K + batch * K * N];
                }
                C[row + col * M + batch * M * N] = tmp + beta
                                                * C[row + col * M + batch * M * N];
            }
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
