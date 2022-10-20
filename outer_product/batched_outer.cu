#include <iostream>
#include <cuda_runtime.h>
#include "timer.h"
#include <cublas_v2.h>
#include <vector>

#define CUDA_CHECK(val) {                                                  \
    if (val != cudaSuccess) {                                              \
        printf("Error: %s in line %d\n",cudaGetErrorName(val), __LINE__);  \
        exit(-1);                                                          \
    }                                                                      \
}

template <typename T>
void host_ger(const size_t, const size_t, const size_t, const T, const T*, const T*, T*);

template <typename T>
T rmse(const size_t, const T*, const T*);

template <typename T>
__global__
void device_ger(const size_t xDim, const size_t yDim, const T a, const T* x, const T* y, T* mat) {
    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int col = threadIdx.y + blockIdx.y * blockDim.y;
    if (row < xDim && col < yDim) {
        mat[row + col * xDim] += a * x[row] * y[col];
    }
}

template <typename T>
__global__
void device_khatriRao(const size_t xDim, const size_t yDim, const size_t nBatches, const T a, const T* x, const T* y, T* mat) {
    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int col = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = threadIdx.z + blockIdx.z * blockDim.z;
    if (row < xDim && col < yDim && batch < nBatches) {
        mat[(batch * xDim * yDim) + row + col * xDim] += a * x[(batch * xDim) + row] * y[(batch * yDim) + col];
    }
}

// template <typename T>

int main() {
    const int runs = 1;
    const bool checkRMSE = false;
    #define FLOAT

    #if defined(DOUBLE)
        #define DTYPE double
    #elif defined(FLOAT)
        #define DTYPE float
    #endif

    CUDA_CHECK(cudaSetDevice(1));


    const DTYPE a = 1.f;
    // const DTYPE b = 2.9f;
    const size_t size = 1 << 10;
    const size_t xDim = size;
    const size_t yDim = size;
    const size_t nBatches = size;

    const double numFlops = 3 * xDim * yDim * 1e-9;
    const double numBytes = (2 * xDim * yDim + xDim + yDim) * nBatches * sizeof(DTYPE) * 1e-9;
    printf("Runs: %d\n", runs);
    printf("xDim: %zu\nyDim: %zu\n", xDim, yDim);
    printf("Memory usage = %f GB\n", numBytes);

    GPUTimer timer;

    DTYPE *x_h, *y_h, *mat_h;

    CUDA_CHECK(cudaMallocHost(&mat_h, nBatches * xDim * yDim * sizeof(DTYPE)))
    CUDA_CHECK(cudaMallocHost(&x_h, nBatches * xDim * sizeof(DTYPE)));
    CUDA_CHECK(cudaMallocHost(&y_h, nBatches * yDim * sizeof(DTYPE)));


    DTYPE *x_d, *y_d, *mat_d;
    CUDA_CHECK(cudaMalloc(&mat_d, nBatches * xDim * yDim * sizeof(DTYPE)));
    CUDA_CHECK(cudaMalloc(&x_d, nBatches * xDim * sizeof(DTYPE)));
    CUDA_CHECK(cudaMalloc(&y_d, nBatches * yDim * sizeof(DTYPE)));

    if (checkRMSE) {
        for (size_t i = 0; i < nBatches * xDim; i++) {
            x_h[i] = (((DTYPE) rand()) / RAND_MAX - 0.5) * 100;
        }
        for (size_t i = 0; i < nBatches * yDim; i++) {
            y_h[i] = (((DTYPE) rand()) / RAND_MAX - 0.5) * 100;
        }
        
        for (size_t i = 0; i <  nBatches * xDim*yDim; i++){
            mat_h[i] = 0.0;
        }
        CUDA_CHECK(cudaMemcpy(x_d, x_h, nBatches * xDim * sizeof(DTYPE), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(y_d, y_h, nBatches * yDim * sizeof(DTYPE), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mat_d, mat_h, nBatches * xDim * yDim * sizeof(DTYPE), cudaMemcpyHostToDevice));
    }

    dim3 numThreads(32, 32);
    dim3 numBlocks(
        (xDim + numThreads.x - 1) / numThreads.x,
        (yDim + numThreads.y - 1) / numThreads.y
    );

    double cpu_time = 0.0;
    if (checkRMSE) {
        for (int iter = 0; iter < runs; iter++) {
            if (!checkRMSE) host_ger(xDim, yDim, 1, a, x_h, y_h, mat_h);
            timer.start();
            host_ger(xDim, yDim, nBatches, a, x_h, y_h, mat_h);
            cpu_time += timer.seconds() / runs;
        }
    }

    DTYPE *tmp_d;
    CUDA_CHECK(cudaMalloc(&tmp_d, nBatches * xDim * yDim * sizeof(DTYPE)));
    CUDA_CHECK(cudaMemset(tmp_d, 0, nBatches * xDim * yDim * sizeof(DTYPE)));

    // double io_time = 0.0;
    // for (int iter = 0; iter < runs; iter++) {
    //     timer.start();
    //     device_io<<<numBlocks, numThreads>>>(n, a, x_d, tmp_d);
    //     io_time += timer.seconds() / runs;
    // }

    double gpu_time = 0.0;
    for (int iter = 0; iter < runs; iter++) {
        if (!checkRMSE) device_ger<<<numBlocks, numThreads>>>(xDim, yDim, a, x_d, y_d, tmp_d);
        timer.start();
        for (size_t batch = 0; batch < nBatches; batch++) {
            device_ger<<<numBlocks, numThreads>>>(xDim, yDim, a, x_d + batch * xDim, y_d + batch * yDim, tmp_d + batch * xDim * yDim);
            // device_ger<<<numBlocks, numThreads>>>(xDim, yDim, a, x_d, y_d, tmp_d);
        }
        gpu_time += timer.seconds() / runs;
    }
    if (checkRMSE) {
        auto myRMSE = rmse(xDim * yDim, mat_h, tmp_d);
        printf("RMSE Batched: %f\n", myRMSE);
    }

    numThreads = dim3(16, 16, 4);
    numBlocks = dim3(
        (xDim + numThreads.x - 1) / numThreads.x,
        (yDim + numThreads.y - 1) / numThreads.y,
        (nBatches + numThreads.z - 1) / numThreads.z
    );

    CUDA_CHECK(cudaMemset(tmp_d, 0, nBatches * xDim * yDim * sizeof(DTYPE)));
    double gpu_batched_time = 0.0;
    for (int iter = 0; iter < runs; iter++) {
        if (!checkRMSE) device_ger<<<numBlocks, numThreads>>>(xDim, yDim, a, x_d, y_d, tmp_d);
        timer.start();
        device_khatriRao<<<numBlocks, numThreads>>>(xDim, yDim, nBatches, a, x_d, y_d, tmp_d);
        gpu_batched_time += timer.seconds() / runs;
    }
    if (checkRMSE) {
        auto myRMSE = rmse(xDim * yDim, mat_h, tmp_d);
        printf("RMSE Tiled: %f\n", myRMSE);
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t error;
    CUDA_CHECK(cudaMemset(tmp_d, 0, nBatches * xDim * yDim * sizeof(DTYPE)));
    double cublas_time = 0.0;
    for (int iter = 0; iter < runs; iter++) {
        if (!checkRMSE) {
            #if defined(DOUBLE)
            error = cublasDger_v2(
                handle, xDim, yDim, &a, x_d, 1, y_d, 1, tmp_d, xDim
            );
            #elif defined(FLOAT)
            error = cublasSger_v2(
                handle, xDim, yDim, &a, x_d, 1, y_d, 1, tmp_d, xDim
            );
            #endif
        }
        timer.start();
        for (size_t batch = 0; batch < nBatches; batch++) {
            #if defined(DOUBLE)
            error = cublasDger_v2(
                handle, xDim, yDim, &a, x_d + batch * xDim, 1, y_d + batch * yDim, 1, tmp_d + batch * xDim * yDim, xDim
            );
            #elif defined(FLOAT)
            error = cublasSger_v2(
                handle, xDim, yDim, &a, x_d + batch * xDim, 1, y_d + batch * yDim, 1, tmp_d + batch * xDim * yDim, xDim
            );
            #endif
        }
        cublas_time += timer.seconds() / runs;

        if (error != CUBLAS_STATUS_SUCCESS) {
            printf("Error\n");
            exit(-1);
        }
    }

    if (checkRMSE) {
        auto myRMSE = rmse(xDim * yDim, mat_h, tmp_d);
        printf("RMSE cuBLAS: %f\n", myRMSE);
    }
    // printf("IO: %f sec\n", io_time);
    printf("Batched:\t%.2f GB/s; %.2f GFLOPS (%f sec) \n", numBytes/gpu_time, numFlops/gpu_time, gpu_time);
    printf("Tiled:  \t%.2f GB/s; %.2f GFLOPS (%f sec) \n", numBytes/gpu_batched_time, numFlops/gpu_batched_time, gpu_batched_time);
    printf("cuBLAS: \t%.2f GB/s; %.2f GFLOPS (%f sec) \n", numBytes/cublas_time, numFlops/cublas_time, cublas_time);
    if (checkRMSE) printf("CPU:\t\t%.2f GB/s; %.2f GFLOPS (%f sec) \n", numBytes/cpu_time, numFlops/cpu_time, cpu_time);

    cublasDestroy_v2(handle);
    CUDA_CHECK(cudaFreeHost(mat_h));
    CUDA_CHECK(cudaFreeHost(x_h));
    CUDA_CHECK(cudaFreeHost(y_h));

    CUDA_CHECK(cudaFree(mat_d));
    CUDA_CHECK(cudaFree(tmp_d));
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
}


template <typename T>
void host_ger(const size_t xDim, const size_t yDim, const size_t nBatches, const T a, const T* x_h, const T* y_h, T* mat_h) {
    for (size_t batch = 0; batch < nBatches; batch++) {
        for (size_t row = 0; row < xDim; row++) {
            for (size_t col = 0; col < yDim; col++) {
                mat_h[(batch * xDim * yDim) + row + xDim * col] = a * x_h[(batch * xDim) + row] * y_h[(batch * yDim) + col];
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
