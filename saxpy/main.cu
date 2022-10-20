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

template <typename T>
void host_saxpy(const size_t, const T, const T*, T*);

template <typename T>
T rmse(const size_t, const T*, const T*);

__global__ void device_copy_vector2_kernel(double* d_in, double* d_out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = idx; i < N/2; i += blockDim.x * gridDim.x) {
    reinterpret_cast<double2*>(d_out)[i] = reinterpret_cast<double2*>(d_in)[i];
  }

  // in only one thread, process final element (if there is one)
  if (idx==N/2 && N%2==1)
    d_out[N-1] = d_in[N-1];
}

__global__ void device_copy_vector4_kernel(double* d_in, double* d_out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = idx; i < N/4; i += blockDim.x * gridDim.x) {
    reinterpret_cast<double4*>(d_out)[i] = reinterpret_cast<double4*>(d_in)[i];
  }

  // in only one thread, process final elements (if there are any)
  int remainder = N%4;
  if (idx==N/4 && remainder!=0) {
    while(remainder) {
      int idx = N - remainder--;
      d_out[idx] = d_in[idx];
    }
  }
}


template <typename T>
__global__
void device_saxpy(const size_t n, const T a, const T* x, T* y) {
    // const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // if (idx < n) {
    //     y[idx] = a * x[idx] + y[idx];
    // }
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < n;
         idx += blockDim.x * gridDim.x)
    {
        y[idx] = a * x[idx] + y[idx];
    }
    // for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //      idx < n/2;
    //      idx += blockDim.x * gridDim.x)
    // {
    //     const double2 x_tmp = reinterpret_cast<const double2*>(x)[idx];
    //     double2 y_tmp = reinterpret_cast<double2*>(y)[idx];
        
    //     y_tmp.x = a * x_tmp.x + y_tmp.x;
    //     y_tmp.y = a * x_tmp.y + y_tmp.y;

    //     reinterpret_cast<double2*>(y)[idx] = y_tmp;
    // }
}

template <typename T>
__global__
void device_io(const size_t n, const T* x, T* y) {
    // const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // if (idx < n) {
    //     y[idx] = x[idx];
    // }
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < n;
         idx += blockDim.x * gridDim.x)
    {
        y[idx] = x[idx];
    }
}

int main() {
    const int runs = 10;
    #define DTYPE double

    CUDA_CHECK(cudaSetDevice(1));
    const bool checkRMSE = false;


    const DTYPE a = 1.7f;
    const size_t n = 1 << 29;
    // const size_t n = 1024 * 1024 * 1024;

    const double numFlops = 2. * n * 1e-9;
    const double numBytes = 3. * n * sizeof(DTYPE) * 1e-9;
    printf("Memory usage = %zu MB\n", 2 * n * sizeof(DTYPE) / 1024 / 1024);

    GPUTimer timer;

    DTYPE *x_h, *y_h;

    CUDA_CHECK(cudaMallocHost(&x_h, n * sizeof(DTYPE)));
    CUDA_CHECK(cudaMallocHost(&y_h, n * sizeof(DTYPE)));


    DTYPE *x_d, *y_d;
    CUDA_CHECK(cudaMalloc(&x_d, n * sizeof(DTYPE)));
    CUDA_CHECK(cudaMalloc(&y_d, n * sizeof(DTYPE)));

    if (checkRMSE) {
        for (size_t i = 0; i < n; i++) {
            x_h[i] = (((DTYPE) rand()) / RAND_MAX - 0.5) * 100;
            y_h[i] = 0.0f;
        }
    }

    CUDA_CHECK(cudaMemcpy(x_d, x_h, n * sizeof(DTYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_d, y_h, n * sizeof(DTYPE), cudaMemcpyHostToDevice));

    const int numThreads = 256;
    const int numBlocks = (n + numThreads - 1) / numThreads;

    // int numSMs;
    // cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 1);
    // const int numBlocks = 32*numSMs;

    double cpu_time = 0.0;
    if (checkRMSE) {
        for (int iter = 0; iter < runs; iter++) {
            timer.start();
            host_saxpy(n, a, x_h, y_h);
            cpu_time += timer.seconds() / runs;

        }
    }

    DTYPE *tmp_d;
    CUDA_CHECK(cudaMalloc(&tmp_d, n * sizeof(DTYPE)));

    double io_time = 0.0;
    for (int iter = 0; iter < runs; iter++) {
        device_io<<<numBlocks, numThreads>>>(n, x_d, tmp_d);
        timer.start();
        device_io<<<numBlocks, numThreads>>>(n, x_d, tmp_d);
        io_time += timer.seconds() / runs;
    }

    double vec2_time = 0.0;
    int threads = 1024;
    int blocks = (n/2 + threads-1) / threads;
    for (int iter = 0; iter < runs; iter++) {
        device_copy_vector2_kernel<<<blocks, threads>>>(x_d, tmp_d, n);
        timer.start();
        device_copy_vector2_kernel<<<blocks, threads>>>(x_d, tmp_d, n);
        vec2_time += timer.seconds() / runs;
    }

    // double vec4_time = 0.0;
    // blocks = (n/4 + threads-1) / threads;
    // for (int iter = 0; iter < runs; iter++) {
    //     device_copy_vector4_kernel<<<blocks, threads>>>(x_d, tmp_d, n);
    //     timer.start();
    //     device_copy_vector4_kernel<<<blocks, threads>>>(x_d, tmp_d, n);
    //     vec4_time += timer.seconds() / runs;
    // }

    // CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, device_saxpy<double>, 0, 0))
    std::cout << "THREADS: " << threads << "; BLOCKS: " << blocks << '\n';
    double gpu_time = 0.0;
    for (int iter = 0; iter < runs; iter++) {
        device_saxpy<<<blocks, threads>>>(n, a, x_d, y_d);
        timer.start();
        device_saxpy<<<blocks, threads>>>(n, a, x_d, y_d);
        gpu_time += timer.seconds() / runs;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t error;
    CUDA_CHECK(cudaMemset(tmp_d, 0, n * sizeof(DTYPE)));
    double cublas_time = 0.0;
    for (int iter = 0; iter < runs; iter++) {
        timer.start();
        error = cublasDaxpy_v2(
            handle, n, &a, x_d, 1, tmp_d, 1
        );
        if (error != CUBLAS_STATUS_SUCCESS) {
            printf("Error\n");
            exit(-1);
        }
        cublas_time += timer.seconds() / runs;
    }

    if (checkRMSE) {
        auto myRMSE = rmse(n, y_h, tmp_d);
        printf("RMSE: %f\n", myRMSE);
    }
    printf("IO:      %.2f GB/s (%f sec)\n", 2.0 / 3.0 * numBytes / io_time, io_time);
    printf("2-Vec:   %.2f GB/s (%f sec)\n", 2.0 / 3.0 * numBytes / vec2_time, vec2_time);
    // printf("4-Vec:   %.2f GB/s (%f sec)\n", 2.0 / 3.0 * numBytes / vec4_time, vec4_time);
    printf("GPU:     %.2f GB/s; %.2f GFLOPS (%f sec) \n", numBytes/gpu_time, numFlops/gpu_time, gpu_time);
    printf("cuBLAS:  %.2f GB/s; %.2f GFLOPS (%f sec) \n", numBytes/cublas_time, numFlops/cublas_time, cublas_time);
    if (checkRMSE)
        printf("CPU:\t%.2f GB/s; %.2f GFLOPS (%f sec) \n", numBytes/cpu_time, numFlops/cpu_time, cpu_time);

    cublasDestroy_v2(handle);
    CUDA_CHECK(cudaFreeHost(x_h));
    CUDA_CHECK(cudaFreeHost(y_h));

    CUDA_CHECK(cudaFree(tmp_d));
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
}


template <typename T>
void host_saxpy(const size_t n, const T a, const T* x_h, T* y_h) {
    for (size_t i = 0; i < n; i++) {
        y_h[i] += a*x_h[i];
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
    }

    free(v_h);
    return diff / n;

}
