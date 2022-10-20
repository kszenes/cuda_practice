#include <iostream>
#include <cuda_runtime.h>
#include "timer.h"

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        printf("Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(-1); \
    } \
}

#define TILE_DIM 32
// const int BLOCK_ROWS = 32;


double random_number();
template <typename T>
T rmse_copy(const size_t, const T*, const T*);
template <typename T>
T rmse_trans(const size_t, const size_t, const T*, const T*);


template <typename T>
__global__
void copy(const size_t rows, const size_t cols, const T *mat, T *copy) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows) {
        copy[x + y * cols] = mat[x + y * cols]; 
    }
}

template <typename T>
__global__
void copy_1d(const size_t n, const T *in, T *out) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

template <typename T>
__global__
void transpose_naive(const size_t rows, const size_t cols, const T *mat, T *copy) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows) {
        copy[y + x * rows] = mat[x + y * cols]; 
    }
}

template <typename T>
__global__ 
void transpose_shmem(const size_t rows, const size_t cols, const T *mat, T *trans) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ T shmem[TILE_DIM * (TILE_DIM+1)];
    shmem[threadIdx.y + threadIdx.x * (TILE_DIM+1)] = mat[x + y * cols];
    __syncthreads();

    x = threadIdx.x + blockIdx.y * blockDim.y;
    y = threadIdx.y + blockIdx.x * blockDim.x;

    if (x < cols && y < rows) {
        trans[x + y * cols] = shmem[threadIdx.x + threadIdx.y * (TILE_DIM+1)];
        // trans[x + y * cols] = mat[y + x * rows]; 
    }

}
// __global__ void copy_ref(float *odata, const float *idata)
// {
//   int x = blockIdx.x * TILE_DIM + threadIdx.x;
//   int y = blockIdx.y * TILE_DIM + threadIdx.y;
//   int width = gridDim.x * TILE_DIM;

//   for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
//     odata[(y+j)*width + x] = idata[(y+j)*width + x];
// }

int main() {
    const int runs = 1000;

    #define DTYPE float

    DTYPE *mat_h, *mat_d, *trans_d;
    const size_t size = 1 << 13;
    const size_t rows = size;
    const size_t cols = size;
    const size_t N = rows * cols;

    CUDA_CHECK(cudaSetDevice(1));

    CUDA_CHECK(cudaMallocHost(&mat_h, N * sizeof(DTYPE)));
    CUDA_CHECK(cudaMalloc(&mat_d, N * sizeof(DTYPE)));
    CUDA_CHECK(cudaMalloc(&trans_d, N * sizeof(DTYPE)));

    const double transferred_bytes = 2 * N * sizeof(DTYPE) * 1e-9;

    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            mat_h[row + rows * col] = random_number();
        }
    }

    CUDA_CHECK(cudaMemcpy(mat_d, mat_h, N * sizeof(DTYPE), cudaMemcpyHostToDevice));

    // dim3 blockSize(TILE_DIM, TILE_DIM);
    // const int bx = (rows + blockSize.x - 1) / blockSize.x;
    // const int by = (cols + blockSize.y - 1) / blockSize.y;
    // dim3 gridSize(bx, by);

    dim3 dimGrid(cols/TILE_DIM, rows/TILE_DIM);
    dim3 dimBlock(TILE_DIM, TILE_DIM);

    GPUTimer timer;
    copy<<<dimGrid, dimBlock>>>(rows, cols, mat_d, trans_d);
    timer.start();
    for (int i = 0; i < runs; i++) {
        copy<<<dimGrid, dimBlock>>>(rows, cols, mat_d, trans_d);
    }
    auto time = timer.seconds() / runs;
    const auto rmseCopy = rmse_copy(N, mat_h, trans_d);
    printf("Copy:  rmse: %f; %f GB/s; (%f sec)\n", rmseCopy, transferred_bytes / time, time);

    int copyNumThreads = 512;
    int copyNumBlocks = (rows*cols + copyNumThreads - 1) / copyNumThreads;
    copy<<<dimGrid, dimBlock>>>(rows, cols, mat_d, trans_d);
    timer.start();
    for (int i = 0; i < runs; i++) {
        copy_1d<<<copyNumBlocks, copyNumThreads>>>(rows*cols, mat_d, trans_d);
    }
    time = timer.seconds() / runs;
    const auto rmseCopy1d = rmse_copy(N, mat_h, trans_d);
    printf("Copy1d:  rmse: %f; %f GB/s; (%f sec)\n", rmseCopy1d, transferred_bytes / time, time);

    timer.start();
    for (int i = 0; i < runs; i++) {
        CUDA_CHECK(cudaMemcpy(mat_d, trans_d, rows*cols*sizeof(DTYPE),cudaMemcpyDeviceToDevice));
    }
    time = timer.seconds() / runs;
    const auto rmseMemcpy = rmse_copy(N, mat_h, trans_d);
    printf("Memcpy:  rmse: %f; %f GB/s; (%f sec)\n", rmseCopy1d, transferred_bytes / time, time);


    // copy_ref<<<dimGrid, dimBlock>>>(mat_d, trans_d);
    // timer.start();
    // for (int i = 0; i < runs; i++) {
    //     copy_ref<<<dimGrid, dimBlock>>>(mat_d, trans_d);
    // }
    // auto time = timer.seconds() / runs;

    timer.start();
    for (int i = 0; i < runs; i++) {
        transpose_naive<<<dimGrid, dimBlock>>>(rows, cols, mat_d, trans_d);
    }
    time = timer.seconds() / runs;
    const auto rmseNaive = rmse_trans(rows, cols, mat_h, trans_d);
    printf("Naive: rmse: %f; %f GB/s; (%f sec)\n", rmseNaive, transferred_bytes / time, time);

    timer.start();
    for (int i = 0; i < runs; i++) {
        transpose_shmem<<<dimGrid, dimBlock>>>(rows, cols, mat_d, trans_d);
    }
    time = timer.seconds() / runs;
    const DTYPE rmseTrans = rmse_trans(rows, cols, mat_h, trans_d);
    printf("Shmem: rmse: %f; %f GB/s; (%f sec)\n", rmseTrans, transferred_bytes / time, time);

    CUDA_CHECK(cudaFreeHost(mat_h));
    CUDA_CHECK(cudaFree(mat_d));
    CUDA_CHECK(cudaFree(trans_d));
}

double random_number() {
    return  (((double) rand()) / RAND_MAX - 0.5) * 100; 
}

template <typename T>
T rmse_copy(const size_t N, const T *ref, const T *mat_d) {
    T* mat_h;
    CUDA_CHECK(cudaMallocHost(&mat_h, N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(mat_h, mat_d, N * sizeof(T), cudaMemcpyDeviceToHost));

    T diff = 0.0f;
    T rmse = 0.0f;
    for (size_t i = 0; i < N; i++) {
        diff = ref[i] - mat_h[i];
        rmse += diff * diff;
    }

    CUDA_CHECK(cudaFreeHost(mat_h));
    return rmse / N;
}

template <typename T>
T rmse_trans(const size_t m, const size_t n, const T *ref, const T *mat_d) {
    T* mat_h;
    const size_t N = m * n;
    CUDA_CHECK(cudaMallocHost(&mat_h, N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(mat_h, mat_d, N * sizeof(T), cudaMemcpyDeviceToHost));

    T diff = 0.0f;
    T rmse = 0.0f;
    for (size_t row = 0; row < m; row++) {
        for (size_t col = 0; col < n; col++) {
            diff = ref[row + col * m] - mat_h[col + row * n];
            rmse += diff * diff;
        }
    }
    CUDA_CHECK(cudaFreeHost(mat_h));
    return rmse / N;
}
