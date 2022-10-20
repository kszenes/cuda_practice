#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        printf("Error %s:%d  %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(-1); \
    } \
}

template <typename T>
__global__ void naiveTranspose(const size_t numRows, const size_t numCols, const T* inMat, T* outMat) {
    const int threadRow = threadIdx.x + blockIdx.x * blockDim.x;
    const int threadCol = threadIdx.y + blockIdx.y * blockDim.y;

    if (threadRow < numRows && threadCol < numCols) {
        outMat[threadRow + threadCol * numCols] = inMat[threadCol + threadRow * numCols];
    }
}

#define TILE_DIM 32

template <typename T>
__global__ void shmemTranspose(const size_t numRows, const size_t numCols, const T* inMat, T* outMat) {
    int threadRow = threadIdx.x + blockIdx.x * blockDim.x;
    int threadCol = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ T shmemMat[TILE_DIM * (TILE_DIM + 1)];

    if (threadRow < numRows && threadCol < numCols) {
        shmemMat[threadIdx.y + threadIdx.x * (TILE_DIM + 1)] = inMat[threadRow + threadCol * numRows];
        __syncthreads();

        threadRow = threadIdx.x + blockIdx.y * blockDim.y;
        threadCol = threadIdx.y + blockIdx.x * blockDim.x;

        outMat[threadRow + threadCol * numRows] = shmemMat[threadIdx.x + threadIdx.y * ( TILE_DIM + 1)];

    }
}

template <typename T>
T rmse(const size_t numRows, const size_t numCols, const T* matRef, const T* mat_d) {
    const size_t n = numRows * numCols;
    T* mat_h;
    CUDA_CHECK(cudaMallocHost(&mat_h, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(mat_h, mat_d, n * sizeof(T), cudaMemcpyDeviceToHost));
    T total = (T) 0.0;
    T diff = (T) 0.0;

    for (size_t row = 0; row < numRows; row++) {
        for (size_t col = 0; col < numCols; col++) {
            diff = mat_h[col + row * numCols] - matRef[row + col * numRows];
            total += diff * diff;
        }
    }
    CUDA_CHECK(cudaFreeHost(mat_h));
    return std::sqrt(total) / n;
}

int main() {
    const size_t size = 1 << 8;
    const size_t numRows = size;
    const size_t numCols = size;
    const size_t n = numRows * numCols;

    double *mat_h;
    CUDA_CHECK(cudaMallocHost(&mat_h, n * sizeof(double)));

    for (size_t i = 0; i < n; i++) {
        mat_h[i] = ((((double) rand()) / RAND_MAX) - 0.5) * 100;
    }

    double *mat_d, *transpose_d;
    CUDA_CHECK(cudaMalloc(&mat_d, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&transpose_d, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(mat_d, mat_h, n * sizeof(double), cudaMemcpyHostToDevice));


    const int numThreads = 32;
    dim3 dimBlock(numThreads, numThreads);
    dim3 dimGrid((numRows + numThreads - 1) / numThreads, (numCols + numThreads - 1) / numThreads);

    shmemTranspose<<<dimGrid, dimBlock>>>(numRows, numCols, mat_d, transpose_d);
    double myRMSE = rmse(numRows, numCols, mat_h, transpose_d);
    printf("RMSE: %f\n", myRMSE);

    CUDA_CHECK(cudaFree(transpose_d));
    CUDA_CHECK(cudaFree(mat_d));
    CUDA_CHECK(cudaFreeHost(mat_h));
}