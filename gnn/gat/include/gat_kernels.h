#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "utils.h"

template <typename T>
__global__ void
computeS_kernel(const size_t *__restrict__ A_rows,
                const size_t *__restrict__ A_cols, const T *__restrict__ vecL,
                const T *__restrict__ vecR, const size_t nnz, T *res) {
  int element = threadIdx.x + blockIdx.x * blockDim.x;
  int row = 0;
  int col = 0;

  for (; element < nnz; element += blockDim.x * gridDim.x) {
    row = A_rows[element];
    col = A_cols[element];
    res[element] = vecL[row] + vecR[col];
  }
}

template <typename T>
void computeS(const size_t *__restrict__ A_rows,
              const size_t *__restrict__ A_cols, const T *__restrict__ vecL,
              const T *__restrict__ vecR, const size_t nnz,
              T *__restrict__ res) {
  const int numThreads = 1 << 9;
  std::cout << "numThreads: " << numThreads << '\n';
  const int numBlocks = (nnz + numThreads - 1) / numThreads;
  computeS_kernel<<<numBlocks, numThreads>>>(A_rows, A_cols, vecL, vecR, nnz,
                                             res);
}