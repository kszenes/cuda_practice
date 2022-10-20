#pragma once

#define FULL_MASK 0xffffffff
#define MAX_GRID_Y 65535

template <typename T> __device__ T sumWarp(T a) {
  a += __shfl_xor_sync(FULL_MASK, a, 16);
  a += __shfl_xor_sync(FULL_MASK, a, 8);
  a += __shfl_xor_sync(FULL_MASK, a, 4);
  a += __shfl_xor_sync(FULL_MASK, a, 2);
  a += __shfl_xor_sync(FULL_MASK, a, 1);
  return a;
}
template <typename T> __device__ double dotBlock(T a, T b) {
  int idx = threadIdx.x;
  __shared__ T warp_sums[32];

  T warp_sum = sumWarp(a * b);
  if (idx < 32) {
    warp_sums[idx] = 0;
  }
  __syncthreads();

  if ((idx & 31) == 0) {
    warp_sums[idx >> 5] = warp_sum;
  }
  __syncthreads();

  if (idx < 32) {
    a = sumWarp(warp_sums[idx]);
  }
  return a;
}
template <typename T>
__global__ void
dotKernel2d_coo(const size_t *__restrict__ A_rows_coo_d,
                const size_t *__restrict__ A_cols_d, const size_t nnz,
                const size_t cols, const T *__restrict__ H_d,
                const T *__restrict__ HT_d, T *__restrict__ tmp_d) {
  // assumes that column fits into blockDim.x
  const int col = threadIdx.x;
  const int mat_element = threadIdx.y + blockIdx.y * blockDim.y;
  // printf("tix: %d; tiy %d\n", blockDim.x, blockDim.y);

  int idx;
  for (idx = mat_element; idx < nnz; idx += blockDim.y * gridDim.y) {
    T a = H_d[col + A_rows_coo_d[idx] * cols];
    T b = HT_d[col + A_cols_d[idx] * cols];
    __syncthreads();
    a = dotBlock(a, b);
    if (threadIdx.x == 0) {
      *(tmp_d + idx) = a;
    }
  }

  // remainder
  if (col < cols && idx < nnz) {
    T a = H_d[col + A_rows_coo_d[idx] * cols];
    T b = HT_d[col + A_cols_d[idx] * cols];
    __syncthreads();
    a = dotBlock(a, b);
    if (threadIdx.x == 0) {
      *(tmp_d + idx) = a;
    }
  }
}

template <typename T>
void vanilla_attention_coo(const size_t *__restrict__ A_rows_coo_d,
                      const size_t *__restrict__ A_cols_d, const size_t nnz,
                      const size_t cols, const T *__restrict__ H_d,
                      const T *__restrict__ HT_d, T *__restrict__ res_d) {
  const unsigned int numThreads = cols;
  const unsigned int numBlocks_y = (unsigned int)min(nnz, (size_t)MAX_GRID_Y);
  dotKernel2d_coo<<<{1, numBlocks_y}, {numThreads, 1}>>>(
      A_rows_coo_d, A_cols_d, nnz, cols, H_d, HT_d, res_d);
}