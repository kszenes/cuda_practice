#pragma once

#define FULL_MASK 0xffffffff

template <typename T>
__device__ T sumWarp(T a) {
  a += __shfl_xor_sync(FULL_MASK, a, 16);
  a += __shfl_xor_sync(FULL_MASK, a, 8);
  a += __shfl_xor_sync(FULL_MASK, a, 4);
  a += __shfl_xor_sync(FULL_MASK, a, 2);
  a += __shfl_xor_sync(FULL_MASK, a, 1);
  return a;
}
template <typename T>
__device__ T dotBlock(T a, T b) {
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
__global__ void dotKernel2d_coo(const size_t *__restrict__ A_rows_coo_d,
                                const size_t *__restrict__ A_cols_d,
                                const size_t nnz, const size_t cols,
                                const T *__restrict__ H_d,
                                const T *__restrict__ HT_d,
                                T *__restrict__ tmp_d) {
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
      // printf("bix: %d; biy: %d; biz: %d: a = %f\n",
      // blockIdx.x, blockIdx.y, blockIdx.z, a);
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
      // printf("bix: %d; biy: %d; biz: %d: a = %f\n",
      // blockIdx.x, blockIdx.y, blockIdx.z, a);
      *(tmp_d + idx) = a;
    }
  }
}

// template <typename T>
// __global__ void dotKernel2d_csr(const size_t *A_rows_csr_d,
//                                 const size_t *A_cols_d, const size_t rows,
//                                 const size_t cols, const T *x_d,
//                                 T *tmp_d) {
//   // assumes that column fits into blockDim.x
//   const int col = threadIdx.x;
//   const int row_ptr = threadIdx.y + blockIdx.y * blockDim.y;
//   const int tiz = threadIdx.z + blockIdx.z * blockDim.z;

//   int curr_row = 0;
//   int next_row = 0;

//   int idx;
//   for (idx = row_ptr; idx < rows; idx += blockDim.y * gridDim.y) {
//     // computes two rows for dot product
//     curr_row = A_rows_csr_d[idx];
//     next_row = A_rows_csr_d[idx + 1];
//     if (tiz < (next_row - curr_row)) {
//       double a = x_d[col + idx * cols];
//       double b = x_d[col + A_cols_d[curr_row + tiz] * cols];
//       __syncthreads();
//       a = dotBlock(a, b);
//       if (threadIdx.x == 0) {
//         // printf("bix: %d; biy: %d; biz: %d: a = %f\n",
//         // blockIdx.x, blockIdx.y, blockIdx.z, a);
//         *(tmp_d + curr_row + tiz) = a;
//       }
//     }
//   }

  // remainder
  // if (col < cols && idx < nnz) {
  //   double a = x_d[col + A_rows_d[idx] * cols];
  //   double b = x_d[col + A_cols_d[idx] * cols];
  //   __syncthreads();
  //   a = dotBlock(a, b);
  //   if (threadIdx.x == 0) {
  //     // printf("bix: %d; biy: %d; biz: %d: a = %f\n",
  //     // blockIdx.x, blockIdx.y, blockIdx.z, a);
  //     *(tmp_d + idx) = a;
  //   }
  // }
}