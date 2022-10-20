#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <utils.h>

template <typename t>
void host_batched_dot(const size_t *A_rows_h, const size_t *A_cols_h,
                      const size_t nnz, const size_t cols, const t *x_h,
                      t *res_h) {
  for (size_t pair_idx = 0; pair_idx < nnz; pair_idx++) {
    t tmp_res = 0.0;
    for (size_t col = 0; col < cols; col++) {
      tmp_res += x_h[col + A_rows_h[pair_idx] * cols] *
                 x_h[col + A_cols_h[pair_idx] * cols];
    }
    res_h[pair_idx] = tmp_res;
  }
}

template <typename T> T rmse(const size_t n, const T *v_ref, const T *v_d) {
  T *v_h = (T *)malloc(n * sizeof(T));
  CUDA_CHECK(cudaMemcpy(v_h, v_d, n * sizeof(T), cudaMemcpyDeviceToHost));

  T diff = 0.0f;
  for (size_t i = 0; i < n; i++) {
    // printf("Ref: %f; Dev: %f\n", v_ref[i], v_h[i]);
    diff += std::sqrt((v_ref[i] - v_h[i]) * (v_ref[i] - v_h[i]));
  }

  free(v_h);
  return diff / n;
}

template <typename T>
void printVector(const size_t n, const T *vec_d) {
  T *vec_h;
  CUDA_CHECK(cudaMallocHost(&vec_h, n * sizeof(T)));
  CUDA_CHECK(
      cudaMemcpy(vec_h, vec_d, n * sizeof(T), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; i++) {
    std::cout << vec_h[i] << ' ';
  }
  std::cout << '\n';
  CUDA_CHECK(cudaFreeHost(vec_h));
}
