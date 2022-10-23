#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <utils.h>

template <typename T>
void host_computeS(const size_t *A_rows, const size_t *A_cols, const T *vecL,
                   const T *vecR, const size_t nnz, T *res) {

  for (size_t element = 0; element < nnz; element++) {
    size_t row = A_rows[element];
    size_t col = A_cols[element];

    res[element] = vecL[row] + vecR[col];
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

template <typename T> void printVector(const size_t n, const T *vec_d) {
  T *vec_h;
  CUDA_CHECK(cudaMallocHost(&vec_h, n * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(vec_h, vec_d, n * sizeof(T), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; i++) {
    std::cout << vec_h[i] << ' ';
  }
  std::cout << '\n';
  CUDA_CHECK(cudaFreeHost(vec_h));
}
