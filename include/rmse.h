#pragma once

#include <cuda_runtime.h>

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); return err; } \
};


template <typename T>
T rmse(const int n, const T* dVec1, const T* dVec2)
{
  T* hVec1;
  T* hVec2;
  HANDLE_CUDA_ERROR(cudaMallocHost(&hVec1, n*sizeof(T)));
  HANDLE_CUDA_ERROR(cudaMallocHost(&hVec2, n*sizeof(T)));
  
  HANDLE_CUDA_ERROR(cudaMemcpy(hVec1, dVec1, n*sizeof(T), cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(hVec2, dVec2, n*sizeof(T), cudaMemcpyDeviceToHost));

  T rmse = 0.;
  for (int i(0); i < n; ++i)
  {
    T diff = hVec1[i] - hVec2[i];
    rmse += (diff*diff);
  }

  HANDLE_CUDA_ERROR(cudaFreeHost(hVec1));
  HANDLE_CUDA_ERROR(cudaFreeHost(hVec2));

  return std::sqrt(rmse/n);
}
