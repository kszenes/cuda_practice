// Computes the operation $A \odot (1 * vL^T) + (vR * 1^T)$ for the GAT model
// where vectors $vL = aL^T * W^T * H^T$ and $vR = H * W * aR$ and
// matrix A is sparse stored and stored in COO format
#include <cassert>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "gat_kernels.h"
#include "host_utils.h"
#include "sort_vector.h"
#include "timer.h"
#include "utils.h"

#define ERR_NE(X, Y)                                                           \
  do {                                                                         \
    if ((X) != (Y)) {                                                          \
      fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)
#define CUDA_CALL(X) ERR_NE((X), cudaSuccess)

int main() {
  const int runs = 1;
#define FLOAT

#if defined(FLOAT)
  printf("Single precision\n");
  typedef float floatType;
#undef TENSOR
#elif defined(TENSOR)
  printf("Tensor float precision\n");
  typedef float floatType;
#undef TENSOR
#elif defined(DOUBLE)
  printf("Double precision\n");
  typedef double floatType;
#undef DOUBLE
#endif

  CUDA_CHECK(cudaSetDevice(1));
  const bool checkRMSE = true;
  const bool print_debug = false;


  const size_t rows = 100000;    // rows = Hrows
  const size_t cols = 128;       // cols = Wcols
  const double sparsity_density = 0.01;
  const size_t nnz = floor(rows * rows * sparsity_density);

  std::cout << "Rows: " << rows << '\n';
  std::cout << "Cols: " << cols << '\n';
  std::cout << "nnz: " << nnz << '\n';
  // S_{ij} = vL{i} + vR{j}
  const double numFlops = nnz * 1e-9;
  // size(A) + vL + vR = 3 * nnz + 2 * rows
  const double numBytes =
      (3 * nnz * sizeof(floatType) + 2 * rows * sizeof(size_t)) * 1e-9;
  printf("Memory usage =         %f GB\n", numBytes);
  printf("Arithmetic Intensity:  %f FLOPs/Bytes\n", numFlops / numBytes);

  std::vector<size_t> A_rows_coo_h(nnz, 0);
  std::vector<size_t> A_cols_h(nnz, 0);

  for (size_t iter_idx = 0; iter_idx < nnz; iter_idx++) {
    A_rows_coo_h[iter_idx] = rand() % rows;
    A_cols_h[iter_idx] = rand() % rows;
  }
  sort_vectors_by_row(A_rows_coo_h, A_cols_h);

  size_t *A_rows_coo_d;
  size_t *A_cols_d;

  CUDA_CHECK(cudaMalloc(&A_rows_coo_d, nnz * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&A_cols_d, nnz * sizeof(size_t)));

  CUDA_CHECK(cudaMemcpy(A_rows_coo_d, A_rows_coo_h.data(), nnz * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(A_cols_d, A_cols_h.data(), nnz * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  GPUTimer timer;

  floatType  *vecL_h, *vecR_h, *res_h;
  CUDA_CHECK(cudaMallocHost(&vecL_h, rows * sizeof(floatType)));
  CUDA_CHECK(cudaMallocHost(&vecR_h, rows * sizeof(floatType)));
  CUDA_CHECK(cudaMallocHost(&res_h, nnz * sizeof(floatType)));

  floatType *vecL_d, *vecR_d, *res_d;
  CUDA_CHECK(cudaMalloc(&vecL_d, rows * sizeof(floatType)));
  CUDA_CHECK(cudaMalloc(&vecR_d, rows * sizeof(floatType)));
  CUDA_CHECK(cudaMalloc(&res_d, nnz * sizeof(floatType)));

  for (size_t i = 0; i < rows; i++) {
    vecL_h[i] = (((floatType)rand()) / RAND_MAX - 0.5) * 100;
    if (print_debug) {
      std::cout << vecL_h[i] << ' ';
    }
  }
  std::cout << "\n";
  for (size_t i = 0; i < rows; i++) {
    vecR_h[i] = (((floatType)rand()) / RAND_MAX - 0.5) * 100;
    if (print_debug) {
      std::cout << vecR_h[i] << ' ';
    }
  }
  std::cout << "\n";
  memset(res_h, 0, nnz * sizeof(floatType));
  if (print_debug)
    std::cout << "Vector init complete\n";

  CUDA_CHECK(cudaMemcpy(vecL_d, vecL_h, rows * sizeof(floatType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(vecR_d, vecR_h, rows * sizeof(floatType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(res_d, res_h, nnz * sizeof(floatType),
                        cudaMemcpyHostToDevice));

  double cpu_time = 0.0;
  if (checkRMSE) {
    for (int iter = 0; iter < runs; iter++) {
      host_computeS(A_rows_coo_h.data(), A_cols_h.data(), vecL_h, vecR_h, nnz,
                    res_h);
      timer.start();
      host_computeS(A_rows_coo_h.data(), A_cols_h.data(), vecL_h, vecR_h, nnz,
                    res_h);
      cpu_time += timer.seconds() / runs;
    }
    if (print_debug) {
      std::cout << "Host: ";
      for (size_t iter = 0; iter < nnz; iter++) {
        std::cout << res_h[iter] << ' ';
      }
      std::cout << '\n';
    }
  }
  // === Cublas ===
  double coo_time = 0.0;
  for (int iter = 0; iter < runs; iter++) {
    computeS(A_rows_coo_d, A_cols_d, vecL_d, vecR_d, nnz, res_d);
    timer.start();
    computeS(A_rows_coo_d, A_cols_d, vecL_d, vecR_d, nnz, res_d);
    coo_time += timer.seconds() / runs;
  }

  if (print_debug) {
    std::cout << "COO: ";
    printVector(nnz, res_d);
  }

  if (checkRMSE) {
    auto myRMSE = rmse(nnz, res_h, res_d);
    printf("RMSE: %f\n", myRMSE);
  }

  printf("COO:     %.4f GB/s;\t%.4f GFLOPS (%f sec) \n", numBytes / coo_time,
         numFlops / coo_time, coo_time);
  if (checkRMSE)
    printf("CPU:     %.4f GB/s;\t%.4f GFLOPS (%f sec) \n", numBytes / cpu_time,
           numFlops / cpu_time, cpu_time);
          
  CUDA_CHECK(cudaFreeHost(res_h));
  CUDA_CHECK(cudaFreeHost(vecL_h));
  CUDA_CHECK(cudaFreeHost(vecR_h));

  CUDA_CHECK(cudaFree(res_d));
  CUDA_CHECK(cudaFree(vecL_d));
  CUDA_CHECK(cudaFree(vecR_d));
  CUDA_CHECK(cudaFree(A_rows_coo_d));
  CUDA_CHECK(cudaFree(A_cols_d));
}
