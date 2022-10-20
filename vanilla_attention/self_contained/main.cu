// Computes the L2-norm (dot product) between all rows of a matrix
// Matrix is stored in row-major format!
#include <cassert>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "host_utils.h"
#include "timer.h"
#include "utils.h"
#include "vanilla_attention_kernels.h"

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

  const size_t rows = 10000;
  const size_t cols = 1024; // assumes that cols < 1024 !
  assert((cols <= 1024) &&
         "Cols must be smaller than 1024 to fit in thread block");
  const double sparsity_density = 0.01;
  const size_t nnz = floor(rows * rows * sparsity_density);
  // size_t nnz = rows * rows;
  std::cout << "Rows: " << rows;
  std::cout << "; Cols: " << cols << '\n';
  std::cout << "nnz: " << nnz << '\n';
  const double numFlops = 2 * nnz * cols * 1e-9;
  const double numBytes = (2 * nnz * cols + 3 * nnz) * sizeof(floatType) * 1e-9;
  printf("Memory usage =         %f GB\n", numBytes);
  printf("Arithmetic Intensity:  %f FLOPs/Bytes\n", numFlops / numBytes);

  std::vector<size_t> A_rows_coo_h(nnz, 0);
  std::vector<size_t> A_cols_h(nnz, 0);

  for (size_t iter_idx = 0; iter_idx < nnz; iter_idx++) {
    A_rows_coo_h[iter_idx] = rand() % rows;
    A_cols_h[iter_idx] = rand() % rows;
  }

  size_t *A_rows_coo_d;
  size_t *A_cols_d;

  CUDA_CHECK(cudaMalloc(&A_rows_coo_d, nnz * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&A_cols_d, nnz * sizeof(size_t)));

  CUDA_CHECK(cudaMemcpy(A_rows_coo_d, A_rows_coo_h.data(), nnz * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(A_cols_d, A_cols_h.data(), nnz * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  GPUTimer timer;

  floatType *H_h, *HT_h, *res_h;
  CUDA_CHECK(cudaMallocHost(&H_h, rows * cols * sizeof(floatType)));
  CUDA_CHECK(cudaMallocHost(&HT_h, rows * cols * sizeof(floatType)));
  CUDA_CHECK(cudaMallocHost(&res_h, nnz * sizeof(floatType)));

  floatType *H_d, *HT_d, *res_d;
  CUDA_CHECK(cudaMalloc(&H_d, rows * cols * sizeof(floatType)));
  CUDA_CHECK(cudaMalloc(&HT_d, rows * cols * sizeof(floatType)));
  CUDA_CHECK(cudaMalloc(&res_d, nnz * sizeof(floatType)));

  for (size_t i = 0; i < rows * cols; i++) {
    H_h[i] = (((floatType)rand()) / RAND_MAX - 0.5) * 100;
    HT_h[i] = H_h[i];
    if (print_debug) {
      if (i % cols == 0) {
        std::cout << "\n";
      }
      std::cout << H_h[i] << ' ';
    }
  }
  std::cout << "\n";
  memset(res_h, 0, nnz * sizeof(floatType));
  if (print_debug)
    std::cout << "Vector init complete\n";

  CUDA_CHECK(cudaMemcpy(H_d, H_h, rows * cols * sizeof(floatType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(HT_d, HT_h, rows * cols * sizeof(floatType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(res_d, res_h, nnz * sizeof(floatType),
                        cudaMemcpyHostToDevice));

  double cpu_time = 0.0;
  if (checkRMSE) {
    for (int iter = 0; iter < runs; iter++) {
      timer.start();
      host_batched_dot(A_rows_coo_h.data(), A_cols_h.data(), nnz, cols, H_h,
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

  CUDA_CHECK(cudaMemset(res_d, 0, nnz * sizeof(floatType)));

  // Parallelizes numBlocks_y over number of nnz up to max limit
  unsigned int numBlocks_y = (unsigned int)min(nnz, (size_t)MAX_GRID_Y);
  std::cout << "NumBlocks_y: " << numBlocks_y << '\n';

  double coo_time = 0.0;
  for (int iter = 0; iter < runs; iter++) {
    vanilla_attention_coo(A_rows_coo_d, A_cols_d, nnz, cols, H_d, HT_d, res_d);
    timer.start();
    vanilla_attention_coo(A_rows_coo_d, A_cols_d, nnz, cols, H_d, HT_d, res_d);
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
  CUDA_CHECK(cudaFreeHost(H_h));
  CUDA_CHECK(cudaFreeHost(HT_h));

  CUDA_CHECK(cudaFree(res_d));
  CUDA_CHECK(cudaFree(H_d));
  CUDA_CHECK(cudaFree(HT_d));
  CUDA_CHECK(cudaFree(A_rows_coo_d));
  CUDA_CHECK(cudaFree(A_cols_d));
}
