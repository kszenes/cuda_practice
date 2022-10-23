// Computes the L2-norm (dot product) between all rows of a matrix
// Matrix is stored in row-major format!
#include <cassert>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

// #include "cuda_kernels.h"
#include "host_utils.h"
#include "sort_vector.h"
#include "timer.h"
#include "utils.h"
#include "vanilla_attention_kernels.h"

#define FULL_MASK 0xffffffff
#define MAX_GRID_Y 65535

#define ERR_NE(X, Y)                                                           \
  do {                                                                         \
    if ((X) != (Y)) {                                                          \
      fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)
#define CUDA_CALL(X) ERR_NE((X), cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X), CUSPARSE_STATUS_SUCCESS)

int main() {
  const int runs = 1;
#define FLOAT

#if defined(FLOAT)
  printf("Single precision\n");
  typedef float floatType;
// cudaDataType_t cublasType = CUDA_R_32F;
// cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_32F;
#undef TENSOR
#elif defined(TENSOR)
  printf("Tensor float precision\n");
  typedef float floatType;
// cudaDataType_t cublasType = CUDA_R_32F;
// cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_32F_FAST_TF32;
#undef TENSOR
#elif defined(DOUBLE)
  printf("Double precision\n");
  typedef double floatType;
// cudaDataType_t cublasType = CUDA_R_64F;
// cublasComputeType_t cublasComputeType = CUBLAS_COMPUTE_64F;
#undef DOUBLE
#endif

  CUDA_CHECK(cudaSetDevice(1));
  const bool checkRMSE = true;
  const bool print_debug = false;

  const size_t rows = 10000;
  const size_t cols = 1024; // assumes that col < 1024 !
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

  // for (auto& e : A_rows_coo_h) std::cout << e << '\t';
  // std::cout << '\n';
  // for (auto& e : A_cols_h) std::cout << e << '\t' ;
  // std::cout << '\n';
  sort_vectors_by_row(A_rows_coo_h, A_cols_h);
  // for (auto& e : A_rows_coo_h) std::cout << e << '\t';
  // std::cout << '\n';
  // for (auto& e : A_cols_h) std::cout << e << '\t';
  // std::cout << '\n';

  auto A_rows_csr_h = coo_to_crs(A_rows_coo_h, rows);

  // This loop prints duplicates
  // for (size_t row_ptr = 0; row_ptr < A_rows_csr_h.size() - 1; row_ptr++) {
  //   size_t start = A_rows_csr_h[row_ptr];
  //   size_t end = A_rows_csr_h[row_ptr + 1];
  //   size_t prev = RAND_MAX;
  //   for (size_t col_ptr = start; col_ptr < end; col_ptr++) {
  //     if (prev == A_cols_h[col_ptr]) {
  //       std::cout << "ERROR at row << "row_ptr: " << prev << " == " <<
  //       A_cols_h[col_ptr] << '\n';
  //     }
  //     prev = A_cols_h[col_ptr];
  //   }
  // }

  size_t *A_rows_coo_d;
  size_t *A_rows_csr_d;
  size_t *A_cols_d;

  CUDA_CHECK(cudaMalloc(&A_rows_coo_d, nnz * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&A_cols_d, nnz * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&A_rows_csr_d, (rows + 1) * sizeof(size_t)));

  CUDA_CHECK(cudaMemcpy(A_rows_coo_d, A_rows_coo_h.data(), nnz * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(A_cols_d, A_cols_h.data(), nnz * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(A_rows_csr_d, A_rows_csr_h.data(),
                        (rows + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

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
  // memset(H_h, 1, rows / 2 * cols * sizeof(floatType));
  // memset(H_h + rows / 2 * cols, 3, rows / 2 * cols * sizeof(floatType));
  memset(res_h, 0, nnz * sizeof(floatType));
  if (print_debug)
    std::cout << "Vector init complete\n";

  CUDA_CHECK(cudaMemcpy(H_d, H_h, rows * cols * sizeof(floatType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(HT_d, HT_h, rows * cols * sizeof(floatType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(res_d, res_h, nnz * sizeof(floatType),
                        cudaMemcpyHostToDevice));

  const int numThreads = cols;

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
    dotKernel2d_coo<<<{1, numBlocks_y}, {numThreads, 1}>>>(
        A_rows_coo_d, A_cols_d, nnz, cols, H_d, HT_d, res_d);
    timer.start();
    // dotKernel2d_coo<<<{1, numBlocks_y}, {numThreads, 1}>>>(
    //     A_rows_coo_d, A_cols_d, nnz, cols, H_d, HT_d, res_d);
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

  CUDA_CHECK(cudaMemset(res_d, 0, nnz * sizeof(floatType)));

  // Parallelizes numBlocks_y over the number of rows
  numBlocks_y = (unsigned int)min(rows, (size_t)MAX_GRID_Y);
  std::cout << "NumBlocks_y: " << numBlocks_y << '\n';
  assert(numBlocks_y <= MAX_GRID_Y &&
         "Matrix has too many rows for this implementations");
  // Parallelzes numBlocks_z over the number off nzz in cols
  double csr_time = 0.0;
  // for (int iter = 0; iter < runs; iter++) {
  //   dotKernel2d_csr<<<{1, numBlocks_y, numBlocks_y}, {numThreads, 1}>>>(
  //       A_rows_csr_d, A_cols_d, rows, cols, H_d, res_d);
  //   timer.start();
  //   dotKernel2d_csr<<<{1, numBlocks_y, numBlocks_y}, {numThreads, 1}>>>(
  //       A_rows_csr_d, A_cols_d, rows, cols, H_d, res_d);
  //   csr_time += timer.seconds() / runs;
  // }

  if (print_debug) {
    std::cout << "CSR: ";
    printVector(nnz, res_d);
  }

  if (checkRMSE) {
    auto myRMSE = rmse(nnz, res_h, res_d);
    printf("RMSE: %f\n", myRMSE);
  }

  cusparseHandle_t cusparse_handle;
  CUSPARSE_CALL(cusparseCreate(&cusparse_handle));
  // cusparseSpMatDescr_t matA;
  // cusparseDnMatDescr_t matB, matC;
  // void *dBuffer = NULL;
  // size_t bufferSize = 0;

  // CUSPARSE_CALL(cusparseCreateCoo(&matA, rows, rows, nnz, A_rows_coo_d,
  //                                 A_cols_d, res_d, CUSPARSE_INDEX_32I,
  //                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

  // double alpha = 1.0;
  // double beta = 0.0;

  printf("COO:     %.4f GB/s;\t%.4f GFLOPS (%f sec) \n", numBytes / coo_time,
         numFlops / coo_time, coo_time);
  printf("CSR:     %.4f GB/s;\t%.4f GFLOPS (%f sec) \n", numBytes / csr_time,
         numFlops / csr_time, csr_time);
  // printf("cusparse:  %.2f GB/s;  %.2f GFLOPS (%f sec) \n", numBytes /
  // cublas_time,
  //        numFlops / cublas_time, cublas_time);
  if (checkRMSE)
    printf("CPU:     %.4f GB/s;\t%.4f GFLOPS (%f sec) \n", numBytes / cpu_time,
           numFlops / cpu_time, cpu_time);

  cusparseDestroy(cusparse_handle);
  CUDA_CHECK(cudaFreeHost(res_h));
  CUDA_CHECK(cudaFreeHost(H_h));
  CUDA_CHECK(cudaFreeHost(HT_h));

  CUDA_CHECK(cudaFree(res_d));
  CUDA_CHECK(cudaFree(H_d));
  CUDA_CHECK(cudaFree(HT_d));
  CUDA_CHECK(cudaFree(A_rows_csr_d));
  CUDA_CHECK(cudaFree(A_rows_coo_d));
  CUDA_CHECK(cudaFree(A_cols_d));
}
