#include <benchmark/benchmark.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <timer.h>

template <typename T>
__global__
void device_io(const size_t n, const T* x, T* y) {
    // const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // if (idx < n) {
    //     y[idx] = x[idx];
    // }
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < n;
         idx += blockDim.x * gridDim.x)
    {
        y[idx] = x[idx];
    }
}

static void copy_benchmark(benchmark::State& state) {

    const size_t N = 1 << 29;
    // state.counters["Bandwidth"] = Counter(2 * N * sizeof(double), benchmark::Counter::kIsRate);
    double giga_byte = 2 * N * sizeof(double);
    state.counters["Memory Transfer"] = giga_byte;
    state.counters["FooRate"] = benchmark::Counter(giga_byte, benchmark::Counter::kIsRate);

    double* v_h;
    double* v1_d;
    double* v2_d;

    GPUTimer timer;

    cudaMallocHost(&v_h, N * sizeof(double));
    std::generate(v_h, v_h + N, []{ return rand() % 100; });

    cudaMalloc(&v1_d, N * sizeof(double));
    cudaMalloc(&v2_d, N * sizeof(double));
    cudaMemcpy(v1_d, v_h, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(v2_d, 0, N * sizeof(double));

    const int numThreads = 256;
    const int numBlocks = (N + numThreads - 1) / numThreads;
    
    // Perform setup here
    for (auto _ : state) {
        // This code gets timed
        // benchmark::DoNotOptimize(
        //     cudaMemcpy(v2_d, v1_d, N * sizeof(double), cudaMemcpyDeviceToDevice));
        // timer.start();
        // benchmark::DoNotOptimize(
        //     cudaMemcpy(v2_d, v1_d, N * sizeof(double), cudaMemcpyDeviceToDevice));
        // state.SetIterationTime(timer.seconds());
        device_io<<<numBlocks, numThreads>>>(N, v1_d, v2_d);
        timer.start();
        device_io<<<numBlocks, numThreads>>>(N, v1_d, v2_d);
        state.SetIterationTime(timer.seconds());
    }
    cudaFree(v1_d);
    cudaFree(v2_d);
    cudaFreeHost(v_h);
}

BENCHMARK(copy_benchmark)->UseManualTime();

BENCHMARK_MAIN();