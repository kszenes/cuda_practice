#pragma once

#include <cuda_runtime.h>

struct GPUTimer
{
    GPUTimer() 
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~GPUTimer() 
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() 
    {
        cudaEventRecord(start_, 0);
    }

    float seconds() 
    {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time * 1e-3;
    }
    private:
    cudaEvent_t start_, stop_;
};
