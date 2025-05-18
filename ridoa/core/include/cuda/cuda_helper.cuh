// 修改 src/include/cuda/cuda_helper.cuh
#pragma once

#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <random>
#include <cuComplex.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <memory>
#include <cmath>
#include <stdexcept>

// CUDA错误检查宏
#define CHECK_CUDA_ERROR(call)                                   \
    do                                                           \
    {                                                            \
        cudaError_t error = call;                                \
        if (error != cudaSuccess)                                \
        {                                                        \
            std::cerr << "CUDA错误在行 " << __LINE__ << ": "     \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)
    