// include/hip/hip_helper.hip
#pragma once

#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <memory>
#include <cmath>
#include <stdexcept>

// 包含HIP标准库
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

// HIP错误检查宏
#define CHECK_HIP_ERROR(call)                                      \
    do                                                             \
    {                                                              \
        hipError_t error = call;                                   \
        if (error != hipSuccess)                                   \
        {                                                          \
            std::cerr << "HIP错误在行 " << __LINE__ << ": "        \
                      << hipGetErrorString(error) << std::endl;    \
            throw std::runtime_error("HIP操作失败");                \
        }                                                          \
    } while (0)

// HIP复数类型与std::complex的转换辅助函数
inline hipFloatComplex makeComplexHIP(const std::complex<float>& c) {
    return make_hipFloatComplex(c.real(), c.imag());
}

inline hipDoubleComplex makeComplexHIP(const std::complex<double>& c) {
    return make_hipDoubleComplex(c.real(), c.imag());
}

inline std::complex<float> fromComplexHIP(const hipFloatComplex& c) {
    return std::complex<float>(c.x, c.y);
}

inline std::complex<double> fromComplexHIP(const hipDoubleComplex& c) {
    return std::complex<double>(c.x, c.y);
}

// 判断当前是否有可用的AMD GPU
inline bool hasAMDGPU() {
    int deviceCount = 0;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    return (error == hipSuccess && deviceCount > 0);
}

// 获取并打印设备信息
inline void printDeviceInfo(int deviceId) {
    hipDeviceProp_t deviceProp;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProp, deviceId));
    
    std::cout << "HIP设备信息:" << std::endl;
    std::cout << "  设备名称: " << deviceProp.name << std::endl;
    std::cout << "  计算能力: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "  多处理器数量: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "  总内存: " << (deviceProp.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  最大线程数/块: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  最大共享内存/块: " << (deviceProp.sharedMemPerBlock / 1024) << " KB" << std::endl;
    std::cout << "  时钟速率: " << (deviceProp.clockRate / 1000) << " MHz" << std::endl;
}

// 初始化HIP设备
inline bool initializeHIP(int deviceId = 0) {
    try {
        int deviceCount = 0;
        CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));
        
        if (deviceCount == 0) {
            std::cerr << "没有找到HIP兼容的设备" << std::endl;
            return false;
        }
        
        if (deviceId >= deviceCount) {
            std::cerr << "指定的设备ID超出范围, 切换到设备0" << std::endl;
            deviceId = 0;
        }
        
        CHECK_HIP_ERROR(hipSetDevice(deviceId));
        printDeviceInfo(deviceId);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "HIP初始化失败: " << e.what() << std::endl;
        return false;
    }
}