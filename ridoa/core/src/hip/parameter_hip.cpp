// src/hip/parameter_hip.cpp
#include "hip/hip_helper.hip"
#include "hip/parameter_hip.hip"
#include "doa_config.h"

// 论文中的公式：θ(t,k) = arcsin(sinβcos(ωt-α) ± 2k/m)
// 这里 m = 阵元间距/(λ/2)，也就是以半波长为单位的阵元间距
// HIP核函数：执行参数空间投影
__global__ void parameterSpaceProjectionKernelHIP(
    const double *peakAngles,
    const double *timepoints,
    int numPeaks,
    double elevationStart,
    double elevationEnd,
    double azimuthStart,
    double azimuthEnd,
    double precision,
    double omega,
    double elementSpacing, // 以半波长为单位
    double *accumulator,
    int elevationSize,
    int azimuthSize)
{
    int peakIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (peakIdx >= numPeaks)
        return;

    double theta = peakAngles[peakIdx] * M_PI / 180.0; // 转为弧度
    double t = timepoints[peakIdx];

    // 遍历所有可能的方位角
    for (int azIdx = 0; azIdx < azimuthSize; ++azIdx)
    {
        double azimuth = azimuthStart + azIdx * precision;
        double azimuthRad = azimuth * M_PI / 180.0; // 转为弧度

        // 根据论文公式计算可能的俯仰角
        // sinθ = sinβcos(ωt-α)
        double denominator = cos(omega * t - azimuthRad);

        // 避免除以零
        if (fabs(denominator) < 1e-10)
            continue;

        double sinBeta = sin(theta) / denominator;

        // 检查sinBeta是否在有效范围内
        if (fabs(sinBeta) <= 1.0)
        {
            double beta = asin(sinBeta) * 180.0 / M_PI; // 转为角度

            // 检查俯仰角是否在有效范围内
            if (beta >= elevationStart && beta <= elevationEnd)
            {
                int elevIdx = static_cast<int>((beta - elevationStart) / precision);

                // 安全检查
                if (elevIdx >= 0 && elevIdx < elevationSize)
                {
                    // 使用原子操作更新累加器
                    atomicAdd(&accumulator[elevIdx * azimuthSize + azIdx], 1.0);
                }
            }
        }
    }
}

// 执行参数空间投影和累加
RealMatrix ParameterSpaceHIP(
    const std::vector<double> &peakAngles,
    const std::vector<double> &timepoints,
    double elevationStart,
    double elevationEnd,
    double azimuthStart,
    double azimuthEnd,
    double precision)
{
    // 检查是否有可用的HIP设备
    if (!initializeHIP(DOAConfig::getInstance().gpuDeviceId))
    {
        throw std::runtime_error("无法初始化HIP设备，无法执行GPU加速");
    }

    int numPeaks = peakAngles.size();
    if (numPeaks == 0 || peakAngles.size() != timepoints.size())
    {
        throw std::invalid_argument("峰值角度和时间点数组长度不匹配或为零");
    }

    // 计算累加器尺寸
    int elevationSize = static_cast<int>((elevationEnd - elevationStart) / precision) + 1;
    int azimuthSize = static_cast<int>((azimuthEnd - azimuthStart) / precision) + 1;

    // HIP内存
    double *d_peakAngles = nullptr;
    double *d_timepoints = nullptr;
    double *d_accumulator = nullptr;

    // 初始化累加器
    RealMatrix accumulator = RealMatrix::Zero(elevationSize, azimuthSize);

    try
    {
        // 分配HIP内存
        CHECK_HIP_ERROR(hipMalloc(&d_peakAngles, numPeaks * sizeof(double)));
        CHECK_HIP_ERROR(hipMalloc(&d_timepoints, numPeaks * sizeof(double)));
        CHECK_HIP_ERROR(hipMalloc(&d_accumulator, elevationSize * azimuthSize * sizeof(double)));

        // 复制数据到GPU
        CHECK_HIP_ERROR(hipMemcpy(d_peakAngles, peakAngles.data(),
                                  numPeaks * sizeof(double), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_timepoints, timepoints.data(),
                                  numPeaks * sizeof(double), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemset(d_accumulator, 0,
                                  elevationSize * azimuthSize * sizeof(double)));

        // 执行HIP核函数
        int blockSize = 256;
        int numBlocks = (numPeaks + blockSize - 1) / blockSize;

        hipLaunchKernelGGL(parameterSpaceProjectionKernelHIP,
                           dim3(numBlocks), dim3(blockSize), 0, 0,
                           d_peakAngles, d_timepoints, numPeaks,
                           elevationStart, elevationEnd,
                           azimuthStart, azimuthEnd,
                           precision, DOAConfig::getInstance().omega, DOAConfig::getInstance().elementSpacing,
                           d_accumulator, elevationSize, azimuthSize);

        // 检查HIP核函数执行是否有错误
        CHECK_HIP_ERROR(hipGetLastError());
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // 创建临时数组存储累加器值
        std::vector<double> hostAccumulator(elevationSize * azimuthSize);

        // 复制结果回主机
        CHECK_HIP_ERROR(hipMemcpy(hostAccumulator.data(), d_accumulator,
                                  elevationSize * azimuthSize * sizeof(double), hipMemcpyDeviceToHost));

        // 将一维数组转回Eigen矩阵
        for (int i = 0; i < elevationSize; ++i)
        {
            for (int j = 0; j < azimuthSize; ++j)
            {
                accumulator(i, j) = hostAccumulator[i * azimuthSize + j];
            }
        }
    }
    catch (const std::exception &e)
    {
        // 确保HIP内存被释放
        if (d_peakAngles)
            hipFree(d_peakAngles);
        if (d_timepoints)
            hipFree(d_timepoints);
        if (d_accumulator)
            hipFree(d_accumulator);
        throw; // 重新抛出异常
    }

    // 释放HIP内存
    hipFree(d_peakAngles);
    hipFree(d_timepoints);
    hipFree(d_accumulator);

    return accumulator;
}