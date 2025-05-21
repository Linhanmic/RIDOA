// include/doa_config.h - 修改以支持HIP
#pragma once

#include <string>
#include <iostream>
#include <cmath>

/**
 * @brief 旋转干涉仪系统配置类
 */
struct DOAConfig
{
    // 单例模式
private:
    DOAConfig() = default;
    DOAConfig(const DOAConfig &) = delete;
    DOAConfig &operator=(const DOAConfig &) = delete;

public:
    int nElements = 8;                 // 阵元数目
    double elementSpacing = 10.0;      // 阵元间距(以半波长为单位)
    double omega = 2 * M_PI;           // 干涉仪旋转角速度(rad/s)
    int samplingRate = int(40e3);      // 采样频率(Hz)
    double carrierFrequency = 3e9;     // 载波频率(Hz)
    int estimateRate = 100;            // 角度估计频率(Hz)
    double thetaPrecision = 0.1;       // 一维角度估计精度(度)
    double precision = 1.0;            // 二维角度估计精度(度)
    double accumulatorThreshold = 0.5; // 累加器阈值
    double spectrumThreshold = 0.5;    // 谱峰阈值
    int fastSnapshots = 200;           // 快拍数量
    int maxSources = 5;                // 最大估计信号源数量
    int forwardSmoothingSize = 0;      // 前向平滑子阵大小，0表示不进行平滑
    bool useGPU = true;                // 是否使用GPU加速
    int gpuDeviceId = 0;               // GPU设备ID
    std::string accelerator = "auto";  // 加速器类型：auto, cuda, hip, cpu
    int minDistance = 5;               // 峰值合并最小距离

    // 获取单例实例
    static DOAConfig &getInstance()
    {
        static DOAConfig instance;
        return instance;
    }

    // 打印配置信息
    void print() const
    {
        std::cout << "DOAConfig: " << std::endl;
        std::cout << "  nElements = " << nElements << std::endl;
        std::cout << "  elementSpacing = " << elementSpacing << std::endl;
        std::cout << "  omega = " << omega << std::endl;
        std::cout << "  samplingRate = " << samplingRate << std::endl;
        std::cout << "  carrierFrequency = " << carrierFrequency << std::endl;
        std::cout << "  estimateRate = " << estimateRate << std::endl;
        std::cout << "  thetaPrecision = " << thetaPrecision << std::endl;
        std::cout << "  precision = " << precision << std::endl;
        std::cout << "  accumulatorThreshold = " << accumulatorThreshold << std::endl;
        std::cout << "  spectrumThreshold = " << spectrumThreshold << std::endl;
        std::cout << "  fastSnapshots = " << fastSnapshots << std::endl;
        std::cout << "  maxSources = " << maxSources << std::endl;
        std::cout << "  forwardSmoothingSize = " << forwardSmoothingSize << std::endl;
        std::cout << "  useGPU = " << (useGPU ? "true" : "false") << std::endl;
        std::cout << "  gpuDeviceId = " << gpuDeviceId << std::endl;
        std::cout << "  accelerator = " << accelerator << std::endl;
    }
};