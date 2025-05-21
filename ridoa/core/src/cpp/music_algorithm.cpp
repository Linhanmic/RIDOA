// src/cpp/music_algorithm.cpp - 修改以支持HIP
#include "music_algorithm.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <thread>

// 在这里声明CUDA和HIP函数的接口
#ifdef USE_CUDA
extern std::tuple<std::vector<double>, std::vector<double>>
computeMusicSpectrumCUDA(const ComplexMatrix &noiseSubspaceProduct,
                         double startAngle, double endAngle, double stepAngle);
#endif

#ifdef USE_HIP
extern std::tuple<std::vector<double>, std::vector<double>>
computeMusicSpectrumHIP(const ComplexMatrix &noiseSubspaceProduct,
                        double startAngle, double endAngle, double stepAngle);
#endif

MusicAlgorithm::MusicAlgorithm()
{
}

std::tuple<std::vector<double>, std::vector<double>> MusicAlgorithm::computeSpectrum(
    const ComplexMatrix &noiseSubspaceProduct,
    double startAngle,
    double endAngle,
    double stepAngle) const
{
    // 根据配置决定使用CPU、CUDA还是HIP
    if (config_.useGPU)
    {
#ifdef USE_HIP
        try
        {
            return computeSpectrumHIP(noiseSubspaceProduct, startAngle, endAngle, stepAngle);
        }
        catch (const std::exception &e)
        {
            std::cerr << "HIP加速失败，切换到CPU: " << e.what() << std::endl;
            return computeSpectrumCPU(noiseSubspaceProduct, startAngle, endAngle, stepAngle);
        }
#elif defined(USE_CUDA)
        try
        {
            return computeSpectrumCUDA(noiseSubspaceProduct, startAngle, endAngle, stepAngle);
        }
        catch (const std::exception &e)
        {
            std::cerr << "CUDA加速失败，切换到CPU: " << e.what() << std::endl;
            return computeSpectrumCPU(noiseSubspaceProduct, startAngle, endAngle, stepAngle);
        }
#else
        std::cerr << "未启用GPU加速，使用CPU计算" << std::endl;
#endif
    }

    return computeSpectrumCPU(noiseSubspaceProduct, startAngle, endAngle, stepAngle);
}

std::tuple<std::vector<double>, std::vector<double>> MusicAlgorithm::computeSpectrumCPU(
    const ComplexMatrix &noiseSubspaceProduct,
    double startAngle,
    double endAngle,
    double stepAngle) const
{
    // CPU实现代码，保持不变
    int numAngles = static_cast<int>((endAngle - startAngle) / stepAngle) + 1;
    std::vector<double> spectrum(numAngles);
    std::vector<double> angles(numAngles);

    // 计算导向矢量和谱值
    for (int i = 0; i < numAngles; ++i)
    {
        angles[i] = startAngle + i * stepAngle;

        // 转换为弧度
        double angle = angles[i] * M_PI / 180.0;

        // 如果使用了空间平滑，噪声子空间投影矩阵的大小会小于阵元数目
        int nElements = config_.nElements;
        if (config_.forwardSmoothingSize > 0)
        {
            nElements = config_.forwardSmoothingSize;
        }
        if (noiseSubspaceProduct.rows() != nElements || noiseSubspaceProduct.cols() != nElements)
        {
            throw std::invalid_argument("噪声子空间投影矩阵的大小不匹配");
        }
        // 计算导向矢量
        Eigen::VectorXcd steeringVector(nElements);
        for (int e = 0; e < nElements; ++e)
        {
            // 导向矢量计算，使用半波长为单位的阵元间距
            double phase = M_PI * config_.elementSpacing * e * std::sin(angle);
            steeringVector(e) = std::exp(ComplexType(0, phase));
        }

        // 归一化导向矢量
        steeringVector.normalize();

        // 计算MUSIC谱
        ComplexType numerator = steeringVector.adjoint() * noiseSubspaceProduct * steeringVector;
        double denominator = std::abs(numerator);

        // 避免除以零
        if (denominator < 1e-10)
        {
            spectrum[i] = 1e10; // 设置一个很大的值
        }
        else
        {
            spectrum[i] = 1.0 / denominator;
        }
    }

    // 归一化谱值
    double maxVal = *std::max_element(spectrum.begin(), spectrum.end());
    if (maxVal > 0)
    {
        for (auto &val : spectrum)
        {
            val /= maxVal;
        }
    }

    return {spectrum, angles};
}

#ifdef USE_CUDA
std::tuple<std::vector<double>, std::vector<double>> MusicAlgorithm::computeSpectrumCUDA(
    const ComplexMatrix &noiseSubspaceProduct,
    double startAngle,
    double endAngle,
    double stepAngle) const
{
    // 调用CUDA实现
    return computeMusicSpectrumCUDA(noiseSubspaceProduct, startAngle, endAngle, stepAngle);
}
#else
std::tuple<std::vector<double>, std::vector<double>> MusicAlgorithm::computeSpectrumCUDA(
    const ComplexMatrix &noiseSubspaceProduct,
    double startAngle,
    double endAngle,
    double stepAngle) const
{
    // 如果没有启用CUDA，直接调用CPU实现
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return computeSpectrumCPU(noiseSubspaceProduct, startAngle, endAngle, stepAngle);
}
#endif

#ifdef USE_HIP
std::tuple<std::vector<double>, std::vector<double>> MusicAlgorithm::computeSpectrumHIP(
    const ComplexMatrix &noiseSubspaceProduct,
    double startAngle,
    double endAngle,
    double stepAngle) const
{
    // 调用HIP实现
    return computeMusicSpectrumHIP(noiseSubspaceProduct, startAngle, endAngle, stepAngle);
}
#else
std::tuple<std::vector<double>, std::vector<double>> MusicAlgorithm::computeSpectrumHIP(
    const ComplexMatrix &noiseSubspaceProduct,
    double startAngle,
    double endAngle,
    double stepAngle) const
{
    // 如果没有启用HIP，直接调用CPU实现
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return computeSpectrumCPU(noiseSubspaceProduct, startAngle, endAngle, stepAngle);
}
#endif

std::vector<int> MusicAlgorithm::findPeaks(
    const std::vector<double> &spectrum,
    double threshold) const
{
    // 峰值搜索代码保持不变
    std::vector<int> peaks;
    if (spectrum.empty())
    {
        return peaks;
    }

    double maxVal = *std::max_element(spectrum.begin(), spectrum.end());
    double thresholdValue = threshold * maxVal;

    // 寻找局部最大值
    for (int i = 1; i < spectrum.size() - 1; ++i)
    {
        if (spectrum[i] > thresholdValue &&
            spectrum[i] > spectrum[i - 1] &&
            spectrum[i] > spectrum[i + 1])
        {
            peaks.push_back(i);
        }
    }

    return peaks;
}