// src/cpp/parameter_space.cpp - 修改以支持HIP
#include "parameter_space.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// 在这里声明CUDA和HIP函数的接口
#ifdef USE_CUDA
extern Eigen::MatrixXd ParameterSpaceCUDA(
    const std::vector<double> &peakAngles,
    const std::vector<double> &timepoints,
    double elevationStart, double elevationEnd,
    double azimuthStart, double azimuthEnd,
    double precision);
#endif

#ifdef USE_HIP
extern Eigen::MatrixXd ParameterSpaceHIP(
    const std::vector<double> &peakAngles,
    const std::vector<double> &timepoints,
    double elevationStart, double elevationEnd,
    double azimuthStart, double azimuthEnd,
    double precision);
#endif

ParameterSpace::ParameterSpace()
{
}

Eigen::MatrixXd ParameterSpace::projectToParameterSpace(
    const std::vector<double> &peakAngles,
    const std::vector<double> &timepoints,
    double elevationStart,
    double elevationEnd,
    double azimuthStart,
    double azimuthEnd,
    double precision) const
{

    // 根据配置决定使用CPU、CUDA还是HIP
    if (config_.useGPU)
    {
#ifdef USE_HIP
        try
        {
            return projectToParameterSpaceHIP(peakAngles, timepoints,
                                              elevationStart, elevationEnd,
                                              azimuthStart, azimuthEnd,
                                              precision);
        }
        catch (const std::exception &e)
        {
            std::cerr << "HIP加速失败，切换到CPU: " << e.what() << std::endl;
            return projectToParameterSpaceCPU(peakAngles, timepoints,
                                              elevationStart, elevationEnd,
                                              azimuthStart, azimuthEnd,
                                              precision);
        }
#elif defined(USE_CUDA)
        try
        {
            return projectToParameterSpaceCUDA(peakAngles, timepoints,
                                               elevationStart, elevationEnd,
                                               azimuthStart, azimuthEnd,
                                               precision);
        }
        catch (const std::exception &e)
        {
            std::cerr << "CUDA加速失败，切换到CPU: " << e.what() << std::endl;
            return projectToParameterSpaceCPU(peakAngles, timepoints,
                                              elevationStart, elevationEnd,
                                              azimuthStart, azimuthEnd,
                                              precision);
        }
#else
        std::cerr << "未启用GPU加速，使用CPU计算" << std::endl;
#endif
    }

    return projectToParameterSpaceCPU(peakAngles, timepoints,
                                      elevationStart, elevationEnd,
                                      azimuthStart, azimuthEnd,
                                      precision);
}

Eigen::MatrixXd ParameterSpace::projectToParameterSpaceCPU(
    const std::vector<double> &peakAngles,
    const std::vector<double> &timepoints,
    double elevationStart,
    double elevationEnd,
    double azimuthStart,
    double azimuthEnd,
    double precision) const
{

    int numPeaks = peakAngles.size();
    if (numPeaks == 0 || peakAngles.size() != timepoints.size())
    {
        throw std::invalid_argument("峰值角度和时间点数组长度不匹配或为零");
    }

    // 计算累加器尺寸
    int elevationSize = static_cast<int>((elevationEnd - elevationStart) / precision) + 1;
    int azimuthSize = static_cast<int>((azimuthEnd - azimuthStart) / precision) + 1;

    // 初始化累加器
    Eigen::MatrixXd accumulator = Eigen::MatrixXd::Zero(elevationSize, azimuthSize);

    // 对每个峰值角度和时间点进行投影
    for (int p = 0; p < numPeaks; ++p)
    {
        double theta = peakAngles[p] * M_PI / 180.0; // 转为弧度
        double t = timepoints[p];

        // 遍历所有可能的方位角
        for (int azIdx = 0; azIdx < azimuthSize; ++azIdx)
        {
            double azimuth = azimuthStart + azIdx * precision;
            double azimuthRad = azimuth * M_PI / 180.0; // 转为弧度

            // 根据论文公式计算可能的俯仰角
            // sinθ = sinβcos(ωt-α)
            double denominator = std::cos(config_.omega * t - azimuthRad);

            // 避免除以零
            if (std::fabs(denominator) < 1e-10)
                continue;

            double sinBeta = std::sin(theta) / denominator;

            // 检查sinBeta是否在有效范围内
            if (std::fabs(sinBeta) <= 1.0)
            {
                double beta = std::asin(sinBeta) * 180.0 / M_PI; // 转为角度

                // 检查俯仰角是否在有效范围内
                if (beta >= elevationStart && beta <= elevationEnd)
                {
                    int elevIdx = static_cast<int>((beta - elevationStart) / precision);

                    // 安全检查
                    if (elevIdx >= 0 && elevIdx < elevationSize)
                    {
                        accumulator(elevIdx, azIdx) += 1.0;
                    }
                }
            }
        }
    }

    return accumulator;
}

#ifdef USE_CUDA
Eigen::MatrixXd ParameterSpace::projectToParameterSpaceCUDA(
    const std::vector<double> &peakAngles,
    const std::vector<double> &timepoints,
    double elevationStart,
    double elevationEnd,
    double azimuthStart,
    double azimuthEnd,
    double precision) const
{

    // 调用CUDA实现
    return ParameterSpaceCUDA(
        peakAngles, timepoints,
        elevationStart, elevationEnd,
        azimuthStart, azimuthEnd,
        precision);
}
#else
Eigen::MatrixXd ParameterSpace::projectToParameterSpaceCUDA(
    const std::vector<double> &peakAngles,
    const std::vector<double> &timepoints,
    double elevationStart,
    double elevationEnd,
    double azimuthStart,
    double azimuthEnd,
    double precision) const
{

    // 如果没有启用CUDA，直接调用CPU实现
    return projectToParameterSpaceCPU(
        peakAngles, timepoints,
        elevationStart, elevationEnd,
        azimuthStart, azimuthEnd,
        precision);
}
#endif

#ifdef USE_HIP
Eigen::MatrixXd ParameterSpace::projectToParameterSpaceHIP(
    const std::vector<double> &peakAngles,
    const std::vector<double> &timepoints,
    double elevationStart,
    double elevationEnd,
    double azimuthStart,
    double azimuthEnd,
    double precision) const
{

    // 调用HIP实现
    return ParameterSpaceHIP(
        peakAngles, timepoints,
        elevationStart, elevationEnd,
        azimuthStart, azimuthEnd,
        precision);
}
#else
Eigen::MatrixXd ParameterSpace::projectToParameterSpaceHIP(
    const std::vector<double> &peakAngles,
    const std::vector<double> &timepoints,
    double elevationStart,
    double elevationEnd,
    double azimuthStart,
    double azimuthEnd,
    double precision) const
{

    // 如果没有启用HIP，直接调用CPU实现
    return projectToParameterSpaceCPU(
        peakAngles, timepoints,
        elevationStart, elevationEnd,
        azimuthStart, azimuthEnd,
        precision);
}
#endif

std::vector<std::pair<int, int>> ParameterSpace::findPeaks(
    const Eigen::MatrixXd &accumulator,
    double threshold) const
{

    std::vector<std::pair<int, int>> peaks;

    if (accumulator.rows() <= 2 || accumulator.cols() <= 2)
    {
        return peaks; // 矩阵太小，无法找到有效的峰值
    }

    // 计算阈值
    double maxVal = accumulator.maxCoeff();
    double thresholdValue = threshold * maxVal;

    // 寻找局部最大值
    for (int i = 1; i < accumulator.rows() - 1; ++i)
    {
        for (int j = 1; j < accumulator.cols() - 1; ++j)
        {
            if (accumulator(i, j) > thresholdValue)
            {
                // 检查是否是局部最大值
                bool isLocalMax = true;
                for (int di = -1; di <= 1 && isLocalMax; ++di)
                {
                    for (int dj = -1; dj <= 1; ++dj)
                    {
                        if (di == 0 && dj == 0)
                            continue;
                        if (accumulator(i, j) < accumulator(i + di, j + dj))
                        {
                            isLocalMax = false;
                            break;
                        }
                    }
                }

                if (isLocalMax)
                {
                    peaks.emplace_back(i, j);
                }
            }
        }
    }
    // 对于距离小于config_.minDistance的峰值进行合并
    std::vector<std::pair<int, int>> mergedPeaks;
    for (const auto &peak : peaks)
    {
        bool isMerged = false;
        for (auto &mergedPeak : mergedPeaks)
        {
            if (std::abs(peak.first - mergedPeak.first) < config_.minDistance &&
                std::abs(peak.second - mergedPeak.second) < config_.minDistance)
            {
                // 合并峰值
                mergedPeak.first = (mergedPeak.first + peak.first) / 2;
                mergedPeak.second = (mergedPeak.second + peak.second) / 2;
                isMerged = true;
                break;
            }
        }
        if (!isMerged)
        {
            mergedPeaks.push_back(peak);
        }
    }
    peaks = mergedPeaks;

    // 按照峰值大小排序，保留最大的config_.maxSources个峰值
    if (peaks.size() > config_.maxSources)
    {
        std::vector<std::pair<double, std::pair<int, int>>> peaksWithValues;
        for (const auto &peak : peaks)
        {
            peaksWithValues.emplace_back(accumulator(peak.first, peak.second), peak);
        }

        std::sort(peaksWithValues.begin(), peaksWithValues.end(),
                  [](const auto &a, const auto &b)
                  {
                      return a.first > b.first;
                  });

        peaks.clear();
        for (int i = 0; i < std::min(config_.maxSources, static_cast<int>(peaksWithValues.size())); ++i)
        {
            peaks.push_back(peaksWithValues[i].second);
        }
    }

    return peaks;
}