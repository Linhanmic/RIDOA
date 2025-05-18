// include/parameter_space.h
#pragma once

#include <vector>
#include <Eigen/Dense>
#include "doa_config.h"
#include "types.h"

/**
 * @brief 参数空间处理类
 */
class ParameterSpace
{
public:
    /**
     * @brief 构造函数
     * @param config 系统配置
     */
    explicit ParameterSpace();

    /**
     * @brief 将角度模糊曲线投影到参数空间
     * @param peakAngles 峰值角度数组
     * @param timepoints 时间点数组
     * @param elevationStart 俯仰角起始值
     * @param elevationEnd 俯仰角结束值
     * @param azimuthStart 方位角起始值
     * @param azimuthEnd 方位角结束值
     * @param precision 精度
     * @return 参数空间累加器
     */
    RealMatrix projectToParameterSpace(
        const std::vector<double> &peakAngles,
        const std::vector<double> &timepoints,
        double elevationStart,
        double elevationEnd,
        double azimuthStart,
        double azimuthEnd,
        double precision) const;

    /**
     * @brief 寻找参数空间中的峰值
     * @param accumulator 参数空间累加器
     * @param threshold 阈值
     * @return 峰值索引对数组
     */
    std::vector<std::pair<int, int>> findPeaks(
        const RealMatrix &accumulator,
        double threshold) const;

private:
    DOAConfig &config_ = DOAConfig::getInstance();

    /**
     * @brief CPU实现的参数空间投影
     */
    RealMatrix projectToParameterSpaceCPU(
        const std::vector<double> &peakAngles,
        const std::vector<double> &timepoints,
        double elevationStart,
        double elevationEnd,
        double azimuthStart,
        double azimuthEnd,
        double precision) const;

    /**
     * @brief CUDA实现的参数空间投影
     */
    RealMatrix projectToParameterSpaceCUDA(
        const std::vector<double> &peakAngles,
        const std::vector<double> &timepoints,
        double elevationStart,
        double elevationEnd,
        double azimuthStart,
        double azimuthEnd,
        double precision) const;
};