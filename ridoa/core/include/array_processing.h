// include/array_processing.h
#pragma once

#include <Eigen/Dense>
#include "doa_config.h"
#include "types.h"

/**
 * @brief 阵列信号处理类
 */
class ArrayProcessor
{
public:
    /**
     * @brief 构造函数
     * @param config 系统配置
     */
    explicit ArrayProcessor();

    /**
     * @brief 生成模拟信号数据
     * @param elevations 俯仰角数组(度)
     * @param azimuths 方位角数组(度)
     * @param duration 信号持续时间(秒)
     * @param snrDb 信噪比(dB)
     * @return 信号数据矩阵
     */
    ComplexMatrix generateSimulationData(
        const std::vector<double> &elevations,
        const std::vector<double> &azimuths,
        double duration,
        double snrDb) const;

    /**
     * @brief 前向空间平滑处理
     * @param dataMatrix 数据矩阵
     * @param subArraySize 子阵大小
     * @return 平滑后的协方差矩阵
     */
    ComplexMatrix forwardSpatialSmoothing(
        const ComplexMatrix &dataMatrix,
        int subArraySize) const;

    /**
     * @brief 计算噪声子空间投影矩阵
     * @param dataMatrix 数据矩阵
     * @param numSignals 信号源数量，-1表示自动估计
     * @return 噪声子空间投影矩阵
     */
    ComplexMatrix computeNoiseSubspace(
        const ComplexMatrix &dataMatrix,
        int numSignals = -1) const;

    /**
     * @brief 估计信号源数量
     * @param eigenvalues 特征值数组
     * @param threshold 能量阈值
     * @return 估计的信号源数量
     */
    int estimateNumSources(
        const Eigen::VectorXd &eigenvalues,
        double threshold = 0.95) const;

private:
    DOAConfig &config_ = DOAConfig::getInstance();
};