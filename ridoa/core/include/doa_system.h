// include/doa_system.h
#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "doa_config.h"
#include "doa_result.h"
#include "phase_matrix.h"
#include "array_processing.h"
#include "music_algorithm.h"
#include "parameter_space.h"

using ComplexMatrix = Eigen::MatrixXcd;

/**
 * @brief 旋转干涉仪DOA估计系统主类
 */
class DOASystem
{
public:
    /**
     * @brief 构造函数
     * @param config 系统配置
     */
    explicit DOASystem();

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
     * @brief 执行DOA估计
     * @param signalData 信号数据矩阵
     * @param numSignals 信号源数量，-1表示自动估计
     * @return DOA估计结果
     */
    DOAResult estimateDOA(
        const ComplexMatrix &signalData,
        int numSignals = -1) const;

private:
    DOAConfig &config_ = DOAConfig::getInstance();
    std::unique_ptr<PhaseMatrix> phaseMatrix_;
    std::unique_ptr<ArrayProcessor> arrayProcessor_;
    std::unique_ptr<MusicAlgorithm> musicAlgorithm_;
    std::unique_ptr<ParameterSpace> parameterSpace_;
};