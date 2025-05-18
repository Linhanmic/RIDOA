// include/phase_matrix.h
#pragma once

#include <vector>
#include <complex>
#include <Eigen/Dense>
#include "doa_config.h"
#include "types.h"

/**
 * @brief 相位差矩阵计算类
 */
class PhaseMatrix
{
public:
    /**
     * @brief 构造函数
     */
    explicit PhaseMatrix();

    /**
     * @brief 计算旋转干涉仪阵元间相位差矩阵
     * @param time 时间
     * @param elevations 俯仰角数组(度)
     * @param azimuths 方位角数组(度)
     * @return 相位差矩阵
     */
    ComplexMatrix compute(double time,
                          const std::vector<double> &elevations,
                          const std::vector<double> &azimuths) const;

private:
    DOAConfig &config_ = DOAConfig::getInstance();
};