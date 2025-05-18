// include/music_algorithm.h
// 修改版本以支持HIP
#pragma once

#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include "doa_config.h"
#include "types.h"

/**
 * @brief MUSIC算法实现类
 */
class MusicAlgorithm
{
public:
    /**
     * @brief 构造函数
     */
    explicit MusicAlgorithm();

    /**
     * @brief 计算MUSIC谱
     * @param noiseSubspaceProduct 噪声子空间投影矩阵
     * @param startAngle 开始角度
     * @param endAngle 结束角度
     * @param stepAngle 步长
     * @return {谱值数组, 角度数组}
     */
    std::tuple<std::vector<double>, std::vector<double>> computeSpectrum(
        const ComplexMatrix &noiseSubspaceProduct,
        double startAngle,
        double endAngle,
        double stepAngle) const;

    /**
     * @brief 寻找一维谱中的峰值
     * @param spectrum 谱值数组
     * @param threshold 阈值
     * @return 峰值索引数组
     */
    std::vector<int> findPeaks(
        const std::vector<double> &spectrum,
        double threshold) const;

private:
    DOAConfig &config_ = DOAConfig::getInstance();

    /**
     * @brief CPU实现的MUSIC谱计算
     */
    std::tuple<std::vector<double>, std::vector<double>> computeSpectrumCPU(
        const ComplexMatrix &noiseSubspaceProduct,
        double startAngle,
        double endAngle,
        double stepAngle) const;

    /**
     * @brief CUDA实现的MUSIC谱计算
     */
    std::tuple<std::vector<double>, std::vector<double>> computeSpectrumCUDA(
        const ComplexMatrix &noiseSubspaceProduct,
        double startAngle,
        double endAngle,
        double stepAngle) const;

    /**
     * @brief HIP实现的MUSIC谱计算
     */
    std::tuple<std::vector<double>, std::vector<double>> computeSpectrumHIP(
        const ComplexMatrix &noiseSubspaceProduct,
        double startAngle,
        double endAngle,
        double stepAngle) const;
};