// src/cpp/doa_system.cpp
#include "doa_system.h"
#include <iostream>

DOASystem::DOASystem()
    : phaseMatrix_(std::make_unique<PhaseMatrix>()),
      arrayProcessor_(std::make_unique<ArrayProcessor>()),
      musicAlgorithm_(std::make_unique<MusicAlgorithm>()),
      parameterSpace_(std::make_unique<ParameterSpace>())
{

    std::cout << "初始化DOA系统..." << std::endl;
    if (DOAConfig::getInstance().useGPU)
    {
        std::cout << "使用GPU加速" << std::endl;
    }
    else
    {
        std::cout << "使用CPU计算" << std::endl;
    }
}

ComplexMatrix DOASystem::generateSimulationData(
    const std::vector<double> &elevations,
    const std::vector<double> &azimuths,
    double duration,
    double snrDb) const
{

    return arrayProcessor_->generateSimulationData(elevations, azimuths, duration, snrDb);
}

DOAResult DOASystem::estimateDOA(
    const ComplexMatrix &signalData,
    int numSignals) const
{

    std::vector<double> musicPeakAngles;
    std::vector<double> musicTimepoints;
    int musicFrameSize = config_.samplingRate / config_.estimateRate;

    // 确保帧大小合理
    if (musicFrameSize <= 0)
    {
        throw std::runtime_error("MUSIC帧大小必须为正数");
    }

    std::cout << "开始DOA估计..." << std::endl;

    // 对每个时间点进行MUSIC算法角度估计
    for (int i = 0; i < config_.estimateRate; ++i)
    {
        // 计算当前时间
        double currentTime = static_cast<double>(i) / config_.estimateRate;

        // 提取当前时间窗口的数据
        int startCol = i * musicFrameSize;
        int endCol = std::min(startCol + musicFrameSize, static_cast<int>(signalData.cols()));

        if (startCol >= signalData.cols())
        {
            break;
        }

        int actualFrameSize = endCol - startCol;
        ComplexMatrix dataMatrix = signalData.block(0, startCol, config_.nElements, actualFrameSize);

        // 计算噪声子空间投影矩阵
        ComplexMatrix noiseSubspaceProduct = arrayProcessor_->computeNoiseSubspace(dataMatrix, numSignals);

        // 执行MUSIC算法
        auto [spectrum, angles] = musicAlgorithm_->computeSpectrum(
            noiseSubspaceProduct, -90.0, 90.0, config_.thetaPrecision);

        // 找到谱峰对应的角度
        std::vector<int> peaks = musicAlgorithm_->findPeaks(spectrum, config_.spectrumThreshold);

        // 记录峰值角度和时间点
        for (int peak : peaks)
        {
            musicPeakAngles.push_back(angles[peak]);
            musicTimepoints.push_back(currentTime);
        }
    }

    std::cout << "检测到 " << musicPeakAngles.size() << " 个峰值" << std::endl;

    // 如果没有检测到足够的峰值，可能需要调整参数
    if (musicPeakAngles.size() < 10)
    {
        std::cout << "警告：检测到的峰值数量较少，可能需要调整参数" << std::endl;
    }

    // 参数空间投影
    double elevationStart = 0.0; // 论文中俯仰角范围为0-90度
    double elevationEnd = 90.0;
    double azimuthStart = 0.0;
    double azimuthEnd = 360.0;

    Eigen::MatrixXd accumulator = parameterSpace_->projectToParameterSpace(
        musicPeakAngles, musicTimepoints,
        elevationStart, elevationEnd,
        azimuthStart, azimuthEnd,
        config_.precision);

    // 寻找参数空间中的峰值
    auto peaks2D = parameterSpace_->findPeaks(accumulator, config_.accumulatorThreshold);

    // 转换索引到实际角度
    DOAResult result;
    result.accumulator = accumulator;
    result.timepoint = 1.0; // 使用1秒作为默认时间点
    for (int i = 0; i < accumulator.rows(); ++i)
    {
        result.elevations.push_back(elevationStart + i * config_.precision);
    }
    for (int j = 0; j < accumulator.cols(); ++j)
    {
        result.azimuths.push_back(azimuthStart + j * config_.precision);
    }
    result.angles = musicPeakAngles;
    result.anglesTimepoints = musicTimepoints;

    for (const auto &peak : peaks2D)
    {
        double elevation = elevationStart + peak.first * config_.precision;
        double azimuth = azimuthStart + peak.second * config_.precision;
        result.estElevations.push_back(elevation);
        result.estAzimuths.push_back(azimuth);
    }

    return result;
}